//! 流式缓冲层：无锁 SPSC 环形缓冲 + 分块重叠窗口（OLA）调度器。
//!
//! ## 1.0 的复读 bug 就出在这里
//! 1.0 的窗口锚在 `history` 的**尾部**（`len - window`），却从**头部** `drain(block)`。
//! drain 之后 `base += B`、`len -= B`，窗口末端 `base + len` **完全不变** ⇒
//! 内层循环每轮拿到的是逐样本相同的窗口 ⇒ 同一块音频被重复输出 k 次，
//! 同时 (k-1)·B 个真实输入被 drain 掉却从未渲染。k≥2 就能听见复读，
//! 而 k≥2 的条件仅仅是「上一批的墙钟时间超过约一个 chunk」（推理慢一点、GPU 抖一下就够）。
//!
//! 这一版：窗口锚在 `history` 的**头部**，渲染完 `drain(block)`，窗口每轮严格前进 block ⇒
//! ① 每块只输出一次 ② 输出严格按时间顺序 ③ OLA 的 tail / head 落在同一段绝对时间上，
//! 交叉淡化才第一次真正成立。
//!
//! ## 延迟模型
//! 输出块覆盖窗口内 `[left-crossfade, left+block-crossfade)`，窗口末端永远是「最新输入」，
//! 所以端到端算法延迟 ≈ `lookahead + block + crossfade + jitter`，**与 left_context 无关**——
//! 左上下文只提升质量并增加算力开销，不再像 1.0 那样和 lookahead 共用一个 `extra_ms` 旋钮。
//!
//! ## 过载保护
//! τ ≥ block 时长时积压只增不减。`max_latency_ms` 给积压封顶：超限就丢弃最旧音频、
//! 复位 OLA、重灌 jitter 缓冲垫。结果是「偶发一次可听见的短断流」，
//! 而不是 1.0 的「无限延迟 + 整句复读 + history 无界增长」。

use std::cell::UnsafeCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;

use crate::rvc::{F0Extractor, HubertExtractor, RvcSynth, HOP_48K};
use crate::{config, logger, rvc};

// ────────────────────────── 无锁 SPSC 环形缓冲 ──────────────────────────
pub struct AudioRingBuffer {
    buffer: UnsafeCell<Vec<f32>>,
    capacity: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
}

// SAFETY: 见模块级文档。跨线程共享的只有两个原子游标，数据区被游标切分成
// “只写段”和“只读段”，不存在对同一元素的同时读写。
unsafe impl Send for AudioRingBuffer {}
unsafe impl Sync for AudioRingBuffer {}

impl AudioRingBuffer {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity >= 2, "ring buffer capacity must be >= 2");
        Self {
            buffer: UnsafeCell::new(vec![0.0; capacity]),
            capacity,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
        }
    }

    /// 生产者可写入的剩余空间（样本数）。
    #[inline]
    pub fn free(&self) -> usize {
        let w = self.write_pos.load(Ordering::Acquire);
        let r = self.read_pos.load(Ordering::Acquire);
        if w >= r {
            self.capacity - (w - r) - 1
        } else {
            r - w - 1
        }
    }

    /// 消费者可读的数据量（样本数）。
    #[inline]
    pub fn available(&self) -> usize {
        let w = self.write_pos.load(Ordering::Acquire);
        let r = self.read_pos.load(Ordering::Acquire);
        if w >= r {
            w - r
        } else {
            self.capacity - r + w
        }
    }

    /// 写入 `data`。空间不足时**一个样本都不写**，返回 0（调用方据此统计丢弃量）。
    pub fn push(&self, data: &[f32]) -> usize {
        if data.is_empty() {
            return 0;
        }
        if data.len() > self.free() {
            return 0;
        }
        let w = self.write_pos.load(Ordering::Relaxed);
        // SAFETY: 上面已确认 [w, w+len) 这段（模 capacity）不与读区间重叠，
        // 且只有生产者会写这段。
        let buffer = unsafe { &mut *self.buffer.get() };
        let end = w + data.len();
        if end <= self.capacity {
            buffer[w..end].copy_from_slice(data);
        } else {
            let first = self.capacity - w;
            buffer[w..self.capacity].copy_from_slice(&data[..first]);
            buffer[0..data.len() - first].copy_from_slice(&data[first..]);
        }
        self.write_pos.store(end % self.capacity, Ordering::Release);
        data.len()
    }

    /// 写入 `n` 个静音样本（用于给输出环预灌缓冲垫）。空间不足则写不进去，返回实际写入量。
    pub fn push_silence(&self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let room = self.free().min(n);
        if room == 0 {
            return 0;
        }
        let w = self.write_pos.load(Ordering::Relaxed);
        // SAFETY: 同 push。
        let buffer = unsafe { &mut *self.buffer.get() };
        let end = w + room;
        if end <= self.capacity {
            for s in buffer[w..end].iter_mut() {
                *s = 0.0;
            }
        } else {
            for s in buffer[w..self.capacity].iter_mut() {
                *s = 0.0;
            }
            for s in buffer[0..end - self.capacity].iter_mut() {
                *s = 0.0;
            }
        }
        self.write_pos.store(end % self.capacity, Ordering::Release);
        room
    }

    /// 读出至多 `out.len()` 个样本，返回实际读出的数量（可能小于 out.len()）。
    pub fn pop(&self, out: &mut [f32]) -> usize {
        let n = out.len().min(self.available());
        if n == 0 {
            return 0;
        }
        let r = self.read_pos.load(Ordering::Relaxed);
        // SAFETY: [r, r+n) 已由生产者在 write_pos 上 Release 发布，只有消费者会读它。
        let buffer = unsafe { &*self.buffer.get() };
        let end = r + n;
        if end <= self.capacity {
            out[..n].copy_from_slice(&buffer[r..end]);
        } else {
            let first = self.capacity - r;
            out[..first].copy_from_slice(&buffer[r..self.capacity]);
            out[first..n].copy_from_slice(&buffer[0..n - first]);
        }
        self.read_pos.store(end % self.capacity, Ordering::Release);
        n
    }

    /// 消费者直接丢弃 `n` 个样本（用于主动丢弃积压、把延迟拉回目标值）。
    pub fn discard(&self, n: usize) -> usize {
        let drop_n = n.min(self.available());
        if drop_n == 0 {
            return 0;
        }
        let r = self.read_pos.load(Ordering::Relaxed);
        self.read_pos.store((r + drop_n) % self.capacity, Ordering::Release);
        drop_n
    }
}


// ────────────────────────── 流式调度器 ──────────────────────────
#[derive(Debug, Clone, PartialEq)]
pub struct Status {
    pub phase: Phase,
    /// 例如 `user_models/my_voice.onnx (用户模型)`
    pub model_label: String,
    /// 加载失败原因、CUDA 回退提示等
    pub detail: String,
    pub avg_tau_ms: f32,
    pub max_tau_ms: f32,
    pub blocks: u64,
    pub flushes: u32,
    /// 每次状态变化 +1，宿主侧据此判断要不要通知用户
    pub revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Loading,
    Ready,
    Error,
}

impl Default for Status {
    fn default() -> Self {
        Self {
            phase: Phase::Loading,
            model_label: String::new(),
            detail: String::new(),
            avg_tau_ms: 0.0,
            max_tau_ms: 0.0,
            blocks: 0,
            flushes: 0,
            revision: 0,
        }
    }
}

pub type SharedStatus = Arc<Mutex<Status>>;

pub fn new_status() -> SharedStatus {
    Arc::new(Mutex::new(Status::default()))
}

fn publish(status: &SharedStatus, f: impl FnOnce(&mut Status)) {
    if let Ok(mut s) = status.lock() {
        f(&mut s);
        s.revision += 1;
    }
}

// ─────────────────────────── 引擎（三个会话）───────────────────────────

struct Engine {
    hubert: HubertExtractor,
    f0: F0Extractor,
    rvc: RvcSynth,
    set: rvc::ModelSet,
    /// 加载期探测结果：模型是否写死了帧数、实测 hop 是多少
    shape: rvc::ModelShape,
}

/// 模型写死了帧数时，把窗口撑到 `frames * 480`。
///
/// 多出来的长度**全部塞进 left context**，`block` 与 `lookahead` 一动不动 ⇒
/// 端到端延迟（≈ lookahead + block + crossfade）完全不变，只是每块的算力变多。
/// 这是能同时满足「用得上这个模型」和「延迟不爆炸」的唯一解。
fn fit_fixed_frames(mut g: Geometry, shape: rvc::ModelShape) -> Geometry {
    let frames = match shape {
        rvc::ModelShape::Fixed { frames, .. } => frames,
        _ => return g,
    };
    let need = frames * HOP_48K;
    if need >= g.block + g.lookahead {
        g.left = need - g.block - g.lookahead;
    } else {
        // 模型要求的窗口比 chunk+lookahead 还短：只能压缩这两者，延迟会跟着降
        g.lookahead = (need / 3).max(HOP_48K);
        g.block = need.saturating_sub(g.lookahead).max(HOP_48K);
        g.left = need.saturating_sub(g.block + g.lookahead);
    }
    g.window = g.left + g.block + g.lookahead;
    g.frames = g.window / HOP_48K;
    g.crossfade = g.crossfade.min(g.block / 2).min(g.lookahead).min(g.left);
    g
}

impl Engine {
    /// 用已经解析好的 `rvc::ModelSet` 建三个会话。
    /// 分开 discover / load 是为了在「配置变了但解析出的模型路径没变」时跳过重载
    /// （重载要 1~3 秒，期间输出会断流，不能白断）。
    fn load_from(
        set: rvc::ModelSet,
        plugin_dir: &PathBuf,
        frames_hint: usize,
        status: &SharedStatus,
    ) -> anyhow::Result<Self> {
        publish(status, |s| {
            s.phase = Phase::Loading;
            s.model_label = set.rvc_label.clone();
            s.detail = "正在加载模型…".into();
        });

        init_ort_once(plugin_dir);

        let mut detail = String::new();
        let load = |path: &PathBuf, tag: &str, detail: &mut String| -> anyhow::Result<ort::session::Session> {
            match rvc::load_session(&path.to_string_lossy(), false) {
                Ok(s) => {
                    Ok(s)
                }
                Err(e) => {
                    logger::log(&format!("[Model] {tag} CUDA EP 加载失败: {e}"));
                    logger::log(&format!("[Model] {tag} 回退 CPU（会明显变慢，请检查 libs/ 里的 CUDA 运行库）"));
                    *detail = format!("{tag}: CUDA 不可用，已回退 CPU（推理会明显变慢）");
                    let s = rvc::load_session(&path.to_string_lossy(), true).map_err(|e2| {
                        anyhow::anyhow!("{tag} 加载失败 {}\n  CPU 回退也失败: {e2}", path.display())
                    })?;
                    Ok(s)
                }
            }
        };

        let hubert = HubertExtractor::new(load(&set.hubert, "HuBERT", &mut detail)?);
        let f0 = F0Extractor::new(load(&set.rmvpe, "RMVPE", &mut detail)?);
        let mut rvc = RvcSynth::new(load(&set.rvc, "RVC", &mut detail)?);

        // ── 加载期探测：这个模型能吃哪些帧数？hop 是多少（=> 采样率）？──
        let shape = rvc.probe(frames_hint);
        match shape {
            rvc::ModelShape::Dynamic { hop } => logger::log(&format!(
                "[RVC] 探测: 帧数动态 ✓ 可用任意 chunk；hop={hop} => 模型采样率 {}Hz{}",
                hop * 100,
                if hop == HOP_48K { "" } else { "（≠48000，输出会自动重采样）" }
            )),
            rvc::ModelShape::Fixed { frames, hop } => logger::log(&format!(
                "[RVC] 探测: ⚠️ 该模型导出时写死了序列长度，只有 {frames} 帧能用（= {:.0}ms @48k）。\n\
                 \x20      插件会把窗口自动撑到 {frames} 帧：多出来的部分全部放进 left context，\n\
                 \x20      chunk 与 lookahead 不变 ⇒ 延迟不受影响，但每块算力会增加。\n\
                 \x20      hop={hop} => 模型采样率 {}Hz。想要更省算力，请用 dynamic axes 重新导出模型。",
                frames as f32 * 10.0,
                hop * 100
            )),
            rvc::ModelShape::Unknown { hop } => logger::log(&format!(
                "[RVC] 探测: ⚠️ 所有候选帧数都失败了（hop 假定 {hop}）。仍会继续尝试真实推理，\n\
                 \x20      每块的失败原因会记在日志里。多半是输入签名不兼容，请把 [RVC] 输入 ... 那几行发给开发者。"
            )),
        }

        publish(status, |s| {
            s.phase = Phase::Ready;
            s.model_label = set.rvc_label.clone();
            s.detail = detail.clone();
            s.avg_tau_ms = 0.0;
            s.max_tau_ms = 0.0;
            s.blocks = 0;
            s.flushes = 0;
        });
        logger::log(&format!("[Model] 全部就绪，RVC = {}{}", set.rvc_label,
            if detail.is_empty() { String::new() } else { format!(" | {detail}") }));
        if set.rvc_source == rvc::Source::Bundled {
            logger::log("[Model] 未发现用户模型，正在使用插件自带模型。想用自己的音色：把 .onnx 放进 user_models/，或在设置的 model_file 里填相对路径");
        }
        Ok(Self { hubert, f0, rvc, set, shape })
    }
}

/// ORT 环境只能初始化一次（进程级单例）。
fn init_ort_once(plugin_dir: &PathBuf) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        rvc::prepare_runtime(plugin_dir);
        match rvc::ort_dylib_path(plugin_dir) {
            // 显式绝对路径加载：失败会返回 Err 并带原因，不再依赖运行期 set_var("ORT_DYLIB_PATH")
            // （在多线程宿主里改进程环境变量是 UB 风险，Rust 2024 已把 set_var 标为 unsafe）
            Some(path) => match ort::init_from(&path) {
                Ok(builder) => {
                    builder.with_name("mambo-rvc").commit();
                    logger::log(&format!("[ORT] loaded {}", path.display()));
                }
                Err(e) => {
                    logger::log(&format!("[ORT] init_from({}) 失败: {e}；改用默认查找", path.display()));
                    ort::init().with_name("mambo-rvc").commit();
                }
            },
            None => {
                logger::log("[ORT] 未找到 libs/onnxruntime.*，改用默认查找（ORT_DYLIB_PATH / 系统路径）");
                ort::init().with_name("mambo-rvc").commit();
            }
        }
    });
}

// ─────────────────────────── 窗口几何 ───────────────────────────

/// 所有时长都对齐到 10ms（= 480 样本 = RVC 的一帧），否则
/// `window/480` 与 `window/3` 会取整不一致，导致每块有几毫秒的时间轴漂移并逐块累积。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Geometry {
    left: usize,
    block: usize,
    lookahead: usize,
    crossfade: usize,
    window: usize,
    frames: usize,
}

impl Geometry {
    fn new(p: &config::Params, sr: usize) -> Self {
        let snap = |ms: u32| -> usize { ((ms + 5) / 10 * 10) as usize * sr / 1000 };
        let mut left = snap(p.left_context_ms);
        let block = snap(p.chunk_ms).max(HOP_48K);
        let lookahead = snap(p.lookahead_ms);
        let crossfade = snap(p.crossfade_ms)
            .min(block / 2)
            .min(lookahead)
            .min(left);

        // HuBERT 的特征提取器每 320 个 16kHz 样本出一帧，而 mask 长度是另一套取整，
        // 两者差 1 就会在 self_attn 的 Where 节点上报「Attempting to broadcast ... 27 by 28」。
        // 窗口对应的 16k 长度 = window/3，要它是 320 的整数倍 <=> window 是 960 的整数倍。
        // 补齐的量全部加到 left 上 ⇒ block 与 lookahead 不动 ⇒ **端到端延迟完全不变**。
        let rem = (left + block + lookahead) % (HOP_48K * 2);
        if rem != 0 {
            left += HOP_48K * 2 - rem;
        }
        let window = left + block + lookahead;
        let crossfade = crossfade.min(left);
        Self {
            left,
            block,
            lookahead,
            crossfade,
            window,
            frames: window / HOP_48K,
        }
    }

    /// 输出块在窗口内的起点（含交叉淡化区）
    #[inline]
    fn out_start(&self) -> usize {
        self.left - self.crossfade
    }
    /// 交给下一块做交叉淡化的尾巴在窗口内的起点
    #[inline]
    fn tail_start(&self) -> usize {
        self.left + self.block - self.crossfade
    }
}

// ─────────────────────────── 主循环 ───────────────────────────

pub struct WorkerHandles {
    pub input_rb: Arc<AudioRingBuffer>,
    pub output_rb: Arc<AudioRingBuffer>,
    pub signal_rx: Receiver<()>,
    pub is_running: Arc<AtomicBool>,
    /// 音频线程丢过输入（环溢出）时 +1，worker 据此复位 OLA，避免跨空洞交叉淡化
    pub resync: Arc<AtomicU64>,
    /// 置位后由**音频线程**清空输出环（模型热重载后不要继续播旧音色的残留音频）。
    /// output_rb 的 read_pos 只能由它的消费者动，所以 worker 只能请求、不能自己动手。
    pub flush_out: Arc<AtomicBool>,
    pub status: SharedStatus,
}

pub fn worker_loop(sample_rate: usize, h: WorkerHandles) {
    // 兜底：绝不让 panic 穿过 FFI 边界，也绝不让 worker 线程因为一次 panic 就永久死掉
    // （1.0 的行为是线程退出 ⇒ 插件从此只输出静音，且日志里只有一行 PANIC）。
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        worker_main(sample_rate, h);
    }));
    if let Err(e) = result {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".into()
        };
        logger::log(&format!("[Worker] PANIC，线程退出: {msg}"));
    }
}

fn worker_main(sample_rate: usize, h: WorkerHandles) {
    logger::log("[Worker] 线程启动");
    let Some(plugin_dir) = rvc::plugin_dir() else {
        logger::log("[Worker] FATAL: 无法定位插件目录");
        publish(&h.status, |s| {
            s.phase = Phase::Error;
            s.detail = "无法定位插件目录".into();
        });
        return;
    };

    let mut engine: Option<Engine> = None;
    let mut loaded_epoch: u32 = u32::MAX;
    let mut next_retry = Instant::now();

    // 流状态（跨批次保留）
    let mut history: Vec<f32> = Vec::with_capacity(sample_rate * 2);
    let mut frame_buf = vec![0.0f32; 480];
    let mut win16k: Vec<f32> = Vec::new();
    let mut rendered: Vec<f32> = Vec::new();
    let mut out_chunk: Vec<f32> = Vec::new();
    let mut geom = Geometry::new(&config::Params::load(), sample_rate);
    // ★ 淡化曲线必须在构造时就按 crossfade 建好。
    //   v1.1 的第一版把它放在「几何参数变化」的分支里，而 geom 的初值就是用同一份参数算的，
    //   首次进循环时 new_geom == geom ⇒ 曲线永远是空 Vec ⇒ 第三块（第一次真正走混合分支）
    //   直接 panic: index out of bounds: the len is 0 but the index is 0，worker 线程当场死掉。
    //   现在曲线长度由 Ola 的构造函数保证，类型层面就没法漏。
    let mut ola = Ola::new(geom.crossfade);
    let mut need_prime = true;
    let mut last_resync = h.resync.load(Ordering::Relaxed);

    // 统计
    let mut tau_sum = Duration::ZERO;
    let mut tau_max = Duration::ZERO;
    let mut tau_n = 0u32;
    let mut blocks_since_log = 0u32;
    let mut slow_warnings = 0u32;

    while h.is_running.load(Ordering::Relaxed) {
        // ── 模型加载 / 热重载 ──
        let epoch = config::model_epoch();
        let want_reload = (engine.is_none() && Instant::now() >= next_retry)
            || (engine.is_some() && epoch != loaded_epoch);
        if want_reload {
            let want = rvc::discover(&plugin_dir, &config::model_file());
            let unchanged = engine
                .as_ref()
                .map(|e| {
                    e.set.rvc == want.rvc && e.set.hubert == want.hubert && e.set.rmvpe == want.rmvpe
                })
                .unwrap_or(false);
            if unchanged {
                logger::log("[Model] 配置变了但解析出的模型路径没变，跳过热重载");
                loaded_epoch = epoch;
            } else {
                let frames_hint = Geometry::new(&config::Params::load(), sample_rate).frames;
                match Engine::load_from(want, &plugin_dir, frames_hint, &h.status) {
                    Ok(e) => {
                        engine = Some(e);
                        loaded_epoch = epoch;
                        history.clear();
                        ola.invalidate();
                        need_prime = true;
                        // 换模型了：别让旧音色的残留音频继续播出去
                        h.flush_out.store(true, Ordering::Release);
                    }
                    Err(e) => {
                        let msg = format!("{e:#}");
                        logger::log(&format!("[Worker] 模型加载失败: {msg}"));
                        publish(&h.status, |s| {
                            s.phase = Phase::Error;
                            s.detail = msg.chars().take(300).collect();
                        });
                        engine = None;
                        loaded_epoch = epoch;
                        next_retry = Instant::now() + Duration::from_secs(15);
                    }
                }
            }
        }
        let Some(engine) = engine.as_mut() else {
            // 没有可用模型：不烧 CPU，等退出或用户改配置（下一轮循环会按 next_retry 重试）
            let _ = h.signal_rx.recv_timeout(Duration::from_millis(250));
            continue;
        };

        // ── 等一个新批次（超时也返回，以便响应退出与热重载）──
        match h.signal_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(()) => {}
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
        if !h.is_running.load(Ordering::Relaxed) {
            break;
        }

        // ── 抽干输入环 ──
        while h.input_rb.available() >= frame_buf.len() {
            let n = h.input_rb.pop(&mut frame_buf);
            if n == 0 {
                break;
            }
            history.extend_from_slice(&frame_buf[..n]);
        }
        // 让多余的排空信号不要堆积成一串空批次
        while h.signal_rx.try_recv().is_ok() {}

        // ── 参数快照（每批一次，实时线程不参与）──
        let params = config::Params::load();
        let new_geom = fit_fixed_frames(Geometry::new(&params, sample_rate), engine.shape);
        if new_geom != geom {
            logger::log(&format!(
                "[Geom] 窗口变更: left={}ms block={}ms lookahead={}ms crossfade={}ms (window={}ms, 延迟≈{}ms, 算力≈{:.1}x)",
                new_geom.left * 1000 / sample_rate,
                new_geom.block * 1000 / sample_rate,
                new_geom.lookahead * 1000 / sample_rate,
                new_geom.crossfade * 1000 / sample_rate,
                new_geom.window * 1000 / sample_rate,
                (new_geom.block + new_geom.lookahead + new_geom.crossfade) * 1000 / sample_rate,
                new_geom.window as f32 / new_geom.block as f32,
            ));
            geom = new_geom;
            // 窗口长度变了，旧 history 的语义也变了：只保留最新一个窗口，避免错位
            if history.len() > geom.window {
                let drop = history.len() - geom.window;
                history.drain(..drop);
            }
            ola.invalidate();
            ola.set_crossfade(geom.crossfade);
            need_prime = true;
            h.flush_out.store(true, Ordering::Release);
        }

        // ── 音频线程丢过输入 ⇒ history 里有空洞，OLA 必须断开 ──
        let resync = h.resync.load(Ordering::Relaxed);
        if resync != last_resync {
            last_resync = resync;
            ola.invalidate();
            need_prime = true;
            logger::log("[Worker] 输入环溢出，已复位 OLA（时间轴出现空洞）");
        }

        if geom.window == 0 || geom.block == 0 || geom.frames == 0 {
            continue;
        }

        // ── 延迟上限：积压超限就丢最旧的音频，把延迟拉回目标 ──
        let backlog_limit = geom.window + (params.max_latency_ms as usize * sample_rate / 1000);
        if history.len() > backlog_limit {
            let keep = geom.window.saturating_sub(geom.block);
            let dropped = history.len() - keep;
            history.drain(..dropped);
            ola.invalidate();
            need_prime = true;
            if let Ok(mut s) = h.status.lock() {
                s.flushes += 1;
                s.revision += 1;
            }
            logger::log(&format!(
                "[Flush] 积压 {}ms 超过上限 {}ms，丢弃 {}ms 最旧音频（说明单块推理 τ 已经大于 chunk_ms，\
                 请调大 Chunk Size 或降低 left/lookahead）",
                (dropped + keep) * 1000 / sample_rate,
                params.max_latency_ms,
                dropped * 1000 / sample_rate
            ));
        }

        // ── 逐块渲染 ──
        ola.set_crossfade(geom.crossfade);
        let gate_rms = params.gate_rms();
        let jitter = (params.jitter_ms as usize + 5) / 10 * 10 * sample_rate / 1000;
        win16k.clear();
        win16k.resize(geom.window / 3, 0.0);
        out_chunk.clear();
        out_chunk.resize(geom.block, 0.0);

        // 游标式推进：一批里可能要渲染 k 个窗口，用 cursor 走完后只做一次 drain。
        // （1.0 是每块 drain 一次，k 块就是 k 次 O(len) memmove，积压时白白烧 CPU。）
        let mut cursor = 0usize;
        while history.len() - cursor >= geom.window {
            let t0 = Instant::now();

            // ★ 窗口取 history 的**头部**（最旧的 window 个样本），游标每轮前进 block。
            //   这是不复读的前提：1.0 锚在尾部（len - window）却从头部 drain，
            //   base 与 len 同增同减 ⇒ 窗口末端恒定 ⇒ 每轮都是同一个窗口。
            {
                let w = &history[cursor..cursor + geom.window];
                for (i, s) in win16k.iter_mut().enumerate() {
                    *s = (w[i * 3] + w[i * 3 + 1] + w[i * 3 + 2]) / 3.0;
                }
            }

            // 门限只看**将要输出的那一段**（body），不看上下文：
            // 否则“安静的左上下文 + 响亮的 body”会被误判为静音而整块丢掉。
            let body16 = geom.left / 3..(geom.left + geom.block) / 3;
            let body_rms = {
                let seg = &win16k[body16.start.min(win16k.len())..body16.end.min(win16k.len())];
                if seg.is_empty() {
                    0.0
                } else {
                    (seg.iter().map(|x| x * x).sum::<f32>() / seg.len() as f32).sqrt()
                }
            };

            let chunk_result: Result<(), String> = if body_rms < gate_rms {
                // 静音块：不跑推理（省算力，也让句间停顿能自动把积压排干）
                ola.emit_silence(geom, &mut out_chunk);
                Ok(())
            } else {
                match render_window(engine, &params, geom, &win16k, &mut rendered) {
                    Ok(()) => {
                        ola.emit(geom, &rendered, &mut out_chunk);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            };

            let tau = t0.elapsed();

            match chunk_result {
                Ok(()) => {
                    if need_prime {
                        // 预灌缓冲垫：吸收推理抖动，代价是 jitter_ms 的固定延迟
                        let want = jitter.saturating_sub(h.output_rb.available());
                        if want > 0 {
                            h.output_rb.push_silence(want);
                        }
                        need_prime = false;
                    }
                    let _ = h.output_rb.push(&out_chunk);
                }
                Err(reason) => {
                    // 失败也必须照常推进游标：1.0 在这里 `continue`，跳过了 drain，
                    // 于是循环条件永远成立 ⇒ 在同一个窗口上无限重跑推理（GPU 100%、输出停摆、只能重启宿主）。
                    logger::log(&format!("[Worker] 本块推理失败，输出静音并继续: {reason}"));
                    publish(&h.status, |s| s.detail = format!("推理失败: {reason}"));
                    for s in out_chunk.iter_mut() {
                        *s = 0.0;
                    }
                    let _ = h.output_rb.push(&out_chunk);
                    ola.invalidate();
                    need_prime = true;
                }
            }

            cursor += geom.block;

            // ── 统计 ──
            tau_sum += tau;
            tau_max = tau_max.max(tau);
            tau_n += 1;
            blocks_since_log += 1;
            if let Ok(mut s) = h.status.lock() {
                s.blocks += 1;
            }
            let block_ms = geom.block as f32 * 1000.0 / sample_rate as f32;
            if tau.as_secs_f32() * 1000.0 > block_ms && slow_warnings < 8 {
                slow_warnings += 1;
                logger::log(&format!(
                    "[Perf] 警告: 单块推理 {:.0}ms > chunk {:.0}ms，会持续积压并触发断流。\
                     请调大 Chunk Size，或调小 Left Context / Lookahead",
                    tau.as_secs_f32() * 1000.0,
                    block_ms
                ));
            }
            if blocks_since_log >= 50 {
                let avg = tau_sum.as_secs_f32() * 1000.0 / tau_n as f32;
                let max = tau_max.as_secs_f32() * 1000.0;
                logger::log(&format!(
                    "[Perf] 最近 {tau_n} 块: τ 平均 {avg:.1}ms / 峰值 {max:.1}ms（chunk={block_ms:.0}ms，\
                     历史 {hist:.0}ms，输出环 {out:.0}ms）",
                    hist = history.len() as f32 * 1000.0 / sample_rate as f32,
                    out = h.output_rb.available() as f32 * 1000.0 / sample_rate as f32,
                ));
                publish(&h.status, |s| {
                    s.avg_tau_ms = avg;
                    s.max_tau_ms = max;
                });
                tau_sum = Duration::ZERO;
                tau_max = Duration::ZERO;
                tau_n = 0;
                blocks_since_log = 0;
                slow_warnings = 0;
            }
        }
        if cursor > 0 {
            history.drain(..cursor);
        }
    }
    logger::log("[Worker] 线程退出");
}

/// OLA（重叠相加）拼接状态。
///
/// 把「淡化曲线长度必须等于 crossfade」「tail 必须与下一块的 head 对齐」这两条不变量
/// 收进构造函数与 `set_crossfade`，避免出现「忘了初始化就索引」这类只在第三块才炸的 bug。
struct Ola {
    cf: usize,
    /// 上一块渲染结果里、紧跟在本块 head 之前的那 crossfade 个样本
    tail: Vec<f32>,
    fade_in: Vec<f32>,
    fade_out: Vec<f32>,
    /// tail 是否真的对应「上一块的紧邻后续」。任何时间轴断裂（积压 flush / 换模型 /
    /// 输入环溢出 / 参数变化 / 推理失败）都必须置 false，否则会把两段无关音频混在一起。
    valid: bool,
}

impl Ola {
    fn new(cf: usize) -> Self {
        let mut o = Self { cf: 0, tail: Vec::new(), fade_in: Vec::new(), fade_out: Vec::new(), valid: false };
        o.set_crossfade(cf);
        o
    }

    /// 改交叉淡化长度。长度没变就什么都不做（每批都会调，必须幂等且廉价）。
    fn set_crossfade(&mut self, cf: usize) {
        if cf == self.cf && self.fade_in.len() == cf && self.tail.len() == cf {
            return;
        }
        self.cf = cf;
        // raised-cosine **幅度**淡化：fade_in + fade_out ≡ 1。
        // 这里刻意不用等功率（sin/cos）：tail 与 head 是**同一段绝对时间**在两个不同上下文窗口下的
        // 两次渲染，高度相关；相关信号用幅度互补才能保持电平恒定，等功率会在中点抬高 +3dB。
        self.fade_in = (0..cf)
            .map(|i| 0.5 * (1.0 - (std::f32::consts::PI * i as f32 / cf.max(1) as f32).cos()))
            .collect();
        self.fade_out = self.fade_in.iter().map(|v| 1.0 - v).collect();
        self.tail = vec![0.0; cf];
        self.valid = false;
    }

    /// 时间轴断裂：丢掉过期尾巴，下一块直接用本窗口的 head，不与旧音频混合。
    fn invalidate(&mut self) {
        self.valid = false;
        for s in self.tail.iter_mut() {
            *s = 0.0;
        }
    }

    /// 静音块：把上一块的尾巴淡出，其余填 0。
    fn emit_silence(&mut self, geom: Geometry, out: &mut [f32]) {
        for s in out.iter_mut() {
            *s = 0.0;
        }
        let n = self.cf.min(geom.crossfade).min(out.len());
        if self.valid && self.fade_out.len() >= n && self.tail.len() >= n {
            for i in 0..n {
                out[i] = self.tail[i] * self.fade_out[i];
            }
        }
        for s in self.tail.iter_mut() {
            *s = 0.0;
        }
        self.valid = true; // 全 0 的尾巴与下一块的 head 仍然对齐
    }

    /// 正常块：`out[i] = rendered[out_start + i]`，前 cf 个样本与上一块的尾巴做等功率交叉淡化。
    ///
    /// 为什么 tail 和 head 是同一段绝对时间：上一块的 tail 覆盖
    /// `[base+left+block-cf, base+left+block)`，本块（base' = base + block）的 head 覆盖
    /// `[base'+left-cf, base'+left)` = `[base+left-cf+block, base+left+block)` ⇒ 完全重合 ✓
    /// （这正是 v1.0 做不到的：它的窗口锚在尾部，base 前进时窗口末端不动，tail/head 差了整整一个 block。）
    fn emit(&mut self, geom: Geometry, rendered: &[f32], out: &mut [f32]) {
        let cf = self.cf.min(geom.crossfade).min(out.len());
        let os = geom.out_start();
        let ts = geom.tail_start();
        if rendered.len() < os + geom.block || rendered.len() < ts + cf {
            // 理论上不会发生（调用方已校验长度）；真发生了就退化成填 0，绝不越界 panic
            for s in out.iter_mut() {
                *s = 0.0;
            }
            self.invalidate();
            return;
        }
        if cf > 0 && self.valid && self.fade_in.len() >= cf && self.tail.len() >= cf {
            for i in 0..cf {
                out[i] = self.tail[i] * self.fade_out[i] + rendered[os + i] * self.fade_in[i];
            }
        } else if cf > 0 {
            out[..cf].copy_from_slice(&rendered[os..os + cf]);
        }
        out[cf..].copy_from_slice(&rendered[os + cf..os + geom.block]);
        if self.tail.len() == cf && cf > 0 {
            self.tail.copy_from_slice(&rendered[ts..ts + cf]);
        }
        self.valid = true;
    }
}

/// 渲染一个窗口：HuBERT → RMVPE → 变调 → RVC →（必要时重采样）。
/// 结果写进 `rendered`（geom.window 个 48k 样本），OLA 拼接由 `Ola::emit` 负责。
fn render_window(
    engine: &mut Engine,
    params: &config::Params,
    geom: Geometry,
    win16k: &[f32],
    rendered: &mut Vec<f32>,
) -> Result<(), String> {
    let phone = engine
        .hubert
        .extract(win16k)
        .map_err(|e| format!("HuBERT: {e}"))?;
    if phone.len() < rvc::FEAT_DIM {
        // 1.0 在这里会 panic（frames_50 == 0 时 build_inputs 索引越界），
        // panic 被 catch_unwind 吃掉后 worker 线程直接退出 ⇒ 插件从此永久静音。
        return Err(format!("HuBERT 特征为空（模型输出维度不是 {} 的整数倍？）", rvc::FEAT_DIM));
    }

    let f0 = engine
        .f0
        .extract(win16k, geom.frames)
        .map_err(|e| format!("RMVPE: {e}"))?;

    // 变调（在 f0 域做，RVC 的 pitch/nsff0 都跟着走）
    let key = params.f0_up_key;
    let f0_shifted: Vec<f32> = if key == 0 {
        f0.to_vec()
    } else {
        let factor = 2.0f32.powf(key as f32 / 12.0);
        f0.iter().map(|&f| if f > 0.0 { f * factor } else { 0.0 }).collect()
    };

    let audio = engine
        .rvc
        .synthesize(geom.frames, params.speaker_id as i64, &f0_shifted, &phone)
        .map_err(|e| format!("RVC: {e}"))?;

    // 模型输出长度 != window ⇒ 不是 48kHz 模型（例如 40k 模型 hop=400、32k 模型 hop=320），
    // 按比例重采样到 window，避免整块被判为“输出过短”而变成静音。
    if audio.len() == geom.window {
        rendered.clear();
        rendered.extend_from_slice(&audio);
    } else {
        let hop = audio.len() as f32 / geom.frames.max(1) as f32;
        logger::log(&format!(
            "[RVC] 输出 {} 样本 ≠ 窗口 {} 样本（推断 hop={hop:.0}，模型采样率≈{:.0}Hz），已重采样到 48kHz",
            audio.len(),
            geom.window,
            hop * 100.0
        ));
        rvc::resample_into(&audio, rendered, geom.window);
    }
    if rendered.len() < geom.tail_start() + geom.crossfade {
        return Err(format!("重采样后长度不足: {} < {}", rendered.len(), geom.tail_start() + geom.crossfade));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraparound_is_lossless() {
        let rb = AudioRingBuffer::new(16);
        let mut out = [0.0f32; 8];
        for round in 0..8u32 {
            let data: Vec<f32> = (0..8).map(|i| (round * 8 + i) as f32).collect();
            assert_eq!(rb.push(&data), 8);
            assert_eq!(rb.pop(&mut out), 8);
            assert_eq!(out[..], data[..], "round {round}");
        }
    }

    #[test]
    fn full_buffer_rejects_push() {
        let rb = AudioRingBuffer::new(8);
        assert_eq!(rb.push(&[0.0; 7]), 7); // capacity-1 是上限
        assert_eq!(rb.push(&[0.0; 1]), 0);
        assert_eq!(rb.available(), 7);
        let mut out = [0.0f32; 7];
        assert_eq!(rb.pop(&mut out), 7);
        assert_eq!(rb.available(), 0);
        assert_eq!(rb.push(&[0.0; 7]), 7);
    }

    #[test]
    fn discard_skips_samples() {
        let rb = AudioRingBuffer::new(32);
        rb.push(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rb.discard(2), 2);
        let mut out = [0.0f32; 4];
        assert_eq!(rb.pop(&mut out), 2);
        assert_eq!(&out[..2], &[3.0, 4.0]);
    }

    #[test]
    fn spsc_threads_stay_consistent() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        let rb = Arc::new(AudioRingBuffer::new(1024));
        let total = 64_000usize;
        let chunk = 64usize;

        // 写线程：推送 0,1,2,... 的递增序列。
        // 关键：失败重试必须有上限，否则读线程一旦 panic，写线程会在“环满”里空转，
        // 把整个 cargo test 挂死（而不是干脆地失败）。
        let w = {
            let rb = rb.clone();
            thread::spawn(move || {
                let mut i = 0usize;
                let mut spins = 0u32;
                while i < total {
                    let data: Vec<f32> = (i..i + chunk).map(|v| v as f32).collect();
                    if rb.push(&data) == chunk {
                        i += chunk;
                        spins = 0;
                    } else {
                        spins += 1;
                        assert!(spins < 2_000_000, "writer starved at {i} (reader died?)");
                        thread::sleep(Duration::from_micros(10));
                    }
                }
            })
        };
        let r = {
            let rb = rb.clone();
            thread::spawn(move || {
                let mut out = vec![0.0f32; chunk];
                let mut got = 0usize;
                let mut spins = 0u32;
                while got < total {
                    let n = rb.pop(&mut out);
                    for k in 0..n {
                        assert_eq!(out[k], (got + k) as f32, "sample {}", got + k);
                    }
                    got += n;
                    if n == 0 {
                        spins += 1;
                        assert!(spins < 2_000_000, "reader starved at {got} (writer died?)");
                        thread::sleep(Duration::from_micros(10));
                    } else {
                        spins = 0;
                    }
                }
                got
            })
        };
        w.join().unwrap();
        assert_eq!(r.join().unwrap(), total);
    }

    // ══════════════ 1.0「复读」bug 的回归测试 ══════════════
    //
    // 用「绝对样本区间」表示每个窗口渲染出的内容落在输入时间轴的哪一段。
    // - 1.0（锚尾部 + 从头 drain）：窗口末端 base+len 恒定 ⇒ k 个窗口逐样本相同 ⇒
    //   同一块音频被输出 k 遍，且 (k-1)*block 个真实输入被 drain 掉却从未渲染；
    // - 现在（锚头部 + 游标前进）：窗口每轮严格前进 block ⇒ 不重、不漏、不倒流。

    fn test_params() -> crate::config::Params {
        crate::config::Params {
            chunk_ms: 100,
            lookahead_ms: 150,
            left_context_ms: 300,
            crossfade_ms: 20,
            max_latency_ms: 300,
            jitter_ms: 40,
            f0_up_key: 0,
            speaker_id: 0,
            gate_enabled: true,
            gate_db: -70.0,
        }
    }

    fn test_geom() -> Geometry {
        Geometry::new(&test_params(), 48_000)
    }

    /// 模拟一批内层循环，返回每块输出覆盖的绝对样本区间。
    fn window_track(anchored_at_tail: bool, base0: usize, len0: usize, g: Geometry) -> Vec<(usize, usize)> {
        let (mut base, mut len) = (base0, len0);
        let mut out = Vec::new();
        while len >= g.window {
            let start = if anchored_at_tail { base + len - g.window } else { base };
            out.push((start + g.out_start(), start + g.out_start() + g.block));
            base += g.block; // drain(..block) / cursor += block
            len -= g.block;
        }
        out
    }

    #[test]
    fn legacy_tail_anchor_repeats_the_same_window() {
        let g = test_geom();
        // 一批里积压了 6 个 block（一次 GPU 抖动就够）
        let track = window_track(true, 0, g.window + 6 * g.block, g);
        assert_eq!(track.len(), 7, "积压 6 块时应渲染 7 块");
        assert!(
            track.windows(2).all(|w| w[0] == w[1]),
            "1.0 的 7 个窗口本应逐样本相同（这正是复读的成因）: {track:?}"
        );
    }

    #[test]
    fn head_anchor_emits_each_block_exactly_once_in_order() {
        let g = test_geom();
        let track = window_track(false, 0, g.window + 6 * g.block, g);
        assert_eq!(track.len(), 7);
        for w in track.windows(2) {
            assert_eq!(w[1].0, w[0].1, "块必须首尾相接：不重叠（复读）、不留缝（丢音频）: {w:?}");
        }
        assert_eq!(track[0].0, g.out_start());
        assert_eq!(track.last().unwrap().1, g.out_start() + 7 * g.block);
    }

    #[test]
    fn backlog_size_never_reorders_output() {
        let g = test_geom();
        for blocks in [0usize, 1, 2, 5, 20, 100] {
            let track = window_track(false, 12_345, g.window + blocks * g.block, g);
            assert_eq!(track.len(), blocks + 1);
            assert!(
                track.windows(2).all(|w| w[1].0 > w[0].0),
                "积压 {blocks} 块时出现时间倒流: {track:?}"
            );
        }
    }

    #[test]
    fn geometry_is_hop_aligned_and_in_bounds() {
        let g = test_geom();
        assert_eq!(g.window, g.left + g.block + g.lookahead);
        assert_eq!(g.window % HOP_48K, 0, "窗口必须是 hop(480) 的整数倍，否则每块都会有时间轴漂移并逐块累积");
        assert_eq!(g.frames * HOP_48K, g.window);
        assert!(g.crossfade <= g.block / 2 && g.crossfade <= g.left && g.crossfade <= g.lookahead);
        assert!(g.out_start() + g.block <= g.window, "输出块越界");
        assert!(g.tail_start() + g.crossfade <= g.window, "OLA 尾巴越界");

        // 非 10ms 倍数的设置必须被吸附到 hop 的整数倍
        let p = crate::config::Params { chunk_ms: 64, lookahead_ms: 33, left_context_ms: 178, ..test_params() };
        let g2 = Geometry::new(&p, 48_000);
        assert_eq!(g2.block % HOP_48K, 0);
        assert_eq!(g2.window % HOP_48K, 0);
        // 极端配置也不能越界或除零
        let p = crate::config::Params { chunk_ms: 20, lookahead_ms: 0, left_context_ms: 0, crossfade_ms: 100, ..test_params() };
        let g3 = Geometry::new(&p, 48_000);
        assert_eq!(g3.crossfade, 0, "left/lookahead 为 0 时交叉淡化必须退化为 0（硬拼接）");
        assert!(g3.out_start() + g3.block <= g3.window);
    }

    // ══════════════ OLA 状态机 ══════════════
    //
    // v1.1 第一版把淡化曲线的构建放在「几何参数变化」分支里，而 geom 的初值就是用同一份
    // 参数算出来的 ⇒ 首次进循环 new_geom == geom ⇒ 曲线永远是空 Vec ⇒
    // 第三块（第一次真正走混合分支）panic: index out of bounds: the len is 0 but the index is 0，
    // worker 线程当场死掉，插件从此永久静音。下面这几个测试把这条不变量钉死。

    #[test]
    fn ola_new_builds_fade_curves_immediately() {
        let ola = Ola::new(960);
        assert_eq!(ola.fade_in.len(), 960, "构造时就必须建好淡化曲线（v1.1 第一版在这里是空的）");
        assert_eq!(ola.fade_out.len(), 960);
        assert_eq!(ola.tail.len(), 960);
        assert!(!ola.valid, "初始状态没有可用的尾巴");
        assert!(ola.fade_in[0].abs() < 1e-6, "淡入曲线应从 0 开始");
        assert!(ola.fade_in[959] > 0.99, "淡入曲线应升到 1");
        assert!((ola.fade_in[480] - 0.5).abs() < 0.01, "中点应为 0.5");
        for i in 0..960 {
            // 幅度互补（不是等功率）：tail 与 head 高度相关，互补才能保证电平恒定
            assert!((ola.fade_in[i] + ola.fade_out[i] - 1.0).abs() < 1e-6, "互补性 @{i}");
            if i > 0 {
                assert!(ola.fade_in[i] >= ola.fade_in[i - 1], "淡入必须单调不降 @{i}");
            }
        }
    }

    #[test]
    fn ola_crossfade_zero_is_safe() {
        let mut ola = Ola::new(0);
        assert!(ola.fade_in.is_empty());
        let g = Geometry { left: 960, block: 480, lookahead: 480, crossfade: 0, window: 1920, frames: 4 };
        let rendered: Vec<f32> = (0..1920).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; g.block];
        ola.emit(g, &rendered, &mut out);
        // crossfade=0 时应硬拼接：out == rendered[left .. left+block]
        assert_eq!(out[..], rendered[g.left..g.left + g.block]);
    }

    #[test]
    fn ola_emit_advances_exactly_one_block_and_stays_in_bounds() {
        let g = Geometry { left: 960, block: 480, lookahead: 480, crossfade: 96, window: 1920, frames: 4 };
        let mut ola = Ola::new(g.crossfade);
        // 连续 5 个「窗口」，每个都比上一个前移 block，内容用绝对下标标记
        for k in 0..5usize {
            let base = k * g.block;
            let rendered: Vec<f32> = (0..g.window).map(|i| (base + i) as f32).collect();
            let mut out = vec![0.0f32; g.block];
            ola.emit(g, &rendered, &mut out);
            // 输出块覆盖 [base+left-cf, base+left+block-cf)
            let start = base + g.out_start();
            if k == 0 {
                // 第一块 ola.valid == false => 不混合，直接取本窗口的 head
                for i in 0..g.block {
                    assert_eq!(out[i], (start + i) as f32, "块 0 位置 {i}");
                }
            } else {
                // 交叉淡化区是「上一块尾巴」与「本块 head」的混合，两者本应是同一段绝对时间，
                // 所以混合结果仍应落在这段区间内（这正好验证 tail/head 对齐）
                for i in 0..g.crossfade {
                    let v = out[i];
                    let lo = (start + i) as f32;
                    assert!((v - lo).abs() < 1e-3, "块 {k} 交叉淡化区 {i}: {v} vs {lo}");
                }
                for i in g.crossfade..g.block {
                    assert_eq!(out[i], (start + i) as f32, "块 {k} 位置 {i}");
                }
            }
            assert!(ola.valid);
        }
    }

    #[test]
    fn ola_invalidate_stops_mixing_stale_tail() {
        let g = Geometry { left: 960, block: 480, lookahead: 480, crossfade: 96, window: 1920, frames: 4 };
        let mut ola = Ola::new(g.crossfade);
        let r1: Vec<f32> = (0..g.window).map(|i| 1000.0 + i as f32).collect();
        let mut out = vec![0.0f32; g.block];
        ola.emit(g, &r1, &mut out);
        ola.invalidate();
        // 复位后不应再混入上一块（值域 1000+）的尾巴
        let r2: Vec<f32> = (0..g.window).map(|i| i as f32 * 0.001).collect();
        ola.emit(g, &r2, &mut out);
        for i in 0..g.crossfade {
            assert!(out[i] < 1.0, "复位后仍在混入过期尾巴: out[{i}] = {}", out[i]);
        }
    }

    #[test]
    fn ola_set_crossfade_is_idempotent_and_rebuilds_on_change() {
        let mut ola = Ola::new(96);
        let first = ola.fade_in.clone();
        ola.set_crossfade(96);
        assert_eq!(ola.fade_in, first, "长度没变就不该重算");
        ola.valid = true;
        ola.set_crossfade(192);
        assert_eq!(ola.fade_in.len(), 192);
        assert_eq!(ola.tail.len(), 192);
        assert!(!ola.valid, "改了交叉淡化长度必须复位，否则会拿长度不匹配的尾巴去混合");
    }
}
