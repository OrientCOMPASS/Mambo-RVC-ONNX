use std::cell::UnsafeCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;

use crate::rvc::{F0Extractor, HubertExtractor, RvcSynth, HOP_48K};
use crate::{config, logger, rvc};

pub struct AudioRingBuffer {
    buffer: UnsafeCell<Vec<f32>>,
    capacity: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
}

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

    pub fn push(&self, data: &[f32]) -> usize {
        if data.is_empty() {
            return 0;
        }
        if data.len() > self.free() {
            return 0;
        }
        let w = self.write_pos.load(Ordering::Relaxed);

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

    pub fn push_silence(&self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let room = self.free().min(n);
        if room == 0 {
            return 0;
        }
        let w = self.write_pos.load(Ordering::Relaxed);

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

    pub fn pop(&self, out: &mut [f32]) -> usize {
        let n = out.len().min(self.available());
        if n == 0 {
            return 0;
        }
        let r = self.read_pos.load(Ordering::Relaxed);

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

#[derive(Debug, Clone, PartialEq)]
pub struct Status {
    pub phase: Phase,

    pub model_label: String,

    pub detail: String,
    pub blocks: u64,
    pub flushes: u32,

    /// 三个会话是否都真实跑在 CUDA EP 上（面板展示 / rt_status 用）
    pub cuda: bool,

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
            blocks: 0,
            flushes: 0,
            cuda: false,
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

struct Engine {
    hubert: HubertExtractor,
    f0: F0Extractor,
    rvc: RvcSynth,
    set: rvc::ModelSet,

    shape: rvc::ModelShape,

    opt: u32,
}

fn fit_fixed_frames(mut g: Geometry, shape: rvc::ModelShape) -> Geometry {
    let frames = match shape {
        rvc::ModelShape::Fixed { frames, .. } => frames,
        _ => return g,
    };
    let need = frames * HOP_48K;
    if need >= g.block + g.lookahead {
        g.left = need - g.block - g.lookahead;
    } else {

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

    fn load_from(
        set: rvc::ModelSet,
        plugin_dir: &PathBuf,
        frames_hint: usize,
        opt: u32,
        status: &SharedStatus,
    ) -> anyhow::Result<Self> {
        publish(status, |s| {
            s.phase = Phase::Loading;
            s.model_label = set.rvc_label.clone();
            s.detail = "正在加载模型…".into();
        });

        init_ort_once(plugin_dir).map_err(|e| {
            logger::log(&format!("[ORT] {e}"));
            anyhow::anyhow!(e)
        })?;

        // CUDA 门控：libs/ 里 19 个运行库齐备才尝试 CUDA EP。
        // 缺库时不做无谓的 provider 加载尝试，日志与面板给出明确指引（面板可一键拉取）。
        let inv = crate::fetch::runtime_inventory(&plugin_dir.join("libs"));
        let cuda_ready = inv.cuda_ready();
        if !cuda_ready {
            logger::log(&format!(
                "[Model] CUDA 运行库未就绪（{}/{}，缺 {} 个文件），本次使用 CPU 推理；\n\
                 \x20      可在 设置 → 曼波RVC 面板「CUDA 运行库」卡片一键拉取（国内镜像），完成后自动切回 GPU",
                inv.present, inv.total, inv.missing.len()
            ));
        }

        let mut detail = String::new();
        let mut cuda_ok = cuda_ready;
        if !cuda_ready {
            detail = format!("CPU 推理（CUDA 运行库缺 {} 个，可在面板拉取）", inv.missing.len());
        }
        let load = |path: &PathBuf,
                    tag: &str,
                    detail: &mut String,
                    cuda_ok: &mut bool|
         -> anyhow::Result<ort::session::Session> {
            let p = path.to_string_lossy();
            if *cuda_ok {
                match rvc::load_session(&p, true, opt) {
                    Ok(s) => return Ok(s),
                    Err(e) => {
                        *cuda_ok = false;
                        logger::log(&format!("[Model] CUDA EP 不可用（{e}），改用 CPU；请检查显卡驱动与 libs/ 运行库"));
                        *detail = "CUDA 不可用，已回退 CPU".to_string();
                    }
                }
            }
            rvc::load_session(&p, false, opt)
                .map_err(|e2| anyhow::anyhow!("{tag} 加载失败 {}: {e2}", path.display()))
        };

        let hubert = HubertExtractor::new(load(&set.hubert, "HuBERT", &mut detail, &mut cuda_ok)?);
        let f0 = F0Extractor::new(load(&set.rmvpe, "RMVPE", &mut detail, &mut cuda_ok)?);
        let mut rvc = RvcSynth::new(load(&set.rvc, "RVC", &mut detail, &mut cuda_ok)?);

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
            s.blocks = 0;
            s.flushes = 0;
            s.cuda = cuda_ok;
        });
        logger::log(&format!("[Model] 全部就绪，RVC = {} | EP = {}{}", set.rvc_label,
            if cuda_ok { "CUDA" } else { "CPU" },
            if detail.is_empty() { String::new() } else { format!(" | {detail}") }));
        if set.rvc_source == rvc::Source::Bundled {
            logger::log("[Model] user_models/ 下没有可用模型，使用插件自带模型");
        }
        Ok(Self { hubert, f0, rvc, set, shape, opt })
    }
}

fn init_ort_once(plugin_dir: &PathBuf) -> Result<(), String> {
    // 只缓存成功结果：失败（如 libs/ 里还没有 onnxruntime.dll）必须允许 15 秒后重试，
    // 否则用户补齐文件 / 面板拉取运行库后，本次进程内永远无法恢复。
    static DONE: AtomicBool = AtomicBool::new(false);
    static LOCK: Mutex<()> = Mutex::new(());
    if DONE.load(Ordering::Acquire) {
        return Ok(());
    }
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if DONE.load(Ordering::Acquire) {
        return Ok(());
    }
    let res = (|| {
        let Some(path) = rvc::ort_dylib_path(plugin_dir) else {
            return Err(format!(
                "未找到 onnxruntime 动态库（期望在 {}）",
                plugin_dir.join("libs").display()
            ));
        };
        let builder = ort::init_from(&path)
            .map_err(|e| format!("加载 {} 失败: {e}", path.display()))?;
        builder.with_name("mambo-rvc").commit();
        logger::log(&format!("[ORT] {}", path.display()));
        rvc::prepare_runtime(plugin_dir);
        Ok(())
    })();
    if res.is_ok() {
        DONE.store(true, Ordering::Release);
    }
    res
}

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

    #[inline]
    fn out_start(&self) -> usize {
        self.left - self.crossfade
    }

    #[inline]
    fn tail_start(&self) -> usize {
        self.left + self.block - self.crossfade
    }
}

pub struct WorkerHandles {
    pub input_rb: Arc<AudioRingBuffer>,
    pub output_rb: Arc<AudioRingBuffer>,
    pub signal_rx: Receiver<()>,
    pub is_running: Arc<AtomicBool>,

    pub resync: Arc<AtomicU64>,

    pub flush_out: Arc<AtomicBool>,

    pub underruns: Arc<AtomicU64>,
    pub status: SharedStatus,
}

pub fn worker_loop(sample_rate: usize, h: WorkerHandles) {

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

    let mut history: Vec<f32> = Vec::with_capacity(sample_rate * 2);
    let mut frame_buf = vec![0.0f32; 480];
    let mut win16k: Vec<f32> = Vec::new();
    let mut rendered: Vec<f32> = Vec::new();
    let mut out_chunk: Vec<f32> = Vec::new();
    let mut geom = Geometry::new(&config::Params::load(), sample_rate);

    let mut ola = Ola::new(geom.crossfade);
    let mut need_prime = true;
    let mut last_resync = h.resync.load(Ordering::Relaxed);

    let mut last_tau_ms = 0.0f32;
    let mut slow_logged = false;
    let mut und_logged = 0u64;

    while h.is_running.load(Ordering::Relaxed) {

        let epoch = config::model_epoch();
        let want_reload = (engine.is_none() && Instant::now() >= next_retry)
            || (engine.is_some() && epoch != loaded_epoch);
        if want_reload {
            let want = rvc::discover(&plugin_dir);
            let opt = config::Params::load().ort_opt;
            let unchanged = engine
                .as_ref()
                .map(|e| {
                    e.opt == opt
                        && e.set.rvc == want.rvc
                        && e.set.hubert == want.hubert
                        && e.set.rmvpe == want.rmvpe
                })
                .unwrap_or(false);
            if unchanged {
                loaded_epoch = epoch;
            } else {
                logger::log(&format!("[Model] {}", want.describe()));
                let frames_hint = Geometry::new(&config::Params::load(), sample_rate).frames;
                match Engine::load_from(want, &plugin_dir, frames_hint, opt, &h.status) {
                    Ok(e) => {
                        engine = Some(e);
                        loaded_epoch = epoch;
                        history.clear();
                        ola.invalidate();
                        need_prime = true;

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

            let _ = h.signal_rx.recv_timeout(Duration::from_millis(250));
            continue;
        };

        match h.signal_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(()) => {}
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
        if !h.is_running.load(Ordering::Relaxed) {
            break;
        }

        while h.input_rb.available() >= frame_buf.len() {
            let n = h.input_rb.pop(&mut frame_buf);
            if n == 0 {
                break;
            }
            history.extend_from_slice(&frame_buf[..n]);
        }

        while h.signal_rx.try_recv().is_ok() {}

        let params = config::Params::load();
        let new_geom = fit_fixed_frames(Geometry::new(&params, sample_rate), engine.shape);
        if new_geom != geom {
            logger::log(&format!(
                "[Geom] window={}ms (left {} + chunk {} + lookahead {}), crossfade={}ms, {:.1}x realtime",
                new_geom.window * 1000 / sample_rate,
                new_geom.left * 1000 / sample_rate,
                new_geom.block * 1000 / sample_rate,
                new_geom.lookahead * 1000 / sample_rate,
                new_geom.crossfade * 1000 / sample_rate,
                new_geom.window as f32 / new_geom.block as f32,
            ));
            geom = new_geom;

            if history.len() > geom.window {
                let drop = history.len() - geom.window;
                history.drain(..drop);
            }
            ola.invalidate();
            ola.set_crossfade(geom.crossfade);
            need_prime = true;
            h.flush_out.store(true, Ordering::Release);
        }

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
                "[Flush] 断流：积压超过 {}ms 上限，丢弃 {}ms 最旧音频（上一块推理 {last_tau_ms:.0}ms）",
                params.max_latency_ms,
                dropped * 1000 / sample_rate
            ));
        }

        ola.set_crossfade(geom.crossfade);
        let gate_rms = params.gate_rms();
        let jitter = (params.jitter_ms as usize + 5) / 10 * 10 * sample_rate / 1000;
        win16k.clear();
        win16k.resize(geom.window / 3, 0.0);
        out_chunk.clear();
        out_chunk.resize(geom.block, 0.0);

        let mut cursor = 0usize;
        while history.len() - cursor >= geom.window {
            let t0 = Instant::now();

            {
                let w = &history[cursor..cursor + geom.window];
                for (i, s) in win16k.iter_mut().enumerate() {
                    *s = (w[i * 3] + w[i * 3 + 1] + w[i * 3 + 2]) / 3.0;
                }
            }

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

                        let want = jitter.saturating_sub(h.output_rb.available());
                        if want > 0 {
                            h.output_rb.push_silence(want);
                        }
                        need_prime = false;
                    }
                    let _ = h.output_rb.push(&out_chunk);
                }
                Err(reason) => {

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

            last_tau_ms = tau.as_secs_f32() * 1000.0;
            if let Ok(mut s) = h.status.lock() {
                s.blocks += 1;
            }
            let block_ms = geom.block as f32 * 1000.0 / sample_rate as f32;
            if !slow_logged && last_tau_ms > block_ms {
                slow_logged = true;
                logger::log(&format!(
                    "[Perf] 单块推理 {last_tau_ms:.0}ms 已超过 chunk {block_ms:.0}ms，开始积压"
                ));
            } else if slow_logged && last_tau_ms * 2.0 < block_ms {
                slow_logged = false;
            }
            let und = h.underruns.load(Ordering::Relaxed) / 20;
            if und > und_logged {
                und_logged = und;
                logger::log(&format!("[Perf] 输出欠载累计 {} 次", und * 20));
            }
        }
        if cursor > 0 {
            history.drain(..cursor);
        }
    }
    logger::log("[Worker] 线程退出");
}

struct Ola {
    cf: usize,

    tail: Vec<f32>,
    fade_in: Vec<f32>,
    fade_out: Vec<f32>,

    valid: bool,
}

impl Ola {
    fn new(cf: usize) -> Self {
        let mut o = Self { cf: 0, tail: Vec::new(), fade_in: Vec::new(), fade_out: Vec::new(), valid: false };
        o.set_crossfade(cf);
        o
    }

    fn set_crossfade(&mut self, cf: usize) {
        if cf == self.cf && self.fade_in.len() == cf && self.tail.len() == cf {
            return;
        }
        self.cf = cf;

        self.fade_in = (0..cf)
            .map(|i| 0.5 * (1.0 - (std::f32::consts::PI * i as f32 / cf.max(1) as f32).cos()))
            .collect();
        self.fade_out = self.fade_in.iter().map(|v| 1.0 - v).collect();
        self.tail = vec![0.0; cf];
        self.valid = false;
    }

    fn invalidate(&mut self) {
        self.valid = false;
        for s in self.tail.iter_mut() {
            *s = 0.0;
        }
    }

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
        self.valid = true;
    }

    fn emit(&mut self, geom: Geometry, rendered: &[f32], out: &mut [f32]) {
        let cf = self.cf.min(geom.crossfade).min(out.len());
        let os = geom.out_start();
        let ts = geom.tail_start();
        if rendered.len() < os + geom.block || rendered.len() < ts + cf {

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

        return Err(format!("HuBERT 特征为空（模型输出维度不是 {} 的整数倍？）", rvc::FEAT_DIM));
    }

    let f0 = engine
        .f0
        .extract(win16k, geom.frames)
        .map_err(|e| format!("RMVPE: {e}"))?;

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
        assert_eq!(rb.push(&[0.0; 7]), 7);
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

    fn test_params() -> crate::config::Params {
        crate::config::Params {
            chunk_ms: 200,
            lookahead_ms: 80,
            left_context_ms: 720,
            crossfade_ms: 50,
            max_latency_ms: 300,
            jitter_ms: 0,
            f0_up_key: 0,
            speaker_id: 0,
            gate_enabled: true,
            gate_db: -80.0,
            ort_opt: 2,
        }
    }

    fn test_geom() -> Geometry {
        Geometry::new(&test_params(), 48_000)
    }

    fn window_track(anchored_at_tail: bool, base0: usize, len0: usize, g: Geometry) -> Vec<(usize, usize)> {
        let (mut base, mut len) = (base0, len0);
        let mut out = Vec::new();
        while len >= g.window {
            let start = if anchored_at_tail { base + len - g.window } else { base };
            out.push((start + g.out_start(), start + g.out_start() + g.block));
            base += g.block;
            len -= g.block;
        }
        out
    }

    #[test]
    fn tail_anchored_window_repeats_same_block() {
        let g = test_geom();

        let track = window_track(true, 0, g.window + 6 * g.block, g);
        assert_eq!(track.len(), 7, "积压 6 块时应渲染 7 块");
        assert!(
            track.windows(2).all(|w| w[0] == w[1]),
            "1.0 的 7 个窗口本应逐样本相同（这正是复读的成因）: {track:?}"
        );
    }

    #[test]
    fn head_anchored_window_emits_each_block_once() {
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

        let p = crate::config::Params { chunk_ms: 64, lookahead_ms: 33, left_context_ms: 178, ..test_params() };
        let g2 = Geometry::new(&p, 48_000);
        assert_eq!(g2.block % HOP_48K, 0);
        assert_eq!(g2.window % HOP_48K, 0);

        let p = crate::config::Params { chunk_ms: 20, lookahead_ms: 0, left_context_ms: 0, crossfade_ms: 100, ..test_params() };
        let g3 = Geometry::new(&p, 48_000);
        assert_eq!(g3.crossfade, 0, "left/lookahead 为 0 时交叉淡化必须退化为 0（硬拼接）");
        assert!(g3.out_start() + g3.block <= g3.window);
    }

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

        assert_eq!(out[..], rendered[g.left..g.left + g.block]);
    }

    #[test]
    fn ola_emit_advances_exactly_one_block_and_stays_in_bounds() {
        let g = Geometry { left: 960, block: 480, lookahead: 480, crossfade: 96, window: 1920, frames: 4 };
        let mut ola = Ola::new(g.crossfade);

        for k in 0..5usize {
            let base = k * g.block;
            let rendered: Vec<f32> = (0..g.window).map(|i| (base + i) as f32).collect();
            let mut out = vec![0.0f32; g.block];
            ola.emit(g, &rendered, &mut out);

            let start = base + g.out_start();
            if k == 0 {

                for i in 0..g.block {
                    assert_eq!(out[i], (start + i) as f32, "块 0 位置 {i}");
                }
            } else {

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
    #[test]
    fn default_window_is_at_least_one_second_and_aligned() {
        let g = test_geom();
        let ms = |n: usize| n * 1000 / 48_000;
        assert!(ms(g.window) >= 1000, "默认推理窗口应 >= 1s，实际 {}ms", ms(g.window));
        assert_eq!(g.window % 960, 0, "窗口必须是 960 的整数倍（48k 下 20ms）");
        assert_eq!((g.window / 3) % 320, 0, "16k 长度必须是 320 的整数倍（HuBERT 帧对齐）");
        assert_eq!(g.frames, g.window / HOP_48K);
        assert_eq!(g.window, g.left + g.block + g.lookahead);
        assert!(g.crossfade <= g.block / 2 && g.crossfade <= g.lookahead && g.crossfade <= g.left);
        assert!(g.out_start() + g.block <= g.window);
        assert!(g.tail_start() + g.crossfade <= g.window);
    }
}
