//! MicYou native DSP 插件入口（C ABI）。
//!
//! 线程模型（严格遵守 `api-reference.md` 的三条硬规则）：
//! - `micyou_plugin_process` 跑在宿主的实时音频线程：**不调用任何 Host API、不分配堆内存、不加锁**，
//!   只碰两个无锁 SPSC 环形缓冲和几个原子量；
//! - `init` / `deinit` / `handle_message` / `handle_event` 跑在宿主分发的线程：只有这里才调用 Host API；
//! - worker 是插件自建子线程：**同样禁止调用 Host API**，所以它只写自己的日志文件，
//!   需要给用户看的消息通过 `SharedStatus` 交给 `interval:tick`（宿主线程）转发到 `host.log` / `host.notify`。

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::cell::UnsafeCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, Sender};

mod rvc;
mod stream;

use stream::AudioRingBuffer;

const MPL_ABI_VERSION: u32 = 1;
const MPL_API_VERSION: u32 = 1;
const PLUGIN_ID: &[u8] = b"opss.mambo-rvc-onnx\0";
const PLUGIN_VERSION: &[u8] = b"1.1.0\0";

/// 48kHz 下 1 秒。宿主每次交给 process 的是 `N*480*channels` 个交织样本
/// （N 取决于网络包大小与抖动），超过这个上限才会 bypass。
/// 1.0 版这里是 4096 帧：5 个 20ms 包一起到（mono 4800 帧）就会越界，
/// 而 bypass 意味着**把未变声的原声直接送给对方**。
const MAX_FRAMES: usize = 48_000;
/// 静音/音频切换时的淡入淡出长度（2ms @48k），避免每次欠载都产生硬跳变爆音
const FADE_SAMPLES: usize = 96;
/// 输入/输出环形缓冲容量（秒）。worker 落后时输入环会溢出丢弃，
/// 之后由 worker 的积压上限（max_latency_ms）把延迟拉回目标值。
const RING_SECONDS: usize = 4;
/// 状态轮询定时器周期
const STATUS_INTERVAL_MS: u64 = 1000;
/// 每 N 个 tick 才向宿主兜底轮询一次配置（get_config 每次都要锁宿主的 plugin manager）
const CONFIG_POLL_TICKS: u8 = 5;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum mpl_result_t {
    MPL_OK = 0,
    MPL_ERR_NOT_IMPLEMENTED = 1,
    MPL_ERR_INVALID_ARG = 2,
    MPL_ERR_RUNTIME = 3,
    MPL_ERR_BUFFER_TOO_SMALL = 4,
    MPL_ERR_PERMISSION = 5,
}

#[repr(C)]
pub struct mpl_plugin_info_t {
    pub abi_version: u32,
    pub api_version: u32,
    pub id: *const c_char,
    pub version: *const c_char,
}
unsafe impl Sync for mpl_plugin_info_t {}

/// 宿主回调表。**按值拷贝**保存（api-reference.md 明确要求：init 返回后宿主可能释放原内存，
/// 严禁保存 host 指针本身）。字段顺序必须与 `micyou_plugin_abi.h` 一致；新字段只追加在 `ctx`
/// 之后，所以这里只声明到 `set_panel_icon`（API v1）是安全的——宿主表更长，读前缀没问题。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct mpl_host_api_t {
    pub log: Option<unsafe extern "C" fn(*mut c_void, i32, *const c_char)>,
    pub get_config: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_char, *mut u32) -> mpl_result_t>,
    pub set_config: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> mpl_result_t>,
    pub emit_event: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> mpl_result_t>,
    pub send_message: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const u8, u32) -> mpl_result_t>,
    pub audio_state: Option<unsafe extern "C" fn(*mut c_void, *mut c_char, *mut u32) -> mpl_result_t>,
    pub connected_devices: Option<unsafe extern "C" fn(*mut c_void, *mut c_char, *mut u32) -> mpl_result_t>,
    pub ctx: *mut c_void,
    pub play_sound: Option<unsafe extern "C" fn(*mut c_void, *const c_char) -> mpl_result_t>,
    pub plugin_dir: Option<unsafe extern "C" fn(*mut c_void, *mut c_char, *mut u32) -> mpl_result_t>,
    pub register_hotkey: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *mut u64) -> mpl_result_t>,
    pub open_window: Option<unsafe extern "C" fn(*mut c_void, *const c_char) -> mpl_result_t>,
    pub fs_read: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_char, *mut u32) -> mpl_result_t>,
    pub fs_write: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> mpl_result_t>,
    pub set_timeout: Option<unsafe extern "C" fn(*mut c_void, u64, *const c_char, *mut u64) -> mpl_result_t>,
    pub clear_timeout: Option<unsafe extern "C" fn(*mut c_void, u64) -> mpl_result_t>,
    pub http_request: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, *const c_char, *const c_char, *mut u64) -> mpl_result_t>,
    pub set_interval: Option<unsafe extern "C" fn(*mut c_void, u64, *const c_char, *mut u64) -> mpl_result_t>,
    pub clear_interval: Option<unsafe extern "C" fn(*mut c_void, u64) -> mpl_result_t>,
    pub open_url: Option<unsafe extern "C" fn(*mut c_void, *const c_char) -> mpl_result_t>,
    pub notify: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> mpl_result_t>,
    pub locale: Option<unsafe extern "C" fn(*mut c_void, *mut c_char, *mut u32) -> mpl_result_t>,
    pub host_info: Option<unsafe extern "C" fn(*mut c_void, *mut c_char, *mut u32) -> mpl_result_t>,
    pub clipboard_read: Option<unsafe extern "C" fn(*mut c_void, *mut c_char, *mut u32) -> mpl_result_t>,
    pub clipboard_write: Option<unsafe extern "C" fn(*mut c_void, *const c_char) -> mpl_result_t>,
    pub set_panel_icon: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> mpl_result_t>,
}
unsafe impl Send for mpl_host_api_t {}
unsafe impl Sync for mpl_host_api_t {}

/// 宿主回调表的值拷贝。只在宿主分发的线程（init / handle_message / handle_event）里使用。
static HOST_API: Mutex<Option<mpl_host_api_t>> = Mutex::new(None);

struct PluginState {
    input_rb: Arc<AudioRingBuffer>,
    output_rb: Arc<AudioRingBuffer>,
    signal_tx: Sender<()>,
    is_running: Arc<AtomicBool>,
    /// 音频线程因为输入环满而丢过数据时 +1（worker 据此复位 OLA）
    resync: Arc<AtomicU64>,
    /// worker 请求音频线程清空输出环（模型热重载后，别把旧音色的残留音频继续播出去）。
    /// 只有 output_rb 的消费者（音频线程）能动 read_pos，所以必须由它自己执行。
    flush_out: Arc<AtomicBool>,
    status: stream::SharedStatus,
    worker: Option<JoinHandle<()>>,
    interval_id: AtomicU64,
    // ── 以下 UnsafeCell 字段各自只允许一个线程访问，见各字段注释 ──
    /// 仅音频线程：下混/上混用的单声道暂存
    scratch: UnsafeCell<Vec<f32>>,
    /// 仅音频线程：淡入淡出与溢出锁存状态
    fade: UnsafeCell<FadeState>,
    /// 仅宿主分发线程：上一次已转发的状态
    reported: UnsafeCell<Reported>,
}

#[derive(Clone, Copy)]
struct FadeState {
    last_val: f32,
    silence_run: u32,
    /// 输入环溢出锁存：溢出期间只通知 worker 一次
    overflow_latched: bool,
}

impl Default for FadeState {
    fn default() -> Self {
        Self { last_val: 0.0, silence_run: 0, overflow_latched: false }
    }
}

#[derive(Clone, Copy, Default)]
struct Reported {
    revision: u64,
    phase: u8,
    ticks: u8,
}

/// 用 `AtomicPtr` 而不是 `static mut`：后者在 Rust 2024 里是硬错误（`static_mut_refs`），
/// 1.0 版编译时已经在报 `creating a mutable reference to mutable static` 警告。
///
/// SAFETY（不变量）：宿主的调用顺序保证 init / process / deinit 不会并发——
/// `PluginDspRegistry::unregister` 取写锁时会等所有正在进行的 `process_all` 读者退出，
/// 节点移除后不会再有新的 process 调用，实例才可能被 drop（触发 deinit）。
static STATE: AtomicPtr<PluginState> = AtomicPtr::new(std::ptr::null_mut());

fn guard<F: FnOnce() -> mpl_result_t + std::panic::UnwindSafe>(f: F) -> mpl_result_t {
    // panic 绝不能穿过 FFI 边界（会直接 abort 整个宿主进程）
    std::panic::catch_unwind(f).unwrap_or(mpl_result_t::MPL_ERR_RUNTIME)
}

#[no_mangle]
pub extern "C" fn micyou_plugin_info() -> *const mpl_plugin_info_t {
    static INFO: mpl_plugin_info_t = mpl_plugin_info_t {
        abi_version: MPL_ABI_VERSION,
        api_version: MPL_API_VERSION,
        id: PLUGIN_ID.as_ptr() as *const c_char,
        version: PLUGIN_VERSION.as_ptr() as *const c_char,
    };
    &INFO
}

// ─────────────────────────── 参数 ───────────────────────────

pub(crate) mod config {
//! 插件参数：全部放在原子变量里，音频线程与推理线程各自无锁读取。
//!
//! - `process()`（实时音频线程）**完全不读配置**，只碰环形缓冲与淡入淡出状态；
//! - 只有 worker 在每个批次开头取一次快照（`Params::load()`），所以改参数永远不会撕裂一个批次；
//! - 模型文件名这类字符串走 `Mutex<String>`，同样只在批次边界读取。

    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
    use std::sync::Mutex;

    pub mod range {
        pub const CHUNK_MS: (u32, u32) = (20, 500);
        pub const LOOKAHEAD_MS: (u32, u32) = (0, 500);
        pub const LEFT_CONTEXT_MS: (u32, u32) = (0, 2000);
        pub const CROSSFADE_MS: (u32, u32) = (0, 100);
        pub const MAX_LATENCY_MS: (u32, u32) = (50, 3000);
        pub const JITTER_MS: (u32, u32) = (0, 300);
        pub const F0_UP_KEY: (i32, i32) = (-24, 24);
        pub const SPEAKER_ID: (i32, i32) = (0, 255);
        pub const GATE_DB: (i32, i32) = (-100, -20);
    }

    static CHUNK_MS: AtomicU32 = AtomicU32::new(100);
    static LOOKAHEAD_MS: AtomicU32 = AtomicU32::new(150);
    static LEFT_CONTEXT_MS: AtomicU32 = AtomicU32::new(300);
    static CROSSFADE_MS: AtomicU32 = AtomicU32::new(20);
    static MAX_LATENCY_MS: AtomicU32 = AtomicU32::new(300);
    static JITTER_MS: AtomicU32 = AtomicU32::new(40);
    static F0_UP_KEY: AtomicI32 = AtomicI32::new(0);
    static SPEAKER_ID: AtomicI32 = AtomicI32::new(0);
    static GATE_ENABLED: AtomicBool = AtomicBool::new(true);
    /// 存 f32 的 bit pattern（-70 dBFS）
    static GATE_DB_BITS: AtomicU32 = AtomicU32::new((-70.0f32).to_bits());

    /// 用户自定义 RVC 模型（相对插件目录）。空 = 自动发现 user_models/ 下的模型。
    static MODEL_FILE: Mutex<String> = Mutex::new(String::new());
    /// 每次配置变化 +1，worker 据此决定是否要重新解析/加载模型。
    static MODEL_EPOCH: AtomicU32 = AtomicU32::new(0);

    /// 一份一致性快照。worker 每个批次取一次。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Params {
        pub chunk_ms: u32,
        pub lookahead_ms: u32,
        pub left_context_ms: u32,
        pub crossfade_ms: u32,
        pub max_latency_ms: u32,
        pub jitter_ms: u32,
        pub f0_up_key: i32,
        pub speaker_id: i32,
        pub gate_enabled: bool,
        pub gate_db: f32,
    }

    impl Params {
        #[inline]
        pub fn load() -> Self {
            Self {
                chunk_ms: CHUNK_MS.load(Ordering::Relaxed),
                lookahead_ms: LOOKAHEAD_MS.load(Ordering::Relaxed),
                left_context_ms: LEFT_CONTEXT_MS.load(Ordering::Relaxed),
                crossfade_ms: CROSSFADE_MS.load(Ordering::Relaxed),
                max_latency_ms: MAX_LATENCY_MS.load(Ordering::Relaxed),
                jitter_ms: JITTER_MS.load(Ordering::Relaxed),
                f0_up_key: F0_UP_KEY.load(Ordering::Relaxed),
                speaker_id: SPEAKER_ID.load(Ordering::Relaxed),
                gate_enabled: GATE_ENABLED.load(Ordering::Relaxed),
                gate_db: f32::from_bits(GATE_DB_BITS.load(Ordering::Relaxed)),
            }
        }

        /// RMS 静音门限（线性幅度）。
        #[inline]
        pub fn gate_rms(&self) -> f32 {
            if self.gate_enabled {
                10.0f32.powf(self.gate_db / 20.0)
            } else {
                0.0
            }
        }
    }

    /// 把解析出来的整数夹到合法区间（负数一律当 0，避免 `-1 as u32` 变成天文数字）。
    #[inline]
    fn clamp_u32(v: i32, r: (u32, u32)) -> u32 {
        (v.max(0) as u32).clamp(r.0, r.1)
    }

    /// 写入单个配置项。`key`/`raw` 来自宿主的 `get_config`（JSON 文本）。
    /// 返回 Some(规范化的值字符串) 表示该项被接受。
    pub fn set_from_raw(key: &str, raw: &str) -> Option<String> {
        let text = raw.trim().trim_matches('"').trim();
        if text.is_empty() || text == "null" {
            return None;
        }
        match key {
            "chunk_ms" => {
                let v = clamp_u32(parse_num(text)?, range::CHUNK_MS);
                CHUNK_MS.store(v, Ordering::Relaxed);
                Some(v.to_string())
            }
            "lookahead_ms" => {
                let v = clamp_u32(parse_num(text)?, range::LOOKAHEAD_MS);
                LOOKAHEAD_MS.store(v, Ordering::Relaxed);
                Some(v.to_string())
            }
            "left_context_ms" => {
                let v = clamp_u32(parse_num(text)?, range::LEFT_CONTEXT_MS);
                LEFT_CONTEXT_MS.store(v, Ordering::Relaxed);
                Some(v.to_string())
            }
            // 兼容 1.0 的 extra_ms：老版本用它同时充当“左上下文”和“右 lookahead”。
            // 只有当宿主存档里没有新键时 lib.rs 才会把 extra_ms 喂进来（见 reload_config），
            // 所以这里无条件套用：左上下文 = extra，lookahead = extra/2（上限 200ms），
            // 既保留老用户的质量设置，又不会让升级后的延迟翻倍。
            "extra_ms" => {
                let v = parse_num(text)?;
                let half = (v / 2).min(200);
                LOOKAHEAD_MS.store(clamp_u32(half, range::LOOKAHEAD_MS), Ordering::Relaxed);
                LEFT_CONTEXT_MS.store(clamp_u32(v, range::LEFT_CONTEXT_MS), Ordering::Relaxed);
                Some(format!("extra_ms={v} -> left_context_ms={v} lookahead_ms={half}"))
            }
            "crossfade_ms" => {
                let v = clamp_u32(parse_num(text)?, range::CROSSFADE_MS);
                CROSSFADE_MS.store(v, Ordering::Relaxed);
                Some(v.to_string())
            }
            "max_latency_ms" => {
                let v = clamp_u32(parse_num(text)?, range::MAX_LATENCY_MS);
                MAX_LATENCY_MS.store(v, Ordering::Relaxed);
                Some(v.to_string())
            }
            "jitter_ms" => {
                let v = clamp_u32(parse_num(text)?, range::JITTER_MS);
                JITTER_MS.store(v, Ordering::Relaxed);
                Some(v.to_string())
            }
            "f0_up_key" => {
                let v = parse_num(text)?.clamp(range::F0_UP_KEY.0, range::F0_UP_KEY.1);
                F0_UP_KEY.store(v, Ordering::Relaxed);
                Some(v.to_string())
            }
            "speaker_id" => {
                let v = parse_num(text)?.clamp(range::SPEAKER_ID.0, range::SPEAKER_ID.1);
                SPEAKER_ID.store(v, Ordering::Relaxed);
                Some(v.to_string())
            }
            "gate_db" => {
                let v = parse_num(text)?.clamp(range::GATE_DB.0, range::GATE_DB.1);
                GATE_DB_BITS.store((v as f32).to_bits(), Ordering::Relaxed);
                Some(v.to_string())
            }
            "gate_enabled" => {
                let v = parse_bool(text)?;
                GATE_ENABLED.store(v, Ordering::Relaxed);
                Some(v.to_string())
            }
            "model_file" => {
                if let Ok(mut slot) = MODEL_FILE.lock() {
                    // 只接受相对路径，且不允许 .. 穿越（插件目录沙箱）
                    let clean = text.replace('\\', "/");
                    let clean = clean.trim_start_matches("/").trim_start_matches("./");
                    if clean.contains("..") {
                        return Some("(rejected: path escape)".to_string());
                    }
                    *slot = clean.to_string();
                    MODEL_EPOCH.fetch_add(1, Ordering::Relaxed);
                    Some(clean.to_string())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn model_file() -> String {
        MODEL_FILE.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn model_epoch() -> u32 {
        MODEL_EPOCH.load(Ordering::Relaxed)
    }

    /// 数值解析：宿主回传的是 JSON 文本，数字可能写成 `100`、`100.0` 或 `"100"`，
    /// 一律先按 f64 解析再取整，避免 `parse::<u32>()` 在 `100.0` 上静默失败。
    fn parse_num(text: &str) -> Option<i32> {
        text.parse::<i32>().ok().or_else(|| text.parse::<f64>().ok().map(|v| v.round() as i32))
    }

    fn parse_bool(text: &str) -> Option<bool> {
        match text {
            "true" | "1" | "on" | "yes" => Some(true),
            "false" | "0" | "off" | "no" => Some(false),
            _ => None,
        }
    }
}

// ─────────────────────────── 文件日志 ───────────────────────────

/// 插件自己的文件日志（带重复消息限流）。
///
/// 为什么不用宿主的 `host.log`：worker 是插件自建的子线程，而 `api-reference.md` 明确规定
/// **禁止在自定义子线程中调用任何 Host API**（宿主内部的锁与状态机不是跨线程安全的）。
/// 所以推理线程只写文件；需要给用户看的消息由 `report_status()` 在宿主分发的线程
/// （`interval:tick`）里转发到 `host.log` 与 `host.notify`。
pub(crate) mod logger {
    use std::fs::{File, OpenOptions};
    use std::io::Write;
    use std::path::Path;
    use std::sync::Mutex;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    struct Inner {
        file: Option<File>,
        last_msg: String,
        /// `Instant::now()` 不是 const fn，所以用 Option 让静态初始化保持 const。
        last_emit: Option<Instant>,
        suppressed: u32,
    }

    static LOG: Mutex<Inner> = Mutex::new(Inner {
        file: None,
        last_msg: String::new(),
        last_emit: None,
        suppressed: 0,
    });

    /// 同一条消息在这个时间窗内只落盘一次（推理失败时可能每块都报同一个错）。
    const DEDUP_WINDOW: Duration = Duration::from_secs(2);

    pub fn init(plugin_dir: &Path) {
        let path = plugin_dir.join("rvc_plugin.log");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .ok();
        if let Ok(mut g) = LOG.lock() {
            g.file = file;
            g.last_msg.clear();
            g.suppressed = 0;
        }
        log(&format!("[Log] opened {}", path.display()));
    }

    pub fn log(msg: &str) {
        let Ok(mut g) = LOG.lock() else { return };
        let now = Instant::now();
        let recent = matches!(g.last_emit, Some(t) if now.duration_since(t) < DEDUP_WINDOW);
        if g.last_msg == msg {
            if recent {
                g.suppressed += 1;
                return;
            }
            let extra = if g.suppressed > 0 {
                format!(" (x{})", g.suppressed + 1)
            } else {
                String::new()
            };
            g.suppressed = 0;
            write_line(&mut g, &format!("{msg}{extra}"));
        } else {
            if g.suppressed > 0 {
                let line = format!("{} (x{})", g.last_msg, g.suppressed + 1);
                g.suppressed = 0;
                write_line(&mut g, &line);
            }
            g.last_msg = msg.to_string();
            write_line(&mut g, msg);
        }
        g.last_emit = Some(now);
    }

    fn write_line(g: &mut Inner, msg: &str) {
        if let Some(f) = g.file.as_mut() {
            let _ = writeln!(f, "[{}] {}", timestamp(), msg);
            let _ = f.flush();
        }
    }

    fn timestamp() -> String {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_millis())
            .unwrap_or(0);
        format!("{secs}.{ms:03}")
    }
}

// ─────────────────────────── Host API 封装（仅宿主线程）───────────────────────────

/// 缓冲区契约（api-reference.md）：先用小缓冲探测拿到所需大小，再分配后重取。
/// 成功时 `*out_size` 是字节数（不含 NUL）；键不存在时宿主返回 MPL_OK 且 `*out_size == 0`。
unsafe fn read_config(host: &mpl_host_api_t, key: &str) -> Option<String> {
    let get_config = host.get_config?;
    let c_key = CString::new(key).ok()?;

    let mut size: u32 = 0;
    let mut probe = [0u8; 1];
    get_config(host.ctx, c_key.as_ptr(), probe.as_mut_ptr() as *mut c_char, &mut size);
    if size == 0 || size > 4096 {
        return None;
    }
    let mut buf = vec![0u8; size as usize + 1];
    let mut size2 = size;
    let res = get_config(host.ctx, c_key.as_ptr(), buf.as_mut_ptr() as *mut c_char, &mut size2);
    if res != mpl_result_t::MPL_OK {
        return None;
    }
    buf.truncate(size2 as usize);
    Some(String::from_utf8_lossy(&buf).into_owned())
}

fn with_host<R>(f: impl FnOnce(&mpl_host_api_t) -> R) -> Option<R> {
    let g = HOST_API.lock().ok()?;
    let host = (*g)?;
    drop(g);
    Some(f(&host))
}

unsafe fn host_log(level: i32, msg: &str) {
    with_host(|h| {
        if let (Some(f), Ok(c)) = (h.log, CString::new(msg)) {
            f(h.ctx, level, c.as_ptr());
        }
    });
}

unsafe fn host_notify(title: &str, body: &str) {
    with_host(|h| {
        if let (Some(f), Ok(t), Ok(b)) = (h.notify, CString::new(title), CString::new(body)) {
            f(h.ctx, t.as_ptr(), b.as_ptr());
        }
    });
}

/// 从宿主拉取全部参数。只在 init / handle_message（宿主线程）里调用。
fn reload_config() {
    let keys = [
        "chunk_ms",
        "lookahead_ms",
        "left_context_ms",
        "crossfade_ms",
        "max_latency_ms",
        "jitter_ms",
        "f0_up_key",
        "speaker_id",
        "gate_enabled",
        "gate_db",
        "model_file",
    ];
    let mut applied = Vec::new();
    let mut has_window_keys = false;
    unsafe {
        with_host(|host| {
            for key in keys {
                if let Some(raw) = read_config(host, key) {
                    if matches!(key, "lookahead_ms" | "left_context_ms") {
                        has_window_keys = true;
                    }
                    if let Some(v) = config::set_from_raw(key, &raw) {
                        applied.push(format!("{key}={v}"));
                    }
                }
            }
            // 1.0 的老配置只有一个 extra_ms（同时充当左上下文与右 lookahead）。
            // 升级安装时宿主存档里没有新键，这里把它拆开，避免用户升级后设置全丢。
            if !has_window_keys {
                if let Some(raw) = read_config(host, "extra_ms") {
                    if let Some(v) = config::set_from_raw("extra_ms", &raw) {
                        applied.push(v);
                    }
                }
            }
        });
    }
    if !applied.is_empty() {
        logger::log(&format!("[Config] 已应用 {}", applied.join(" ")));
    }
}

fn log_effective_params() {
    let p = config::Params::load();
    logger::log(&format!(
        "[Config] 生效: chunk={}ms lookahead={}ms left={}ms crossfade={}ms max_latency={}ms jitter={}ms key={} sid={} gate={}({}dB) model='{}'",
        p.chunk_ms, p.lookahead_ms, p.left_context_ms, p.crossfade_ms,
        p.max_latency_ms, p.jitter_ms, p.f0_up_key, p.speaker_id,
        p.gate_enabled, p.gate_db, config::model_file()
    ));
}

// ─────────────────────────── 生命周期 ───────────────────────────

#[no_mangle]
pub unsafe extern "C" fn micyou_plugin_init(host: *const mpl_host_api_t) -> mpl_result_t {
    guard(|| {
        if host.is_null() {
            return mpl_result_t::MPL_ERR_INVALID_ARG;
        }
        // 按值拷贝宿主表
        if let Ok(mut slot) = HOST_API.lock() {
            // SAFETY: 宿主保证 host 在 init 期间有效
            *slot = Some(unsafe { *host });
        } else {
            return mpl_result_t::MPL_ERR_RUNTIME;
        }

        let plugin_dir = rvc::plugin_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        logger::init(&plugin_dir);
        logger::log(&format!(
            "[Init] mambo-rvc-onnx v{} | abi={} api={} | dir={}",
            std::env!("CARGO_PKG_VERSION"), MPL_ABI_VERSION, MPL_API_VERSION, plugin_dir.display()
        ));

        reload_config();
        log_effective_params();

        let sample_rate = 48_000usize; // 宿主进 DSP 链之前已把输入重采样到 48kHz
        let capacity = sample_rate * RING_SECONDS;
        let input_rb = Arc::new(AudioRingBuffer::new(capacity));
        let output_rb = Arc::new(AudioRingBuffer::new(capacity));
        // 信号通道只承载“有新数据”这一个语义，深度 4 足够（worker 每批都会抽干输入环）
        let (signal_tx, signal_rx) = bounded::<()>(4);
        let is_running = Arc::new(AtomicBool::new(true));
        let resync = Arc::new(AtomicU64::new(0));
        let flush_out = Arc::new(AtomicBool::new(false));
        let status = stream::new_status();

        let handles = stream::WorkerHandles {
            input_rb: input_rb.clone(),
            output_rb: output_rb.clone(),
            signal_rx,
            is_running: is_running.clone(),
            resync: resync.clone(),
            flush_out: flush_out.clone(),
            status: status.clone(),
        };
        let spawned = thread::Builder::new()
            .name("mambo-rvc-worker".into())
            .spawn(move || stream::worker_loop(sample_rate, handles));
        let worker = match spawned {
            Ok(h) => h,
            Err(e) => {
                logger::log(&format!("[Init] 无法创建 worker 线程: {e}"));
                return mpl_result_t::MPL_ERR_RUNTIME;
            }
        };

        // 状态轮询定时器：回调走 handle_message（宿主线程），因此可以在里面安全调用 Host API，
        // 把 worker（自建子线程，禁止调 Host API）的状态转发到 GUI 插件日志与系统通知。
        let mut interval_id: u64 = 0;
        with_host(|h| {
            if let (Some(set_interval), Ok(payload)) = (h.set_interval, CString::new("status")) {
                let mut id: u64 = 0;
                // SAFETY: h.ctx 由宿主提供，set_interval 在宿主线程调用
                unsafe {
                    if set_interval(h.ctx, STATUS_INTERVAL_MS, payload.as_ptr(), &mut id)
                        == mpl_result_t::MPL_OK
                    {
                        interval_id = id;
                    }
                }
            }
        });
        if interval_id == 0 {
            logger::log("[Init] set_interval 不可用，状态将只写入 rvc_plugin.log");
        }

        let state = Box::new(PluginState {
            input_rb,
            output_rb,
            signal_tx,
            is_running,
            resync,
            flush_out,
            status,
            worker: Some(worker),
            interval_id: AtomicU64::new(interval_id),
            scratch: UnsafeCell::new(vec![0.0f32; MAX_FRAMES]),
            fade: UnsafeCell::new(FadeState::default()),
            reported: UnsafeCell::new(Reported::default()),
        });
        let prev = STATE.swap(Box::into_raw(state), Ordering::AcqRel);
        if !prev.is_null() {
            // 宿主保证会先 deinit 再 init；真出现残留就回收，避免泄漏
            // SAFETY: 该指针由上面的 Box::into_raw 产生，且此后无人再使用
            drop(unsafe { Box::from_raw(prev) });
        }
        mpl_result_t::MPL_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn micyou_plugin_deinit() {
    let _ = guard(|| {
        let ptr = STATE.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if ptr.is_null() {
            return mpl_result_t::MPL_OK;
        }
        // SAFETY: 见 STATE 的不变量——此时不会再有 process 调用进来，指针由 init 的 Box 产生
        let mut state = unsafe { Box::from_raw(ptr) };

        // 先停定时器（deinit 由宿主调用，属宿主线程，调用 Host API 合规）
        let id = state.interval_id.load(Ordering::Relaxed);
        if id != 0 {
            with_host(|h| {
                if let Some(clear) = h.clear_interval {
                    // SAFETY: 同上
                    unsafe { clear(h.ctx, id) };
                }
            });
        }

        state.is_running.store(false, Ordering::Relaxed);
        // worker 用 recv_timeout(100ms) 轮询 is_running，这里再推一把让它立刻醒
        let _ = state.signal_tx.try_send(());
        if let Some(handle) = state.worker.take() {
            let _ = handle.join();
        }
        logger::log("[Deinit] 完成");
        // state 在此处 drop：signal_tx / 两个环 / status 一并释放
        mpl_result_t::MPL_OK
    });
}

// ─────────────────────────── 实时音频 ───────────────────────────

/// 宿主契约：`data` 是 `samples` 个交织 f32（`samples = frames * channels`），原地处理；
/// `bypass = 1` 表示本帧旁路（宿主保留缓冲原样，也就是**干声直通**）。
///
/// 实时安全：无堆分配、无锁、无 Host API、无系统调用，只有环形缓冲的原子游标操作。
#[no_mangle]
pub unsafe extern "C" fn micyou_plugin_process(
    data: *mut f32,
    samples: u32,
    channels: u32,
    _queued_ms: f64,
    bypass: *mut u32,
) -> mpl_result_t {
    guard(|| {
        if data.is_null() || bypass.is_null() {
            return mpl_result_t::MPL_ERR_INVALID_ARG;
        }
        let total = samples as usize;
        let ch = channels as usize;
        // SAFETY: 宿主保证 bypass 可写
        unsafe { *bypass = 1 };
        if total == 0 || ch == 0 || total % ch != 0 {
            return mpl_result_t::MPL_OK;
        }
        let frames = total / ch;

        let ptr = STATE.load(Ordering::Acquire);
        if ptr.is_null() {
            return mpl_result_t::MPL_OK;
        }
        // SAFETY: 见 STATE 的不变量；下面的 UnsafeCell 字段均只被音频线程访问
        let state = unsafe { &*ptr };
        if frames > MAX_FRAMES {
            // 几乎不可能走到（1 秒的音频块）；bypass 会泄漏干声，所以宁可留大上限
            return mpl_result_t::MPL_OK;
        }

        // SAFETY: 宿主保证 data 指向 total 个可写 f32
        let buf = unsafe { std::slice::from_raw_parts_mut(data, total) };
        // SAFETY: scratch / fade 只被音频线程访问
        let (scratch, fade) = unsafe { (&mut *state.scratch.get(), &mut *state.fade.get()) };

        // worker 请求清空输出环（模型热重载 / 参数突变后）
        if state.flush_out.swap(false, Ordering::AcqRel) {
            state.output_rb.discard(state.output_rb.available());
            *fade = FadeState::default();
        }

        // 1) 下混到单声道（RVC 是单声道模型）
        if ch == 1 {
            scratch[..frames].copy_from_slice(&buf[..frames]);
        } else {
            let inv = 1.0 / ch as f32;
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..ch {
                    sum += buf[i * ch + c];
                }
                scratch[i] = sum * inv;
            }
        }

        // 2) 送入输入环。环满说明 worker 严重落后：丢弃这一块并通知 worker 复位 OLA
        if state.input_rb.push(&scratch[..frames]) == 0 {
            if !fade.overflow_latched {
                fade.overflow_latched = true;
                state.resync.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            fade.overflow_latched = false;
        }
        let _ = state.signal_tx.try_send(());

        // 3) 取出变声结果
        if state.output_rb.available() >= frames {
            state.output_rb.pop(&mut scratch[..frames]);
            if fade.silence_run > 0 {
                // 从静音恢复：淡入，避免硬切爆音
                let n = FADE_SAMPLES.min(frames);
                for i in 0..n {
                    scratch[i] *= i as f32 / FADE_SAMPLES as f32;
                }
                fade.silence_run = 0;
            }
            fade.last_val = scratch[frames - 1];
            for i in 0..frames {
                let v = scratch[i];
                for c in 0..ch {
                    buf[i * ch + c] = v;
                }
            }
        } else {
            // 欠载：冻结最后一个样本并淡出（与宿主 output stream 的 underrun 行为一致），
            // 比 1.0 的硬写 0 少一次爆音。
            let base = fade.silence_run;
            fade.silence_run = base.saturating_add(frames as u32);
            let last = fade.last_val;
            for i in 0..frames {
                let run = base + i as u32;
                let g = if run < FADE_SAMPLES as u32 {
                    1.0 - run as f32 / FADE_SAMPLES as f32
                } else {
                    0.0
                };
                let v = last * g;
                for c in 0..ch {
                    buf[i * ch + c] = v;
                }
            }
            fade.last_val = 0.0;
        }

        // SAFETY: 宿主保证 bypass 可写；0 = 本帧已被插件处理
        unsafe { *bypass = 0 };
        mpl_result_t::MPL_OK
    })
}

// ─────────────────────────── 消息 / 事件（宿主线程）───────────────────────────

#[no_mangle]
pub extern "C" fn micyou_plugin_handle_message(
    _source: *const c_char,
    topic: *const c_char,
    payload: *const u8,
    len: u32,
) -> mpl_result_t {
    guard(|| {
        if topic.is_null() {
            return mpl_result_t::MPL_ERR_INVALID_ARG;
        }
        // SAFETY: 宿主保证 topic 是 NUL 结尾的 UTF-8 C 字符串
        let topic = unsafe { CStr::from_ptr(topic) }.to_str().unwrap_or("");
        match topic {
            "config:changed" => {
                // 注意：宿主派发用的是 try_lock，音频线程正忙时这条消息会被**直接丢弃**
                // （宿主日志 "skip message for busy instance"）。所以下面还有定时兜底轮询。
                reload_config();
                log_effective_params();
            }
            "interval:tick" => {
                // SAFETY: 宿主保证 payload/len 描述一段有效内存
                let body = unsafe { payload_string(payload, len) };
                if body.contains("status") {
                    report_status();
                    let ptr = STATE.load(Ordering::Acquire);
                    let do_poll = if ptr.is_null() {
                        false
                    } else {
                        // SAFETY: reported 只被宿主分发线程访问
                        let reported = unsafe { &mut *(*ptr).reported.get() };
                        reported.ticks = reported.ticks.wrapping_add(1);
                        reported.ticks % CONFIG_POLL_TICKS == 0
                    };
                    if do_poll {
                        let before = config::Params::load();
                        let before_model = config::model_file();
                        reload_config();
                        if config::Params::load() != before || config::model_file() != before_model {
                            logger::log("[Config] 定时兜底同步到了新配置（config:changed 可能被宿主 try_lock 丢弃）");
                            log_effective_params();
                        }
                    }
                }
            }
            _ => {}
        }
        mpl_result_t::MPL_OK
    })
}

#[no_mangle]
pub extern "C" fn micyou_plugin_handle_event(event_type: *const c_char, json: *const c_char) -> mpl_result_t {
    guard(|| {
        // SAFETY: 宿主保证两个参数是 NUL 结尾的 C 字符串（可能为 null）
        let (t, j) = unsafe {
            (
                if event_type.is_null() { "" } else { CStr::from_ptr(event_type).to_str().unwrap_or("") },
                if json.is_null() { "" } else { CStr::from_ptr(json).to_str().unwrap_or("") },
            )
        };
        logger::log(&format!("[Event] {t} {j}"));
        // 插件被启用/禁用时复位流水线，别用上一轮的残留状态。
        // 环形缓冲只能由各自的消费者清空，所以这里只发信号，由对应线程自己执行。
        if t == "state_changed" {
            let ptr = STATE.load(Ordering::Acquire);
            if !ptr.is_null() {
                // SAFETY: 见 STATE 的不变量；这里只碰跨线程安全的原子量
                let state = unsafe { &*ptr };
                state.resync.fetch_add(1, Ordering::Relaxed);
                state.flush_out.store(true, Ordering::Release);
            }
        }
        mpl_result_t::MPL_OK
    })
}

/// SAFETY: 调用方保证 payload/len 描述一段有效的 UTF-8（或可损失转换的）内存。
unsafe fn payload_string(payload: *const u8, len: u32) -> String {
    if payload.is_null() || len == 0 {
        return String::new();
    }
    String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(payload, len as usize) }).into_owned()
}

/// 把 worker 的状态转发到宿主日志 + 系统通知（运行在宿主分发线程，调用 Host API 合规）。
fn report_status() {
    let ptr = STATE.load(Ordering::Acquire);
    if ptr.is_null() {
        return;
    }
    // SAFETY: 见 STATE 的不变量；reported 只被宿主分发线程访问
    let state = unsafe { &*ptr };
    let snapshot = match state.status.lock() {
        Ok(s) => s.clone(),
        Err(_) => return,
    };
    // SAFETY: reported 只被宿主分发线程访问
    let reported = unsafe { &mut *state.reported.get() };
    let phase = phase_u8(snapshot.phase);
    if reported.revision == snapshot.revision && reported.phase == phase {
        return;
    }
    let changed_phase = reported.phase != phase;
    reported.revision = snapshot.revision;
    reported.phase = phase;

    let line = format!(
        "[RVC] {:?} | 模型 {} | τ 平均 {:.0}ms 峰值 {:.0}ms | 已处理 {} 块 | 断流 {} 次{}",
        snapshot.phase,
        snapshot.model_label,
        snapshot.avg_tau_ms,
        snapshot.max_tau_ms,
        snapshot.blocks,
        snapshot.flushes,
        if snapshot.detail.is_empty() { String::new() } else { format!(" | {}", snapshot.detail) }
    );
    logger::log(&line);
    // SAFETY: host_log 内部按值使用已拷贝的宿主表
    unsafe { host_log(2, &line) };

    // 只在用户真正需要知道时弹通知：加载完成 / 加载失败
    if changed_phase {
        // SAFETY: 同上
        unsafe {
            match snapshot.phase {
                stream::Phase::Ready => host_notify(
                    "曼波 RVC 已就绪",
                    &format!("模型：{}{}", snapshot.model_label,
                        if snapshot.detail.is_empty() { String::new() } else { format!("\n{}", snapshot.detail) }),
                ),
                stream::Phase::Error => host_notify(
                    "曼波 RVC 加载失败",
                    &snapshot.detail.chars().take(180).collect::<String>(),
                ),
                stream::Phase::Loading => {}
            }
        }
    }
}

fn phase_u8(p: stream::Phase) -> u8 {
    match p {
        stream::Phase::Loading => 0,
        stream::Phase::Ready => 1,
        stream::Phase::Error => 2,
    }
}
