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

const PLUGIN_VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

const MAX_FRAMES: usize = 48_000;

const FADE_SAMPLES: usize = 96;

const RING_SECONDS: usize = 4;

const STATUS_INTERVAL_MS: u64 = 1000;

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

static HOST_API: Mutex<Option<mpl_host_api_t>> = Mutex::new(None);

struct PluginState {
    input_rb: Arc<AudioRingBuffer>,
    output_rb: Arc<AudioRingBuffer>,
    signal_tx: Sender<()>,
    is_running: Arc<AtomicBool>,

    resync: Arc<AtomicU64>,

    flush_out: Arc<AtomicBool>,

    underruns: Arc<AtomicU64>,
    status: stream::SharedStatus,
    worker: Option<JoinHandle<()>>,
    interval_id: AtomicU64,

    scratch: UnsafeCell<Vec<f32>>,

    fade: UnsafeCell<FadeState>,

    reported: UnsafeCell<Reported>,
}

#[derive(Clone, Copy)]
struct FadeState {
    last_val: f32,
    silence_run: u32,

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

static STATE: AtomicPtr<PluginState> = AtomicPtr::new(std::ptr::null_mut());

fn guard<F: FnOnce() -> mpl_result_t + std::panic::UnwindSafe>(f: F) -> mpl_result_t {

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

pub(crate) mod config {

    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

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

    static CHUNK_MS: AtomicU32 = AtomicU32::new(200);
    static LOOKAHEAD_MS: AtomicU32 = AtomicU32::new(80);
    static LEFT_CONTEXT_MS: AtomicU32 = AtomicU32::new(720);
    static CROSSFADE_MS: AtomicU32 = AtomicU32::new(50);
    static MAX_LATENCY_MS: AtomicU32 = AtomicU32::new(300);
    static JITTER_MS: AtomicU32 = AtomicU32::new(0);
    static F0_UP_KEY: AtomicI32 = AtomicI32::new(0);
    static SPEAKER_ID: AtomicI32 = AtomicI32::new(0);
    static GATE_ENABLED: AtomicBool = AtomicBool::new(true);

    static GATE_DB_BITS: AtomicU32 = AtomicU32::new((-80.0f32).to_bits());

    static ORT_OPT: AtomicU32 = AtomicU32::new(2);

    static MODEL_EPOCH: AtomicU32 = AtomicU32::new(0);

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

        pub ort_opt: u32,
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
                ort_opt: ORT_OPT.load(Ordering::Relaxed),
            }
        }

        #[inline]
        pub fn gate_rms(&self) -> f32 {
            if self.gate_enabled {
                10.0f32.powf(self.gate_db / 20.0)
            } else {
                0.0
            }
        }
    }

    #[inline]
    fn clamp_u32(v: i32, r: (u32, u32)) -> u32 {
        (v.max(0) as u32).clamp(r.0, r.1)
    }

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
            "ort_opt_level" => {
                let v = match text {
                    "basic" | "0" | "level1" => 0,
                    "extended" | "1" | "level2" => 1,
                    "all" | "2" | "level3" | "99" => 2,
                    _ => return None,
                };
                if ORT_OPT.swap(v, Ordering::Relaxed) == v {
                    return Some(text.to_string());
                }

                MODEL_EPOCH.fetch_add(1, Ordering::Relaxed);
                Some(text.to_string())
            }
            _ => None,
        }
    }

    pub fn model_epoch() -> u32 {
        MODEL_EPOCH.load(Ordering::Relaxed)
    }

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

pub(crate) mod logger {
    use std::fs::{File, OpenOptions};
    use std::io::Write;
    use std::path::Path;
    use std::sync::Mutex;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    struct Inner {
        file: Option<File>,
        last_msg: String,

        last_emit: Option<Instant>,
        suppressed: u32,
    }

    static LOG: Mutex<Inner> = Mutex::new(Inner {
        file: None,
        last_msg: String::new(),
        last_emit: None,
        suppressed: 0,
    });

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
        format!("{secs}")
    }
}

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

fn reload_config() -> bool {
    let before = config::Params::load();
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
        "ort_opt_level",
    ];
    unsafe {
        with_host(|host| {
            let mut has_window_keys = false;
            for key in keys {
                if let Some(raw) = read_config(host, key) {
                    if matches!(key, "lookahead_ms" | "left_context_ms") {
                        has_window_keys = true;
                    }
                    config::set_from_raw(key, &raw);
                }
            }
            if !has_window_keys {
                if let Some(raw) = read_config(host, "extra_ms") {
                    config::set_from_raw("extra_ms", &raw);
                }
            }
        });
    }
    config::Params::load() != before
}

fn log_effective_params() {
    let p = config::Params::load();
    logger::log(&format!(
        "[Config] chunk={}ms lookahead={}ms left={}ms crossfade={}ms max_latency={}ms jitter={}ms key={} sid={} gate={}({}dB) ort_opt={}",
        p.chunk_ms, p.lookahead_ms, p.left_context_ms, p.crossfade_ms,
        p.max_latency_ms, p.jitter_ms, p.f0_up_key, p.speaker_id,
        p.gate_enabled, p.gate_db, ["basic", "extended", "all"][p.ort_opt as usize % 3]
    ));
}

#[no_mangle]
pub unsafe extern "C" fn micyou_plugin_init(host: *const mpl_host_api_t) -> mpl_result_t {
    guard(|| {
        if host.is_null() {
            return mpl_result_t::MPL_ERR_INVALID_ARG;
        }

        if let Ok(mut slot) = HOST_API.lock() {

            *slot = Some(unsafe { *host });
        } else {
            return mpl_result_t::MPL_ERR_RUNTIME;
        }

        let plugin_dir = rvc::plugin_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        logger::init(&plugin_dir);
        logger::log(&format!(
            "[Init] v{} dir={}",
            std::env!("CARGO_PKG_VERSION"), plugin_dir.display()
        ));

        reload_config();
        log_effective_params();

        let sample_rate = 48_000usize;
        let capacity = sample_rate * RING_SECONDS;
        let input_rb = Arc::new(AudioRingBuffer::new(capacity));
        let output_rb = Arc::new(AudioRingBuffer::new(capacity));

        let (signal_tx, signal_rx) = bounded::<()>(4);
        let is_running = Arc::new(AtomicBool::new(true));
        let resync = Arc::new(AtomicU64::new(0));
        let flush_out = Arc::new(AtomicBool::new(false));
        let underruns = Arc::new(AtomicU64::new(0));
        let status = stream::new_status();

        let handles = stream::WorkerHandles {
            input_rb: input_rb.clone(),
            output_rb: output_rb.clone(),
            signal_rx,
            is_running: is_running.clone(),
            resync: resync.clone(),
            flush_out: flush_out.clone(),
            underruns: underruns.clone(),
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

        let mut interval_id: u64 = 0;
        with_host(|h| {
            if let (Some(set_interval), Ok(payload)) = (h.set_interval, CString::new("status")) {
                let mut id: u64 = 0;

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
            underruns,
            status,
            worker: Some(worker),
            interval_id: AtomicU64::new(interval_id),
            scratch: UnsafeCell::new(vec![0.0f32; MAX_FRAMES]),
            fade: UnsafeCell::new(FadeState::default()),
            reported: UnsafeCell::new(Reported::default()),
        });
        let prev = STATE.swap(Box::into_raw(state), Ordering::AcqRel);
        if !prev.is_null() {

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

        let mut state = unsafe { Box::from_raw(ptr) };

        let id = state.interval_id.load(Ordering::Relaxed);
        if id != 0 {
            with_host(|h| {
                if let Some(clear) = h.clear_interval {

                    unsafe { clear(h.ctx, id) };
                }
            });
        }

        state.is_running.store(false, Ordering::Relaxed);

        let _ = state.signal_tx.try_send(());
        if let Some(handle) = state.worker.take() {
            let _ = handle.join();
        }
        logger::log("[Deinit] 完成");

        mpl_result_t::MPL_OK
    });
}

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

        unsafe { *bypass = 1 };
        if total == 0 || ch == 0 || total % ch != 0 {
            return mpl_result_t::MPL_OK;
        }
        let frames = total / ch;

        let ptr = STATE.load(Ordering::Acquire);
        if ptr.is_null() {
            return mpl_result_t::MPL_OK;
        }

        let state = unsafe { &*ptr };
        if frames > MAX_FRAMES {

            return mpl_result_t::MPL_OK;
        }

        let buf = unsafe { std::slice::from_raw_parts_mut(data, total) };

        let (scratch, fade) = unsafe { (&mut *state.scratch.get(), &mut *state.fade.get()) };

        if state.flush_out.swap(false, Ordering::AcqRel) {
            state.output_rb.discard(state.output_rb.available());
            *fade = FadeState::default();
        }

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

        if state.input_rb.push(&scratch[..frames]) == 0 {
            if !fade.overflow_latched {
                fade.overflow_latched = true;
                state.resync.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            fade.overflow_latched = false;
        }
        let _ = state.signal_tx.try_send(());

        if state.output_rb.available() >= frames {
            state.output_rb.pop(&mut scratch[..frames]);
            if fade.silence_run > 0 {

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

            state.underruns.fetch_add(1, Ordering::Relaxed);
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

        unsafe { *bypass = 0 };
        mpl_result_t::MPL_OK
    })
}

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

        let topic = unsafe { CStr::from_ptr(topic) }.to_str().unwrap_or("");
        match topic {
            "config:changed" => {
                if reload_config() {
                    log_effective_params();
                }
            }
            "interval:tick" => {

                let body = unsafe { payload_string(payload, len) };
                if body.contains("status") {
                    report_status();
                    let ptr = STATE.load(Ordering::Acquire);
                    let do_poll = if ptr.is_null() {
                        false
                    } else {

                        let reported = unsafe { &mut *(*ptr).reported.get() };
                        reported.ticks = reported.ticks.wrapping_add(1);
                        reported.ticks % CONFIG_POLL_TICKS == 0
                    };
                    if do_poll && reload_config() {
                        log_effective_params();
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

        let (t, j) = unsafe {
            (
                if event_type.is_null() { "" } else { CStr::from_ptr(event_type).to_str().unwrap_or("") },
                if json.is_null() { "" } else { CStr::from_ptr(json).to_str().unwrap_or("") },
            )
        };
        logger::log(&format!("[Event] {t} {j}"));

        if t == "state_changed" {
            let ptr = STATE.load(Ordering::Acquire);
            if !ptr.is_null() {

                let state = unsafe { &*ptr };
                state.resync.fetch_add(1, Ordering::Relaxed);
                state.flush_out.store(true, Ordering::Release);
            }
        }
        mpl_result_t::MPL_OK
    })
}

unsafe fn payload_string(payload: *const u8, len: u32) -> String {
    if payload.is_null() || len == 0 {
        return String::new();
    }
    String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(payload, len as usize) }).into_owned()
}

fn report_status() {
    let ptr = STATE.load(Ordering::Acquire);
    if ptr.is_null() {
        return;
    }

    let state = unsafe { &*ptr };
    let snapshot = match state.status.lock() {
        Ok(s) => s.clone(),
        Err(_) => return,
    };

    let reported = unsafe { &mut *state.reported.get() };
    let phase = phase_u8(snapshot.phase);
    if reported.revision == snapshot.revision && reported.phase == phase {
        return;
    }
    let changed_phase = reported.phase != phase;
    reported.revision = snapshot.revision;
    reported.phase = phase;

    let line = format!(
        "[RVC] {:?} | 模型 {} | 已处理 {} 块 | 断流 {} 次{}",
        snapshot.phase,
        snapshot.model_label,
        snapshot.blocks,
        snapshot.flushes,
        if snapshot.detail.is_empty() { String::new() } else { format!(" | {}", snapshot.detail) }
    );
    logger::log(&line);

    unsafe { host_log(2, &line) };

    if changed_phase {

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
