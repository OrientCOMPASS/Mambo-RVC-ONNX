#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use crossbeam_channel::{bounded, Sender};

mod logger;
mod path_resolver;
mod ring_buffer;
mod worker;

use ring_buffer::AudioRingBuffer;

const MPL_ABI_VERSION: u32 = 1;
const MPL_API_VERSION: u32 = 1;
const PLUGIN_ID: &[u8] = b"opss.mambo-rvc-onnx\0";

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum mpl_result_t { 
    MPL_OK = 0, MPL_ERR_NOT_IMPLEMENTED = 1, MPL_ERR_INVALID_ARG = 2,
    MPL_ERR_RUNTIME = 3, MPL_ERR_BUFFER_TOO_SMALL = 4, MPL_ERR_PERMISSION = 5,
}

#[repr(C)]
pub struct mpl_plugin_info_t {
    pub abi_version: u32,
    pub api_version: u32,
    pub id: *const c_char,
    pub version: *const c_char,
}
unsafe impl Sync for mpl_plugin_info_t {}

// 严格复刻 MicYou 官方完整 API 布局，使用 Option 安全处理 NULL 指针
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

// 全局线程安全配置
static CHUNK_MS: AtomicU32 = AtomicU32::new(100);
static EXTRA_MS: AtomicU32 = AtomicU32::new(400);
static CROSSFADE_MS: AtomicU32 = AtomicU32::new(20);
static F0_UP_KEY: AtomicI32 = AtomicI32::new(0);

static HOST_API: Mutex<Option<mpl_host_api_t>> = Mutex::new(None);

struct PluginState {
    input_rb: Arc<AudioRingBuffer>,
    output_rb: Arc<AudioRingBuffer>,
    signal_tx: Sender<()>,
    is_running: Arc<AtomicBool>,
    worker_handle: Option<thread::JoinHandle<()>>,
    mono_buffer: Vec<f32>,
    max_frames: usize,
}

static mut STATE: Option<PluginState> = None;

fn guard<F: FnOnce() -> mpl_result_t + std::panic::UnwindSafe>(f: F) -> mpl_result_t {
    std::panic::catch_unwind(f).unwrap_or(mpl_result_t::MPL_ERR_RUNTIME)
}

#[no_mangle]
pub extern "C" fn micyou_plugin_info() -> *const mpl_plugin_info_t {
    static INFO: mpl_plugin_info_t = mpl_plugin_info_t {
        abi_version: MPL_ABI_VERSION,
        api_version: MPL_API_VERSION,
        id: PLUGIN_ID.as_ptr() as *const c_char,
        version: b"0.1.0\0".as_ptr() as *const c_char,
    };
    &INFO
}

// 安全的两步配置读取法
unsafe fn get_config_string(host: &mpl_host_api_t, key: &str) -> Option<String> {
    let get_config = host.get_config?;
    let c_key = CString::new(key).ok()?;
    
    let mut out_size: u32 = 0;
    let mut dummy = [0u8; 1];
    get_config(host.ctx, c_key.as_ptr(), dummy.as_mut_ptr() as *mut c_char, &mut out_size);
    
    if out_size > 0 {
        let mut buffer = vec![0u8; out_size as usize + 1];
        let res = get_config(host.ctx, c_key.as_ptr(), buffer.as_mut_ptr() as *mut c_char, &mut out_size);
        if res == mpl_result_t::MPL_OK {
            buffer.truncate(out_size as usize);
            let s = String::from_utf8_lossy(&buffer).to_string();
            return Some(s.trim_matches('"').to_string());
        }
    }
    None
}

fn reload_config() {
    if let Ok(guard) = HOST_API.lock() {
        if let Some(host) = guard.as_ref() {
            unsafe {
                if let Some(val) = get_config_string(host, "chunk_ms") {
                    if let Ok(ms) = val.parse::<u32>() {
                        CHUNK_MS.store(ms.clamp(50, 1000), Ordering::Relaxed);
                    }
                }
                if let Some(val) = get_config_string(host, "extra_ms") {
                    if let Ok(ms) = val.parse::<u32>() {
                        EXTRA_MS.store(ms.clamp(50, 1000), Ordering::Relaxed);
                    }
                }
                if let Some(val) = get_config_string(host, "crossfade_ms") {
                    if let Ok(ms) = val.parse::<u32>() {
                        CROSSFADE_MS.store(ms.clamp(5, 100), Ordering::Relaxed);
                    }
                }
                if let Some(val) = get_config_string(host, "f0_up_key") {
                    if let Ok(key) = val.parse::<i32>() {
                        F0_UP_KEY.store(key.clamp(-24, 24), Ordering::Relaxed);
                    }
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn micyou_plugin_init(host: *const mpl_host_api_t) -> mpl_result_t {
    guard(|| {
        if host.is_null() { return mpl_result_t::MPL_ERR_INVALID_ARG; }
        
        let h = &*host;
        *HOST_API.lock().unwrap() = Some(*h);

        if let Some(plugin_dir) = path_resolver::get_plugin_dir() {
            logger::init(&plugin_dir);
        } else {
            logger::init(std::path::Path::new("."));
        }

        reload_config();

        let sample_rate = 48000;
        let hop = 480;
        let speaker = 0;
        let max_frames = 4096;

        let buffer_capacity = sample_rate * 10; 
        let input_rb = Arc::new(AudioRingBuffer::new(buffer_capacity));
        let output_rb = Arc::new(AudioRingBuffer::new(buffer_capacity));

        let (signal_tx, signal_rx) = bounded::<()>(200);
        let is_running = Arc::new(AtomicBool::new(true));

        let cfg = worker::WorkerConfig {
            sample_rate, hop, speaker,
            model_path: "models/uma-Matikane_Tannhauser.onnx".to_string(),
            hubert_path: "models/hubert_base.onnx".to_string(),
            rmvpe_path: "models/rmvpe.onnx".to_string(),
        };

        let input_rb_worker = input_rb.clone();
        let output_rb_worker = output_rb.clone();
        let is_running_worker = is_running.clone();

        let handle = thread::spawn(move || {
            worker::worker_loop(cfg, input_rb_worker, output_rb_worker, signal_rx, is_running_worker);
        });

        unsafe {
            STATE = Some(PluginState {
                input_rb, output_rb, signal_tx, is_running,
                worker_handle: Some(handle),
                mono_buffer: vec![0.0; max_frames],
                max_frames,
            });
        }
        mpl_result_t::MPL_OK
    })
}

#[no_mangle]
pub extern "C" fn micyou_plugin_handle_message(
    _source: *const c_char, topic: *const c_char, _payload: *const u8, _len: u32,
) -> mpl_result_t {
    unsafe {
        if topic.is_null() { return mpl_result_t::MPL_ERR_INVALID_ARG; }
        let topic_str = CStr::from_ptr(topic).to_str().unwrap_or("");
        if topic_str == "config:changed" {
            reload_config();
        }
    }
    mpl_result_t::MPL_OK
}

#[no_mangle]
pub unsafe extern "C" fn micyou_plugin_process(
    data: *mut f32, samples: u32, channels: u32, _queued_ms: f64, bypass: *mut u32,
) -> mpl_result_t {
    guard(|| {
        if data.is_null() || bypass.is_null() { return mpl_result_t::MPL_ERR_INVALID_ARG; }
        let total_samples = samples as usize;
        let num_channels = channels as usize;
        if total_samples == 0 || num_channels == 0 { unsafe { *bypass = 1; } return mpl_result_t::MPL_OK; }
        let num_frames = total_samples / num_channels;

        let state = match unsafe { STATE.as_mut() } {
            Some(s) => s, None => { unsafe { *bypass = 1; } return mpl_result_t::MPL_OK; }
        };
        if num_frames > state.max_frames { unsafe { *bypass = 1; } return mpl_result_t::MPL_OK; }

        let data_slice = std::slice::from_raw_parts_mut(data, total_samples);

        for i in 0..num_frames {
            let mut sum = 0.0f32;
            for ch in 0..num_channels { sum += data_slice[i * num_channels + ch]; }
            state.mono_buffer[i] = sum / num_channels as f32;
        }

        let _ = state.input_rb.push(&state.mono_buffer[..num_frames]);
        let _ = state.signal_tx.try_send(());

        if state.output_rb.available() >= num_frames {
            state.output_rb.pop(&mut state.mono_buffer[..num_frames]);
            for i in (0..num_frames).rev() {
                let val = state.mono_buffer[i];
                for ch in 0..num_channels { data_slice[i * num_channels + ch] = val; }
            }
        } else {
            for i in 0..total_samples { data_slice[i] = 0.0; }
        }

        unsafe { *bypass = 0; }
        mpl_result_t::MPL_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn micyou_plugin_deinit() {
    let _ = guard(|| {
        if let Some(mut state) = unsafe { STATE.take() } {
            state.is_running.store(false, Ordering::Relaxed);
            drop(state.signal_tx);
            if let Some(handle) = state.worker_handle.take() { let _ = handle.join(); }
        }
        mpl_result_t::MPL_OK
    });
}