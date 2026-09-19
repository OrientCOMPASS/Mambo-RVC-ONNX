//! 模型层：定位插件目录与 CUDA 运行库、发现并加载模型、跑 HuBERT / RMVPE / RVC。
//!
//! ## 用户模型优先
//! `discover()` 的搜索顺序（见函数文档）保证：**用户自己放进插件目录的 RVC 模型优先**，
//! 找不到才回退到插件自带模型。设置里的 `model_file` 可以显式指定；不填就自动挑
//! `user_models/` 下修改时间最新的 `.onnx`。
//!
//! ## 性能
//! 单块推理耗时 τ 必须小于 chunk_ms，否则会持续积压并触发断流。为此这里做了三件事：
//! - `GraphOptimizationLevel::Level3`（1.0 用的 Level1 少了 attention/layout 融合，对 transformer 慢数倍）；
//! - RMVPE 的 mel 前端全部缓存：Hann 窗、128×513 滤波器组、FFT plan、所有中间缓冲只建一次
//!   （1.0 每块都重建 planner + 重算 mel_basis + 分配数 MB）；magnitude 改成 `[frame][freq]`
//!   连续布局，mel 矩阵乘从跨步访存变成顺序访存；
//! - f0 后处理不再每帧分配 Vec；中值滤波用 `total_cmp`，遇到 NaN 不会 panic。

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{bail, Result};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionOutputs};
use ort::value::{DynValue, Tensor};
use rustfft::{num_complex::Complex, FftPlanner};

use crate::logger;

// ────────────────────────── 插件目录 / DLL 引导 ──────────────────────────
#[cfg(target_os = "windows")]
mod imp {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::path::PathBuf;

    type HMODULE = *mut std::ffi::c_void;
    type WCHAR = u16;
    type DWORD = u32;
    type BOOL = i32;

    extern "system" {
        fn GetModuleHandleExW(dw_flags: DWORD, lp_module_name: *const WCHAR, ph_module: *mut HMODULE) -> BOOL;
        fn GetModuleFileNameW(h_module: HMODULE, lp_filename: *mut WCHAR, n_size: DWORD) -> DWORD;
        fn SetDllDirectoryW(lp_path_name: *const WCHAR) -> BOOL;
    }

    const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: DWORD = 0x0000_0004;
    const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: DWORD = 0x0000_0002;

    /// 本 cdylib 自己所在的目录（不是宿主 exe 的目录）。
    pub fn current_module_dir() -> Option<PathBuf> {
        unsafe {
            let mut module: HMODULE = std::ptr::null_mut();
            // 用本函数的地址反查所属模块，避免拿到宿主 exe 的路径
            let address = (current_module_dir as usize) as *const WCHAR;
            let flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
            if GetModuleHandleExW(flags, address, &mut module) == 0 || module.is_null() {
                return None;
            }
            let mut buffer = vec![0u16; 1024];
            let len = GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as DWORD) as usize;
            if len == 0 || len >= buffer.len() {
                return None;
            }
            PathBuf::from(OsString::from_wide(&buffer[..len])).parent().map(|p| p.to_path_buf())
        }
    }

    /// 把 `dir` 加入进程 DLL 搜索路径。
    ///
    /// ⚠️ `SetDllDirectoryW` 是**进程级全局单槽**设置：会影响宿主和其他插件，
    /// 多个插件同时调用会互相覆盖。这里之所以还要用它，是因为 onnxruntime 本体
    /// 已经用 `ort::init_from(绝对路径)` 显式加载了，但它的 CUDA EP
    /// (`onnxruntime_providers_cuda.dll`) 依赖的 cudart/cublas/cudnn 仍然要走
    /// 标准搜索顺序，而这些 DLL 只存在于插件的 `libs/` 里。
    pub fn add_dll_directory(dir: &std::path::Path) -> bool {
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        unsafe { SetDllDirectoryW(wide.as_ptr()) != 0 }
    }
}

#[cfg(not(target_os = "windows"))]
mod imp {
    use std::path::PathBuf;

    pub fn current_module_dir() -> Option<PathBuf> {
        // 非 Windows 平台仅用于开发期 cargo check；dladdr 才是正确做法。
        std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()))
    }

    pub fn add_dll_directory(_dir: &std::path::Path) -> bool {
        true
    }
}

pub fn plugin_dir() -> Option<PathBuf> {
    imp::current_module_dir()
}

/// onnxruntime 动态库的绝对路径（存在则返回）。
pub fn ort_dylib_path(plugin_dir: &std::path::Path) -> Option<PathBuf> {
    let libs = plugin_dir.join("libs");
    let name = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    let path = libs.join(name);
    path.exists().then_some(path)
}

/// 在加载 ORT 之前调用一次：把 `libs/` 加入 DLL 搜索路径。
pub fn prepare_runtime(plugin_dir: &std::path::Path) {
    let libs = plugin_dir.join("libs");
    if libs.is_dir() {
        imp::add_dll_directory(&libs);
    }
}

// ────────────────────────── 模型发现（用户模型优先） ──────────────────────────
pub const USER_MODEL_DIR: &str = "user_models";
pub const BUNDLED_RVC: &str = "models/uma-Matikane_Tannhauser.onnx";
pub const BUNDLED_HUBERT: &str = "models/hubert_base.onnx";
pub const BUNDLED_RMVPE: &str = "models/rmvpe.onnx";

#[derive(Debug, Clone)]
pub struct ModelSet {
    pub rvc: PathBuf,
    pub hubert: PathBuf,
    pub rmvpe: PathBuf,
    /// 给用户看的来源说明，例如 `user_models/my_voice.onnx (用户模型)`
    pub rvc_label: String,
    pub rvc_source: Source,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// 配置项显式指定
    Config,
    /// user_models/ 自动发现
    UserDir,
    /// 插件自带
    Bundled,
}

impl Source {
    pub fn label(&self) -> &'static str {
        match self {
            Source::Config => "配置指定",
            Source::UserDir => "用户模型",
            Source::Bundled => "插件自带",
        }
    }
}

/// 确保 `user_models/` 存在，并放一个说明文件（失败不影响功能）。
pub fn ensure_user_dir(plugin_dir: &Path) -> PathBuf {
    let dir = plugin_dir.join(USER_MODEL_DIR);
    if fs::create_dir_all(&dir).is_ok() {
        let readme = dir.join("把你的模型放这里.txt");
        if !readme.exists() {
            let _ = fs::write(
                &readme,
                "把你自己的 RVC 模型（.onnx）直接放进这个文件夹即可，插件会优先加载它。\r\n\
                 \r\n\
                 规则：\r\n\
                 1. 放多个模型时，使用「修改时间最新」的那个；\r\n\
                 2. 也可以在插件设置里用 model_file 明确指定文件名（例如 user_models/我的音色.onnx）；\r\n\
                 3. 文件名里含 hubert / contentvec 的会当成内容编码器，含 rmvpe 的会当成音高提取器；\r\n\
                 4. 找不到用户模型时自动回退到插件自带的 models/uma-Matikane_Tannhauser.onnx；\r\n\
                 5. 实际加载了哪个模型，看插件目录下的 rvc_plugin.log，或启用插件后弹出的系统通知。\r\n",
            );
        }
    }
    dir
}

fn list_onnx(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .map(|x| x.to_string_lossy().eq_ignore_ascii_case("onnx"))
                    .unwrap_or(false)
        })
        .collect();
    // 修改时间最新的排前面；同时间按文件名，保证可复现
    files.sort_by(|a, b| mtime(b).cmp(&mtime(a)).then_with(|| a.file_name().cmp(&b.file_name())));
    files
}

fn mtime(p: &Path) -> SystemTime {
    fs::metadata(p)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

fn rel(plugin_dir: &Path, p: &Path) -> String {
    p.strip_prefix(plugin_dir)
        .map(|r| r.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| p.to_string_lossy().into_owned())
}

/// 解析用户填的 `model_file`：允许 `my.onnx`、`user_models/my.onnx`、`models/my.onnx`。
fn resolve_user_spec(plugin_dir: &Path, spec: &str) -> Option<PathBuf> {
    let clean = spec.trim().replace('\\', "/");
    if clean.is_empty() {
        return None;
    }
    if clean.starts_with('/') || clean.contains(":") || clean.contains("..") {
        logger::log(&format!("[Model] model_file 被拒绝（必须是插件目录内的相对路径）: {clean}"));
        return None;
    }
    let direct = plugin_dir.join(&clean);
    if direct.is_file() {
        return Some(direct);
    }
    // 只给了文件名时，按搜索目录顺序找一遍
    for dir in [USER_MODEL_DIR, "models/user", "models", ""] {
        let cand = if dir.is_empty() {
            plugin_dir.join(&clean)
        } else {
            plugin_dir.join(dir).join(&clean)
        };
        if cand.is_file() {
            return Some(cand);
        }
    }
    logger::log(&format!("[Model] model_file 指定的文件不存在: {clean}"));
    None
}

fn pick_by_keyword(files: &[PathBuf], keywords: &[&str]) -> Option<PathBuf> {
    for kw in keywords {
        if let Some(p) = files
            .iter()
            .find(|p| {
                p.file_stem()
                    .map(|s| s.to_string_lossy().to_lowercase().contains(kw))
                    .unwrap_or(false)
            })
        {
            return Some(p.clone());
        }
    }
    None
}

/// 完整的模型发现流程。`user_model_file` 来自配置（可为空）。
pub fn discover(plugin_dir: &Path, user_model_file: &str) -> ModelSet {
    let user_dir = ensure_user_dir(plugin_dir);
    let user_files = list_onnx(&user_dir);
    let models_user_files = list_onnx(&plugin_dir.join("models/user"));
    let models_files = list_onnx(&plugin_dir.join("models"));

    // ---- RVC 主模型 ----
    let (rvc, source) = if let Some(p) = resolve_user_spec(plugin_dir, user_model_file) {
        (p, Source::Config)
    } else if let Some(p) = user_files
        .iter()
        .find(|p| !is_feature_model(p))
        .cloned()
        .or_else(|| models_user_files.iter().find(|p| !is_feature_model(p)).cloned())
    {
        (p, Source::UserDir)
    } else {
        let bundled = plugin_dir.join(BUNDLED_RVC);
        if bundled.is_file() {
            (bundled, Source::Bundled)
        } else {
            // 兜底：models/ 下任何不是 hubert/rmvpe 的 onnx
            let fallback = models_files.iter().find(|p| !is_feature_model(p)).cloned();
            (fallback.unwrap_or(bundled), Source::Bundled)
        }
    };

    // ---- HuBERT / ContentVec（也允许用户覆盖）----
    let hubert_kw = ["hubert", "contentvec", "content_vec"];
    let hubert = pick_by_keyword(&user_files, &hubert_kw)
        .or_else(|| pick_by_keyword(&models_files, &hubert_kw))
        .unwrap_or_else(|| plugin_dir.join(BUNDLED_HUBERT));

    // ---- RMVPE ----
    let rmvpe_kw = ["rmvpe", "f0"];
    let rmvpe = pick_by_keyword(&user_files, &rmvpe_kw)
        .or_else(|| pick_by_keyword(&models_files, &rmvpe_kw))
        .unwrap_or_else(|| plugin_dir.join(BUNDLED_RMVPE));

    let rvc_label = format!("{} ({})", rel(plugin_dir, &rvc), source.label());
    logger::log(&format!("[Model] RVC    = {rvc_label}"));
    logger::log(&format!("[Model] HuBERT = {}", rel(plugin_dir, &hubert)));
    logger::log(&format!("[Model] RMVPE  = {}", rel(plugin_dir, &rmvpe)));
    if !user_files.is_empty() {
        let names: Vec<String> = user_files.iter().map(|p| rel(plugin_dir, p)).collect();
        logger::log(&format!("[Model] user_models/ 候选: {}", names.join(", ")));
    }

    ModelSet {
        rvc,
        hubert,
        rmvpe,
        rvc_label,
        rvc_source: source,
    }
}

/// hubert / rmvpe 这类“特征模型”不能当 RVC 主模型用，自动发现时要跳过。
fn is_feature_model(p: &Path) -> bool {
    let name = p
        .file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    ["hubert", "contentvec", "content_vec", "rmvpe", "crepe", "fcpe", "rmvpe_v2"]
        .iter()
        .any(|kw| name.contains(kw))
}

// ────────────────────────── ONNX 会话与推理 ──────────────────────────
pub const HOP_48K: usize = 480; // 48kHz / 100fps
pub const FEAT_DIM: usize = 768; // HuBERT 隐藏维度

// ─────────────────────────── 会话构建 ───────────────────────────

/// 用 CUDA EP 建会话；CUDA 不可用时**直接报错**而不是静默回退 CPU
/// （CPU 推理必然慢于实时，会立刻表现为复读/断流，必须让它响亮地失败）。
pub fn load_session(path: &str, allow_cpu_fallback: bool) -> Result<Session> {
    let cuda = ort::ep::CUDA::default().with_device_id(0).build();
    let mut builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("builder: {e}"))?
        // Level3 = 全部图优化（含 attention / layout 融合）。
        // 1.0 用的 Level1 只做基础重写，对 transformer 结构会慢数倍。
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("optimization level: {e}"))?;

    if allow_cpu_fallback {
        builder = builder
            .with_execution_providers([cuda])
            .map_err(|e| anyhow::anyhow!("EP register: {e}"))?;
    } else {
        builder = builder
            .with_execution_providers([cuda.error_on_failure()])
            .map_err(|e| anyhow::anyhow!("EP register: {e}"))?;
    }

    builder
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("load {path}: {e}"))
}

/// 打印模型输入签名，方便用户排查“我自己的模型为什么跑不起来”。
pub fn log_inputs(tag: &str, session: &Session) {
    let names: Vec<String> = session
        .inputs()
        .iter()
        .map(|i| format!("{}:{:?}", i.name(), i.dtype()))
        .collect();
    logger::log(&format!("[Model] {tag} inputs = [{}]", names.join(", ")));
}

// ─────────────────────────── HuBERT ───────────────────────────

pub struct HubertExtractor {
    session: Session,
    /// 归一化后的输入缓冲，复用避免每块分配
    buf: Vec<f32>,
}

impl HubertExtractor {
    pub fn new(session: Session) -> Self {
        Self { session, buf: Vec::new() }
    }

    /// 返回展平的内容特征，长度是 `FEAT_DIM` 的整数倍（帧数 = len / 768，约 50fps）。
    pub fn extract(&mut self, pcm16k: &[f32]) -> Result<Vec<f32>> {
        if pcm16k.is_empty() {
            bail!("empty input");
        }
        // 峰值归一化到 0.9（增益上限 10 倍），与 RVC 参考实现一致
        self.buf.clear();
        self.buf.extend_from_slice(pcm16k);
        let peak = self.buf.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        if peak > 1e-6 {
            let gain = (0.9 / peak).min(10.0);
            for s in self.buf.iter_mut() {
                *s *= gain;
            }
        }

        let len = self.buf.len() as i64;
        let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(self.session.inputs().len());
        for input in self.session.inputs().iter() {
            let name = input.name().to_lowercase();
            if name.contains("feats")
                || name.contains("source")
                || name.contains("input_values")
                || name.contains("hubert")
                || name == "input"
                || name == "audio"
            {
                inputs.push((
                    input.name().to_string(),
                    Tensor::from_array((vec![1, 1, len], self.buf.clone()))?.into_dyn(),
                ));
            } else if name.contains("mask") {
                inputs.push((
                    input.name().to_string(),
                    Tensor::from_array((vec![1, len], vec![1i64; len as usize]))?.into_dyn(),
                ));
            } else if name.contains("length") || name.contains("_len") {
                inputs.push((
                    input.name().to_string(),
                    Tensor::from_array((vec![1], vec![len]))?.into_dyn(),
                ));
            }
        }
        if inputs.is_empty() {
            bail!("no recognizable audio input (names: {:?})",
                self.session.inputs().iter().map(|i| i.name().to_string()).collect::<Vec<_>>());
        }

        let outputs = self.session.run(inputs)?;
        for (_name, value) in outputs.iter() {
            if let Ok(v) = value.try_extract_tensor::<f32>() {
                let data = v.1;
                if data.len() >= FEAT_DIM && data.len() % FEAT_DIM == 0 {
                    return Ok(data.to_vec());
                }
            }
        }
        bail!("failed to extract phone features (no [.., 768] output)");
    }
}

// ─────────────────────────── RMVPE mel 前端（全缓存）───────────────────────────

const N_FFT: usize = 1024;
const MEL_HOP: usize = 160; // 16kHz
const N_MELS: usize = 128;
const N_FREQS: usize = N_FFT / 2 + 1;
const MEL_FMIN: f32 = 30.0;
const MEL_FMAX: f32 = 8000.0;
const MEL_CLAMP: f32 = 1e-5;
const MEL_SR: f32 = 16000.0;

struct MelFrontend {
    window: Vec<f32>,
    /// `[N_MELS][N_FREQS]` 行主序
    mel_basis: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32>>,
    // 复用缓冲
    padded: Vec<f32>,
    spec: Vec<Complex<f32>>,
    /// `[num_frames][N_FREQS]`
    mag: Vec<f32>,
    /// `[num_frames][N_MELS]`
    mel_frames: Vec<f32>,
    cap_frames: usize,
}

impl MelFrontend {
    fn new() -> Self {
        let hann = |n: usize, i: usize| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
        let window: Vec<f32> = (0..N_FFT).map(|i| hann(N_FFT, i)).collect();

        let mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
        let hz = |m: f32| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0);
        let mel_min = mel(MEL_FMIN);
        let mel_max = mel(MEL_FMAX);
        let mut points = vec![0.0f32; N_MELS + 2];
        for i in 0..N_MELS + 2 {
            points[i] = hz(mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32);
        }
        let mut mel_basis = vec![0.0f32; N_MELS * N_FREQS];
        for m in 0..N_MELS {
            let (left, center, right) = (points[m], points[m + 1], points[m + 2]);
            let enorm = 2.0 / (right - left);
            for k in 0..N_FREQS {
                let f = (MEL_SR / N_FFT as f32) * k as f32;
                let w = if center > left && f >= left && f <= center {
                    (f - left) / (center - left)
                } else if right > center && f > center && f <= right {
                    (right - f) / (right - center)
                } else {
                    0.0
                };
                mel_basis[m * N_FREQS + k] = w * enorm;
            }
        }

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        Self {
            window,
            mel_basis,
            fft,
            padded: Vec::new(),
            spec: vec![Complex::new(0.0, 0.0); N_FFT],
            mag: Vec::new(),
            mel_frames: Vec::new(),
            cap_frames: 0,
        }
    }

    fn ensure_cap(&mut self, frames: usize) {
        if frames > self.cap_frames {
            self.cap_frames = frames;
            self.mag = vec![0.0; frames * N_FREQS];
            self.mel_frames = vec![0.0; frames * N_MELS];
        }
    }

    /// 返回 `(num_frames, total_frames)`，并把 log-mel 写进 `channel_major`
    /// （形状 `[N_MELS][total_frames]`，正是 RMVPE 的输入布局）。
    fn compute(&mut self, pcm16k: &[f32], channel_major: &mut Vec<f32>) -> (usize, usize) {
        let pad = N_FFT / 2;
        let n = pcm16k.len();
        self.padded.clear();
        self.padded.reserve(n + 2 * pad);
        // 反射填充（与 librosa reflect 一致，不含端点重复）
        for i in (1..=pad).rev() {
            self.padded.push(pcm16k[i.min(n.saturating_sub(1))]);
        }
        self.padded.extend_from_slice(pcm16k);
        for i in 1..=pad {
            self.padded.push(pcm16k[n.saturating_sub(2 + i)]);
        }

        let num_frames = if self.padded.len() >= N_FFT {
            (self.padded.len() - N_FFT) / MEL_HOP + 1
        } else {
            1
        };
        self.ensure_cap(num_frames);

        // STFT magnitude，写成 [frame][freq] 连续布局
        for f in 0..num_frames {
            let start = f * MEL_HOP;
            for i in 0..N_FFT {
                self.spec[i] = Complex::new(self.padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut self.spec);
            let row = &mut self.mag[f * N_FREQS..(f + 1) * N_FREQS];
            for k in 0..N_FREQS {
                row[k] = self.spec[k].norm();
            }
        }

        // mel 矩阵乘 + log，先写成 [frame][mel]（顺序写），最后一次性转置成 [mel][frame]
        for f in 0..num_frames {
            let mag_row = &self.mag[f * N_FREQS..(f + 1) * N_FREQS];
            let out_row = &mut self.mel_frames[f * N_MELS..(f + 1) * N_MELS];
            for m in 0..N_MELS {
                let basis = &self.mel_basis[m * N_FREQS..(m + 1) * N_FREQS];
                let mut sum = 0.0f32;
                for k in 0..N_FREQS {
                    sum += basis[k] * mag_row[k];
                }
                out_row[m] = sum.max(MEL_CLAMP).ln();
            }
        }

        // 帧数补齐到 32 的倍数（RMVPE 的卷积下采样要求）
        let pad_time = 32 * ((num_frames.saturating_sub(1)) / 32 + 1) - num_frames;
        let total = num_frames + pad_time;
        channel_major.clear();
        channel_major.reserve(N_MELS * total);
        for m in 0..N_MELS {
            for f in 0..num_frames {
                channel_major.push(self.mel_frames[f * N_MELS + m]);
            }
            // 反射填充尾部
            for i in 0..pad_time {
                let idx = num_frames.saturating_sub(2 + i);
                channel_major.push(self.mel_frames[idx * N_MELS + m]);
            }
        }
        (num_frames, total)
    }
}

pub struct F0Extractor {
    session: Session,
    mel: MelFrontend,
    mel_buf: Vec<f32>,
    cents: Vec<f32>,
    f0: Vec<f32>,
    salience: Vec<f32>,
}

impl F0Extractor {
    pub fn new(session: Session) -> Self {
        let mut cents = vec![0.0f32; 368];
        for i in 0..360 {
            cents[i + 4] = 20.0 * i as f32 + 1997.3794084376191;
        }
        Self {
            session,
            mel: MelFrontend::new(),
            mel_buf: Vec::new(),
            cents,
            f0: Vec::new(),
            salience: vec![0.0; 360],
        }
    }

    /// 提取 `frames` 个 100fps 的 f0（Hz，0 表示清音段）。返回的切片长度恒为 `frames`。
    pub fn extract(&mut self, pcm16k: &[f32], frames: usize) -> Result<&[f32]> {
        if frames == 0 {
            bail!("frames == 0");
        }
        let (num_frames, total) = self.mel.compute(pcm16k, &mut self.mel_buf);

        let inputs: Vec<(String, DynValue)> = self
            .session
            .inputs()
            .iter()
            .map(|i| {
                Ok((
                    i.name().to_string(),
                    Tensor::from_array((vec![1, N_MELS as i64, total as i64], self.mel_buf.clone()))?
                        .into_dyn(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let outputs = self.session.run(inputs)?;
        let hidden = pick_f32_output(&outputs, num_frames * 360)
            .ok_or_else(|| anyhow::anyhow!("RMVPE: no f32 output with len >= {}", num_frames * 360))?;
        let hidden = hidden.as_slice();

        self.f0.clear();
        self.f0.resize(frames, 0.0);
        let usable = frames.min(num_frames).min(hidden.len() / 360);
        for f in 0..usable {
            let base = f * 360;
            let mut center = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for b in 0..360 {
                let v = hidden[base + b];
                self.salience[b] = v;
                if v > max_val {
                    max_val = v;
                    center = b;
                }
            }
            if !(max_val > 0.03) {
                self.f0[f] = 0.0;
                continue;
            }
            // 抛物线插值（用 salience 加权重心，边界补零）
            let at = |i: usize| if i >= 4 && i < 364 { self.salience[i - 4] } else { 0.0 };
            let c = center + 4;
            let start = c.saturating_sub(4);
            let end = (c + 5).min(368);
            let mut product_sum = 0.0f32;
            let mut weight_sum = 0.0f32;
            for i in start..end {
                product_sum += at(i) * self.cents[i];
                weight_sum += at(i);
            }
            let divided = if weight_sum > 1e-8 { product_sum / weight_sum } else { 0.0 };
            let freq = 10.0 * 2.0f32.powf(divided / 1200.0);
            self.f0[f] = if (freq - 10.0).abs() < 1e-5 { 0.0 } else { freq };
        }

        // 3 点中值滤波，抑制八度跳变。
        // 注意必须读**原始**序列（和 RVC 参考实现一致），所以用两个滚动变量保存原值，
        // 不能直接读 self.f0[i-1]——它在上一轮已经被覆写了。
        if usable >= 3 {
            let mut orig_prev = self.f0[0];
            let mut orig_cur = self.f0[1];
            for i in 1..usable - 1 {
                let orig_next = self.f0[i + 1];
                let mid = median3(orig_prev, orig_cur, orig_next);
                self.f0[i] = mid;
                orig_prev = orig_cur;
                orig_cur = orig_next;
            }
        }
        Ok(&self.f0)
    }
}

fn median3(a: f32, b: f32, c: f32) -> f32 {
    let mut arr = [a, b, c];
    // total_cmp 对 NaN 也有全序，不会像 partial_cmp().unwrap() 那样 panic
    arr.sort_by(|x, y| x.total_cmp(y));
    arr[1]
}

/// 取第一个长度足够的 f32 输出。
///
/// 返回 owned `Vec` 而不是借用切片：`SessionOutputs::iter()` 产出的是临时借用，
/// 借用检查器不允许把它带出函数（1.0 版同样是 `to_vec()` 才编过）。
/// 拷贝量每块约 100KB（RVC 音频）/ 80KB（RMVPE hidden），10 块/秒 ≈ 1.8MB/s，
/// 相对几十毫秒的推理耗时可以忽略。
fn pick_f32_output(outputs: &SessionOutputs, min_len: usize) -> Option<Vec<f32>> {
    // 优先按下标 0（RVC/RMVPE 的主输出），否则找第一个够长的 f32 张量
    if let Ok(v) = outputs[0].try_extract_tensor::<f32>() {
        if v.1.len() >= min_len {
            return Some(v.1.to_vec());
        }
    }
    for (_name, value) in outputs.iter() {
        if let Ok(v) = value.try_extract_tensor::<f32>() {
            if v.1.len() >= min_len {
                return Some(v.1.to_vec());
            }
        }
    }
    None
}

// ─────────────────────────── RVC 合成 ───────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Phone,
    PhoneLen,
    Pitch,
    NsfF0,
    Sid,
    Unknown,
}

fn classify(name: &str) -> Role {
    let n = name.to_lowercase();
    // 顺序很重要：phone_lengths 同时含 "phone" 和 "len"
    if n.contains("len") {
        return Role::PhoneLen;
    }
    if n.contains("nsff0") || (n.contains("nsf") && n.contains("f0")) {
        return Role::NsfF0;
    }
    if n.contains("pitch") {
        return Role::Pitch;
    }
    if n.contains("phone") || n.contains("content") || n.contains("feats") || n.contains("hubert") {
        return Role::Phone;
    }
    if n.contains("sid") || n.contains("speaker") {
        return Role::Sid;
    }
    Role::Unknown
}

pub struct RvcSynth {
    session: Session,
    /// 预先算好的「输入名 -> 角色」映射，避免每块都做字符串匹配
    plan: Vec<(String, Role)>,
    phone_100: Vec<f32>,
    pitch: Vec<i64>,
    nsff0: Vec<f32>,
    logged: bool,
}

impl RvcSynth {
    pub fn new(session: Session) -> Self {
        let plan: Vec<(String, Role)> = session.inputs().iter().map(|i| (i.name().to_string(), classify(i.name()))).collect();
        let unknown: Vec<&str> = plan
            .iter()
            .filter(|(_, r)| *r == Role::Unknown)
            .map(|(n, _)| n.as_str())
            .collect();
        if !unknown.is_empty() {
            logger::log(&format!("[RVC] 无法识别的输入名: {:?}（将按位置回退）", unknown));
        }
        Self { session, plan, phone_100: Vec::new(), pitch: Vec::new(), nsff0: Vec::new(), logged: false }
    }

    /// `phone_50fps` 是展平的 HuBERT 特征（约 50fps），内部插值到 `frames`（100fps）。
    /// 返回模型输出的原始音频（采样率由调用方按长度推断）。
    pub fn synthesize(&mut self, frames: usize, sid: i64, f0: &[f32], phone_50fps: &[f32]) -> Result<Vec<f32>> {
        let frames_50 = phone_50fps.len() / FEAT_DIM;
        if frames_50 == 0 {
            bail!("hubert features empty");
        }
        if f0.len() < frames {
            bail!("f0 too short: {} < {}", f0.len(), frames);
        }

        // 50fps -> 100fps 线性插值（复用缓冲，不每块分配）
        self.phone_100.clear();
        self.phone_100.resize(frames * FEAT_DIM, 0.0);
        let denom = if frames > 1 { (frames - 1) as f32 } else { 1.0 };
        for i in 0..frames {
            let t = if frames > 1 { i as f32 * (frames_50 - 1) as f32 / denom } else { 0.0 };
            let t0 = (t.floor() as usize).min(frames_50 - 1);
            let t1 = (t0 + 1).min(frames_50 - 1);
            let alpha = t - t0 as f32;
            let src0 = &phone_50fps[t0 * FEAT_DIM..(t0 + 1) * FEAT_DIM];
            let src1 = &phone_50fps[t1 * FEAT_DIM..(t1 + 1) * FEAT_DIM];
            let dst = &mut self.phone_100[i * FEAT_DIM..(i + 1) * FEAT_DIM];
            for j in 0..FEAT_DIM {
                dst[j] = (1.0 - alpha) * src0[j] + alpha * src1[j];
            }
        }

        self.pitch.clear();
        self.pitch.extend(f0[..frames].iter().map(|&f| f0_coarse(f)));
        self.nsff0.clear();
        self.nsff0.extend_from_slice(&f0[..frames]);

        // 按名字装配；名字全都对不上时按标准顺序回退
        let named: usize = self.plan.iter().filter(|(_, r)| *r != Role::Unknown).count();
        let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(self.plan.len());
        if named > 0 {
            for (name, role) in self.plan.iter() {
                let value: DynValue = match role {
                    Role::Phone => Tensor::from_array((vec![1, frames as i64, FEAT_DIM as i64], self.phone_100.clone()))?.into_dyn(),
                    Role::PhoneLen => Tensor::from_array((vec![1], vec![frames as i64]))?.into_dyn(),
                    Role::Pitch => Tensor::from_array((vec![1, frames as i64], self.pitch.clone()))?.into_dyn(),
                    Role::NsfF0 => Tensor::from_array((vec![1, frames as i64], self.nsff0.clone()))?.into_dyn(),
                    Role::Sid => Tensor::from_array((vec![1], vec![sid]))?.into_dyn(),
                    Role::Unknown => continue,
                };
                inputs.push((name.clone(), value));
            }
        } else {
            // 标准 RVC 导出顺序：phone, phone_lengths, pitch, nsff0, sid
            let order = [Role::Phone, Role::PhoneLen, Role::Pitch, Role::NsfF0, Role::Sid];
            for (idx, role) in order.iter().enumerate() {
                let Some(name) = self.plan.get(idx).map(|(n, _)| n.clone()) else { break };
                let value: DynValue = match role {
                    Role::Phone => Tensor::from_array((vec![1, frames as i64, FEAT_DIM as i64], self.phone_100.clone()))?.into_dyn(),
                    Role::PhoneLen => Tensor::from_array((vec![1], vec![frames as i64]))?.into_dyn(),
                    Role::Pitch => Tensor::from_array((vec![1, frames as i64], self.pitch.clone()))?.into_dyn(),
                    Role::NsfF0 => Tensor::from_array((vec![1, frames as i64], self.nsff0.clone()))?.into_dyn(),
                    Role::Sid => Tensor::from_array((vec![1], vec![sid]))?.into_dyn(),
                    Role::Unknown => continue,
                };
                inputs.push((name, value));
            }
        }
        if inputs.is_empty() {
            bail!("model declares no usable inputs");
        }

        let outputs = self.session.run(inputs)?;
        let audio = pick_f32_output(&outputs, frames)
            .ok_or_else(|| anyhow::anyhow!("RVC: no audio output"))?;
        let audio = audio.as_slice();
        if !self.logged {
            self.logged = true;
            logger::log(&format!(
                "[RVC] 首块成功: frames={frames} 输出样本={} 推断hop={}",
                audio.len(),
                audio.len() / frames.max(1)
            ));
        }
        Ok(audio.to_vec())
    }
}

/// RVC 的 coarse pitch：mel 刻度量化到 1..255，0 表示清音。
pub fn f0_coarse(f: f32) -> i64 {
    if !(f > 0.0) {
        return 0;
    }
    let mel = 1127.0_f32 * (1.0_f32 + f / 700.0_f32).ln();
    let min_mel = 1127.0_f32 * (1.0_f32 + 40.0_f32 / 700.0_f32).ln();
    let max_mel = 1127.0_f32 * (1.0_f32 + 1100.0_f32 / 700.0_f32).ln();
    let x = (mel - min_mel) / (max_mel - min_mel) * 255.0_f32;
    if !x.is_finite() {
        return 0;
    }
    x.round().clamp(1.0_f32, 255.0_f32) as i64
}

// ─────────────────────────── 重采样（非 48k 模型兜底）───────────────────────────

/// 模型输出采样率不是 48kHz 时（例如 40k 模型 hop=400、32k 模型 hop=320），
/// 按实际长度比例线性重采样到目标长度。这是兜底路径：48k 模型 ratio≈1 时直接拷贝。
pub fn resample_into(src: &[f32], dst: &mut Vec<f32>, dst_len: usize) {
    dst.clear();
    if dst_len == 0 || src.is_empty() {
        dst.resize(dst_len, 0.0);
        return;
    }
    dst.reserve(dst_len);
    if src.len() == dst_len {
        dst.extend_from_slice(src);
        return;
    }
    let step = src.len() as f64 / dst_len as f64;
    let last = src.len() - 1;
    for i in 0..dst_len {
        let pos = i as f64 * step;
        let i0 = (pos.floor() as usize).min(last);
        let i1 = (i0 + 1).min(last);
        let frac = (pos - i0 as f64) as f32;
        dst.push(src[i0] * (1.0 - frac) + src[i1] * frac);
    }
}

