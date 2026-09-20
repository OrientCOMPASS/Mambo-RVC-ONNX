use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{bail, Result};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionOutputs};
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};
use rustfft::{num_complex::Complex, FftPlanner};

use crate::logger;

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

    pub fn current_module_dir() -> Option<PathBuf> {
        unsafe {
            let mut module: HMODULE = std::ptr::null_mut();

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

    pub fn add_dll_directory(dir: &std::path::Path) -> bool {
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        unsafe { SetDllDirectoryW(wide.as_ptr()) != 0 }
    }
}

#[cfg(not(target_os = "windows"))]
mod imp {
    use std::path::PathBuf;

    #[repr(C)]
    struct DlInfo {
        dli_fname: *const std::ffi::c_char,
        dli_fbase: *mut std::ffi::c_void,
        dli_sname: *const std::ffi::c_char,
        dli_saddr: *mut std::ffi::c_void,
    }

    extern "C" {
        fn dladdr(addr: *mut std::ffi::c_void, info: *mut DlInfo) -> i32;
    }

    pub fn current_module_dir() -> Option<PathBuf> {
        unsafe {
            let mut info = DlInfo {
                dli_fname: std::ptr::null(),
                dli_fbase: std::ptr::null_mut(),
                dli_sname: std::ptr::null(),
                dli_saddr: std::ptr::null_mut(),
            };
            let addr = current_module_dir as *const () as *mut std::ffi::c_void;
            if dladdr(addr, &mut info) != 0 && !info.dli_fname.is_null() {
                let path = std::ffi::CStr::from_ptr(info.dli_fname).to_string_lossy().into_owned();
                if let Some(parent) = PathBuf::from(&path).parent() {
                    return Some(parent.to_path_buf());
                }
            }
        }
        std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()))
    }

    pub fn add_dll_directory(_dir: &std::path::Path) -> bool {
        true
    }
}

pub fn plugin_dir() -> Option<PathBuf> {
    imp::current_module_dir()
}

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

pub fn prepare_runtime(plugin_dir: &std::path::Path) {
    let libs = plugin_dir.join("libs");
    if libs.is_dir() && imp::add_dll_directory(&libs) {
        logger::log(&format!("[ORT] DLL 搜索路径 = {}", libs.display()));
    }
}

pub const USER_MODEL_DIR: &str = "user_models";
pub const BUNDLED_RVC: &str = "models/uma-Matikane_Tannhauser.onnx";
pub const BUNDLED_HUBERT: &str = "models/hubert_base.onnx";
pub const BUNDLED_RMVPE: &str = "models/rmvpe.onnx";

#[derive(Debug, Clone)]
pub struct ModelSet {
    pub rvc: PathBuf,
    pub hubert: PathBuf,
    pub rmvpe: PathBuf,

    pub rvc_label: String,
    pub rvc_source: Source,
}

impl ModelSet {
    pub fn describe(&self) -> String {
        let name = |p: &Path| {
            p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string())
        };
        format!("RVC={} | HuBERT={} | RMVPE={}", self.rvc_label, name(&self.hubert), name(&self.rmvpe))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {

    UserDir,

    Bundled,
}

impl Source {
    pub fn label(&self) -> &'static str {
        match self {
            Source::UserDir => "用户模型",
            Source::Bundled => "插件自带",
        }
    }
}

pub fn ensure_user_dir(plugin_dir: &Path) -> PathBuf {
    let dir = plugin_dir.join(USER_MODEL_DIR);
    if fs::create_dir_all(&dir).is_ok() {
        let readme = dir.join("README.txt");
        if !readme.exists() {
            let _ = fs::write(
                &readme,
                "把 RVC 模型（.onnx）放进这个文件夹，插件会优先加载它。\r\n\
                 \r\n\
                 规则：\r\n\
                 1. 放多个时取修改时间最新的那个；\r\n\
                 2. 文件名含 hubert / contentvec 的当作内容编码器，含 rmvpe / f0 的当作音高提取器；\r\n\
                 3. 这里没有可用模型时回退到插件自带的 models/uma-Matikane_Tannhauser.onnx；\r\n\
                 4. 增删模型后需要在 MicYou 设置里关闭再启用本插件；\r\n\
                 5. 实际加载了哪个模型，看插件目录下的 rvc_plugin.log 或启用时的系统通知。\r\n"
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

pub fn discover(plugin_dir: &Path) -> ModelSet {
    let user_dir = ensure_user_dir(plugin_dir);
    let user_files = list_onnx(&user_dir);

    let (rvc, source) = match user_files.iter().find(|p| !is_feature_model(p)).cloned() {
        Some(p) => (p, Source::UserDir),
        None => (plugin_dir.join(BUNDLED_RVC), Source::Bundled),
    };
    let hubert_kw = ["hubert", "contentvec", "content_vec"];
    let hubert = pick_by_keyword(&user_files, &hubert_kw).unwrap_or_else(|| plugin_dir.join(BUNDLED_HUBERT));
    let rmvpe_kw = ["rmvpe", "f0"];
    let rmvpe = pick_by_keyword(&user_files, &rmvpe_kw).unwrap_or_else(|| plugin_dir.join(BUNDLED_RMVPE));

    let rvc_label = format!("{} ({})", rel(plugin_dir, &rvc), source.label());
    ModelSet { rvc, hubert, rmvpe, rvc_label, rvc_source: source }
}

fn is_feature_model(p: &Path) -> bool {
    let name = p
        .file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    ["hubert", "contentvec", "content_vec", "rmvpe", "crepe", "fcpe", "rmvpe_v2"]
        .iter()
        .any(|kw| name.contains(kw))
}

pub const HOP_48K: usize = 480;
pub const FEAT_DIM: usize = 768;

pub fn load_session(path: &str, allow_cpu_fallback: bool, opt: u32) -> Result<Session> {
    let cuda = ort::ep::CUDA::default().with_device_id(0).build();

    let level = match opt {
        0 => GraphOptimizationLevel::Level1,
        1 => GraphOptimizationLevel::Level2,
        _ => GraphOptimizationLevel::All,
    };
    let mut builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("builder: {e}"))?
        .with_optimization_level(level)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Phone,
    PhoneLen,
    Pitch,
    NsfF0,
    Sid,
    Noise,
    Mask,
    Audio,
    Scalar,
    Unknown,
}

fn is_int_ty(ty: TensorElementType) -> bool {
    matches!(
        ty,
        TensorElementType::Int8
            | TensorElementType::Int16
            | TensorElementType::Int32
            | TensorElementType::Int64
            | TensorElementType::Uint8
            | TensorElementType::Uint16
            | TensorElementType::Uint32
    )
}

pub fn classify(name: &str, ty: TensorElementType) -> Role {
    let n = name.to_lowercase();
    let int = is_int_ty(ty);
    if n.contains("len") {
        return Role::PhoneLen;
    }
    if n.contains("mask") {
        return Role::Mask;
    }
    if (n.contains("rnd") || n.contains("noise") || n.contains("random") || n == "z") && !n.contains("scale") {
        return Role::Noise;
    }
    if n.contains("nsff0") || n.contains("pitchf") || (n.contains("f0") && !int) {
        return Role::NsfF0;
    }
    if n.contains("pitch") {
        return Role::Pitch;
    }
    if n.contains("phone") || n.contains("content") || n.contains("feats") || n.contains("hubert") || n.contains("cvec")
    {
        return Role::Phone;
    }
    if n.contains("sid") || n.contains("speaker") || n.contains("spk") || n.contains("singer") || n == "ds" {
        return Role::Sid;
    }
    if n.contains("source") || n.contains("input_values") || n.contains("wav") || n.contains("audio") || n == "input" {
        return Role::Audio;
    }
    if n.contains("scale") || n.contains("vol") || n.contains("gain") || n.contains("threshold") {
        return Role::Scalar;
    }
    Role::Unknown
}

fn scalar_default(name: &str) -> f64 {
    let n = name.to_lowercase();
    if n.contains("noise_scale_w") {
        0.8
    } else if n.contains("noise_scale") {
        0.667
    } else if n.contains("length_scale") || n.contains("len_scale") {
        1.0
    } else if n.contains("vol") || n.contains("scale") || n.contains("gain") {
        1.0
    } else {
        0.0
    }
}

#[derive(Debug, Clone)]
pub struct InputSpec {
    pub name: String,
    pub role: Role,
    pub ty: TensorElementType,

    pub dims: Vec<i64>,
}

impl InputSpec {
    pub fn of(name: &str, dtype: &ValueType) -> Option<Self> {
        match dtype {
            ValueType::Tensor { ty, shape, .. } => Some(Self {
                name: name.to_string(),
                role: classify(name, *ty),
                ty: *ty,
                dims: shape.to_vec(),
            }),
            _ => None,
        }
    }

    pub fn shape(&self, frames: usize, audio_len: usize) -> Vec<i64> {
        let n = self.dims.len();
        let mut out = self.dims.clone();
        for (i, d) in out.iter_mut().enumerate() {
            if *d > 0 {
                continue;
            }
            let last = i + 1 == n;
            *d = match self.role {
                Role::Phone => {
                    if last {
                        FEAT_DIM as i64
                    } else if i + 2 == n {
                        frames as i64
                    } else {
                        1
                    }
                }
                Role::PhoneLen | Role::Sid | Role::Scalar => 1,
                Role::Audio | Role::Mask => {
                    if last {
                        audio_len as i64
                    } else {
                        1
                    }
                }

                _ => {
                    if last {
                        frames as i64
                    } else {
                        1
                    }
                }
            };
        }
        if out.is_empty() {
            out.push(1);
        }
        out
    }

    fn mask_is_padding(&self) -> bool {
        self.name.to_lowercase().contains("padding")
    }
}

pub struct BlockData<'a> {
    pub phone: &'a [f32],
    pub pitch: &'a [i64],
    pub nsff0: &'a [f32],
    pub sid: i64,
    pub audio: &'a [f32],
    pub noise: &'a [f32],
    pub frames: usize,
}

fn fit_f32(src: &[f32], n: usize, name: &str, warn: &mut Option<String>) -> Vec<f32> {
    if src.len() == n {
        return src.to_vec();
    }
    if warn.is_none() {
        *warn = Some(format!("输入 {name} 需要 {n} 个元素，实际只有 {}，已补零/截断", src.len()));
    }
    let mut v = vec![0.0f32; n];
    let k = src.len().min(n);
    v[..k].copy_from_slice(&src[..k]);
    v
}

pub fn build_input(spec: &InputSpec, d: &BlockData, warn: &mut Option<String>) -> Result<DynValue> {
    let shape = spec.shape(d.frames, d.audio.len());
    let n: usize = shape.iter().map(|x| (*x).max(1) as usize).product();

    let floats = |role: Role, warn: &mut Option<String>| -> Vec<f32> {
        match role {
            Role::Phone => fit_f32(d.phone, n, &spec.name, warn),
            Role::NsfF0 => fit_f32(d.nsff0, n, &spec.name, warn),
            Role::Noise => fit_f32(d.noise, n, &spec.name, warn),
            Role::Audio => fit_f32(d.audio, n, &spec.name, warn),
            Role::Mask => vec![if spec.mask_is_padding() { 0.0 } else { 1.0 }; n],
            Role::Pitch | Role::PhoneLen | Role::Sid => {
                let v = match role {
                    Role::Pitch => d.pitch.first().copied().unwrap_or(0) as f32,
                    Role::PhoneLen => d.frames as f32,
                    _ => d.sid as f32,
                };
                vec![v; n]
            }
            Role::Scalar | Role::Unknown => vec![scalar_default(&spec.name) as f32; n],
        }
    };
    let ints = |role: Role, warn: &mut Option<String>| -> Vec<i64> {
        match role {
            Role::Pitch => fit_i64(d.pitch, n, &spec.name, warn),
            Role::PhoneLen => vec![d.frames as i64; n],
            Role::Sid => vec![d.sid; n],
            Role::Mask => vec![if spec.mask_is_padding() { 0 } else { 1 }; n],
            _ => vec![scalar_default(&spec.name) as i64; n],
        }
    };

    match spec.ty {
        TensorElementType::Float32 => Ok(Tensor::from_array((shape, floats(spec.role, warn)))?.into_dyn()),
        TensorElementType::Float64 => {
            let v: Vec<f64> = floats(spec.role, warn).into_iter().map(|x| x as f64).collect();
            Ok(Tensor::from_array((shape, v))?.into_dyn())
        }
        TensorElementType::Int64 => Ok(Tensor::from_array((shape, ints(spec.role, warn)))?.into_dyn()),
        TensorElementType::Int32 => {
            let v: Vec<i32> = ints(spec.role, warn).into_iter().map(|x| x as i32).collect();
            Ok(Tensor::from_array((shape, v))?.into_dyn())
        }
        TensorElementType::Bool => {
            let pad = spec.mask_is_padding();
            let v = vec![matches!(spec.role, Role::Mask) && !pad; n];
            Ok(Tensor::from_array((shape, v))?.into_dyn())
        }
        other => bail!("输入 {} 的类型 {:?} 暂不支持", spec.name, other),
    }
}

fn fit_i64(src: &[i64], n: usize, name: &str, warn: &mut Option<String>) -> Vec<i64> {
    if src.len() == n {
        return src.to_vec();
    }
    if warn.is_none() {
        *warn = Some(format!("输入 {name} 需要 {n} 个元素，实际只有 {}，已补零/截断", src.len()));
    }
    let mut v = vec![0i64; n];
    let k = src.len().min(n);
    v[..k].copy_from_slice(&src[..k]);
    v
}

pub struct NoiseGen {
    state: u32,
    buf: Vec<f32>,
}

impl NoiseGen {
    pub fn new() -> Self {
        Self { state: 0x9E37_79B9, buf: Vec::new() }
    }

    pub fn fill(&mut self, n: usize) -> &[f32] {
        self.buf.clear();
        self.buf.reserve(n);
        for _ in 0..n {
            let mut acc = 0.0f32;
            for _ in 0..4 {
                self.state ^= self.state << 13;
                self.state ^= self.state >> 17;
                self.state ^= self.state << 5;
                acc += (self.state as f32) / 2147483648.0 - 1.0;
            }
            self.buf.push(acc * 0.5);
        }
        &self.buf
    }
}

pub fn log_plan(tag: &str, specs: &[InputSpec]) {
    for s in specs {
        logger::log(&format!(
            "[{tag}] 输入 {:<16} 角色={:<9} 类型={:?} 声明形状={:?}",
            s.name,
            format!("{:?}", s.role),
            s.ty,
            s.dims
        ));
    }
    let unknown: Vec<&str> = specs.iter().filter(|s| s.role == Role::Unknown).map(|s| s.name.as_str()).collect();
    if !unknown.is_empty() {
        logger::log(&format!(
            "[{tag}] ⚠️ 无法识别的输入 {unknown:?}：将按声明形状填 0（或用常见默认值）。\
             如果模型输出异常，多半是这里——请把上面几行 [输入] 日志发给开发者。"
        ));
    }
}

pub struct HubertExtractor {
    session: Session,
    specs: Vec<InputSpec>,

    buf: Vec<f32>,

    pad_extra: usize,
    swept: bool,
}

impl HubertExtractor {
    pub fn new(session: Session) -> Self {
        let specs = session
            .inputs()
            .iter()
            .filter_map(|i| InputSpec::of(i.name(), i.dtype()))
            .collect::<Vec<_>>();
        log_plan("HuBERT", &specs);
        Self { session, specs, buf: Vec::new(), pad_extra: 0, swept: false }
    }

    pub fn extract(&mut self, pcm16k: &[f32]) -> Result<Vec<f32>> {
        if pcm16k.is_empty() {
            bail!("empty input");
        }

        self.buf.clear();
        self.buf.extend_from_slice(pcm16k);
        let peak = self.buf.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        if peak > 1e-6 {
            let gain = (0.9 / peak).min(10.0);
            for s in self.buf.iter_mut() {
                *s *= gain;
            }
        }

        if self.specs.is_empty() {
            bail!("模型没有声明任何输入");
        }
        let base_len = self.buf.len();

        let tries: usize = if self.swept { 1 } else { 6 };
        let start = self.pad_extra;
        let mut last_err = String::new();
        for k in 0..tries {
            let attempt = start + k * 320;
            self.buf.resize(base_len + attempt, 0.0);
            match self.run_once() {
                Ok(v) => {
                    if attempt != self.pad_extra {
                        logger::log(&format!(
                            "[HuBERT] 该模型要求 16k 输入补零 +{attempt} 样本（{} 帧）才能跑通，已记住；                             窗口对应的 16k 长度是 {base_len}",
                            attempt / 320
                        ));
                        self.pad_extra = attempt;
                    }
                    self.swept = true;
                    return Ok(v);
                }
                Err(e) => last_err = format!("{e:#}"),
            }
        }
        self.swept = true;
        bail!("HuBERT 推理失败（已尝试补零 0..{}）: {last_err}", (tries - 1) * 320);
    }

    fn run_once(&mut self) -> Result<Vec<f32>> {
        let audio_len = self.buf.len();
        let specs = self.specs.clone();
        let mut warn = None;
        let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(specs.len());
        {
            let data = BlockData {
                phone: &[],
                pitch: &[],
                nsff0: &[],
                sid: 0,
                audio: &self.buf,
                noise: &[],

                frames: audio_len,
            };
            for s in specs.iter() {
                inputs.push((s.name.clone(), build_input(s, &data, &mut warn)?));
            }
        }
        if let Some(w) = warn {
            logger::log(&format!("[HuBERT] {w}"));
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
        bail!("未能取出内容特征（没有长度是 {FEAT_DIM} 整数倍的 f32 输出）");
    }
}

const N_FFT: usize = 1024;
const MEL_HOP: usize = 160;
const N_MELS: usize = 128;
const N_FREQS: usize = N_FFT / 2 + 1;
const MEL_FMIN: f32 = 30.0;
const MEL_FMAX: f32 = 8000.0;
const MEL_CLAMP: f32 = 1e-5;
const MEL_SR: f32 = 16000.0;

struct MelFrontend {
    window: Vec<f32>,

    mel_basis: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32>>,

    padded: Vec<f32>,
    spec: Vec<Complex<f32>>,

    mag: Vec<f32>,

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

    fn compute(&mut self, pcm16k: &[f32], channel_major: &mut Vec<f32>) -> (usize, usize) {
        let pad = N_FFT / 2;
        let n = pcm16k.len();
        self.padded.clear();
        self.padded.reserve(n + 2 * pad);

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

        let pad_time = 32 * ((num_frames.saturating_sub(1)) / 32 + 1) - num_frames;
        let total = num_frames + pad_time;
        channel_major.clear();
        channel_major.reserve(N_MELS * total);
        for m in 0..N_MELS {
            for f in 0..num_frames {
                channel_major.push(self.mel_frames[f * N_MELS + m]);
            }

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

    arr.sort_by(|x, y| x.total_cmp(y));
    arr[1]
}

fn pick_f32_output(outputs: &SessionOutputs, min_len: usize) -> Option<Vec<f32>> {

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelShape {

    Dynamic { hop: usize },

    Fixed { frames: usize, hop: usize },

    Unknown { hop: usize },
}

pub struct RvcSynth {
    session: Session,
    specs: Vec<InputSpec>,
    phone_100: Vec<f32>,
    pitch: Vec<i64>,
    nsff0: Vec<f32>,
    noise: NoiseGen,

    noise_role: bool,
    logged: bool,
}

impl RvcSynth {
    pub fn new(session: Session) -> Self {
        let specs = session
            .inputs()
            .iter()
            .filter_map(|i| InputSpec::of(i.name(), i.dtype()))
            .collect::<Vec<_>>();
        log_plan("RVC", &specs);
        let noise_role = specs.iter().any(|s| s.role == Role::Noise);
        Self {
            session,
            specs,
            phone_100: Vec::new(),
            pitch: Vec::new(),
            nsff0: Vec::new(),
            noise: NoiseGen::new(),
            noise_role,
            logged: false,
        }
    }

    pub fn synthesize(&mut self, frames: usize, sid: i64, f0: &[f32], phone_50fps: &[f32]) -> Result<Vec<f32>> {
        self.run(frames, sid, f0, phone_50fps, false)
    }

    pub fn probe(&mut self, want: usize) -> ModelShape {

        const CANDIDATES: [usize; 12] = [200, 256, 128, 100, 64, 300, 400, 512, 150, 320, 96, 80];
        let first = want.clamp(8, 4096);
        match self.try_frames(first) {
            Ok(hop) => {
                let other = if first > 24 { first - 7 } else { first + 7 };
                match self.try_frames(other) {
                    Ok(_) => ModelShape::Dynamic { hop },
                    Err(_) => ModelShape::Fixed { frames: first, hop },
                }
            }
            Err(e) => {
                for t in CANDIDATES {
                    if t == first {
                        continue;
                    }
                    if let Ok(hop) = self.try_frames(t) {
                        logger::log(&format!("[RVC] 探测: 帧数 {first} 失败（{e}），但 {t} 帧可用"));
                        return ModelShape::Fixed { frames: t, hop };
                    }
                }
                logger::log(&format!("[RVC] 探测: 所有候选帧数都失败，最后一次错误: {e}"));
                ModelShape::Unknown { hop: HOP_48K }
            }
        }
    }

    fn try_frames(&mut self, frames: usize) -> Result<usize, String> {
        if frames < 2 {
            return Err("frames < 2".into());
        }
        let phone = vec![0.0f32; FEAT_DIM * (frames / 2).max(2)];
        let f0 = vec![0.0f32; frames];
        let audio = self.run(frames, 0, &f0, &phone, true).map_err(|e| format!("{e:#}"))?;
        if audio.is_empty() {
            return Err("输出为空".into());
        }
        Ok(audio.len() / frames)
    }

    fn run(&mut self, frames: usize, sid: i64, f0: &[f32], phone_50fps: &[f32], quiet: bool) -> Result<Vec<f32>> {
        let frames_50 = phone_50fps.len() / FEAT_DIM;
        if frames_50 == 0 {

            bail!("HuBERT 特征为空");
        }
        if f0.len() < frames {
            bail!("f0 长度不足: {} < {frames}", f0.len());
        }

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

        let specs = self.specs.clone();
        if specs.is_empty() {
            bail!("模型没有声明任何输入");
        }
        let mut warn = None;
        let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(specs.len());
        {

            let noise_slice: &[f32] = if self.noise_role {
                let n = specs
                    .iter()
                    .find(|s| s.role == Role::Noise)
                    .map(|s| s.shape(frames, 0).iter().map(|x| (*x).max(1) as usize).product())
                    .unwrap_or(0);
                self.noise.fill(n)
            } else {
                &[]
            };
            let data = BlockData {
                phone: &self.phone_100,
                pitch: &self.pitch,
                nsff0: &self.nsff0,
                sid,
                audio: &[],
                noise: noise_slice,
                frames,
            };
            for s in specs.iter() {
                inputs.push((s.name.clone(), build_input(s, &data, &mut warn)?));
            }
        }
        if let Some(w) = warn {
            logger::log(&format!("[RVC] {w}"));
        }

        let outputs = self.session.run(inputs)?;
        let audio = pick_f32_output(&outputs, frames).ok_or_else(|| anyhow::anyhow!("RVC: 没有音频输出"))?;
        if !quiet && !self.logged {
            self.logged = true;
            logger::log(&format!(
                "[RVC] 首块成功: frames={frames} 输出样本={} 推断hop={}",
                audio.len(),
                audio.len() / frames.max(1)
            ));
        }
        Ok(audio)
    }
}

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
