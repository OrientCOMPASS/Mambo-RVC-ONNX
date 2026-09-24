//! CUDA / cuDNN 运行库拉取（面板一键下载）。
//!
//! 设计约束：
//! - 本模块运行在**自建后台线程**里，绝不触碰任何 Host API（宿主规范：Host API 只能在
//!   宿主分发的线程调用）。进度通过 `Arc<Shared>` 暴露，由 lib.rs 的定时器回调在宿主线程
//!   读取并 `set_config`，面板轮询 `get_config` 展示。
//! - 下载源是 PyPI 的 `nvidia-*-cu12` wheel（zip），版本固定 + sha256 校验；
//!   各大镜像（清华/阿里/中科大/腾讯）以相同 `/packages/<path>` 布局镜像 PyPI，
//!   因此只存路径、运行期拼镜像前缀，失败自动轮换镜像。
//! - 断点续传：`.part` 文件 + HTTP Range；ureq 的 `timeout_recv_body` 是**单次调用的总
//!   预算**而非单次 read 超时，所以预算给 120s，超时视为一次尝试失败，靠续传无缝接上。
//! - 磁盘峰值 = 最大单包 wheel(cudnn ≈ 700MB) + 全部解压产物(≈3.2GB)，每个包解压完
//!   立即删除 wheel；开始前用 GetDiskFreeSpaceExW 预检。
//!
//! pin 表由 `tools/gen_pins.py` 生成（校验过 wheel 中央目录里的 dll 精确大小）。

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::logger;

// ───────────────────────────── 静态数据 ─────────────────────────────

/// 插件包自带的 3 个 ONNX Runtime dll（缺失 = 安装包被破坏，提示重装）。
pub const ORT_DLLS: [(&str, &str); 3] = [
    ("core", "onnxruntime.dll"),
    ("cuda_provider", "onnxruntime_providers_cuda.dll"),
    ("shared", "onnxruntime_providers_shared.dll"),
];

pub struct DllSpec {
    pub name: &'static str,
    pub size: u64,
}

pub struct PkgSpec {
    pub pkg: &'static str,
    pub version: &'static str,
    pub wheel_file: &'static str,
    pub sha256: &'static str,
    /// `https://<mirror>/packages/` 之后的部分（PyPI 内容寻址路径，各镜像一致）。
    pub path: &'static str,
    pub wheel_size: u64,
    pub dlls: &'static [DllSpec],
}

// 由 tools/gen_pins.py 生成 —— 共 7 包 / 19 dll / 下载量 2.09 GiB（小包在前）
pub const PKGS: &[PkgSpec] = &[
    PkgSpec {
        pkg: "nvidia-cuda-runtime-cu12",
        version: "12.9.79",
        wheel_file: "nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
        sha256: "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891",
        path: "59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
        wheel_size: 3591604,
        dlls: &[
            DllSpec { name: "cudart64_12.dll", size: 583680 },
        ],
    },
    PkgSpec {
        pkg: "nvidia-curand-cu12",
        version: "10.3.10.19",
        wheel_file: "nvidia_curand_cu12-10.3.10.19-py3-none-win_amd64.whl",
        sha256: "e8129e6ac40dc123bd948e33d3e11b4aa617d87a583fa2f21b3210e90c743cde",
        path: "e5/98/1bd66fd09cbe1a5920cb36ba87029d511db7cca93979e635fd431ad3b6c0/nvidia_curand_cu12-10.3.10.19-py3-none-win_amd64.whl",
        wheel_size: 68774847,
        dlls: &[
            DllSpec { name: "curand64_10.dll", size: 79197696 },
        ],
    },
    PkgSpec {
        pkg: "nvidia-cufft-cu12",
        version: "11.4.1.4",
        wheel_file: "nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl",
        sha256: "8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80",
        path: "20/ee/29955203338515b940bd4f60ffdbc073428f25ef9bfbce44c9a066aedc5c/nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl",
        wheel_size: 200067309,
        dlls: &[
            DllSpec { name: "cufft64_11.dll", size: 287136768 },
            DllSpec { name: "cufftw64_11.dll", size: 163328 },
        ],
    },
    PkgSpec {
        pkg: "nvidia-cusolver-cu12",
        version: "11.7.5.82",
        wheel_file: "nvidia_cusolver_cu12-11.7.5.82-py3-none-win_amd64.whl",
        sha256: "77666337237716783c6269a658dea310195cddbd80a5b2919b1ba8735cec8efd",
        path: "32/5d/feb7f86b809f89b14193beffebe24cf2e4bf7af08372ab8cdd34d19a65a0/nvidia_cusolver_cu12-11.7.5.82-py3-none-win_amd64.whl",
        wheel_size: 326215953,
        dlls: &[
            DllSpec { name: "cusolver64_11.dll", size: 283129856 },
            DllSpec { name: "cusolverMg64_11.dll", size: 187460096 },
        ],
    },
    PkgSpec {
        pkg: "nvidia-cusparse-cu12",
        version: "12.5.10.65",
        wheel_file: "nvidia_cusparse_cu12-12.5.10.65-py3-none-win_amd64.whl",
        sha256: "9e487468a22a1eaf1fbd1d2035936a905feb79c4ce5c2f67626764ee4f90227c",
        path: "73/ef/063500c25670fbd1cbb0cd3eb7c8a061585b53adb4dd8bf3492bb49b0df3/nvidia_cusparse_cu12-12.5.10.65-py3-none-win_amd64.whl",
        wheel_size: 362504719,
        dlls: &[
            DllSpec { name: "cusparse64_12.dll", size: 477564928 },
        ],
    },
    PkgSpec {
        pkg: "nvidia-cublas-cu12",
        version: "12.9.2.10",
        wheel_file: "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
        sha256: "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661",
        path: "20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
        wheel_size: 553162896,
        dlls: &[
            DllSpec { name: "cublas64_12.dll", size: 102518272 },
            DllSpec { name: "cublasLt64_12.dll", size: 668673536 },
        ],
    },
    PkgSpec {
        pkg: "nvidia-cudnn-cu12",
        version: "9.25.1.1",
        wheel_file: "nvidia_cudnn_cu12-9.25.1.1-py3-none-win_amd64.whl",
        sha256: "debb5f5901ae6071f34d0a2b256acecc33dc3277f1fd5a11f8249f921db8a40d",
        path: "0b/ee/b5699f1960e358ec995bb72f71c2ec06c550fd0c8280525796d6646c0299/nvidia_cudnn_cu12-9.25.1.1-py3-none-win_amd64.whl",
        wheel_size: 732338891,
        dlls: &[
            DllSpec { name: "cudnn64_9.dll", size: 267888 },
            DllSpec { name: "cudnn_adv64_9.dll", size: 269022832 },
            DllSpec { name: "cudnn_cnn64_9.dll", size: 2993264 },
            DllSpec { name: "cudnn_engines_precompiled64_9.dll", size: 542110320 },
            DllSpec { name: "cudnn_engines_runtime_compiled64_9.dll", size: 38617200 },
            DllSpec { name: "cudnn_engines_tensor_ir64_9.dll", size: 155760 },
            DllSpec { name: "cudnn_ext64_9.dll", size: 130160 },
            DllSpec { name: "cudnn_graph64_9.dll", size: 99882096 },
            DllSpec { name: "cudnn_heuristic64_9.dll", size: 58741360 },
            DllSpec { name: "cudnn_ops64_9.dll", size: 105601136 },
        ],
    },
];

pub struct Mirror {
    pub id: &'static str,
    pub label: &'static str,
    /// 拼接在 `PkgSpec.path` 前面。
    pub base: &'static str,
}

pub const MIRRORS: &[Mirror] = &[
    Mirror { id: "tuna", label: "清华 TUNA", base: "https://pypi.tuna.tsinghua.edu.cn/packages/" },
    Mirror { id: "aliyun", label: "阿里云", base: "https://mirrors.aliyun.com/pypi/packages/" },
    Mirror { id: "ustc", label: "中科大", base: "https://mirrors.ustc.edu.cn/pypi/packages/" },
    Mirror { id: "tencent", label: "腾讯云", base: "https://mirrors.cloud.tencent.com/pypi/packages/" },
    Mirror { id: "pypi", label: "PyPI 官方", base: "https://files.pythonhosted.org/packages/" },
];

pub fn mirror_by_id(id: &str) -> &'static Mirror {
    MIRRORS.iter().find(|m| m.id == id).unwrap_or(&MIRRORS[0])
}

/// 镜像轮换顺序：偏好镜像在前，其余按默认顺序殿后（自动故障转移）。
pub fn mirror_order(pref: &str) -> Vec<&'static Mirror> {
    let first = mirror_by_id(pref);
    let mut v = vec![first];
    for m in MIRRORS {
        if m.id != first.id {
            v.push(m);
        }
    }
    v
}

pub fn total_wheel_bytes() -> u64 {
    PKGS.iter().map(|p| p.wheel_size).sum()
}

pub fn cuda_dll_count() -> usize {
    PKGS.iter().map(|p| p.dlls.len()).sum()
}

// ───────────────────────────── 目录盘点 ─────────────────────────────

pub struct Inventory {
    pub ort: [bool; 3],
    pub present: usize,
    pub total: usize,
    pub missing: Vec<&'static str>,
    pub missing_bytes: u64,
}

impl Inventory {
    pub fn cuda_ready(&self) -> bool {
        self.missing.is_empty()
    }
    pub fn ort_ready(&self) -> bool {
        self.ort.iter().all(|&b| b)
    }
}

fn file_size(p: &Path) -> Option<u64> {
    fs::metadata(p).ok().map(|m| m.len())
}

/// 某个包已就位（存在且大小精确匹配）的 dll 数。
pub fn pkg_have_count(libs: &Path, p: &PkgSpec) -> usize {
    p.dlls.iter().filter(|d| file_size(&libs.join(d.name)) == Some(d.size)).count()
}

pub fn runtime_inventory(libs: &Path) -> Inventory {
    let ort: [bool; 3] = std::array::from_fn(|i| libs.join(ORT_DLLS[i].1).is_file());
    let mut present = 0usize;
    let mut missing = Vec::new();
    let mut missing_bytes = 0u64;
    for pkg in PKGS {
        for d in pkg.dlls {
            // 只认「存在且大小正确」——半截文件视为缺失，拉取时会覆盖
            match file_size(&libs.join(d.name)) {
                Some(s) if s == d.size => present += 1,
                _ => {
                    missing.push(d.name);
                    missing_bytes += d.size;
                }
            }
        }
    }
    let total = cuda_dll_count();
    Inventory { ort, present, total, missing, missing_bytes }
}

// ───────────────────────────── 进度状态 ─────────────────────────────

#[derive(Serialize, Clone, Debug)]
pub struct PkgProgress {
    pub name: &'static str,
    pub version: &'static str,
    pub bytes: u64,
    pub dlls: usize,
    /// 已就位（存在且大小正确）的 dll 数
    pub have: usize,
    /// wait | skip | downloading | verifying | extracting | done | error
    pub state: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct Progress {
    pub v: u8,
    /// idle | downloading | verifying | extracting | done | error | canceled
    pub state: String,
    pub mirror: String,
    pub pkg_index: i64,
    pub pkg_count: usize,
    pub pkg: String,
    pub pkg_version: String,
    /// 当前包已下载字节
    pub bytes_done: u64,
    /// 当前包 wheel 总字节
    pub bytes_total: u64,
    /// 全局（已完成包 + 当前包 bytes_done）
    pub all_done: u64,
    pub all_total: u64,
    pub speed_bps: f64,
    /// 人类可读的当前步骤
    pub step: String,
    pub error: String,
    pub ts: u64,
    pub pkgs: Vec<PkgProgress>,
}

impl Progress {
    pub fn terminal(&self) -> bool {
        matches!(self.state.as_str(), "done" | "error" | "canceled")
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// 按 libs/ 现状构造初始进度（全部就绪 => state=done；否则 idle）。
pub fn initial_progress(libs: &Path, mirror: &str) -> Progress {
    let pkgs: Vec<PkgProgress> = PKGS
        .iter()
        .map(|p| {
            let have = pkg_have_count(libs, p);
            PkgProgress {
                name: p.pkg,
                version: p.version,
                bytes: p.wheel_size,
                dlls: p.dlls.len(),
                have,
                state: if have == p.dlls.len() { "skip" } else { "wait" }.to_string(),
            }
        })
        .collect();
    let all_done = pkgs.iter().filter(|p| p.state == "skip").map(|p| p.bytes).sum();
    let all_total = total_wheel_bytes();
    let ready = all_done == all_total;
    Progress {
        v: 1,
        state: if ready { "done" } else { "idle" }.to_string(),
        mirror: mirror_by_id(mirror).id.to_string(),
        pkg_index: -1,
        pkg_count: PKGS.len(),
        pkg: String::new(),
        pkg_version: String::new(),
        bytes_done: 0,
        bytes_total: 0,
        all_done,
        all_total,
        speed_bps: 0.0,
        step: if ready { "运行库已就绪" } else { "" }.to_string(),
        error: String::new(),
        ts: now_ms(),
        pkgs,
    }
}

pub struct Shared {
    prog: Mutex<Progress>,
    cancel: AtomicBool,
    finished: AtomicBool,
}

impl Shared {
    fn new(libs: &Path, mirror: &str) -> Self {
        Self {
            prog: Mutex::new(initial_progress(libs, mirror)),
            cancel: AtomicBool::new(false),
            finished: AtomicBool::new(false),
        }
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
    /// 更新进度并返回快照（锁内完成，保证一致性）。
    fn update(&self, f: impl FnOnce(&mut Progress)) -> Progress {
        let mut g = self.prog.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut g);
        // all_done = 已终态包的 wheel 之和 + 当前包已下载字节。
        // verifying/extracting 阶段 bytes_done 已等于整包，也要计入，避免总进度条回跳。
        let base: u64 = g.pkgs.iter().filter(|p| matches!(p.state.as_str(), "done" | "skip")).map(|p| p.bytes).sum();
        let in_flight = matches!(g.state.as_str(), "downloading" | "verifying" | "extracting");
        g.all_done = base + if in_flight { g.bytes_done } else { 0 };
        g.ts = now_ms();
        g.clone()
    }
    fn snapshot(&self) -> Progress {
        self.prog.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

pub struct Handle {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl Handle {
    pub fn snapshot(&self) -> Progress {
        self.shared.snapshot()
    }
    pub fn cancel(&self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
    }
    pub fn is_finished(&self) -> bool {
        self.shared.finished.load(Ordering::Acquire)
    }
    /// 阻塞等待下载线程退出（deinit 前必须调用，否则库卸载后线程仍在执行插件代码会崩）。
    pub fn join(&mut self) {
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// 启动拉取任务。调用方（lib.rs）保证同一时刻只有一个 Handle 存活。
pub fn spawn(plugin_dir: PathBuf, mirror_pref: &str) -> Result<Handle, String> {
    let libs = plugin_dir.join("libs");
    let shared = Arc::new(Shared::new(&libs, mirror_pref));
    let sh = shared.clone();
    let mirror_pref = mirror_pref.to_string();
    let t = std::thread::Builder::new()
        .name("mambo-rvc-fetch".into())
        .spawn(move || run(sh, plugin_dir, &mirror_pref))
        .map_err(|e| format!("无法创建下载线程: {e}"))?;
    Ok(Handle { shared, thread: Some(t) })
}

// ───────────────────────────── 任务主流程 ─────────────────────────────

#[derive(Debug)]
enum JobError {
    Cancelled,
    Fatal(String),
}

type JobResult = Result<(), JobError>;

fn run(shared: Arc<Shared>, plugin_dir: PathBuf, mirror_pref: &str) {
    let res = run_inner(&shared, &plugin_dir, mirror_pref);
    let (state, step, err) = match &res {
        Ok(()) => ("done", format!("完成：{} 个运行库已就位", cuda_dll_count()), String::new()),
        Err(JobError::Cancelled) => ("canceled", "已取消（保留断点，可继续）".to_string(), String::new()),
        Err(JobError::Fatal(m)) => ("error", String::new(), truncate(m, 400)),
    };
    shared.update(|p| {
        p.state = state.to_string();
        p.step = step.clone();
        p.error = err.clone();
        p.speed_bps = 0.0;
        // 把仍处于中间态的包复位，面板不会显示“正在下载”的假象
        for q in p.pkgs.iter_mut() {
            if matches!(q.state.as_str(), "downloading" | "verifying" | "extracting") {
                q.state = "wait".into();
            }
        }
    });
    logger::log(&format!("[Fetch] 结束 state={state}{}", if err.is_empty() { String::new() } else { format!(" err={err}") }));
    shared.finished.store(true, Ordering::Release);
}

fn run_inner(shared: &Arc<Shared>, plugin_dir: &Path, mirror_pref: &str) -> JobResult {
    let libs = plugin_dir.join("libs");
    let cache = plugin_dir.join(".rt_cache");
    fs::create_dir_all(&libs).map_err(|e| JobError::Fatal(format!("无法创建 libs/: {e}")))?;
    fs::create_dir_all(&cache).map_err(|e| JobError::Fatal(format!("无法创建缓存目录: {e}")))?;

    let inv = runtime_inventory(&libs);
    if inv.cuda_ready() {
        shared.update(|p| p.pkgs.iter_mut().for_each(|q| { q.state = "skip".into(); q.have = q.dlls; }));
        return Ok(());
    }

    // 磁盘预检：解压产物 + 最大单包 wheel + 512MB 余量
    let need = inv.missing_bytes + PKGS.iter().map(|p| p.wheel_size).max().unwrap_or(0) + 512 * 1024 * 1024;
    if let Some(free) = free_disk_bytes(&libs) {
        if free < need {
            return Err(JobError::Fatal(format!(
                "磁盘空间不足：约需 {}（含下载缓存与余量），当前盘剩余 {}。请清理后重试",
                human(need),
                human(free)
            )));
        }
    }

    let mirrors = mirror_order(mirror_pref);
    let agent = build_agent();
    shared.update(|p| {
        p.mirror = mirrors[0].id.to_string();
        p.state = "downloading".to_string();
    });
    logger::log(&format!(
        "[Fetch] 开始：缺失 {} 个 dll（{}），镜像顺序 {}",
        inv.missing.len(),
        human(inv.missing_bytes),
        mirrors.iter().map(|m| m.id).collect::<Vec<_>>().join(" → ")
    ));

    for (i, pkg) in PKGS.iter().enumerate() {
        if shared.cancelled() {
            return Err(JobError::Cancelled);
        }
        let have = pkg.dlls.iter().filter(|d| file_size(&libs.join(d.name)) == Some(d.size)).count();
        if have == pkg.dlls.len() {
            shared.update(|p| {
                if let Some(q) = p.pkgs.get_mut(i) {
                    q.state = "skip".into();
                    q.have = have;
                }
            });
            continue;
        }

        let wheel = cache.join(pkg.wheel_file);
        let part = cache.join(format!("{}.part", pkg.wheel_file));

        // 下载（含 sha256 校验；缓存里已有完好 wheel 则跳过）
        let cached_ok = file_size(&wheel) == Some(pkg.wheel_size)
            && sha256_file(&wheel).ok().as_deref() == Some(pkg.sha256);
        if !cached_ok {
            let _ = fs::remove_file(&wheel);
            shared.update(|p| {
                p.state = "downloading".into();
                p.pkg_index = i as i64;
                p.pkg = pkg.pkg.to_string();
                p.pkg_version = pkg.version.to_string();
                p.bytes_total = pkg.wheel_size;
                p.bytes_done = file_size(&part).unwrap_or(0);
                if let Some(q) = p.pkgs.get_mut(i) {
                    q.state = "downloading".into();
                    q.have = have;
                }
            });
            download_wheel(&agent, pkg, &part, &wheel, shared, i, &mirrors)?;
        }

        // 解压
        shared.update(|p| {
            p.state = "extracting".into();
            p.step = format!("解压 {}（{}/{}）", pkg.wheel_file, i + 1, PKGS.len());
            if let Some(q) = p.pkgs.get_mut(i) {
                q.state = "extracting".into();
            }
        });
        extract_pkg(pkg, &wheel, &libs, shared, i)?;

        let _ = fs::remove_file(&wheel);
        shared.update(|p| {
            p.bytes_done = pkg.wheel_size;
            if let Some(q) = p.pkgs.get_mut(i) {
                q.state = "done".into();
                q.have = q.dlls;
            }
        });
        logger::log(&format!("[Fetch] {} 完成（{}/{}）", pkg.pkg, i + 1, PKGS.len()));
    }

    // 收尾：清空缓存目录（.part 断点也一并丢弃，全部包已成功）
    let _ = fs::remove_dir_all(&cache);
    // libs/ 可能是运行中途新建的，重新把它挂进 DLL 搜索路径（进程级、幂等）
    crate::rvc::prepare_runtime(plugin_dir);
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

pub fn human(bytes: u64) -> String {
    const U: [(&str, f64); 4] = [("B", 1.0), ("KB", 1024.0), ("MB", 1024.0 * 1024.0), ("GB", 1024.0 * 1024.0 * 1024.0)];
    let b = bytes as f64;
    if b >= U[3].1 {
        format!("{:.2}GB", b / U[3].1)
    } else if b >= U[2].1 {
        format!("{:.1}MB", b / U[2].1)
    } else if b >= U[1].1 {
        format!("{:.0}KB", b / U[1].1)
    } else {
        format!("{}B", bytes)
    }
}

// ───────────────────────────── 下载 ─────────────────────────────

/// recv_body 是「单次调用的 body 总预算」而非单次 read 超时：
/// 给 120s，慢速链路一次拉不完就靠 Range 续传分多段完成（重试成本 ≈ 一次 TLS 握手）。
const RECV_BODY_BUDGET: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const ATTEMPTS_PER_MIRROR: u32 = 2;
const FULL_ROUNDS: u32 = 2;

fn build_agent() -> ureq::Agent {
    let mut b = ureq::Agent::config_builder()
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_body(Some(RECV_BODY_BUDGET))
        .user_agent(concat!("mambo-rvc-onnx/", env!("CARGO_PKG_VERSION")));
    // 用户显式配置了代理就尊重（镜像已覆盖国内场景，代理仅是兜底）
    if let Some(p) = ureq::Proxy::try_from_env() {
        b = b.proxy(Some(p));
    }
    b.build().new_agent()
}

enum AttemptError {
    /// 同镜像重试（网络抖动 / 超时 / 5xx）
    Retry(String),
    /// 直接换镜像（403/404/451 之类，这个源没有该文件）
    SwitchMirror(String),
    Cancelled,
}

fn download_wheel(
    agent: &ureq::Agent,
    pkg: &PkgSpec,
    part: &Path,
    wheel: &Path,
    shared: &Arc<Shared>,
    pkg_idx: usize,
    mirrors: &[&Mirror],
) -> JobResult {
    let mut last_err = String::from("未知错误");
    for round in 0..FULL_ROUNDS {
        for m in mirrors {
            for attempt in 0..ATTEMPTS_PER_MIRROR {
                if shared.cancelled() {
                    return Err(JobError::Cancelled);
                }
                if round > 0 || attempt > 0 {
                    shared.update(|p| {
                        p.mirror = m.id.to_string();
                        p.step = format!("重试下载 {}（{}，第 {} 轮）", pkg.pkg, m.label, round * 2 + attempt + 1);
                    });
                }
                match try_download(agent, pkg, part, m, shared, pkg_idx) {
                    Ok(()) => {
                        // sha256 校验；不通过 = 这个源的文件坏了，删掉换镜像重来
                        shared.update(|p| {
                            p.state = "verifying".into();
                            p.step = format!("校验 {}", pkg.wheel_file);
                            if let Some(q) = p.pkgs.get_mut(pkg_idx) {
                                q.state = "verifying".into();
                            }
                        });
                        match sha256_file(part) {
                            Ok(h) if h == pkg.sha256 => {
                                fs::rename(part, wheel)
                                    .map_err(|e| JobError::Fatal(format!("无法落盘 {}: {e}", wheel.display())))?;
                                return Ok(());
                            }
                            Ok(h) => {
                                last_err = format!("{} 校验失败（sha256 不符），换镜像重试", m.label);
                                logger::log(&format!("[Fetch] sha256 mismatch from { }: got {h}", m.id));
                                let _ = fs::remove_file(part);
                                break; // 该镜像文件损坏，换下一个
                            }
                            Err(e) => {
                                last_err = format!("校验读取失败: {e}");
                                let _ = fs::remove_file(part);
                            }
                        }
                    }
                    Err(AttemptError::Cancelled) => return Err(JobError::Cancelled),
                    Err(AttemptError::SwitchMirror(msg)) => {
                        last_err = format!("{}: {msg}", m.label);
                        logger::log(&format!("[Fetch] { } 不可用（{msg}），换镜像", m.id));
                        break;
                    }
                    Err(AttemptError::Retry(msg)) => {
                        last_err = format!("{}: {msg}", m.label);
                        logger::log(&format!("[Fetch] { } 第 {attempt} 次尝试失败: {msg}（已下载部分保留，续传）", m.id));
                    }
                }
            }
        }
    }
    Err(JobError::Fatal(format!("所有镜像均失败。最后错误：{last_err}")))
}

fn try_download(
    agent: &ureq::Agent,
    pkg: &PkgSpec,
    part: &Path,
    mirror: &Mirror,
    shared: &Arc<Shared>,
    pkg_idx: usize,
) -> Result<(), AttemptError> {
    let total = pkg.wheel_size;
    let mut have = file_size(part).unwrap_or(0);
    if have == total {
        return Ok(());
    }
    if have > total {
        let _ = fs::remove_file(part);
        have = 0;
    }

    let url = format!("{}{}", mirror.base, pkg.path);
    let mut req = agent.get(&url);
    if have > 0 {
        req = req.header("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::StatusCode(416)) => {
            // 断点超出文件长度（镜像上的文件换了）——丢弃断点从头来
            let _ = fs::remove_file(part);
            return Err(AttemptError::Retry("HTTP 416，已重置断点".into()));
        }
        Err(ureq::Error::StatusCode(c @ (403 | 404 | 451))) => {
            return Err(AttemptError::SwitchMirror(format!("HTTP {c}")));
        }
        Err(ureq::Error::StatusCode(c)) => return Err(AttemptError::Retry(format!("HTTP {c}"))),
        Err(e) => return Err(AttemptError::Retry(format!("{e}"))),
    };

    let status = resp.status().as_u16();
    let mut start = have;
    if status == 200 {
        start = 0; // 服务器没理会 Range，整文件重下
    } else if status == 206 {
        let cr = resp
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        if let Some(cr) = cr {
            // "bytes <start>-<end>/<total>"
            if let Some(s) = cr.split_whitespace().nth(1).and_then(|r| r.split('-').next()) {
                if s.parse::<u64>().map(|v| v != have).unwrap_or(false) {
                    start = 0; // 服务器给的起点和断点对不上，从头来
                }
            }
        }
    } else {
        return Err(AttemptError::Retry(format!("HTTP {status}")));
    }

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(part)
        .map_err(|e| AttemptError::Retry(format!("打开 {}: {e}", part.display())))?;
    if start == 0 {
        file.set_len(0).map_err(|e| AttemptError::Retry(format!("重置断点文件: {e}")))?;
    }
    file.seek(SeekFrom::Start(start)).map_err(|e| AttemptError::Retry(format!("seek: {e}")))?;

    shared.update(|p| {
        p.mirror = mirror.id.to_string();
        p.state = "downloading".into();
        p.pkg_index = pkg_idx as i64;
        p.pkg = pkg.pkg.to_string();
        p.pkg_version = pkg.version.to_string();
        p.bytes_total = total;
        p.bytes_done = start;
        p.step = format!("下载 {}（{}/{}）· {}", pkg.pkg, pkg_idx + 1, PKGS.len(), mirror.label);
        if let Some(q) = p.pkgs.get_mut(pkg_idx) {
            q.state = "downloading".into();
        }
    });

    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut written = start;
    let mut last_pub = Instant::now();
    let mut speed_window_t = Instant::now();
    let mut speed_window_b = written;
    let mut speed = 0.0f64;

    loop {
        if shared.cancelled() {
            let _ = file.flush();
            return Err(AttemptError::Cancelled);
        }
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                file.write_all(&buf[..n]).map_err(|e| AttemptError::Retry(format!("写入 {}: {e}", part.display())))?;
                written += n as u64;
                let now = Instant::now();
                let dt = now.duration_since(speed_window_t).as_secs_f64();
                if dt >= 0.5 {
                    let inst = (written - speed_window_b) as f64 / dt;
                    speed = if speed == 0.0 { inst } else { 0.75 * speed + 0.25 * inst };
                    speed_window_t = now;
                    speed_window_b = written;
                }
                if now.duration_since(last_pub) >= Duration::from_millis(250) || written >= total {
                    last_pub = now;
                    shared.update(|p| {
                        p.bytes_done = written;
                        p.speed_bps = speed;
                    });
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                let _ = file.flush();
                return Err(AttemptError::Retry(format!("传输中断于 {}: {e}", human(written))));
            }
        }
    }
    file.flush().map_err(|e| AttemptError::Retry(format!("flush: {e}")))?;
    if written != total {
        return Err(AttemptError::Retry(format!("连接提前结束（{}/{}）", human(written), human(total))));
    }
    shared.update(|p| p.bytes_done = total);
    Ok(())
}

// ───────────────────────────── sha256 ─────────────────────────────

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

// ───────────────────────────── zip（wheel）读取 ─────────────────────────────
// 只实现 wheel 需要的子集：中央目录解析（含 zip64 扩展字段）+ stored/deflate 条目流式解压。

pub struct ZipEntry {
    pub name: String,
    pub method: u16,
    pub csize: u64,
    pub usize: u64,
    /// 本地头偏移（数据偏移在解压时再解析本地头得到）
    pub lho: u64,
}

pub struct Zip {
    path: PathBuf,
    pub entries: Vec<ZipEntry>,
}

impl Zip {
    pub fn open(path: &Path) -> std::io::Result<Zip> {
        let mut f = File::open(path)?;
        let size = f.metadata()?.len();
        let tail_len = size.min(66_000) as usize;
        let mut tail = vec![0u8; tail_len];
        f.seek(SeekFrom::End(-(tail_len as i64)))?;
        f.read_exact(&mut tail)?;

        let eocd = tail.windows(4).rposition(|w| w == b"PK\x05\x06").ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "不是 zip：找不到 EOCD")
        })?;
        let (mut count, mut cd_size, mut cd_off) = {
            // EOCD: +10 条目数(u16) +12 CD大小(u32) +16 CD偏移(u32)
            let b = &tail[eocd + 10..eocd + 20];
            (
                u16::from_le_bytes([b[0], b[1]]) as u64,
                u32::from_le_bytes([b[2], b[3], b[4], b[5]]) as u64,
                u32::from_le_bytes([b[6], b[7], b[8], b[9]]) as u64,
            )
        };
        if cd_off == u32::MAX as u64 || count == u16::MAX as u64 {
            // zip64：EOCD64 定位器（20 字节）紧挨在 EOCD 前，其 +8 处是 EOCD64 的绝对偏移
            if eocd >= 20 && &tail[eocd - 20..eocd - 16] == b"PK\x06\x07" {
                let loc = &tail[eocd - 12..eocd - 4];
                let eocd64_off = u64::from_le_bytes(loc.try_into().unwrap());
                if eocd64_off + 56 <= size && (size - eocd64_off) <= tail_len as u64 {
                    let base = size - tail_len as u64;
                    let t = (eocd64_off - base) as usize;
                    if &tail[t..t + 4] == b"PK\x06\x06" {
                        count = u64::from_le_bytes(tail[t + 32..t + 40].try_into().unwrap());
                        cd_size = u64::from_le_bytes(tail[t + 40..t + 48].try_into().unwrap());
                        cd_off = u64::from_le_bytes(tail[t + 48..t + 56].try_into().unwrap());
                    }
                }
            }
        }
        if cd_off + cd_size > size {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "zip 中央目录越界"));
        }
        let mut cd = vec![0u8; cd_size as usize];
        f.seek(SeekFrom::Start(cd_off))?;
        f.read_exact(&mut cd)?;

        let mut entries = Vec::with_capacity(count as usize);
        let mut p = 0usize;
        while p + 46 <= cd.len() && &cd[p..p + 4] == b"PK\x01\x02" {
            let method = u16::from_le_bytes([cd[p + 10], cd[p + 11]]);
            let csize32 = u32::from_le_bytes([cd[p + 20], cd[p + 21], cd[p + 22], cd[p + 23]]) as u64;
            let usize32 = u32::from_le_bytes([cd[p + 24], cd[p + 25], cd[p + 26], cd[p + 27]]) as u64;
            let nlen = u16::from_le_bytes([cd[p + 28], cd[p + 29]]) as usize;
            let elen = u16::from_le_bytes([cd[p + 30], cd[p + 31]]) as usize;
            let clen = u16::from_le_bytes([cd[p + 32], cd[p + 33]]) as usize;
            let lho32 = u32::from_le_bytes([cd[p + 42], cd[p + 43], cd[p + 44], cd[p + 45]]) as u64;
            let name = String::from_utf8_lossy(&cd[p + 46..p + 46 + nlen]).into_owned();
            let (mut csize, mut usize_, mut lho) = (csize32, usize32, lho32);
            // zip64 扩展字段（tag 0x0001）：按 8/8/8 顺序替换值为 0xFFFFFFFF 的字段
            if csize32 == u32::MAX as u64 || usize32 == u32::MAX as u64 || lho32 == u32::MAX as u64 {
                let mut e = p + 46 + nlen;
                let end = e + elen;
                while e + 4 <= end && e + 4 <= cd.len() {
                    let tag = u16::from_le_bytes([cd[e], cd[e + 1]]);
                    let len = u16::from_le_bytes([cd[e + 2], cd[e + 3]]) as usize;
                    if tag == 1 {
                        let mut q = e + 4;
                        if usize32 == u32::MAX as u64 && q + 8 <= end {
                            usize_ = u64::from_le_bytes(cd[q..q + 8].try_into().unwrap());
                            q += 8;
                        }
                        if csize32 == u32::MAX as u64 && q + 8 <= end {
                            csize = u64::from_le_bytes(cd[q..q + 8].try_into().unwrap());
                            q += 8;
                        }
                        if lho32 == u32::MAX as u64 && q + 8 <= end {
                            lho = u64::from_le_bytes(cd[q..q + 8].try_into().unwrap());
                        }
                        break;
                    }
                    e += 4 + len;
                }
            }
            entries.push(ZipEntry { name, method, csize, usize: usize_, lho });
            p += 46 + nlen + elen + clen;
        }
        Ok(Zip { path: path.to_path_buf(), entries })
    }

    /// 按 basename 精确匹配条目。
    pub fn find_basename(&self, name: &str) -> Option<&ZipEntry> {
        self.entries.iter().find(|e| {
            e.name == name || e.name.ends_with(&format!("/{name}"))
        })
    }

    /// 流式解压到 dest（覆盖写）。返回写出的字节数。
    pub fn extract_to(&self, entry: &ZipEntry, dest: &Path) -> std::io::Result<u64> {
        let mut f = File::open(&self.path)?;
        // 解析本地头拿到真实数据偏移（本地头的 name/extra 长度可能和中央目录不同）
        f.seek(SeekFrom::Start(entry.lho))?;
        let mut lh = [0u8; 30];
        f.read_exact(&mut lh)?;
        if &lh[0..4] != b"PK\x03\x04" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "本地头签名错误"));
        }
        let nlen = u16::from_le_bytes([lh[26], lh[27]]) as u64;
        let elen = u16::from_le_bytes([lh[28], lh[29]]) as u64;
        let data_off = entry.lho + 30 + nlen + elen;
        f.seek(SeekFrom::Start(data_off))?;

        let limited = (&f).take(entry.csize);
        let mut out = File::create(dest)?;
        let written = match entry.method {
            0 => std::io::copy(&mut BufReader::new(limited), &mut out)?,
            8 => {
                let mut inf = flate2::bufread::DeflateDecoder::new(BufReader::new(limited));
                std::io::copy(&mut inf, &mut out)?
            }
            m => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("不支持的压缩方法 {m}（{}）", entry.name),
                ))
            }
        };
        out.flush()?;
        Ok(written)
    }
}

// ───────────────────────────── 解压一个包 ─────────────────────────────

fn extract_pkg(pkg: &PkgSpec, wheel: &Path, libs: &Path, shared: &Arc<Shared>, pkg_idx: usize) -> JobResult {
    let zip = Zip::open(wheel).map_err(|e| JobError::Fatal(format!("打开 {} 失败: {e}", wheel.display())))?;
    for d in pkg.dlls {
        if shared.cancelled() {
            return Err(JobError::Cancelled);
        }
        let dest = libs.join(d.name);
        if file_size(&dest) == Some(d.size) {
            continue; // 幂等：已就位
        }
        shared.update(|p| p.step = format!("解压 {}", d.name));
        let entry = zip
            .find_basename(d.name)
            .ok_or_else(|| JobError::Fatal(format!("{} 里找不到 {}", pkg.wheel_file, d.name)))?;
        if entry.usize != d.size {
            return Err(JobError::Fatal(format!(
                "{} 里的 {} 声明大小 {} ≠ 预期 {}（pin 表与 wheel 不匹配）",
                pkg.wheel_file, d.name, entry.usize, d.size
            )));
        }
        let tmp = libs.join(format!("{}.dlpart", d.name));
        let n = zip
            .extract_to(entry, &tmp)
            .map_err(|e| JobError::Fatal(format!("解压 {} 失败: {e}", d.name)))?;
        if n != d.size {
            let _ = fs::remove_file(&tmp);
            return Err(JobError::Fatal(format!("{} 大小不符（{} ≠ {}），wheel 可能损坏", d.name, n, d.size)));
        }
        // 替换目标：已存在的旧文件先删（若被进程占用会失败 → 明确提示重启）
        if dest.exists() {
            fs::remove_file(&dest).map_err(|_| {
                JobError::Fatal(format!("{} 被占用（可能已被加载）。请完全退出 MicYou 后删除该文件再重试", d.name))
            })?;
        }
        fs::rename(&tmp, &dest).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            JobError::Fatal(format!("写入 {} 失败: {e}", dest.display()))
        })?;
        shared.update(|p| {
            if let Some(q) = p.pkgs.get_mut(pkg_idx) {
                q.have += 1;
            }
        });
    }
    Ok(())
}

// ───────────────────────────── 磁盘空间（Windows） ─────────────────────────────

#[cfg(target_os = "windows")]
fn free_disk_bytes(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    extern "system" {
        fn GetDiskFreeSpaceExW(
            lp_dir_name: *const u16,
            lp_free_bytes_available: *mut u64,
            lp_total_number_of_bytes: *mut u64,
            lp_total_number_of_free_bytes: *mut u64,
        ) -> i32;
    }
    // 用绝对路径的根目录查询
    let abs = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let wide: Vec<u16> = abs.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let mut avail: u64 = 0;
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut avail, std::ptr::null_mut(), std::ptr::null_mut()) };
    (ok != 0).then_some(avail)
}

#[cfg(not(target_os = "windows"))]
fn free_disk_bytes(_path: &Path) -> Option<u64> {
    None
}

// ───────────────────────────── 测试 ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_dll_names() -> Vec<&'static str> {
        let mut v: Vec<&'static str> = PKGS.iter().flat_map(|p| p.dlls.iter().map(|d| d.name)).collect();
        v.sort();
        v
    }

    #[test]
    fn pkg_table_is_consistent() {
        assert_eq!(PKGS.len(), 7);
        let names = expected_dll_names();
        assert_eq!(names.len(), 19, "19 个运行库");
        let mut dedup = names.clone();
        dedup.dedup();
        assert_eq!(dedup.len(), names.len(), "dll 不允许重复");
        for p in PKGS {
            assert_eq!(p.sha256.len(), 64, "{} sha256 长度", p.pkg);
            assert!(p.sha256.chars().all(|c| c.is_ascii_hexdigit()));
            assert!(p.wheel_size > 0);
            assert!(p.path.ends_with(p.wheel_file), "{} path 应以文件名结尾", p.pkg);
            assert!(p.path.starts_with(&p.sha256[..2].to_lowercase()) || p.path.contains('/'), "path 是内容寻址布局");
            assert!(!p.dlls.is_empty());
            for d in p.dlls {
                assert!(d.size > 0);
                assert!(!d.name.contains('/'));
            }
        }
        // 小包在前（下载顺序 = 数组顺序，先易后难）
        for w in PKGS.windows(2) {
            assert!(w[0].wheel_size <= w[1].wheel_size, "PKGS 应按 wheel 大小升序");
        }
        // 与用户实测的 22 文件清单中的 19 个 CUDA dll 完全一致
        let want = [
            "cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll",
            "cudnn64_9.dll", "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll",
            "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_engines_tensor_ir64_9.dll", "cudnn_ext64_9.dll", "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll", "cudnn_ops64_9.dll",
            "cufft64_11.dll", "cufftw64_11.dll", "curand64_10.dll",
            "cusolver64_11.dll", "cusolverMg64_11.dll", "cusparse64_12.dll",
        ];
        let mut want = want.to_vec();
        want.sort();
        assert_eq!(names, want);
    }

    #[test]
    fn mirror_order_prefers_choice_and_covers_all() {
        let o = mirror_order("aliyun");
        assert_eq!(o[0].id, "aliyun");
        assert_eq!(o.len(), MIRRORS.len());
        let o2 = mirror_order("不存在的镜像");
        assert_eq!(o2[0].id, "tuna", "未知偏好回退清华");
        assert_eq!(o2.len(), MIRRORS.len());
        // 镜像 id 不重复
        let mut ids: Vec<_> = MIRRORS.iter().map(|m| m.id).collect();
        let n = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), n);
        for m in MIRRORS {
            assert!(m.base.starts_with("https://") && m.base.ends_with("/packages/"));
        }
    }

    #[test]
    fn inventory_counts_and_gating() {
        let dir = std::env::temp_dir().join(format!("mambo_inv_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let inv = runtime_inventory(&dir);
        assert!(!inv.cuda_ready());
        assert_eq!(inv.present, 0);
        assert_eq!(inv.total, 19);
        assert_eq!(inv.missing.len(), 19);
        assert!(!inv.ort_ready());

        // 放一个大小正确的 cudart → present=1
        let d = &PKGS.iter().find(|p| p.pkg == "nvidia-cuda-runtime-cu12").unwrap().dlls[0];
        write_zeros(&dir.join(d.name), d.size);
        let inv = runtime_inventory(&dir);
        assert_eq!(inv.present, 1);
        assert!(!inv.missing.contains(&d.name));
        // 大小不对 → 仍算缺失
        write_zeros(&dir.join(d.name), 10);
        let inv = runtime_inventory(&dir);
        assert_eq!(inv.present, 0);
        assert!(inv.missing.contains(&d.name));
        assert!(!runtime_inventory(&dir).cuda_ready());

        for f in ["onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll"] {
            fs::write(dir.join(f), b"x").unwrap();
        }
        assert!(runtime_inventory(&dir).ort_ready());
        fs::remove_dir_all(&dir).unwrap();
    }

    fn write_zeros(p: &Path, n: u64) {
        let f = File::create(p).unwrap();
        f.set_len(n).unwrap();
    }

    #[test]
    fn initial_progress_marks_complete_libs_done() {
        let dir = std::env::temp_dir().join(format!("mambo_prog_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let p = initial_progress(&dir, "tuna");
        assert_eq!(p.state, "idle");
        assert_eq!(p.all_done, 0);
        assert_eq!(p.all_total, total_wheel_bytes());
        assert!(p.pkgs.iter().all(|q| q.state == "wait"));
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"state\":\"idle\"") && json.contains("\"pkgs\""));

        // 全部就位 → done + skip
        for pkg in PKGS {
            for d in pkg.dlls {
                write_zeros(&dir.join(d.name), d.size);
            }
        }
        let p = initial_progress(&dir, "pypi");
        assert_eq!(p.state, "done");
        assert_eq!(p.all_done, p.all_total);
        assert!(p.terminal());
        assert!(p.pkgs.iter().all(|q| q.state == "skip" && q.have == q.dlls));
        fs::remove_dir_all(&dir).unwrap();
    }

    // ── zip 读写往返（手工构造一个含 stored + deflate 条目的 zip）──

    fn build_test_zip(path: &Path, entries: &[(&str, &[u8], u16)]) {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        let mut buf: Vec<u8> = Vec::new();
        let mut cd: Vec<u8> = Vec::new();
        for (name, data, method) in entries {
            let lho = buf.len() as u64;
            let (comp, crc) = if *method == 8 {
                let mut e = DeflateEncoder::new(Vec::new(), Compression::default());
                e.write_all(data).unwrap();
                (e.finish().unwrap(), crc32(data))
            } else {
                (data.to_vec(), crc32(data))
            };
            // local header
            buf.extend_from_slice(b"PK\x03\x04");
            buf.extend_from_slice(&20u16.to_le_bytes()); // version needed
            buf.extend_from_slice(&0u16.to_le_bytes()); // flags
            buf.extend_from_slice(&method.to_le_bytes());
            buf.extend_from_slice(&[0; 4]); // time/date
            buf.extend_from_slice(&crc.to_le_bytes());
            buf.extend_from_slice(&(comp.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
            buf.extend_from_slice(&0u16.to_le_bytes()); // extra len
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&comp);
            // central directory entry
            cd.extend_from_slice(b"PK\x01\x02");
            cd.extend_from_slice(&20u16.to_le_bytes()); // version made by
            cd.extend_from_slice(&20u16.to_le_bytes()); // version needed
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&method.to_le_bytes());
            cd.extend_from_slice(&[0; 4]);
            cd.extend_from_slice(&crc.to_le_bytes());
            cd.extend_from_slice(&(comp.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes()); // extra
            cd.extend_from_slice(&0u16.to_le_bytes()); // comment
            cd.extend_from_slice(&0u16.to_le_bytes()); // disk
            cd.extend_from_slice(&0u16.to_le_bytes()); // int attr
            cd.extend_from_slice(&0u32.to_le_bytes()); // ext attr
            cd.extend_from_slice(&(lho as u32).to_le_bytes());
            cd.extend_from_slice(name.as_bytes());
        }
        let cd_off = buf.len() as u64;
        buf.extend_from_slice(&cd);
        // EOCD
        buf.extend_from_slice(b"PK\x05\x06");
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(cd.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(cd_off as u32).to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        fs::write(path, &buf).unwrap();
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut c: u32 = 0xFFFF_FFFF;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }

    #[test]
    fn zip_roundtrip_stored_and_deflate() {
        let dir = std::env::temp_dir().join(format!("mambo_zip_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let zp = dir.join("t.zip");
        let big: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        build_test_zip(
            &zp,
            &[
                ("nvidia/cuda_runtime/bin/cudart64_12.dll", b"hello dll", 8),
                ("nvidia/cuda_runtime-12.9.dist-info/METADATA", b"meta", 0),
                ("nvidia/cublas/bin/cublas64_12.dll", &big[..], 8),
            ],
        );
        let zip = Zip::open(&zp).unwrap();
        assert_eq!(zip.entries.len(), 3);
        let e = zip.find_basename("cudart64_12.dll").unwrap();
        assert_eq!(e.method, 8);
        assert_eq!(e.usize, 9);
        let out = dir.join("a.dll");
        assert_eq!(zip.extract_to(e, &out).unwrap(), 9);
        assert_eq!(fs::read(&out).unwrap(), b"hello dll");

        let e2 = zip.find_basename("METADATA").unwrap();
        assert_eq!(e2.method, 0);
        let out2 = dir.join("b.txt");
        zip.extract_to(e2, &out2).unwrap();
        assert_eq!(fs::read(&out2).unwrap(), b"meta");

        let e3 = zip.find_basename("cublas64_12.dll").unwrap();
        let out3 = dir.join("c.dll");
        assert_eq!(zip.extract_to(e3, &out3).unwrap(), big.len() as u64);
        assert_eq!(fs::read(&out3).unwrap(), big);

        assert!(zip.find_basename("nonexistent.dll").is_none());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn zip_rejects_garbage() {
        let dir = std::env::temp_dir().join(format!("mambo_zipbad_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let zp = dir.join("bad.zip");
        fs::write(&zp, b"this is not a zip file at all").unwrap();
        assert!(Zip::open(&zp).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sha256_known_vector() {
        let dir = std::env::temp_dir().join(format!("mambo_sha_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("abc");
        fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let p2 = dir.join("empty");
        fs::write(&p2, []).unwrap();
        assert_eq!(
            sha256_file(&p2).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn human_sizes() {
        assert_eq!(human(512), "512B");
        assert_eq!(human(2048), "2KB");
        assert_eq!(human(5 * 1024 * 1024), "5.0MB");
        assert!(human(3_400_000_000).ends_with("GB"));
    }

    /// 真实网络端到端：从清华镜像拉最小的 wheel（cuda-runtime，3.6MB），
    /// 校验 sha256 并解出 cudart64_12.dll。默认忽略（CI 不跑网络测试），
    /// 手工验证：cargo test --release -- --ignored --nocapture
    #[test]
    #[ignore = "需要网络（下载 3.6MB）"]
    fn real_download_smallest_wheel_from_tuna() {
        let dir = std::env::temp_dir().join(format!("mambo_net_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let libs = dir.join("libs");
        fs::create_dir_all(&libs).unwrap();
        let pkg = &PKGS[0];
        assert_eq!(pkg.pkg, "nvidia-cuda-runtime-cu12");
        let shared = Arc::new(Shared::new(&libs, "tuna"));
        let agent = build_agent();
        let part = dir.join("w.part");
        let wheel = dir.join("w.whl");
        let mirrors = mirror_order("tuna");
        download_wheel(&agent, pkg, &part, &wheel, &shared, 0, &mirrors).unwrap();
        assert_eq!(file_size(&wheel), Some(pkg.wheel_size));
        extract_pkg(pkg, &wheel, &libs, &shared, 0).unwrap();
        assert_eq!(file_size(&libs.join("cudart64_12.dll")), Some(583_680));
        assert!(!runtime_inventory(&libs).cuda_ready()); // 只有 1/19
        assert_eq!(runtime_inventory(&libs).present, 1);
        let snap = shared.snapshot();
        assert_eq!(snap.pkgs[0].have, 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// 真实网络断点续传：先手工下载前 1MB 写入 .part，再跑 download_wheel，
    /// 应当带 Range 头续传补齐并通过 sha256。
    #[test]
    #[ignore = "需要网络（下载 3.6MB）"]
    fn real_resume_from_partial_file() {
        let dir = std::env::temp_dir().join(format!("mambo_resume_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let pkg = &PKGS[0];
        let agent = build_agent();
        let part = dir.join("w.part");

        // 手工构造 1MB 断点
        let cut = 1_000_000u64;
        let url = format!("{}{}", mirror_by_id("tuna").base, pkg.path);
        let resp = agent.get(&url).header("Range", format!("bytes=0-{}", cut - 1)).call().unwrap();
        assert_eq!(resp.status().as_u16(), 206, "TUNA 应支持 Range");
        let mut r = resp.into_body().into_reader();
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).unwrap();
        assert_eq!(buf.len() as u64, cut);
        fs::write(&part, &buf).unwrap();

        let libs = dir.join("libs");
        fs::create_dir_all(&libs).unwrap();
        let shared = Arc::new(Shared::new(&libs, "tuna"));
        let wheel = dir.join("w.whl");
        let mirrors = mirror_order("tuna");
        download_wheel(&agent, pkg, &part, &wheel, &shared, 0, &mirrors).unwrap();
        assert_eq!(file_size(&wheel), Some(pkg.wheel_size));
        assert_eq!(sha256_file(&wheel).unwrap(), pkg.sha256, "续传拼接后 sha256 必须一致");
        fs::remove_dir_all(&dir).unwrap();
    }

    /// 编排级真实测试：预置其余 6 包的 dll（稀疏零文件），跑完整 spawn→run 流程，
    /// 只应下载最小的 cuda-runtime 包，其余 skip；结束后 .rt_cache 清理、状态 done。
    #[test]
    #[ignore = "需要网络（下载 3.6MB）"]
    fn real_run_skips_present_packages() {
        let dir = std::env::temp_dir().join(format!("mambo_run_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let libs = dir.join("libs");
        fs::create_dir_all(&libs).unwrap();
        for pkg in &PKGS[1..] {
            for d in pkg.dlls {
                let f = File::create(libs.join(d.name)).unwrap();
                f.set_len(d.size).unwrap(); // 稀疏文件，不占盘
            }
        }
        let mut h = spawn(dir.clone(), "tuna").unwrap();
        let deadline = Instant::now() + Duration::from_secs(90);
        while !h.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(200));
        }
        let p = h.snapshot();
        assert_eq!(p.state, "done", "err={}", p.error);
        assert_eq!(p.all_done, p.all_total);
        assert_eq!(p.pkgs[0].state, "done");
        assert!(p.pkgs[1..].iter().all(|q| q.state == "skip"));
        assert_eq!(file_size(&libs.join("cudart64_12.dll")), Some(583_680), "真实文件已解压");
        assert!(!dir.join(".rt_cache").exists(), "缓存目录应已清理");
        assert!(runtime_inventory(&libs).cuda_ready(), "19/19 就位");
        h.join();
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Content-Range 起点解析（续传对齐逻辑依赖它）
    #[test]
    fn content_range_parsing_helper() {
        fn start_of(cr: &str) -> Option<u64> {
            cr.split_whitespace().nth(1).and_then(|r| r.split('-').next()).and_then(|s| s.parse().ok())
        }
        assert_eq!(start_of("bytes 100-200/300"), Some(100));
        assert_eq!(start_of("garbage"), None);
    }
}
