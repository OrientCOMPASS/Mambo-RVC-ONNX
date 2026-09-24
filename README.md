<div align="center"><p><em>“曼波~曼波~哦马自立曼波~🎵”</em></p></div>

# 🎤 Mambo RVC ONNX — MicYou 实时 AI 变声插件

在本地做实时流式 AI 变声的 [MicYou](https://github.com/LanRhyme/MicYou) 原生 DSP 插件。
推理链路是 **HuBERT（内容） → RMVPE（音高） → RVC（声码器）**，全部跑在 ONNX Runtime 上，默认使用 CUDA。

它的工作方式是：把连续的麦克风音频切成**带上下文的重叠窗口**，每个窗口整体推理出一段音频，
只取窗口中间那一段输出，相邻两段用交叉淡化（OLA）拼接。因此有两个互相制约的量：

- **推理音频长度**（= 左上下文 + 输出块 + 右预读）越长，模型看到的上下文越充分，音质越好，但每块的计算量也越大；
- **输出块（Chunk）** 越短，延迟越低，但每秒的推理次数越多。

所有参数都在 **设置 → 曼波RVC 控制台面板**（或插件卡片的表单）里，
用来匹配你自己的硬件和延迟目标；面板会实时显示推理音频长度与估算延迟。

---

## 📦 安装

### 1. 安装插件

用 MicYou 安装 Release 里的插件包。

### 2. 启用插件

重启 MicYou，在设置里启用 `opss.mambo-rvc-onnx`。启用后会弹一条系统通知，
告诉你**实际加载了哪个模型**；同样的信息也写在插件目录的 `rvc_plugin.log` 里。

插件包自带的 `libs/` 里已经有 3 个 ONNX Runtime 文件
（`onnxruntime.dll` / `onnxruntime_providers_cuda.dll` / `onnxruntime_providers_shared.dll`），
启用后即可用 **CPU 推理**先跑起来——CPU 是可用的，只是每块耗时 τ 会显著变大，
需要按下面的方法把 Chunk Size 调大来匹配，不是"不能用"，而是"要换一组参数"。

### 3. 一键拉取 CUDA 运行库（推荐）

要用 GPU，需要 19 个 CUDA/cuDNN 运行库（约 **2.1GB 下载 / 3.2GB 磁盘**）。
它们**不在插件包里**，而是在面板中按需拉取：

1. 打开 **设置 → 侧边栏「曼波RVC · 控制台」**（插件专属面板）；
2. 缺库时顶部会有黄色提醒，「CUDA 运行库」卡片自动展开；
3. 选择下载源（**清华 TUNA**（默认）/ 阿里云 / 中科大 / 腾讯云 / PyPI 官方），点 **开始拉取**；
4. 面板实时显示每个包的下载 / 校验 / 解压进度、速度与剩余时间；
5. 全部完成后插件**自动重建推理会话切换到 CUDA**（约 1~3 秒静音），无需重启。

拉取的文件来自 **NVIDIA 官方发布在 PyPI 的 wheel**，版本固定并逐个做 **SHA256 校验**；
下载到插件目录 `.rt_cache/`（每包解压完立即删除，磁盘峰值约 4GB），
最终解压到 📂 `%APPDATA%\micyou\plugins\opss.mambo-rvc-onnx\libs\`。
支持**断点续传**（取消 / 断网后点开始拉取即可继续），单个镜像失败会**自动轮换**下一个。

> 手动方式也支持：自行把 19 个 dll 放进 `libs/` 即可（清单见「预期文件」）。
> `onnxruntime.dll` 由插件按绝对路径显式加载；其余依赖靠插件启动时把 `libs/`
> 加入进程 DLL 搜索路径（`SetDllDirectoryW`）解析。

<details>
<summary><b>预期文件（libs/ 共 22 个 dll）</b></summary>

| 来源 | 文件 |
| :--- | :--- |
| 插件自带（ORT 1.27.1 cuda12） | `onnxruntime.dll` `onnxruntime_providers_cuda.dll` `onnxruntime_providers_shared.dll` |
| `nvidia-cuda-runtime-cu12==12.9.79` | `cudart64_12.dll` |
| `nvidia-cublas-cu12==12.9.2.10` | `cublas64_12.dll` `cublasLt64_12.dll` |
| `nvidia-cudnn-cu12==9.25.1.1` | `cudnn64_9.dll` `cudnn_adv64_9.dll` `cudnn_cnn64_9.dll` `cudnn_engines_precompiled64_9.dll` `cudnn_engines_runtime_compiled64_9.dll` `cudnn_engines_tensor_ir64_9.dll` `cudnn_ext64_9.dll` `cudnn_graph64_9.dll` `cudnn_heuristic64_9.dll` `cudnn_ops64_9.dll` |
| `nvidia-cufft-cu12==11.4.1.4` | `cufft64_11.dll` `cufftw64_11.dll` |
| `nvidia-curand-cu12==10.3.10.19` | `curand64_10.dll` |
| `nvidia-cusolver-cu12==11.7.5.82` | `cusolver64_11.dll` `cusolverMg64_11.dll` |
| `nvidia-cusparse-cu12==12.5.10.65` | `cusparse64_12.dll` |

</details>

> **DSP 链位置**：宿主目前把插件节点固定插在 **AEC 之后**（manifest 里的 `insertAfter` 暂未被使用），
> 也就是说变声结果还会再经过宿主的降噪 / EQ / AGC / VAD。宿主的降噪是针对真人语音训练的，
> 对合成音色可能吃掉气声、引入金属感。如果发现音色发闷或有金属声，
> 到「设置 → 音频 → 处理链」把 `Plugins` 节点拖到链尾（VAD 之后）再对比。

---

## 🧬 用你自己的模型

把 RVC 模型（`.onnx`）放进插件目录的 **`user_models/`** 文件夹即可，插件会优先加载它。
首次运行会自动创建这个文件夹并放一个说明文件。

| 优先级 | 位置 |
| :--- | :--- |
| 1 | `user_models/*.onnx` —— 有多个时取**修改时间最新**的那个 |
| 2 | `models/uma-Matikane_Tannhauser.onnx`（插件自带，回退） |

- 文件名含 `hubert` / `contentvec` 的会被当成内容编码器，含 `rmvpe` / `f0` 的会被当成音高提取器。
  所以你可以只覆盖其中一个，其余继续用自带的。
- **增删模型后，在 设置 → 曼波RVC 控制台面板 点「重载模型」**（或关闭再启用本插件）。
- 多音色模型用 **Speaker ID** 选音色；单音色模型保持 0。

### 兼容性：签名不一致的模型也能用

社区导出的模型命名和张量签名差别很大，所以插件**不按名字硬编码形状和类型**，而是读取模型
自己声明的 dtype 与秩来装配输入，只把符号维（`-1`）实例化成帧数/音频长度，其余照抄声明：

| 差异 | 处理 |
| :--- | :--- |
| `nsff0` / `pitchf` / 别的名字 | 按「名字 + 声明 dtype」联合判断：float 的 f0 → 连续 f0，int 的 → coarse pitch |
| `sid` / `ds` / `speaker` / `spk` | 都识别为音色 id |
| 额外的 `rnd: [1,192,T]` 噪声输入 | 每块自动生成新噪声（RVC 的 flow 解码器靠它产生音色细节） |
| `noise_scale` / `length_scale` / `vol` 等标量 | 用 VITS/RVC 的通用默认值（0.667 / 1.0 / 1.0） |
| HuBERT 的 `source` 是 `[B,L]`（秩 2）而不是 `[1,1,L]` | 按声明的秩装配 |
| `padding_mask` 是 `Bool` 而不是 `Int64` | 按声明的 dtype 装配；`padding_mask` 填 False（fairseq 语义 True = 填充），`attention_mask` 填 True |
| 16k 输入长度不是 320 的整数倍 | 窗口按 960 对齐（= 48k 下 20ms = 16k 下 320 样本）；仍失败时按 320 递增补零重试并记住有效值 |
| **模型把序列长度写死了**（导出时没开 dynamic axes） | 加载期探测出唯一可用的帧数 T，把窗口撑到 `T×480`，多出来的长度全部塞进左上下文 ⇒ **延迟不变**，只是每块算力变多 |
| 非 48kHz 模型（hop=400 / 320） | 按输出长度反推采样率并重采样到 48kHz |
| 认不出来的输入 | 按声明形状填 0，并在日志里列出名字 |

加载时日志会逐条打印 `输入 <名字> 角色=<Role> 类型=<dtype> 声明形状=[...]` 和探测结论，
换模型出问题时可以据此定位。

> ⚠️ 有些模型导出时把序列长度写死了（例如只接受 200 帧 = 2 秒），这种模型可以用，
> 但每块的计算量会被迫按那个长度算。想要更省算力，请用 dynamic axes 重新导出。

---

## 🎛️ 参数怎么调

### 两条硬约束

**约束 A（不断流）：单块推理耗时 τ 必须小于 Chunk Size。**

否则积压只增不减，涨到 `Max Backlog` 上限就会丢弃最旧的音频，表现为一次短断流。
τ 怎么估：推理要处理"推理音频长度"这么多的音频，受内存带宽限制，**普通 PC 大致能做到 3~4 倍实时**，
也就是

```
τ ≈ 推理音频长度 ÷ 3.5        （纯 CPU，粗略估算）
```

推理音频长度 1 秒 ⇒ τ ≈ 290ms ⇒ Chunk Size 要大于这个值。
GPU 上 τ 通常只有几十毫秒，Chunk Size 可以取到 100~200ms。
实测值不用猜：日志里出现 `[Perf] 单块推理 XXms 已超过 chunk XXms` 就是明确信号。

**约束 B（音质）：推理音频长度 = 左上下文 + Chunk + 右预读，尽量 ≥ 1000ms。**

模型看到的上下文太短时，块边界处的音高和咬字会失准，听感就是**电音 / 金属声 / 发虚**。
如果音质不满意，第一件事是把这三项之和拉到 1 秒以上（优先加**左上下文**，因为它不增加延迟）。

### 延迟与算力

```
端到端算法延迟 ≈ 右预读 + 交叉淡化 + 抖动缓冲 + (0 ~ Chunk)
每块计算量    ≈ 推理音频长度 ÷ Chunk        倍实时
```

**左上下文不进入延迟公式**，它只换音质和算力。所以调延迟请动「右预读」和「Chunk」，
调音质请动「左上下文」。

### 各项说明

| 参数 | 默认 | 作用 |
| :--- | :--- | :--- |
| **Chunk Size (ms)** | 200 | 每次推理输出的音频块。稳定性主旋钮，必须 > τ |
| **Lookahead (ms)** | 80 | 块尾预读（右上下文）。延迟的主要来源之一 |
| **Left Context (ms)** | 720 | 块前历史（左上下文）。不增加延迟，成比例增加算力 |
| **Crossfade (ms)** | 50 | 块拼接处的交叉淡化，掩盖边界毛刺 |
| **Jitter Buffer (ms)** | 0 | 输出缓冲垫，吸收推理耗时抖动。代价是等量的固定延迟 |
| **Max Backlog (ms)** | 300 | 积压上限，超过就丢弃最旧音频把延迟拉回来 |
| **Pitch Shift** | 0 | 变调半音数。男变女 +12，女变男 −12，同声线微调 ±2 |
| **Speaker ID** | 0 | 多音色模型的 sid |
| **Silence Gate** | 开 | 安静的块不跑推理，省算力，也让句间停顿自动排空积压 |
| **Gate Threshold (dBFS)** | −80 | 静音判定电平 |
| **ORT 图优化等级** | all | `all` 最快；怀疑图优化改变了声码器数值时可切 `basic` 对比 |

所有时长参数会被吸附到 **10ms 的整数倍**，窗口再按 960 样本对齐（保证 48k 帧与 16k 帧同时对齐，
否则每块会有几毫秒的时间轴漂移并逐块累积）。

### 建议的调参顺序

1. 先定**推理音频长度**：左上下文 + Chunk + 右预读 ≥ 1000ms（音质底线）。
2. 再定 **Chunk**：按你的硬件估 τ，取 Chunk > τ 且留有余量。
3. 然后定**右预读**：在能接受的延迟里尽量给大，剩下的预算全给左上下文。
4. 最后微调 **Crossfade**（边界有毛刺就加大）和 **Jitter Buffer**（偶发断流就加大）。

> 📌 设置值是**按安装持久化**的。改代码里的默认值不会影响已装好的实例，必须动滑杆，或卸载重装插件。

---

## 🩺 症状对照

| 症状 | 原因 | 处理 |
| :--- | :--- | :--- |
| **偶发短断流**，日志有 `[Flush] 断流` | τ 逼近或超过 Chunk Size，积压撞上上限 | 调大 Chunk Size；或缩短推理音频长度（减左上下文）。**如果确认只是推理性能到了参数边界、偶发一次可以接受，就适当加大 Jitter Buffer 来吸收抖动** |
| **持续断流** | τ 明显大于 Chunk Size | 必须调大 Chunk Size 或缩短推理音频长度；加大缓冲没用 |
| **电音 / 金属声 / 发虚** | 推理音频长度太短，块边界的音高与声码器状态欠约束 | **把左上下文 + Chunk + 右预读拉到 ≥ 1000ms**（优先加左上下文，不增加延迟） |
| 电音，且日志显示频繁 `输出欠载` | 欠载会插入淡出静音、恢复时再淡入，频繁发生时是周期性的电平抖动 | 加大 Jitter Buffer，或调大 Chunk Size 降低推理频率 |
| 电音，参数已经够大 | 可能是宿主的降噪/AGC 在处理合成音色，或 ORT 的激进图优化 | 把 `Plugins` 节点拖到处理链末尾；把 ORT 图优化等级切到 `basic` 对比 |
| 软起音/气声被切掉 | 静音门限太高 | 调低 Gate Threshold（如 −90），或关掉 Silence Gate |
| 对方偶尔听到**你的原声** | 极端情况下单次音频块超过 1 秒触发旁路 | 正常网络下不会发生；若频繁出现请反馈 |
| 完全没声音，日志有 `加载失败` | ORT 库缺失或模型文件缺失 | 检查 `libs/` 的 3 个 onnxruntime dll 和 `models/`；插件会每 15 秒重试，补齐文件后无需重启 |
| 能用但延迟大、CPU 占用高 | CUDA 运行库未拉取，正在 CPU 推理 | 打开面板「CUDA 运行库」卡片一键拉取，完成后自动切到 GPU |

---

## 📄 日志

插件目录下的 `rvc_plugin.log`。只在有信息量时写入，不做周期性刷屏：

- `[Init]` / `[Config]` —— 启动与生效参数（参数变化时才再打一条）
- `[Model]` —— 实际加载的三个模型路径、`user_models/` 候选、每个输入的角色与声明签名
- `[RVC] 探测` —— 帧数是动态的还是写死的、实测 hop 与采样率
- `[Geom]` —— 窗口构成与算力倍数（只在参数变化时）
- `[Perf]` —— 单块推理耗时超过 Chunk 时一条；输出欠载每累计 20 次一条
- `[Flush]` —— 每次断流一条，带当时的推理耗时
- `[Worker]` / `[RVC]` 错误 —— 推理失败原因（相同消息 2 秒内只记一次）

启用插件时还会有一条系统通知，报告加载的模型。

---

## 🛠️ 开发者说明

```
src/lib.rs      宿主接口层：C ABI、生命周期、实时 process、参数、文件日志、状态上报、
                       面板桥接（ui:rt_status / ui:rt_fetch / ui:rt_cancel / ui:reload）
src/stream.rs   流式缓冲层：无锁 SPSC 环形缓冲 + 窗口调度器（OLA / 积压控制 / 热重载编排）
src/rvc.rs      模型层：目录与 CUDA 运行库引导、模型发现、ORT 会话、
                       输入自适应装配、HuBERT / RMVPE(mel 前端) / RVC、重采样
src/fetch.rs    运行库拉取：PyPI pin 表、多镜像断点续传下载、SHA256 校验、
                       wheel(zip) 流式解压、进度状态机（面板轮询 get_config 展示）
panel.html      设置面板（自包含单文件）：参数滑杆 + 可折叠的「CUDA 运行库」拉取卡片
tools/gen_pins.py  重新生成 fetch.rs 的 pin 表（PyPI JSON + Range 读 wheel 中央目录，不下载整包）
```

- **构建**：`cargo build --release` → `target/release/mambo_rvc_onnx.dll`
  （`entry` 写的是无后缀基础名，宿主按平台自动补 `.dll`/`.so`/`.dylib`；
  Linux/macOS 产物需去掉 `lib` 前缀才能跨平台共用一个 ZIP）
- **测试**：`cargo test` —— 23 个单元测试，覆盖环形缓冲的 SPSC 正确性、
  窗口调度的「每块只输出一次且严格按时间顺序」、OLA 状态机的不变量、默认窗口的对齐约束、
  pin 表一致性（7 包 / 19 dll / sha256 格式 / 升序）、zip 中央目录解析与解压往返、sha256 向量。
  另有 2 个 `#[ignore]` 的真实网络集成测试（从清华镜像下载最小 wheel、断点续传拼接后校验 sha256）：
  `cargo test --release -- --ignored`
- **端到端验证**：`../rvc-harness`（独立程序，不属于插件本体）通过真实 C ABI 加载编译产物、
  跑真实 ONNX Runtime 推理，把输出解码回时间轴来判定有没有复读/倒流/丢块/卡死
- **CI**：`.github/workflows/main.yml` 每次 push 构建 dll + 跑单测；
  `.github/workflows/release.yml` 在推 `v*` 标签时校验三处版本号一致、构建、检查 ABI 导出符号、
  下载 ORT 官方 cuda12 包取 3 个 dll、复用 v1.1.0 Release 的模型（可用 `models_url` 输入覆盖），
  打包 `opss.mambo-rvc-onnx.zip` + `plugin.json`（updateUrl 资产）发布 Release
- **实时安全**：`process()` 内无堆分配、无锁、无 Host API 调用；推理线程与下载线程都是自建子线程，
  按宿主规范同样不调用任何 Host API，需要给用户看的消息通过定时器回调在宿主线程转发。
  拉取进度 = 下载线程写 `Arc<Mutex<Progress>>`，宿主线程的 500ms 定时器读快照写 `set_config("rt_state")`，
  面板 iframe 每 500ms 轮询 `get_config`（面板桥只有 get/set_config + trigger，没有事件订阅）
- **运行库版本 pin 在 `src/fetch.rs` 的 `PKGS` 表**：与 ORT 的 cuda12 构建匹配（CUDA 12.9 + cuDNN 9.25）。
  ONNX Runtime 官方从 1.27 起 PyPI wheel 只发 CUDA 13 版，**cuda12 构建只存在于 GitHub Release 资产**
  （`onnxruntime-win-x64-gpu_cuda12-<ver>.zip`），CI 里同样 pin 了 1.27.1。
  升级组合时：改 release.yml 的 `ORT_ZIP_URL` + 跑 `python3 tools/gen_pins.py --latest` 重新生成 pin 表，
  两者的大版本必须匹配（ORT cuda12 ↔ `nvidia-*-cu12`）
- **CUDA EP 门控**：`libs/` 里 19 个运行库**全部就位（大小精确匹配）才会尝试注册 CUDA EP**，
  缺库直接建 CPU 会话，不做无谓的 provider 加载尝试；拉取完成后 `MODEL_EPOCH` +1 触发会话重建，
  ORT 的 provider 加载失败不缓存（`ProviderLibrary::Get()` 失败即 Unload），因此**不用重启就能热切到 CUDA**
- **`init_ort_once` 只缓存成功**：失败（如 `onnxruntime.dll` 暂时缺失）允许 15 秒重试路径继续尝试
- **不要**在 `Cargo.toml` 里设 `panic = "abort"`：本 cdylib 跑在宿主进程内，abort 会带走整个 MicYou
- **任何 ort API 都必须在 `ort::init_from` 成功之后调用**：`ort::Error` 的构造内部会走 `ortsys!`，
  dylib 尚未加载时 ort 会用默认库名（`onnxruntime.dll` / `libonnxruntime.so`）懒加载，加载不到就
  `expect` panic —— 而它发生在 worker 线程里，插件会变成永久静音。所以 `init_ort_once` 返回
  `Result`：找不到 dylib 时走正常的错误 + 15 秒重试路径，不会 panic
- **ORT 图优化等级不要用 `Level3`**：ort rc.13 把它映射到 `ORT_ENABLE_LAYOUT`(=3)，
  而 ORT 只接受 `{0,1,2,99}`，运行期会报 `graph_optimization_level is not valid` 导致三个模型全部
  加载失败。全部优化对应的枚举是 `All`(=99)。这类错误 `cargo check` 查不出来

<div align="center"><p><em>Powered by Rust & ONNX Runtime. 让每一次“曼波”都如丝般顺滑。</em></p></div>
