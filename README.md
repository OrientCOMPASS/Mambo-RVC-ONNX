<div align="center"><p><em>“曼波~曼波~哦马自立曼波~🎵”</em></p></div>

# 🎤 Mambo RVC ONNX v1.1 — MicYou 实时 AI 变声插件

本插件专为 [MicYou](https://github.com/LanRhyme/MicYou) 开发，在本地做实时流式 AI 变声（HuBERT → RMVPE → RVC，ONNX Runtime + CUDA）。

**v1.1 是对 v1.0 的重构**，修掉了「变声后一直复读同一小段音频」的问题，把默认延迟从 ~500ms 降到 ~260ms、
每块推理的计算量降到原来的 **61%**，并支持**优先加载你自己的 RVC 模型**。
完整的问题定位与数据见 `docs/诊断报告.md`。

---

## 🚨 极其重要的安装警告 (CUDA DLL)

本插件强依赖 GPU 进行实时推理。**请务必下载 Release 页面中的 `cuda_dll.zip`**，并将其中的 CUDA 运行库解压至插件目录下的 `libs` 文件夹内：

📂 **目标路径**：`%APPDATA%\micyou\plugins\opss.mambo-rvc-onnx\libs\`

⚠️ **验证标准**：预期该 `libs` 文件夹内**一共应有 22 个文件，且全部为 `.dll` 后缀**。

❌ **后果**：缺失这些 DLL 时 ONNX Runtime 找不到 CUDA 环境，只能回退 CPU 推理。CPU 推理必然慢于实时，
会持续积压并触发断流。**v1.1 与 v1.0 的区别是它会把这件事告诉你**：
CUDA EP 注册失败会写进 `rvc_plugin.log` 并弹一条系统通知，而不是像 v1.0 那样静默回退。

---

## 📦 安装

1. 用 MicYou 安装 Release 里的插件包；
2. 把 `cuda_dll.zip` 内的 22 个 `.dll` 解压到插件目录的 `libs/`；
3. 重启 MicYou，在设置里启用 `opss.mambo-rvc-onnx`。

启用后插件会弹一条通知告诉你**实际加载了哪个模型**；同样的信息也会写进
`%APPDATA%\micyou\plugins\opss.mambo-rvc-onnx\rvc_plugin.log`，以及 MicYou 设置里的插件日志面板。

> ⚠️ **DSP 链位置**：manifest 里写了 `dsp.insertAfter: "VAD"`，但当前宿主是把它固定插在 **AEC 之后**
> （`PLUGIN_NODE_AFTER = "AEC"`，`insertAfter` 未被使用）。也就是说变声结果还会再经过宿主的
> 降噪 / EQ / AGC / VAD。如果发现气声被吃掉、尾音被切，请到「设置 → 音频 → 处理链」里
> 把 `Plugins` 节点拖到链尾（VAD 之后）。

---

## 🧬 用你自己的 RVC 模型

把导出的 `.onnx` 模型丢进插件目录的 **`user_models/`** 文件夹就行（首次运行会自动创建这个文件夹，
并在里面放一个说明文件）。搜索优先级：

| 优先级 | 位置 | 说明 |
| :--- | :--- | :--- |
| 1 | 设置里的 **`model_file`** | 相对插件目录的路径，例如 `user_models/我的音色.onnx`。填了就只用它 |
| 2 | **`user_models/*.onnx`** | 有多个时取**修改时间最新**的那个（刚丢进去的模型自动生效） |
| 3 | `models/user/*.onnx` | 备用位置 |
| 4 | `models/uma-Matikane_Tannhauser.onnx` | 插件自带模型（回退） |
| 5 | `models/*.onnx` | 兜底：排除 hubert/rmvpe 后的任意模型 |

补充说明：

- **改完 `model_file` 立即热重载**，不用重启 MicYou。重载期间会有 1~3 秒静音（要重新建 ONNX 会话）；
  如果解析出的模型路径其实没变，会自动跳过重载，不会白断一次流。
- `user_models/` 里文件名含 `hubert` / `contentvec` 的会被当成内容编码器，含 `rmvpe` / `f0` 的会被当成
  音高提取器 —— 所以你也可以只覆盖这两个，主模型继续用自带的。
- **多音色模型**用 `Speaker ID (sid)` 选音色；单音色模型保持 0。
- **非 48kHz 模型也能跑**：插件会按输出长度反推模型采样率（hop=400 → 40k、hop=320 → 32k）并自动重采样到 48kHz，
  同时在日志里提示。48k 模型（hop=480）走零开销直通路径。
- 输入名做了模糊匹配，兼容不同导出脚本（`phone` / `phone_lengths` / `p_len` / `nsff0` / `pitch` / `sid`）；
  名字全对不上时按标准 RVC 导出顺序回退。加载时日志会打印模型的实际输入签名，方便排查。
- 加载失败**不会**让插件变成哑巴：会记日志、弹通知，并每 15 秒自动重试一次（你补上缺失文件后无需重启）。

---

## 🎛️ 参数调节指南

### 延迟是怎么算出来的

```
端到端算法延迟 ≈ lookahead + crossfade + jitter + (0 ~ chunk)
                 └ 块尾最新的样本 ┘            └ 块粒度带来的抖动，平均取 chunk/2
每块推理的计算量 ≈ (left_context + chunk + lookahead) / chunk  倍实时
```

默认配置下就是 `150 + 20 + 40 + (0~100)` = **210~310ms**（平均 ~260ms）。
v1.0 的默认值是 `extra(400) + crossfade(50) + (0~100)` = **450~550ms**。

**关键**：`Left Context` 只影响音质与算力，**不增加延迟**（v1.0 的 `extra_ms` 同时充当左上下文和右 lookahead，
所以调低它会同时牺牲音质，这一版拆开了）。

### 各参数

| 参数 | 默认 | 作用 | 调参建议 |
| :--- | :--- | :--- | :--- |
| **Chunk Size (ms)** | 100 | 每次推理输出的音频块 | **稳定性主旋钮**，必须 > 单块推理耗时 τ。显卡弱就调大（200） |
| **Lookahead (ms)** | 150 | 块尾预读量（右上下文） | 延迟的主要来源，决定块边界的音高/咬字准确度 |
| **Left Context (ms)** | 300 | 块前历史量（左上下文） | 不增加延迟，但成比例增加算力。算力紧张就调小 |
| **Crossfade (ms)** | 20 | OLA 交叉淡化 | 掩盖块拼接毛刺。lookahead/left 很小时适当调大；0 = 硬拼接 |
| **Jitter Buffer (ms)** | 40 | 输出缓冲垫 | 吸收推理耗时抖动，减少断流；代价是等量固定延迟 |
| **Max Backlog (ms)** | 300 | 积压上限 | 超过就丢弃最旧音频把延迟拉回来（一次短断流）。宁可断流也不要无限延迟 |
| **Pitch Shift** | 0 | 变调半音数 | 男变女 +12，女变男 -12，同声线微调 ±2 |
| **Speaker ID** | 0 | 多音色模型的 sid | 单音色模型保持 0 |
| **Silence Gate** | 开 | 安静的块不跑推理 | 省算力，并让句间停顿自动排空积压 |
| **Gate Threshold (dBFS)** | -70 | 静音判定电平 | 环境吵调高（-55），气声被吃掉调低（-85） |

所有时长参数会被吸附到 **10ms 的整数倍**（= 480 样本 = RVC 的一帧），否则每块会有几毫秒的时间轴漂移并逐块累积。

### 按硬件挑预设

| 场景 | chunk | lookahead | left | jitter | 延迟 | 算力 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 低延迟（RTX 3060 及以上） | 60 | 100 | 200 | 20 | 140~200ms | 6.0× |
| **默认（均衡）** | 100 | 150 | 300 | 40 | 210~310ms | 5.5× |
| 音质优先（RTX 4070 及以上） | 100 | 200 | 600 | 40 | 260~360ms | 9.0× |
| 老显卡 / 核显 | 200 | 200 | 300 | 60 | 280~480ms | 3.5× |

> 宿主自己的输出缓冲（设置 → 音频 → `output_buffer_ms`，默认 300ms）会**叠加**在上面。
> 想要极致低延迟，把它也调小。

### 怎么知道 τ（单块推理耗时）

日志每 50 块打一行：

```
[Perf] 最近 50 块: τ 平均 43.2ms / 峰值 88.7ms（chunk=100ms，历史 560ms，输出环 41ms）
```

- **τ 平均 < chunk_ms** → 健康；
- **τ 接近或超过 chunk_ms** → 会持续积压，日志里开始出现 `[Flush] 积压 ... 超过上限`，听感是周期性短断流。
  此时要么调大 `Chunk Size`，要么调小 `Left Context` / `Lookahead`；
- 日志里出现 `[Perf] 警告: 单块推理 xxms > chunk xxms` 就是明确信号。

---

## 🔧 v1.1 相对 v1.0 的改动

### 修掉的 bug

| # | 问题 | v1.0 的表现 | v1.1 |
| :-- | :--- | :--- | :--- |
| 1 | **窗口锚在 history 尾部却从头部 drain** ⇒ 窗口末端 `base+len` 恒定 ⇒ 同一窗口被重复推理 k 次 | 同一块音频连播 k 遍、(k-1)·chunk 的真实语音被丢弃；k 随 τ/chunk 几何级增长。一次 900ms 的 GPU 抖动就能让同一段复读 9 遍 | 窗口锚头部 + 游标前进，每块只输出一次、严格按时间顺序（含回归测试） |
| 2 | 推理失败时 `continue` 跳过 `drain` ⇒ 循环条件永远成立 | 在同一个窗口上**无限重跑推理**：GPU 100%、输出停摆、只能重启 MicYou | 失败照常推进游标，输出静音并记日志 |
| 3 | HuBERT 特征为空时 `build_inputs` 索引越界 panic | panic 被 `catch_unwind` 吃掉 ⇒ **worker 线程永久退出**，此后插件只输出静音，日志只有一行 PANIC | 提前拦下并转成可恢复的错误；worker 也不会因为一次 panic 就死掉 |
| 4 | `history_buf` 无上限增长 | τ > chunk 时延迟与内存一起爆炸（10 分钟 ~58MB），且 `drain` 的 O(n) memmove 让 τ 更大 ⇒ 正反馈 | `max_latency_ms` 封顶；游标式推进让每批只 memmove 一次 |
| 5 | OLA 交叉淡化两端不在同一段时间上 | 拼接处混入错位音频（轻微咔哒） | 窗口锚点修正后 tail/head 天然对齐；flush、换模型、参数突变时主动复位 OLA |
| 6 | `max_frames = 4096`，超限 `bypass=1` | 5 个 20ms 包一起到就 bypass ⇒ **把未变声的原声直接送给对方** | 上限提到 48000 帧（1 秒），实际不可达 |
| 7 | 欠载时硬写 0 | 每个 gap 边界一次爆音 | 冻结末样本 + 2ms 淡出，恢复时 2ms 淡入 |
| 8 | `static mut STATE` | 编译期已报 `creating a mutable reference to mutable static`；Rust 2024 是硬错误 | 换成 `AtomicPtr` + 明确的 SAFETY 不变量说明 |
| 9 | 参数热更新只认 `config:changed` | 宿主派发用 `try_lock`，音频线程正忙时消息被**直接丢弃** ⇒ 拖了滑杆没反应 | 保留消息处理，另加 5 秒一次的定时兜底轮询 |
| 10 | 改 `chunk_ms` / `extra_ms` 不复位窗口状态 | 换滑杆当场一段乱码 | 窗口形状变化时重基 history 并复位 OLA |
| 11 | 静音门限看整个 900ms 窗口的 RMS | 「安静的上下文 + 响亮的 body」会被整块判为静音丢掉 | 只统计将要输出的 body 段；门限改成可配置的 dBFS |
| 12 | 运行期 `set_var("ORT_DYLIB_PATH")`、`SetDllDirectoryW` 静默失败、CUDA 静默回退 CPU | 排查困难 | 改用 `ort::init_from(绝对路径)`，失败返回原因；CUDA EP 加 `error_on_failure()` |

### 性能

- **图优化等级 `Level1` → `Level3`**：Level1 只做语义保持的基础重写，没有 extended/full 优化
  （含 attention、layout 融合），对 HuBERT/RMVPE 这类 transformer 会慢数倍。这是把 τ 压到 chunk 以下最有效的一刀；
- **RMVPE 的 mel 前端全部缓存**：Hann 窗、128×513 滤波器组、FFT plan、所有中间缓冲只建一次
  （v1.0 每块都重建 planner、重算 mel_basis、分配数 MB）；magnitude 从 `[freq][frame]` 转置布局改成
  `[frame][freq]` 连续布局，mel 矩阵乘由跨步访存变成顺序访存；
- **f0 后处理**不再每帧分配两个 `Vec`（90 帧/块 ⇒ 少 180 次分配）；中值滤波用 `total_cmp`，遇 NaN 不 panic；
- **默认窗口从 900ms 缩到 550ms**（left 300 + chunk 100 + lookahead 150），每块计算量降到 v1.0 的 61%，
  同时延迟从 ~500ms 降到 ~260ms。

### 文件结构（8 → 3）

```
src/lib.rs      宿主接口层：C ABI、生命周期、实时 process、参数、文件日志、状态上报
src/stream.rs   流式缓冲层：无锁 SPSC 环形缓冲 + 窗口调度器（OLA / 积压控制 / 统计 / 热重载编排）
src/rvc.rs      模型层：插件目录与 CUDA 运行库引导、模型发现（用户优先）、ORT 会话、
                       HuBERT / RMVPE(mel 前端) / RVC 合成、非 48k 模型重采样
```

v1.0 的 `config.rs` 与 `logger.rs` 并入 `lib.rs`，`ring_buffer.rs` 并入 `stream.rs`，
`models.rs` / `inference.rs` / `path_resolver.rs` 合并为 `rvc.rs`。

---

## 🛠️ 开发者说明

- **语言**：Rust（edition 2021）
- **推理引擎**：ONNX Runtime 2.0.0-rc.13（`ort`，CUDA EP，`load-dynamic`）
- **构建**：`cargo build --release`，产物 `target/release/mambo_rvc_onnx.dll`
  （`entry` 写的是无后缀基础名，宿主会按平台自动补 `.dll`/`.so`/`.dylib`；
  Linux/macOS 产物需去掉 `lib` 前缀才能跨平台共用一个 ZIP）
- **测试**：`cargo test` —— 8 个测试，其中 4 个是 v1.0 复读 bug 的回归测试
  （`legacy_tail_anchor_repeats_the_same_window` 会把 v1.0 的错误行为原样复现出来作为对照）
- **实时安全**：`process()` 里无堆分配、无锁、无 Host API 调用、无系统调用；
  worker 是自建子线程，按 `api-reference.md` 的规定**不调用任何 Host API**，
  需要给用户看的消息通过 `interval:tick` 在宿主线程里转发到 `host.log` / `host.notify`
- **不要**在 `Cargo.toml` 里设 `panic = "abort"`：本 cdylib 跑在宿主进程内，abort 会带走整个 MicYou

<div align="center"><p><em>Powered by Rust & ONNX Runtime. 让每一次“曼波”都如丝般顺滑。</em></p></div>
