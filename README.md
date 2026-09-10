<div align="center"><p><em>“曼波~曼波~哦马自立曼波~🎵”</em></p></div>

# 🎤 Mambo RVC ONNX - MicYou 实时 AI 变声插件

本插件专为 [MicYou](https://github.com/LanRhyme/MicYou) 开发，致力于在本地实时流式 AI 变声。

---

## 🚨 极其重要的安装警告 (CUDA DLL)

本插件强依赖 GPU 进行实时推理。**请务必下载 Release 页面中的 `cuda_dll.zip`**，并将其中的 CUDA 运行库解压至插件目录下的 `libs` 文件夹内：

📂 **目标路径**：`%APPDATA%\micyou\plugins\opss.mambo-rvc-onnx\libs\`

⚠️ **验证标准**：预期该 `libs` 文件夹内**一共应有 22 个文件，且全部为 `.dll` 后缀**。
❌ **严重后果**：如果缺失这些 DLL，ONNX Runtime 将无法找到 CUDA 环境，**强制回退到 CPU 推理**。CPU 推理速度极慢，会造成严重的变声卡顿、延迟爆炸以及音频断流！

---

## 📦 安装指南

### 1. 安装插件
使用 MicYou 安装 Release 中的插件包

### 2. 注入 CUDA 运行库 (关键！)
将 `cuda_dll.zip` 内的所有 `.dll` 文件解压到 `libs` 目录，确保 `libs` 下有 **22 个 dll 文件**。

### 3. 启用插件
重启 MicYou，在设置中启用 `opss.mambo-rvc-onnx` 插件，即可开始变声！

---

## 🎛️ 参数调节指南 (延迟 vs 音质)

本插件将核心 DSP 参数暴露给了 MicYou 的 Config UI，你可以根据自己的显卡性能和需求实时寻找最佳平衡点：

| 参数名 | 作用说明 | 调参建议 |
| :--- | :--- | :--- |
| **Chunk Size (ms)** | 每次推理输出的音频块大小。 | 越小延迟越低，但 GPU 调用频率翻倍。推荐 `100 ~ 200`。 |
| **Extra Context (ms)** | 前后保护带（上下文窗口）。 | **延迟的主要来源**。决定 RMVPE 预测 F0 的准确度。越小延迟越低，但边缘可能出现音高瑕疵。推荐 `200 ~ 400`。 |
| **Crossfade (ms)** | OLA 交叉淡化长度。 | 用于掩盖音频块拼接处的边界毛刺。**Extra 越小，此值应适当调大**以抹平瑕疵。推荐 `20 ~ 50`。 |
| **Pitch Shift** | 变调半音数。 | 男变女推荐 `12`，女变男推荐 `-12`，同声线微调 `±2`。 |


---

## 🛠️ 开发者说明

- **语言**：Rust
- **推理引擎**：ONNX Runtime (CUDA EP)
- **构建**：`cargo build --release` (确保 `Cargo.toml` 中配置为 `cdylib` 动态链接库)

<div align="center"><p><em>Powered by Rust & ONNX Runtime. 让每一次“曼波”都如丝般顺滑。</em></p></div>