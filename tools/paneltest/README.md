# panel.html 行为测试（jsdom）

不需要启动 MicYou：用 jsdom 加载 panel.html，并模拟宿主的 postMessage 桥
（get_config / set_config / trigger / locale），验证渲染、交互、轮询与进度展示。

```bash
cd tools/paneltest
npm init -y && npm i jsdom     # 首次
node panel.test.mjs            # 全部通过时退出码 0
```

覆盖 6 个场景：缺库初始渲染（横幅/自动展开/包表）、参数交互（滑杆/开关/预设/下拉/
镜像选择/重置/折叠）、下载进度渲染（总进度/单包进度/速度/ETA/错误态/完成态）、
ORT 缺失、英文 locale、中途打开面板恢复轮询。
