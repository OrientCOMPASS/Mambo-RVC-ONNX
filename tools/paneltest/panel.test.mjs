// panel.html 行为测试：jsdom 加载 + 模拟宿主 postMessage 桥
import { JSDOM } from 'jsdom';
import fs from 'fs';

const html = fs.readFileSync(new URL('../../panel.html', import.meta.url), 'utf-8');

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
let failures = 0;
function check(name, cond, extra = '') {
  if (cond) console.log(`  ✓ ${name}`);
  else { failures++; console.error(`  ✗ ${name} ${extra}`); }
}

const PKG_META = [
  ['nvidia-cuda-runtime-cu12', '12.9.79', 3591604, ['cudart64_12.dll']],
  ['nvidia-curand-cu12', '10.3.10.19', 68774847, ['curand64_10.dll']],
  ['nvidia-cufft-cu12', '11.4.1.4', 200067309, ['cufft64_11.dll', 'cufftw64_11.dll']],
  ['nvidia-cusolver-cu12', '11.7.5.82', 326215953, ['cusolver64_11.dll', 'cusolverMg64_11.dll']],
  ['nvidia-cusparse-cu12', '12.5.10.65', 362504719, ['cusparse64_12.dll']],
  ['nvidia-cublas-cu12', '12.9.2.10', 553162896, ['cublas64_12.dll', 'cublasLt64_12.dll']],
  ['nvidia-cudnn-cu12', '9.25.1.1', 732338891, ['cudnn64_9.dll', 'cudnn_adv64_9.dll', 'cudnn_cnn64_9.dll', 'cudnn_engines_precompiled64_9.dll', 'cudnn_engines_runtime_compiled64_9.dll', 'cudnn_engines_tensor_ir64_9.dll', 'cudnn_ext64_9.dll', 'cudnn_graph64_9.dll', 'cudnn_heuristic64_9.dll', 'cudnn_ops64_9.dll']],
];

function makeRtStatus({ present = 0, ortReady = true, phase = 'ready', cuda = false } = {}) {
  const files = PKG_META.flatMap(p => p[3]);
  const missing = files.slice(present);
  let acc = 0;
  return {
    v: 1, ts: Date.now(), plugin_version: '1.2.0', mirror: 'tuna',
    ort: { core: ortReady, cuda_provider: ortReady, shared: ortReady, ready: ortReady },
    cuda: { present, total: 19, missing, missing_bytes: missing.length * 100000000 },
    pkgs: PKG_META.map(([name, version, bytes, fl]) => {
      const have = Math.max(0, Math.min(fl.length, present - acc));
      acc += fl.length;
      return { name, version, bytes, have, files: fl };
    }),
    engine: { phase, model: 'RVC=uma (插件自带)', detail: cuda ? '' : 'CPU 推理（CUDA 运行库缺 19 个，可在面板拉取）', cuda, blocks: 42, flushes: 1 },
  };
}

function makeRtState(over = {}) {
  return {
    v: 1, state: 'idle', mirror: 'tuna', pkg_index: -1, pkg_count: 7, pkg: '', pkg_version: '',
    bytes_done: 0, bytes_total: 0, all_done: 0, all_total: 2246656219, speed_bps: 0, step: '', error: '',
    ts: Date.now(),
    pkgs: Array.from({ length: 7 }, (_, i) => ({ name: PKG_META[i][0], version: PKG_META[i][1], bytes: PKG_META[i][2], dlls: PKG_META[i][3].length, have: 0, state: 'wait' })),
    ...over,
  };
}

async function boot({ locale = 'zh-CN', present = 0, ortReady = true, phase = 'ready', cuda = false } = {}) {
  const store = {
    speaker_id: 0, f0_up_key: 0, chunk_ms: 200, lookahead_ms: 80, left_context_ms: 720,
    crossfade_ms: 50, jitter_ms: 0, max_latency_ms: 300, gate_enabled: true, gate_db: -80,
    ort_opt_level: 'all',
    rt_status: makeRtStatus({ present, ortReady, phase, cuda }),
    rt_state: makeRtState(),
  };
  const writes = [];
  const triggers = [];

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'http://localhost/',
    beforeParse(window) {
      window.Element.prototype.scrollIntoView = function () {};
      window.addEventListener('message', (e) => {
        const d = e.data;
        if (!d || d.__micyou !== 1 || typeof d.api !== 'string' || d.id === undefined) return;
        if (typeof d.id === 'string' && d.id.startsWith('console-')) return;
        let value = null, error = null;
        try {
          switch (d.api) {
            case 'locale': value = locale; break;
            case 'get_config': value = JSON.parse(JSON.stringify(store)); break;
            case 'set_config': store[d.args.key] = d.args.value; writes.push([d.args.key, d.args.value]); break;
            case 'trigger': {
              const action = d.args.action;
              const payload = d.args.payload ? JSON.parse(d.args.payload) : null;
              triggers.push({ action, payload });
              if (action === 'rt_status') store.rt_status = makeRtStatus({ present, ortReady, phase, cuda });
              break;
            }
            default: error = 'unknown api ' + d.api;
          }
        } catch (err) { error = String(err); }
        const resp = error ? { __micyou: 1, id: d.id, ok: false, error } : { __micyou: 1, id: d.id, ok: true, value };
        window.dispatchEvent(new window.MessageEvent('message', { data: resp }));
      });
    },
  });
  return { dom, window: dom.window, store, writes, triggers };
}

function fire(el, type) { el.dispatchEvent(new el.ownerDocument.defaultView.Event(type, { bubbles: true })); }
function click(el) { el.dispatchEvent(new el.ownerDocument.defaultView.MouseEvent('click', { bubbles: true })); }

// ═══════════════ 场景 1：缺库初始渲染 ═══════════════
console.log('场景 1：缺库初始渲染');
{
  const { window, triggers } = await boot({ present: 0 });
  await sleep(400);
  const doc = window.document;
  check('标题中文', doc.getElementById('t-title').textContent === '曼波 RVC 变声器');
  check('版本号显示 v1.2.0', doc.getElementById('ver').textContent === 'v1.2.0', doc.getElementById('ver').textContent);
  check('引擎 pill = CPU', doc.getElementById('enginePillText').textContent === 'CPU', doc.getElementById('enginePillText').textContent);
  check('EP chip = CPU', doc.getElementById('vEp').innerHTML.includes('CPU'));
  check('ORT 完整', doc.getElementById('vOrt').textContent.includes('完整'));
  check('CUDA 缺失提示', doc.getElementById('vCuda').textContent.includes('缺 19 个'));
  check('黄色横幅显示', doc.getElementById('alert').className.includes('show'));
  check('横幅 CTA = 去拉取', doc.getElementById('alertBtn').textContent === '去拉取');
  check('拉取卡片自动展开', !doc.getElementById('cardFetch').className.includes('closed'));
  check('包表 7 行', doc.querySelectorAll('#pkgTable .pkg').length === 7);
  check('镜像 5 个', doc.querySelectorAll('#mirrors button').length === 5);
  check('默认镜像 = 清华', doc.querySelector('#mirrors button.active').textContent.includes('清华'));
  check('推理窗口达标 1000ms', doc.getElementById('readout').innerHTML.includes('1000ms') && doc.getElementById('readout').innerHTML.includes('tag-ok'), doc.getElementById('readout').textContent);
  check('滑杆数量 = 9', doc.querySelectorAll('input[type=range]').length === 9);
  check('chunk 滑杆范围 20-500', doc.getElementById('in-chunk_ms').min === '20' && doc.getElementById('in-chunk_ms').max === '500');
  check('已触发 rt_status', triggers.some(t => t.action === 'rt_status'));
  check('fetch 按钮可点', !doc.getElementById('btnFetch').disabled && doc.getElementById('btnFetch').style.display !== 'none');
  check('包行 tooltip 列出 dll', doc.getElementById('pkg-6') === null || doc.querySelectorAll('#pkgTable .pkg')[6].title.includes('cudnn'));
  window.close();
}

// ═══════════════ 场景 2：参数交互 ═══════════════
console.log('场景 2：参数交互');
{
  const { window, store, writes } = await boot({ present: 0 });
  await sleep(300);
  const doc = window.document;

  const chunk = doc.getElementById('in-chunk_ms');
  chunk.value = '300';
  fire(chunk, 'input');
  check('滑杆气泡即时更新', doc.getElementById('vv-chunk_ms').textContent === '300 ms', doc.getElementById('vv-chunk_ms').textContent);
  check('窗口读数更新 1100ms', doc.getElementById('readout').innerHTML.includes('1100ms'));
  fire(chunk, 'change');
  await sleep(80);
  check('change 写入 chunk_ms=300', store.chunk_ms === 300 && writes.some(w => w[0] === 'chunk_ms' && w[1] === 300));

  const left = doc.getElementById('in-left_context_ms');
  left.value = '200'; fire(left, 'input');
  check('窗口不足警示', doc.getElementById('readout').innerHTML.includes('tag-warn'));
  left.value = '720'; fire(left, 'input');

  click(doc.getElementById('sw-gate_enabled'));
  await sleep(80);
  check('gate 关闭写入 false', store.gate_enabled === false);
  check('gate_db 随之禁用', doc.getElementById('in-gate_db').disabled);
  click(doc.getElementById('sw-gate_enabled'));
  await sleep(80);
  check('gate 恢复 true', store.gate_enabled === true && !doc.getElementById('in-gate_db').disabled);

  const pre = [...doc.querySelectorAll('.presets button')].find(b => b.dataset.v === '-12');
  click(pre);
  await sleep(80);
  check('预设写入 f0=-12', store.f0_up_key === -12);
  check('预设按钮高亮', pre.className.includes('active'));
  check('滑杆联动 -12', doc.getElementById('in-f0_up_key').value === '-12');

  const sel = doc.getElementById('selOrtOpt');
  sel.value = 'basic'; fire(sel, 'change');
  await sleep(80);
  check('ort_opt_level=basic 写入', store.ort_opt_level === 'basic');

  const aliyun = [...doc.querySelectorAll('#mirrors button')].find(b => b.dataset.id === 'aliyun');
  click(aliyun);
  await sleep(80);
  check('rt_mirror 持久化 aliyun', store.rt_mirror === 'aliyun');

  click(doc.getElementById('btnReset'));
  await sleep(120);
  check('重置后 chunk=200', store.chunk_ms === 200);
  check('重置后 f0=0', store.f0_up_key === 0);
  check('重置后 ort=all', store.ort_opt_level === 'all');

  click(doc.querySelector('#cardParams .card-head'));
  check('参数卡片可折叠', doc.getElementById('cardParams').className.includes('closed'));
  click(doc.querySelector('#cardParams .card-head'));
  check('再点展开', !doc.getElementById('cardParams').className.includes('closed'));

  click(doc.getElementById('btnReload'));
  await sleep(60);
  check('无桥错误', doc.getElementById('bridgeErr').style.display !== 'block');
  window.close();
}

// ═══════════════ 场景 3：下载进度渲染 ═══════════════
console.log('场景 3：下载进度');
{
  const { window, store, triggers } = await boot({ present: 0 });
  await sleep(300);
  const doc = window.document;

  const tuna = [...doc.querySelectorAll('#mirrors button')].find(b => b.dataset.id === 'tuna');
  click(tuna);
  click(doc.getElementById('btnFetch'));
  await sleep(60);
  const t = triggers.find(t => t.action === 'rt_fetch');
  check('触发 rt_fetch 带 mirror', t && t.payload && t.payload.mirror === 'tuna', JSON.stringify(triggers));

  const done2 = 3591604 + 68774847;
  store.rt_state = makeRtState({
    state: 'downloading', mirror: 'tuna', pkg_index: 2, pkg: 'nvidia-cufft-cu12', pkg_version: '11.4.1.4',
    bytes_done: 100033855, bytes_total: 200067309, all_done: done2 + 100033855, speed_bps: 20971520,
    step: '下载 nvidia-cufft-cu12（3/7）· 清华 TUNA',
    pkgs: [
      { name: 'nvidia-cuda-runtime-cu12', version: '12.9.79', bytes: 3591604, dlls: 1, have: 1, state: 'done' },
      { name: 'nvidia-curand-cu12', version: '10.3.10.19', bytes: 68774847, dlls: 1, have: 1, state: 'done' },
      { name: 'nvidia-cufft-cu12', version: '11.4.1.4', bytes: 200067309, dlls: 2, have: 0, state: 'downloading' },
      { name: 'nvidia-cusolver-cu12', version: '11.7.5.82', bytes: 326215953, dlls: 2, have: 0, state: 'wait' },
      { name: 'nvidia-cusparse-cu12', version: '12.5.10.65', bytes: 362504719, dlls: 1, have: 0, state: 'wait' },
      { name: 'nvidia-cublas-cu12', version: '12.9.2.10', bytes: 553162896, dlls: 2, have: 0, state: 'wait' },
      { name: 'nvidia-cudnn-cu12', version: '9.25.1.1', bytes: 732338891, dlls: 10, have: 0, state: 'wait' },
    ],
  });
  await sleep(750);
  check('进度区显示', doc.getElementById('progArea').className.includes('show'));
  const allPct = (store.rt_state.all_done / store.rt_state.all_total) * 100;
  check('总进度条百分比', doc.getElementById('progPct').textContent === allPct.toFixed(0) + '%', doc.getElementById('progPct').textContent + ' want ' + allPct.toFixed(0));
  check('总进度条宽度', Math.abs(parseFloat(doc.getElementById('progBarAll').style.width) - allPct) < 0.2, doc.getElementById('progBarAll').style.width);
  check('当前包进度条 ~50%', Math.abs(parseFloat(doc.getElementById('progBarPkg').style.width) - 50.0) < 0.5, doc.getElementById('progBarPkg').style.width);
  check('速度显示 20.0MB/s', doc.getElementById('progSpeed').textContent === '20.0MB/s', doc.getElementById('progSpeed').textContent);
  check('ETA 有值', /^[0-9]+:[0-9]{2}$/.test(doc.getElementById('progEta').textContent), doc.getElementById('progEta').textContent);
  check('步骤文案', doc.getElementById('progStep').textContent.includes('清华'));
  check('包标题 3/7', doc.getElementById('progTitle').textContent.includes('3/7') && doc.getElementById('progTitle').textContent.includes('cufft'));
  check('当前包行高亮', doc.getElementById('pkg-2').className.includes('active'));
  check('完成包行绿勾', doc.getElementById('pkg-0').innerHTML.includes('✓'));
  check('pill = 拉取中', doc.getElementById('fetchPillText').textContent === '拉取中');
  check('取消按钮出现', doc.getElementById('btnCancel').style.display !== 'none');
  check('横幅隐藏（busy）', !doc.getElementById('alert').className.includes('show'));

  store.rt_state = makeRtState({ state: 'error', error: '所有镜像均失败。最后错误: 清华 TUNA: timeout', all_done: done2 });
  await sleep(700);
  check('错误行显示', doc.getElementById('fetchErr').className.includes('show') && doc.getElementById('fetchErr').textContent.includes('镜像'));
  check('错误态按钮变重试', doc.getElementById('btnFetch').textContent === '重试');
  check('错误 pill', doc.getElementById('fetchPill').className.includes('err'));

  store.rt_state = makeRtState({ state: 'done', all_done: 2246656219, step: '完成：19 个运行库已就位' });
  store.rt_status = makeRtStatus({ present: 19, phase: 'ready', cuda: true });
  await sleep(2900); // error 态后进入 2.5s idle 轮询
  check('完成提示显示', doc.getElementById('fetchOk').className.includes('show'));
  check('完成后 fetch 按钮隐藏', doc.getElementById('btnFetch').style.display === 'none');
  check('横幅消失', !doc.getElementById('alert').className.includes('show'));
  check('pill = 全部就绪', doc.getElementById('fetchPillText').textContent === '全部就绪' && doc.getElementById('fetchPill').className.includes('ok'));
  check('引擎切 CUDA', doc.getElementById('enginePillText').textContent === 'CUDA（GPU）', doc.getElementById('enginePillText').textContent);
  check('EP chip = CUDA', doc.getElementById('vEp').innerHTML.includes('gpu'));
  check('CUDA 行 = 已就绪', doc.getElementById('vCuda').textContent.includes('已就绪'));
  window.close();
}

// ═══════════════ 场景 4：ORT 缺失（安装包损坏） ═══════════════
console.log('场景 4：ORT 缺失');
{
  const { window } = await boot({ present: 0, ortReady: false });
  await sleep(350);
  const doc = window.document;
  check('红色横幅', doc.getElementById('alert').className.includes('crit') && doc.getElementById('alert').className.includes('show'));
  check('fetch 禁用', doc.getElementById('btnFetch').disabled);
  check('ORT 行报错', doc.getElementById('vOrt').textContent.includes('重新安装'));
  window.close();
}

// ═══════════════ 场景 5：英文 locale ═══════════════
console.log('场景 5：英文界面');
{
  const { window } = await boot({ locale: 'en', present: 19, phase: 'ready', cuda: true });
  await sleep(350);
  const doc = window.document;
  check('英文标题', doc.getElementById('t-title').textContent === 'Mambo RVC Voice Changer');
  check('英文卡片', doc.getElementById('t-fetch').textContent === 'CUDA Runtime');
  check('全就绪无横幅', !doc.getElementById('alert').className.includes('show'));
  check('拉取卡片默认收起', doc.getElementById('cardFetch').className.includes('closed'));
  check('pill ok', doc.getElementById('fetchPill').className.includes('ok'));
  check('英文 foot', doc.getElementById('foot').textContent.includes('README'));
  window.close();
}

// ═══════════════ 场景 6：中途打开面板（恢复轮询） ═══════════════
console.log('场景 6：中途打开面板');
{
  const { window, store } = await boot({ present: 1 });
  store.rt_state = makeRtState({ state: 'extracting', pkg_index: 0, pkg: 'nvidia-cuda-runtime-cu12', bytes_done: 3591604, bytes_total: 3591604, all_done: 3591604, step: '解压 cudart64_12.dll' });
  await sleep(700);
  const doc = window.document;
  check('进度区恢复显示', doc.getElementById('progArea').className.includes('show'));
  check('pill = 拉取中', doc.getElementById('fetchPillText').textContent === '拉取中');
  check('取消按钮出现', doc.getElementById('btnCancel').style.display !== 'none');
  window.close();
}

console.log(failures === 0 ? '\n== 全部通过 ==' : `\n== ${failures} 个失败 ==`);
process.exit(failures === 0 ? 0 : 1);
