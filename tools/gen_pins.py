#!/usr/bin/env python3
"""生成 src/fetch.rs 里的 PyPI wheel pin 表。

用法:
    python3 tools/gen_pins.py            # 校验 PINNED 里的版本并打印 Rust 常量
    python3 tools/gen_pins.py --latest   # 探测各包最新兼容版本并打印（人工确认后回填 PINNED）

原理:
  1. 查询 PyPI JSON API 拿到 win_amd64 wheel 的 /packages/ 路径、大小、sha256；
  2. 用 HTTP Range 请求只读取 wheel(zip) 的中央目录，列出内部 dll 与精确大小（不下载整包）；
  3. 断言 7 个包解出的 dll 并集 == 期望的 19 个文件，输出可直接粘贴进 fetch.rs 的常量表。

镜像说明: 各大镜像（清华/阿里/腾讯/中科大）都以相同 /packages/<路径> 布局镜像 PyPI 文件，
所以表里只存路径，运行期拼接镜像前缀即可。
"""
import json
import re
import struct
import sys
import urllib.request

UA = {"User-Agent": "mambo-rvc-gen-pins"}

# 期望落盘到 libs/ 的 19 个运行库（与 README / panel 保持一致）
EXPECTED_DLLS = {
    "cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll",
    "cudnn64_9.dll", "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll", "cudnn_ext64_9.dll", "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll", "cudnn_ops64_9.dll",
    "cufft64_11.dll", "cufftw64_11.dll", "curand64_10.dll",
    "cusolver64_11.dll", "cusolverMg64_11.dll", "cusparse64_12.dll",
}

# 固定版本 = 经过实测的组合（CUDA 12.9 + cuDNN 9.25，与 ORT cuda12 构建匹配）
PINNED = {
    "nvidia-cuda-runtime-cu12": "12.9.79",
    "nvidia-curand-cu12": "10.3.10.19",
    "nvidia-cufft-cu12": "11.4.1.4",
    "nvidia-cusparse-cu12": "12.5.10.65",
    "nvidia-cusolver-cu12": "11.7.5.82",
    "nvidia-cublas-cu12": "12.9.2.10",
    "nvidia-cudnn-cu12": "9.25.1.1",
}

# 各大版本上限（--latest 模式用）：ORT cuda12 构建要求 CUDA 12.x + cuDNN 9.x
LATEST_MAJOR = {
    "nvidia-cublas-cu12": 12,
    "nvidia-cuda-runtime-cu12": 12,
    "nvidia-cudnn-cu12": 9,
    "nvidia-cufft-cu12": 11,
    "nvidia-curand-cu12": 10,
    "nvidia-cusolver-cu12": 11,
    "nvidia-cusparse-cu12": 12,
}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=60))


def range_get(url, start, end):
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req, timeout=60).read()


def zip_entries(url):
    """通过 Range 请求读取远端 zip 的中央目录，返回 [(name, usize)]。"""
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
    tail_len = min(size, 262144)
    tail = range_get(url, size - tail_len, size - 1)
    base = size - tail_len
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("EOCD not found")
    n, cd_size, cd_off = struct.unpack("<HII", tail[i + 10 : i + 20])
    if cd_off == 0xFFFFFFFF:  # zip64
        j = tail.rfind(b"PK\x06\x06")
        n = struct.unpack("<Q", tail[j + 32 : j + 40])[0]
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40 : j + 56])
    cd = tail[cd_off - base : cd_off - base + cd_size] if cd_off >= base else range_get(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p + 46 <= len(cd) and cd[p : p + 4] == b"PK\x01\x02":
        fields = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p : p + 46])
        csize, usize, nlen, elen, clen = fields[8], fields[9], fields[10], fields[11], fields[12]
        name = cd[p + 46 : p + 46 + nlen].decode("utf-8", "replace")
        out.append((name, usize))
        p += 46 + nlen + elen + clen
    if len(out) != n:
        print(f"  warn: parsed {len(out)} entries, EOCD says {n}", file=sys.stderr)
    return out


def vkey(v):
    return [int(x) if x.isdigit() else 0 for x in re.split(r"[.\-+]", v)]


def resolve(pkg, ver=None):
    """返回 (version, filename, url, size, sha256, path, dlls)。"""
    d = get_json(f"https://pypi.org/pypi/{pkg}/json")
    if ver is None:
        cands = [
            v for v in d["releases"]
            if not re.search(r"(dev|rc|a|b)\d", v)
            and vkey(v)[0] == LATEST_MAJOR[pkg]
            and any(f["filename"].endswith("win_amd64.whl") for f in d["releases"][v])
        ]
        ver = max(cands, key=vkey)
    files = [f for f in d["releases"][ver] if f["filename"].endswith("win_amd64.whl")]
    if not files:
        raise RuntimeError(f"{pkg}=={ver}: no win_amd64 wheel")
    f = files[0]
    url = f["url"]
    path = url.split("/packages/", 1)[1]
    dlls = [
        (n.rsplit("/", 1)[-1], u)
        for (n, u) in zip_entries(url)
        if n.endswith(".dll") and "/bin/" in n and n.rsplit("/", 1)[-1] in EXPECTED_DLLS
    ]
    dlls.sort()
    return ver, f["filename"], url, f["size"], f["digests"]["sha256"], path, dlls


def main():
    latest = "--latest" in sys.argv
    pkgs = []
    seen = set()
    for pkg in PINNED:
        ver = None if latest else PINNED[pkg]
        v, fn, url, size, sha, path, dlls = resolve(pkg, ver)
        print(f"# {pkg}=={v}  wheel={size:,}B  dlls={[d[0] for d in dlls]}", file=sys.stderr)
        for n, _ in dlls:
            if n in seen:
                raise RuntimeError(f"duplicate dll {n} in {pkg}")
            seen.add(n)
        pkgs.append((pkg, v, sha, path, size, dlls, fn))
    missing = EXPECTED_DLLS - seen
    if missing:
        raise RuntimeError(f"missing dlls: {sorted(missing)}")

    # 小包在前：先易后难，用户能尽快看到进展
    pkgs.sort(key=lambda p: p[4])

    total = sum(p[4] for p in pkgs)
    print(f"// 由 tools/gen_pins.py 生成 —— 共 {len(pkgs)} 包 / {len(seen)} dll / 下载量 {total/2**30:.2f} GiB")
    print("pub const PKGS: &[PkgSpec] = &[")
    for pkg, v, sha, path, size, dlls, fn in pkgs:
        print("    PkgSpec {")
        print(f'        pkg: "{pkg}",')
        print(f'        version: "{v}",')
        print(f'        wheel_file: "{fn}",')
        print(f'        sha256: "{sha}",')
        print(f'        path: "{path}",')
        print(f"        wheel_size: {size},")
        print("        dlls: &[")
        for n, u in dlls:
            print(f'            DllSpec {{ name: "{n}", size: {u} }},')
        print("        ],")
        print("    },")
    print("];")


if __name__ == "__main__":
    main()
