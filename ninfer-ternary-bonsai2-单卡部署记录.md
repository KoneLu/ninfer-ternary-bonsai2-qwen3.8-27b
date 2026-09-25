# NINFER 三元 Bonsai 2 27B 单卡部署记录（58 / GPU6 / 2026-09-25）

## 一、结论

- **一张 RTX 4090D 跑通** Ternary Bonsai 2 27B（2.125 bit/权重），运行时权重 **7.12 GiB**。
- 服务：`ninfer-serve` 监听 `0.0.0.0:8011`，模型 id `qwen3.8-27b`。
- **上下文**：`--kv-capacity auto` + `--kv-dtype fp8` 解出 **387,008 token 的 KV 池**；
  `/v1/models` 报 `max_model_len = 262144`（模型原生上限），即**单卡吃满 256K**（对比：INT6 双卡才 131,072）。
- 实测单流：英文说明文 **133.8 tok/s**、中文说明文 **122.3 tok/s**、短问答 **147.4 tok/s**；
  prefill **460–488 tok/s**；MTP 接受率 49%–70%（短回答最高）。

## 二、硬件与环境

| 项 | 值 |
|---|---|
| 机器 | 192.168.1.*（Ubuntu 22.04.5，96 核，503 GB 内存） |
| GPU | RTX 4090 D |
| CUDA | 13.3.73（pip 旁装 shim；系统 nvcc 是 11.5，不能用） |
| 编译器 | GCC/G++ 11.4.0 |
| CMake | 3.22.1（系统自带；仓库要 3.28，见补丁） |
| Ninja | 1.10.1 |
| 引擎树 | `/data/ninfer-ternary-bonsai-ada`（CraneBW/ninfer-ternary-bonsai-ada，Linux Ada sm_89） |

## 三、制品怎么来的（三件套）

| 件 | 来源 | 大小 / sha256 |
|---|---|---|
| 模板（骨架+MTP+vision） | HF `neroued/Qwen3.8-27B-NInfer` **revision `18dfc887`**（2026-08-19，**容器 v2**） | 18,210,531,328 B；`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| 权重 GGUF | `prism-ml/Ternary-Bonsai-2-27B-gguf` → `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 B；`3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |
| 产物 | `pack.py build` | **8,306,927,628 B（7.74 GiB）**，1126 个对象 |

打包命令（必须在引擎树根 + `PYTHONPATH=. `，因为 `pack.py` 依赖引擎的 `tools.artifact`）：

```bash
cd /data/ninfer-ternary-bonsai-ada
export PYTHONPATH=/data/ninfer-ternary-bonsai-ada
PY=/home/miniconda3/envs/vllm_env/bin/python   # 有 numpy/safetensors/gguf
$PY tools/pack.py check --gguf artifacts/Ternary-Bonsai-2-27B-PQ2_0.gguf --template artifacts/qwen3_8_27b_v2.ninfer
$PY tools/pack.py build artifacts/ternary-bonsai-2-27b.ninfer --gguf artifacts/Ternary-Bonsai-2-27B-PQ2_0.gguf --template artifacts/qwen3_8_27b_v2.ninfer
```

校验要点（全部实测通过）：`zero_share=0.3278`（PQ2_0 理论零码占比）、CHECK 3 全 OK、
`text/hadamard_signs` 114,688 B、`text/hadamard_widths` 12 B。

## 四、为编译打的补丁（都在引擎树内，脚本留档）

| 文件 | 改动 | 原因 |
|---|---|---|
| `CMakeLists.txt` | `cmake_minimum_required` 3.28→3.22 | 系统 CMake 3.22 |
| `CMakeLists.txt` | 补 `CMAKE_CUDA20_STANDARD_COMPILE_OPTION/EXTENSION = "-std=c++20"` | CMake 3.22 不认识 CUDA C++20 dialect |
| `CMakeLists.txt` | FFmpeg/libcurl `REQUIRED`→`QUIET`；补 `PkgConfig::FFMPEG/LIBCURL` 与 `CUDA::nvtx3` 接口桩 | 纯文本构建，无 ffmpeg/curl 开发包 |
| `CMakeLists.txt` | `add_compile_definitions(CCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1)` | CCCL 对 CUDA compiler/toolkit 头版本的严格校验（同 INT6 那套 FlashInfer 的坑） |
| `src/CMakeLists.txt` | `NINFER_HAVE_FFMPEG` 改为只在 `FFMPEG_FOUND` 时定义 | 原逻辑只看 `NINFER_BUILD_MEDIA_ACQUIRE`，会导致无 ffmpeg 却去 include |
| `src/media/decode/decode.cpp` | no-ffmpeg 分支补 `inspect_image` / `inspect_video` 桩 | 否则链接期 undefined reference |

产物：`build/apps/ninfer`、`ninfer-serve`（277 MB）、`ninfer-perplexity`。

## 五、启动 / 停止

```bash
# 启动（脚本：/data/ninfer-ternary-bonsai-ada/start_ninfer.sh）
cd /data/ninfer-ternary-bonsai-ada
export PATH=/home/cuda13-home/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/cuda13-home/lib
CUDA_VISIBLE_DEVICES=6 ./build/apps/ninfer-serve artifacts/ternary-bonsai-2-27b.ninfer \
  --host 0.0.0.0 --port 8011 \
  --max-context 262144 --kv-capacity auto --kv-dtype fp8 \
  --max-concurrency 2 --spec mtp --draft-tokens 2

# 日志
tail -f /data/ninfer-ternary-bonsai-ada/ninfer_serve.log

# 停止 / 恢复 INT6
pkill -x ninfer-serve
systemctl --user start vllm-int6-27b-8011.service
```

启动日志关键行（实测）：

```
engine ready | qwen3.8-27b/groupwise-int | total 19.9s | weights 7.12 GiB
capacity | KV 387,008 tokens, fp8, auto | pages 6,047/8,192 | runtime 13.6 GiB | free 1.12 GiB
listening on http://0.0.0.0:8011 | model qwen3.8-27b | auth disabled
```

## 六、实测

| 场景 | decode | prefill | MTP 接受 |
|---|---:|---:|---:|
| 短问答（The capital of France is） | 147.4 tok/s | 221 tok/s | 14/20 = 70.0% |
| 英文说明文 600 tok | 133.8 tok/s | 488 tok/s | 296/606 |
| 中文说明文 363 tok | 122.3 tok/s | 461 tok/s | 163/400 |

