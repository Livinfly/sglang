# SGLang FastH3 portable benchmark

All scripts live in this checkout and require branch `exp-fixed-version`. The
native inference source is unchanged. `sglang_bench` uses SGLang's supported
plugin registry for V1 admission and measurement hooks. No attention, offload,
FSDP, scheduler or VAE implementation is replaced. The installed plugin is inert
unless the launcher explicitly selects `SGLANG_PLUGINS=sglang_bench`; ordinary
native commands do not acquire measurement hooks.

## Environment and weights

```bash
uv --version
bash benchmarks/setup.sh
# Optional: download hash-verified standalone ffmpeg/ffprobe.
.venv/bin/python benchmarks/prepare_tools.py --media
# Optional, only when the installed driver needs CUDA 13 forward compatibility:
.venv/bin/python benchmarks/prepare_tools.py --cuda-compat
export BENCH_CUDA_COMPAT="$PWD/.cache/benchmark/tools/cuda-compat/usr/local/cuda-13.0/compat"
```

Setup strictly synchronizes the dedicated venv to the dependency lock, removing
unlisted packages from an existing environment before installing this checkout.
The locked environment uses Python 3.12 / Torch 2.14.1 / CUDA 13.0. No GPU model
is hardcoded; the native VSA-H3 backend validates device support. `UV`,
`BENCH_VENV`, `PYTHON_VERSION`, and the standard uv cache/index settings can be
set by the caller. A Rust toolchain may be needed to build locked dependencies
when no platform wheel is available; `UV_FIND_LINKS` can reuse a wheel cache.
Unused SGLang Rust extensions are excluded from this diffusion-only build.
`setup.sh --check` checks an existing environment. A compatible
system driver needs no `BENCH_CUDA_COMPAT`. System ffmpeg can also be used.

Pass complete **SGLang native exports**, not FastVideo snapshots. For official
FastVideo weights, prepare a native export once (CPU conversion, no generation):

```bash
.venv/bin/python benchmarks/prepare_models.py \
  --config benchmarks/configs/fasth3_v1.json \
  --source /your/official-v1-snapshot --output /your/native-v1
.venv/bin/python benchmarks/prepare_models.py \
  --config benchmarks/configs/fasth3_v2.json \
  --source /your/official-v2-snapshot --output /your/native-v2
```

The converter downloads three small pinned overlay files, verifies their SHA256,
adds the derived RoPE tensor and converts the video VAE layout. DiT/TE weights
are linked without changing the originals. Use `--copy-weights` for a standalone
movable export, or `--overlay-path` for an offline overlay copy. Arbitrary model
directory names work via the explicit model ID. Launch checks the native DMD
ladder/shifts, all 50 compression gates, component/index files and shard presence;
config hashes and weight sizes are saved (not a full weight-content hash).

## Configurations and launch

Edit `configs/fasth3_v1.json` / `fasth3_v2.json` or pass additional named JSON
profiles with `--config`. Model names separate outputs and speedup baselines.
Prompt/seed/target/shape, sparsity and documented server overrides are editable.
`target` controls native shape; the expected width/height/frame count must match.
FastH3 V1/V2 require 4/8 forwards respectively; arbitrary step counts are rejected.

| Setting | V1 | V2 |
|---|---|---|
| Model | FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree | FastVideo/FastVideo-FastH3-8-Step-V2 |
| SGLang steps / actual DiT forwards | 4 / 4 | 8 / 8 |
| DMD ladder | 999,749,500,250 | 999,874,749,624,500,375,250,125 |
| Video / audio shift | 12 / 3 | 10 / 3 |
| VSA sparsity / tile | 0.9 / 4×4×4 | 0.8 / 4×4×4 |

Shared: t2va with generated audio, 1344×768×124 frames, 24 FPS, seed 1000,
CFG 1, BF16 DiT attention/MLP with native FP32 projections retained, café prompt, empty negative prompt, `quality=lossless`.
SP/Ulysses = GPU count; TP=1, ring=1, CFG parallel off. All other execution
policies retain native defaults, including placement/folding, conditioning
cache and lossless fusions. In particular **offload is not forced off**.
Actual hardware-dependent ServerArgs and per-rank settings are recorded.

Preview native registry, request shape, and all resolved ServerArgs without
loading model weights (queries available CUDA hardware):

```bash
.venv/bin/python benchmarks/preview.py --config benchmarks/configs/fasth3_v1.json \
  --model-path /your/native-v1 --gpus 1 2 4 8 --output preview-v1.json
.venv/bin/python benchmarks/preview.py --config benchmarks/configs/fasth3_v2.json \
  --model-path /your/native-v2 --gpus 1 2 4 8 --output preview-v2.json
```

Review these configurations before starting measurements:

```bash
.venv/bin/python benchmarks/run.py \
  --model-path fasth3_v1=/your/native-v1 --model-path fasth3_v2=/your/native-v2 \
  --gpus 8 4 2 1 --output outputs/benchmarks/fasth3-native --background
.venv/bin/python benchmarks/monitor.py outputs/benchmarks/fasth3-native --watch
```

Defaults: `--modes dit --warmup 1 --repeats 1` (`--gpus 1` if omitted). `dit`
means E2E **plus** DiT. `--modes dit e2e` runs both, each with its own warmup.
`--prompt`, `--prompt-file`, `--validate-video` (extra output hash), `--wait-for
/path/status.json`, and `--baseline /path/summary.json` are optional. Historical
comparison runs only when explicitly requested. `--dry-run` prints the profile
and commands without requiring weights or a GPU; use `preview.py` for validation.

## Timing boundaries

- **response_e2e_s**: client `DiffGenerator.generate(sample)` entry to return of
  saved media. Includes request preparation/IPC, text encoding/cache lookup,
  noise, all DiT/scheduler/communication, VAE, video/audio encoding and save.
  Loading, queueing, ARM setup, validation and statistics are outside E2E.
- **DiT**: complete transformer root forwards, including existing root hooks and
  SP/FSDP. CUDA event pairs are preallocated/primed during load. Root pre-hook
  records before existing hooks and root post-hook after them; no per-step or
  added pipeline-end synchronization. Events are queried only after client E2E.
  Report `max_rank(sum_steps)` plus `sum_steps(max_rank)`; keep every rank/step.
- **distributed_e2e_s**: same client start to each rank's observer-thread Gloo
  barrier return. Native worker completion (including rank 0 save) wakes these
  threads. A separate CPU process group avoids scheduler collective races and
  does not block the native response. This can end before client response;
  includes thread scheduling/CPU barrier overhead and is not interchangeable
  with the other frameworks' main-thread NCCL barrier metric. Raw records also
  retain `worker_finished_s`. No client-to-worker clock conversion is needed:
  this is a single-host harness using the shared monotonic clock.
- In `e2e` mode no DiT hooks are attached; in an e2e-only suite no CUDA timing
  events are allocated. Both modes have the same observer protocol. Native
  stage timing logs are disabled. There is no model-output tensor retention.

The timing locations are marked directly in `worker.py`, `timing.py`, and
`sglang_bench/observer.py`. Native `async_output_save` remains false so the
worker completion boundary includes saving. Changing it is rejected.

## Outputs, GPU admission, failures

The suite snapshots scripts, plan, config/weight manifest, source commit/diff,
per-rank effective configs, all raw rank measurements, videos and logs.
`report.md` / `summary.json` include inference settings, formal/warmup E2E and
DiT, and per-model/mode single-GPU speedups. A single formal sample has no
estimated variance. Output shape and finite decoded audio are checked outside
timing; `--validate-video` also computes the video SHA256.

Idle checks and same-user UUID file locks in `/tmp/video-benchmark-gpus-UID`
coordinate with the other framework launchers. Busy GPUs cause waiting with
reasons in `status.json`; no GPU polling during inference. Locks only coordinate
cooperating scripts. GPU mask order is checked against actual CUDA UUIDs.

`status.json` records current/completed groups, `has_failures`, `failure_count`,
`last_error`, phase and log/error-file paths. Ordinary failures stop the suite;
only explicit loading/warmup OOM may be recorded and skipped. No automatic
resolution/offload changes or retries. Existing suite directories are never
overwritten. Restart manually into a new output directory. The monitor also
reports a dead launcher; SIGKILL cannot itself write a final status.

```bash
.venv/bin/python -m unittest discover -s benchmarks -p 'test_*.py'
.venv/bin/python benchmarks/report.py outputs/benchmarks/fasth3-native
```

## Sparse attention runtime 计时（scope v2）

添加 `--sparse-attention`（需 `dit` 模式）。起点是 attention 输入通信阶段开始前；终点是最后结果回传/聚合及必要依赖等待完成、输出可用时。区间内包含路由/选块、调度、稀疏计算、KV/元数据通信、结果合并与等待。单卡采用对应 runtime 的入口/出口。

通常不含输入 Norm、通信前的 QKV/QK Norm/RoPE、输出投影和 residual；按源码实际位置计入，不能笼统宣称所有框架这些操作都在区间外。SGLang H3 包含反向 All-to-All 后的 coarse All-Gather 和 gate fold；SSTA 包含文本头 All-Gather；NABLA 多卡通信与投影交错，连续区间会包含 K/V 投影和 QK Norm/RoPE，manifest 明确标注。准确起止点见 [sparse_attention.py](sparse_attention.py) 的运行时 `boundaries`。

事件在请求外预分配并初始化，不新增逐层同步或 collective；异步通信利用原有执行流依赖保证结果可用。保留原生 hooks/编译 helper；NABLA 多卡在 DTensor hooks 内的事件记录显式保持 eager，可能产生 helper 图断点，需单独核实插桩扰动。whole-block compile/graph 不支持，不自动改变原配置。

逐 forward/layer/rank 数据、源码锚点和 `scope_version=2` 保存于 `metrics.json` 的 `per_rank[].sparse_attention`，计划文件也记录版本和范围。主拆分在完整 DiT 最慢的同一 rank 上计算：attention + 其余 DiT = 完整 DiT；另存最大 attention rank 的总和，不相加各层独立最大 rank。

旧 `scope_version=1` 是 Norm → residual，不能重标或混入新结果。`dit` 模式在同一请求记录 E2E、完整 DiT 和 attention 三项；使用 `--modes dit --warmup 1 --repeats 1 --sparse-attention` 可做单次测量。`e2e` 模式移除 attention 插桩，可按需作对照；模式间差异包含整请求波动，不等同于纯插桩成本。各模型的 GPU 验证状态以具体实验结果为准。

SGLang H3 的默认 `bcg_breakpoint=True` 只是原生包装入口；BCG 关闭时保留其 eager 透传并计时，不改配置。真实 BCG 捕获上下文仍会拒绝。
