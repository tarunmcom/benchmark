# +======================================================================+
# |         vLLM Benchmark Suite -- Configuration File                   |
# +======================================================================+
#
# This file is sourced by bash. Use valid bash array/variable syntax.
# Lines beginning with # are comments.

# ----------------------------------------------------------------------
# MODEL DEFINITIONS
# ----------------------------------------------------------------------
# Format:  "model_path | precisions | tp_sizes | extra_vllm_args"
#
#   model_path    : HuggingFace model ID or local path
#   precisions    : Comma-separated: bf16, fp16, fp8, awq, gptq, auto
#   tp_sizes      : Comma-separated tensor-parallel sizes: 1, 2, 4, 8
#   extra_args    : (Optional) additional vllm serve flags for this model
#
# All combinations of (precision x tp_size) are benchmarked per model.

MODEL_CONFIGS=(
  "meta-llama/Llama-3.1-8B-Instruct  | bf16  | 1 "
  #"mistralai/Mistral-7B-Instruct-v0.3 | bf16      | 1"
  # -- Local model examples (absolute or relative paths) --
  # "/data/models/my-finetuned-llama    | bf16      | 1"
  # "./local-models/Mistral-7B          | bf16,fp8  | 1,2 | --enforce-eager"
  # "../shared/checkpoints/my-model     | auto      | 4   | --max-model-len 4096"
)

# ----------------------------------------------------------------------
# BENCHMARK SWEEP PARAMETERS  (applied identically to every model)
# ----------------------------------------------------------------------

INPUT_LENS=(500 2500 5000 10000)       # Random input token lengths to sweep
OUTPUT_LENS=(50 100 1500 5000)            # Random output token lengths to sweep
CONCURRENCIES=(25 50 75)       # Max concurrency levels to sweep
NUM_PROMPTS=500                  # Number of prompts per benchmark run
REPEATS=1                        # Repetitions per combination (inner repeat)
COMPLETE_REPEATS=1               # Full suite repetitions (re-runs all benchmarks from start)
DATASET_NAME="random"            # Dataset: random, sharegpt, etc.
BACKEND="openai"                 # Backend: openai, tgi, etc.

# ----------------------------------------------------------------------
# SERVER SETTINGS
# ----------------------------------------------------------------------

HOST="0.0.0.0"                   # vLLM server bind address
PORT=8000                        # vLLM server port
GPU_MEMORY_UTILIZATION=0.95      # GPU memory fraction (0.0–1.0)
MAX_MODEL_LEN="15000"                 # Empty = auto from model config.json; set to override

# Additional args passed to EVERY vllm serve invocation
EXTRA_SERVER_ARGS=""
# Example: EXTRA_SERVER_ARGS="--enable-prefix-caching --disable-log-requests"

# Additional args passed to EVERY vllm bench serve invocation
EXTRA_BENCH_ARGS=""
# Example: EXTRA_BENCH_ARGS="--request-rate 10 --percentile-metrics p50,p99"

# Compilation config (optional). Set ENABLE_COMPILATION=true to pass --compilation-config to vllm serve.
# COMPILATION_CONFIG can be: (1) path to a .json file, or (2) inline JSON string (single line).
ENABLE_COMPILATION=false
# Example (inline, single line):
# COMPILATION_CONFIG='{"custom_ops":["-rms_norm","-silu_and_mul"],"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true,"fuse_attn_quant":true},"use_inductor_graph_partition":true,"splitting_ops":[]}'
# Or use a file: COMPILATION_CONFIG="./my_compilation_config.json"
COMPILATION_CONFIG=""

# ----------------------------------------------------------------------
# TIMEOUTS  (seconds)
# ----------------------------------------------------------------------

DOWNLOAD_START_TIMEOUT=120       # Max seconds to wait for download to begin
DOWNLOAD_IDLE_TIMEOUT=300        # Max seconds of no download progress before abort
SERVER_START_TIMEOUT=900         # Max seconds AFTER download to reach healthy state
SERVER_STOP_TIMEOUT=60           # Max seconds to wait for graceful stop
BENCHMARK_TIMEOUT=9990           # Max seconds for a single benchmark run

# ----------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------

BASE_OUT_DIR="./benchmark_results"
