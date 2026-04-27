
#
# Usage:
#   ./vllm_benchmark_suite.sh                        Interactive mode
#   ./vllm_benchmark_suite.sh -c benchmark.conf      Use config file
#   ./vllm_benchmark_suite.sh --generate-config      Generate sample config
#   ./vllm_benchmark_suite.sh --help                 Show help
#
# Requirements: bash 4+, curl, vllm (pip install vllm)
# -----------------------------------------------------------------------

set -uo pipefail

# ====================================================================
# SECTION 1 -- Constants & Colours
# ====================================================================

if [[ -t 1 ]]; then
	  readonly C_RED='\033[0;31m'    C_GREEN='\033[0;32m'
	    readonly C_YELLOW='\033[1;33m' C_BLUE='\033[0;34m'
	      readonly C_MAGENTA='\033[0;35m' C_CYAN='\033[0;36m'
	        readonly C_WHITE='\033[1;37m'  C_DIM='\033[2m'
		  readonly C_BOLD='\033[1m'      C_RESET='\033[0m'
	  else
		    readonly C_RED='' C_GREEN='' C_YELLOW='' C_BLUE=''
		      readonly C_MAGENTA='' C_CYAN='' C_WHITE='' C_DIM=''
		        readonly C_BOLD='' C_RESET=''
fi

readonly SCRIPT_VERSION="1.0.0"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ====================================================================
# SECTION 2 -- Defaults
# ====================================================================

MODEL_CONFIGS=()

INPUT_LENS=(1024)
OUTPUT_LENS=(512)
CONCURRENCIES=(128)
NUM_PROMPTS=500
REPEATS=1
COMPLETE_REPEATS=1
DATASET_NAME="random"
BACKEND="openai"

HOST="0.0.0.0"
PORT=8000
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=""  # empty = use model's default from config.json

EXTRA_SERVER_ARGS=""  # extra args for vllm serve
EXTRA_BENCH_ARGS=""   # extra args for vllm bench serve

# Compilation config: set ENABLE_COMPILATION=true and COMPILATION_CONFIG to use --compilation-config
ENABLE_COMPILATION=false
COMPILATION_CONFIG=""  # JSON string, or path to a .json file

DOWNLOAD_START_TIMEOUT=120   # max seconds to wait for download to begin (if model not cached)
DOWNLOAD_IDLE_TIMEOUT=300    # max seconds of no download progress before giving up
SERVER_START_TIMEOUT=600     # max seconds AFTER download completes to reach healthy state
SERVER_STOP_TIMEOUT=60
BENCHMARK_TIMEOUT=7200

BASE_OUT_DIR="./benchmark_results"

# ====================================================================
# SECTION 3 -- Runtime State
# ====================================================================

VLLM_SERVER_PID=""
RUN_DIR=""
ERROR_LOG=""
COMMANDS_LOG=""
SUMMARY_FILE=""
TOTAL_RUNS=0
PASSED_RUNS=0
FAILED_RUNS=0
SKIPPED_RUNS=0
CURRENT_RUN=0

# Resume support
RESUME_DIR=""             # set by --resume; empty = fresh run
RESUMED_RUNS=0            # count of previously completed runs being skipped
declare -A COMPLETED_RUNS # associative array: COMPLETED_RUNS["full_tag"]=1 for passed runs

# ====================================================================
# SECTION 4 -- Printing Utilities
# ====================================================================

banner() {
	  local msg="$1"
	    local len=${#msg}
	      local border
	        border=$(printf '=%.0s' $(seq 1 $((len + 4))))
		  echo ""
		    echo -e "${C_CYAN}+${border}+${C_RESET}"
		      echo -e "${C_CYAN}|  ${C_WHITE}${msg}${C_CYAN}  |${C_RESET}"
		        echo -e "${C_CYAN}+${border}+${C_RESET}"
			  echo ""
		  }

		  section() {
			    echo ""
			      echo -e "${C_BLUE}--- ${C_BOLD}$1${C_RESET}${C_BLUE} ---${C_RESET}"
		      }

		      info()    { echo -e "${C_GREEN}[INFO]${C_RESET}    $*"; }
		      warn()    { echo -e "${C_YELLOW}[WARN]${C_RESET}    $*"; }
		      err()     { echo -e "${C_RED}[ERROR]${C_RESET}   $*" >&2; }
		      detail()  { echo -e "${C_DIM}          $*${C_RESET}"; }
		      step()    { echo -e "${C_MAGENTA}[STEP]${C_RESET}    $*"; }
		      success() { echo -e "${C_GREEN}[  OK  ]${C_RESET}  $*"; }
		      fail()    { echo -e "${C_RED}[FAILED]${C_RESET}  $*"; }

		      progress_bar() {
			        local current=$1 total=$2 width=40
				  if (( total <= 0 )); then return; fi
				    local pct=$(( current * 100 / total ))
				      local filled=$(( current * width / total ))
				        local empty=$(( width - filled ))
					  local bar=""
					    if (( filled > 0 )); then
						        bar=$(printf '#%.0s' $(seq 1 "$filled"))
							  fi
							    if (( empty > 0 )); then
								        bar+=$(printf '.%.0s' $(seq 1 "$empty"))
									  fi
									    printf "\r${C_CYAN}  Progress: [%s] %3d%% (%d/%d)${C_RESET}" "$bar" "$pct" "$current" "$total"
								    }

								    log_error() {
									      local context="$1"
									        local log_file="$2"
										  shift 2
										    local message="$*"

										      if [[ -n "$ERROR_LOG" ]]; then
											          {
													        echo "===================================================="
														      echo "TIMESTAMP : $(date -Is)"
														            echo "CONTEXT   : $context"
															          echo "MESSAGE   : $message"
																        if [[ -n "$log_file" && -f "$log_file" ]]; then
																		        echo ""
																			        echo "LAST 20 LINES OF: $log_file"
																				        echo "----------------------------------------------------"
																					        tail -n 20 "$log_file" 2>/dev/null || echo "(could not read log)"
																						        echo "----------------------------------------------------"
																							      fi
																							            echo "===================================================="
																								          echo ""
																									      } >> "$ERROR_LOG"
																									        fi

																										  err "[$context] $message"
																										    if [[ -n "$log_file" && -f "$log_file" ]]; then
																											        # Extract only lines containing error/exception/traceback text from the log
																												    local error_lines
																												        error_lines=$(grep -i -E 'error|exception|traceback|failed|fatal|abort|CUDA|OOM|out of memory|RuntimeError|ValueError|KeyError|ImportError|ModuleNotFoundError|FileNotFoundError|AssertionError' "$log_file" 2>/dev/null | tail -n 15)
																													    if [[ -n "$error_lines" ]]; then
																														          echo -e "${C_RED}          -- Error lines from ${log_file} --${C_RESET}" >&2
																															        while IFS= read -r line; do
																																	        echo -e "${C_RED}          ${line}${C_RESET}" >&2
																																		      done <<< "$error_lines"
      echo -e "${C_DIM}          -- (full log: ${log_file}) --${C_RESET}" >&2
    else
      echo -e "${C_DIM}          (no error lines found -- see full log: ${log_file})${C_RESET}" >&2
    fi
  fi
}

log_command() {
  local label="$1"
  shift
  [[ -z "$COMMANDS_LOG" ]] && return 0
  {
    echo "------------------------------------------"
    echo "LABEL     : $label"
    echo "TIMESTAMP : $(date -Is)"
    echo "COMMAND   : $*"
    echo "------------------------------------------"
  } >> "$COMMANDS_LOG"
}

# ====================================================================
# SECTION 5 -- Cleanup & Signal Handling
# ====================================================================

cleanup() {
  echo ""
  warn "Caught interrupt -- cleaning up..."
  stop_vllm_server "interrupted"
  if [[ -n "$RUN_DIR" && -d "$RUN_DIR" && -n "$ERROR_LOG" ]]; then
    generate_final_report
  fi
  echo -e "${C_YELLOW}Cleanup complete. Logs are in: ${C_WHITE}${RUN_DIR:-N/A}${C_RESET}"
  exit 130
}
trap cleanup SIGINT SIGTERM

# ====================================================================
# SECTION 6 -- Precision Mapping
# ====================================================================

# Maps user-friendly precision names to vllm serve arguments.
# Returns the extra args string via stdout.
map_precision_to_vllm_args() {
  local precision="$1"
  case "$precision" in
    bf16|bfloat16)   echo "--dtype bfloat16" ;;
    fp16|float16)    echo "--dtype float16" ;;
    fp8)             echo "--dtype auto --quantization fp8" ;;
    fp8_w8a8)        echo "--dtype auto --quantization fp8" ;;
    awq)             echo "--dtype auto --quantization awq" ;;
    gptq)            echo "--dtype auto --quantization gptq" ;;
    squeezellm)      echo "--dtype auto --quantization squeezellm" ;;
    auto)            echo "--dtype auto" ;;
    *)
      warn "Unknown precision '${precision}', passing as --dtype ${precision}"
      echo "--dtype ${precision}"
      ;;
  esac
}

precision_display_name() {
  local precision="$1"
  case "$precision" in
    bf16|bfloat16)   echo "BFloat16" ;;
    fp16|float16)    echo "Float16" ;;
    fp8|fp8_w8a8)    echo "FP8" ;;
    awq)             echo "AWQ (int4)" ;;
    gptq)            echo "GPTQ (int4)" ;;
    auto)            echo "Auto" ;;
    *)               echo "$precision" ;;
  esac
}

# ====================================================================
# SECTION 7 -- Model Config Helpers
# ====================================================================

# MODEL_CONFIGS entries: "model_path | precisions | tp_sizes | extra_args"
# Fields separated by |, precisions and tp_sizes are comma-separated.

parse_model_path()   { echo "$1" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}'; }
parse_precisions()   { echo "$1" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' | tr ',' ' '; }
parse_tp_sizes()     { echo "$1" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}' | tr ',' ' '; }
parse_model_extra()  { echo "$1" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $4); print $4}'; }

is_local_path() {
  local path="$1"
  [[ "$path" == /* || "$path" == ./* || "$path" == ../* || "$path" == ~* ]] && return 0
  # HuggingFace IDs are "org/model" -- at most one slash.
  # Anything with 2+ slashes or that exists as a directory is local.
  local stripped="${path//[^\/]/}"
  (( ${#stripped} >= 2 )) && return 0
  [[ -d "$path" ]] && return 0
  return 1
}

model_short_name() {
  local model_path="$1"
  basename "$model_path" | sed 's/[^a-zA-Z0-9._-]/_/g'
}

# ====================================================================
# SECTION 7b -- Resume Helpers
# ====================================================================

find_latest_run_dir() {
  local base="${1:-$BASE_OUT_DIR}"
  if [[ ! -d "$base" ]]; then
    echo ""
    return
  fi
  # Find the most recent run_* directory by name (lexicographic = chronological with YYYYMMDD_HHMMSS)
  local latest
  latest=$(ls -1d "${base}"/run_* 2>/dev/null | sort | tail -n 1)
  echo "${latest:-}"
}

load_completed_runs() {
  local tracker_file="$1"
  COMPLETED_RUNS=()
  RESUMED_RUNS=0

  if [[ ! -f "$tracker_file" ]]; then
    warn "No results_tracker.txt found in resume directory."
    return
  fi

  while IFS= read -r line; do
    # Skip header, separator, and empty lines
    [[ "$line" == RUN* || "$line" == -* || -z "$line" ]] && continue

    # Parse: full_tag is the first field (up to 70 chars, then whitespace)
    local tag rc_field
    tag=$(echo "$line" | awk '{print $1}')
    rc_field=$(echo "$line" | grep -o 'rc=[^ ]*' | head -1)

    if [[ "$rc_field" == "rc=0" && -n "$tag" ]]; then
      COMPLETED_RUNS["$tag"]=1
      RESUMED_RUNS=$((RESUMED_RUNS + 1))
    fi
  done < "$tracker_file"

  info "Found ${RESUMED_RUNS} previously completed benchmark(s) to skip."
}

is_run_completed() {
  local full_tag="$1"
  [[ -n "${COMPLETED_RUNS[$full_tag]+_}" ]]
}

# Check if ALL benchmarks for a given model/precision/tp combo are already done
all_benchmarks_completed_for_combo() {
  local model_path="$1" precision="$2" tp="$3" iter="${4:-1}"
  local model_name
  model_name=$(model_short_name "$model_path")

  for concurrency in "${CONCURRENCIES[@]}"; do
    for input_len in "${INPUT_LENS[@]}"; do
      for output_len in "${OUTPUT_LENS[@]}"; do
        for rep in $(seq 1 "$REPEATS"); do
          local tag="c${concurrency}_in${input_len}_out${output_len}_rep${rep}"
          local full_tag
          if (( COMPLETE_REPEATS > 1 )); then
            full_tag="iter${iter}/${model_name}/${precision}_tp${tp}/${tag}"
          else
            full_tag="${model_name}/${precision}_tp${tp}/${tag}"
          fi
          if ! is_run_completed "$full_tag"; then
            return 1
          fi
        done
      done
    done
  done
  return 0
}

# ====================================================================
# SECTION 8 -- Generate Sample Config
# ====================================================================

generate_sample_config() {
  local outfile="${1:-benchmark.conf}"
  cat > "$outfile" <<'SAMPLE_EOF'
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
  "meta-llama/Llama-3.1-8B-Instruct  | bf16,fp8  | 1,2"
  "mistralai/Mistral-7B-Instruct-v0.3 | bf16      | 1"
  # -- Local model examples (absolute or relative paths) --
  # "/data/models/my-finetuned-llama    | bf16      | 1"
  # "./local-models/Mistral-7B          | bf16,fp8  | 1,2 | --enforce-eager"
  # "../shared/checkpoints/my-model     | auto      | 4   | --max-model-len 4096"
)

# ----------------------------------------------------------------------
# BENCHMARK SWEEP PARAMETERS  (applied identically to every model)
# ----------------------------------------------------------------------

INPUT_LENS=(512 1024 2048)       # Random input token lengths to sweep
OUTPUT_LENS=(128 512)            # Random output token lengths to sweep
CONCURRENCIES=(64 128 256)       # Max concurrency levels to sweep
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
MAX_MODEL_LEN=""                 # Empty = auto from model config.json; set to override

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
SERVER_START_TIMEOUT=600         # Max seconds AFTER download to reach healthy state
SERVER_STOP_TIMEOUT=60           # Max seconds to wait for graceful stop
BENCHMARK_TIMEOUT=7200           # Max seconds for a single benchmark run

# ----------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------

BASE_OUT_DIR="./benchmark_results"
SAMPLE_EOF

  info "Sample config written to: ${C_WHITE}${outfile}${C_RESET}"
  info "Edit the file, then run:  ${C_CYAN}./vllm_benchmark_suite.sh -c ${outfile}${C_RESET}"
}

# ====================================================================
# SECTION 9 -- Interactive Mode
# ====================================================================

prompt_value() {
  local prompt="$1" default="$2" var_name="$3"
  local input
  echo -en "${C_CYAN}  ${prompt}${C_DIM} [${default}]${C_RESET}: "
  read -r input
  printf -v "$var_name" '%s' "${input:-$default}"
}

prompt_array() {
  local prompt="$1" default="$2" var_name="$3"
  local input
  echo -en "${C_CYAN}  ${prompt}${C_DIM} [${default}]${C_RESET}: "
  read -r input
  input="${input:-$default}"
  read -ra _tmp_arr <<< "$input"
  eval "$var_name=(\"\${_tmp_arr[@]}\")"
  unset _tmp_arr
}

interactive_setup() {
  banner "vLLM Benchmark Suite -- Interactive Setup"

  echo -e "${C_WHITE}  Let's configure your benchmark run step by step.${C_RESET}"
  echo -e "${C_DIM}  Press Enter to accept defaults shown in brackets.${C_RESET}"
  echo ""

  # -- Models --
  section "Model Configuration"
  echo ""
  local num_models
  prompt_value "How many models to benchmark?" "1" num_models
  if ! [[ "$num_models" =~ ^[0-9]+$ ]] || (( num_models < 1 )); then
    err "Invalid number of models: '${num_models}'. Must be a positive integer."
    exit 1
  fi

  MODEL_CONFIGS=()
  for (( i=1; i<=num_models; i++ )); do
    echo ""
    echo -e "  ${C_WHITE}-- Model $i of $num_models --${C_RESET}"
    local m_path m_prec m_tp m_extra=""
    echo -e "  ${C_DIM}  Examples: meta-llama/Llama-3.1-8B-Instruct  or  /data/models/my-model${C_RESET}"
    prompt_value "  Model path (HuggingFace ID or local path)" "meta-llama/Llama-3.1-8B-Instruct" m_path
    if is_local_path "$m_path"; then
      if [[ -d "$m_path" ]]; then
        info "  Local model directory found: ${m_path}"
      else
        warn "  Path '${m_path}' does not exist yet -- make sure it exists before running."
      fi
    fi
    prompt_value "  Precisions (comma-separated: bf16,fp16,fp8,awq,auto)" "bf16" m_prec
    prompt_value "  Tensor-parallel sizes (comma-separated: 1,2,4,8)" "1" m_tp
    prompt_value "  Extra vllm serve args for this model (or empty)" "" m_extra
    MODEL_CONFIGS+=("${m_path} | ${m_prec} | ${m_tp} | ${m_extra}")
  done

  # -- Benchmark params --
  section "Benchmark Parameters"
  echo ""
  prompt_array "Input token lengths (space-separated)" "512 1024" INPUT_LENS
  prompt_array "Output token lengths (space-separated)" "128 512" OUTPUT_LENS
  prompt_array "Concurrency levels (space-separated)" "64 128" CONCURRENCIES
  prompt_value "Number of prompts per run" "500" NUM_PROMPTS
  prompt_value "Repetitions per combination" "1" REPEATS
  prompt_value "Complete suite repetitions (re-runs all benchmarks from start)" "1" COMPLETE_REPEATS
  prompt_value "Dataset name" "random" DATASET_NAME
  prompt_value "Backend" "openai" BACKEND

  # -- Server --
  section "Server Settings"
  echo ""
  prompt_value "Server host" "0.0.0.0" HOST
  prompt_value "Server port" "8000" PORT
  prompt_value "GPU memory utilization (0.0–1.0)" "0.95" GPU_MEMORY_UTILIZATION
  echo -e "  ${C_DIM}  Leave empty to use model's default from config.json${C_RESET}"
  prompt_value "Max model length (empty = auto from model config)" "" MAX_MODEL_LEN
  echo -e "  ${C_DIM}  e.g. --enable-prefix-caching --disable-log-requests${C_RESET}"
  prompt_value "Extra vllm serve args (global, or empty)" "" EXTRA_SERVER_ARGS

  # -- Compilation config --
  section "Compilation Config (optional)"
  echo ""
  echo -e "  ${C_DIM}  Use --compilation-config for custom_ops, cudagraph_mode, pass_config, etc.${C_RESET}"
  prompt_value "Enable compilation config? (y/n)" "n" enable_comp
  if [[ "$enable_comp" =~ ^[Yy] ]]; then
    ENABLE_COMPILATION=true
    echo -e "  ${C_DIM}  Enter path to a .json file, or paste single-line JSON${C_RESET}"
    prompt_value "Compilation config (file path or JSON string)" "" COMPILATION_CONFIG
  else
    ENABLE_COMPILATION=false
    COMPILATION_CONFIG=""
  fi

  # -- Benchmark client --
  section "Benchmark Client Settings"
  echo ""
  echo -e "  ${C_DIM}  e.g. --request-rate 10 --percentile-metrics p50,p99${C_RESET}"
  prompt_value "Extra vllm bench serve args (global, or empty)" "" EXTRA_BENCH_ARGS

  # -- Timeouts --
  section "Timeouts"
  echo ""
  echo -e "  ${C_DIM}  Download timeouts apply when model is not cached locally.${C_RESET}"
  prompt_value "Download start timeout -- abort if download doesn't begin (seconds)" "120" DOWNLOAD_START_TIMEOUT
  prompt_value "Download idle timeout -- abort if download stalls (seconds)" "300" DOWNLOAD_IDLE_TIMEOUT
  echo -e "  ${C_DIM}  Server start timeout begins AFTER download completes.${C_RESET}"
  prompt_value "Server start timeout -- model loading + ready (seconds)" "600" SERVER_START_TIMEOUT
  prompt_value "Benchmark run timeout (seconds)" "7200" BENCHMARK_TIMEOUT

  # -- Output --
  section "Output"
  echo ""
  prompt_value "Base output directory" "./benchmark_results" BASE_OUT_DIR

  echo ""

  # Offer to save config
  local save_config
  prompt_value "Save this configuration to a file? (y/n)" "y" save_config
  if [[ "$save_config" =~ ^[Yy] ]]; then
    local config_file
    prompt_value "Config filename" "benchmark.conf" config_file
    save_interactive_config "$config_file"
    info "Config saved to ${C_WHITE}${config_file}${C_RESET} -- reuse with:  ${C_CYAN}-c ${config_file}${C_RESET}"
  fi
}

save_interactive_config() {
  local outfile="$1"
  {
    echo "# Auto-generated by vLLM Benchmark Suite (interactive mode)"
    echo "# Generated: $(date -Is)"
    echo ""
    echo "MODEL_CONFIGS=("
    for entry in "${MODEL_CONFIGS[@]}"; do
      echo "  \"${entry}\""
    done
    echo ")"
    echo ""
    echo "INPUT_LENS=(${INPUT_LENS[*]})"
    echo "OUTPUT_LENS=(${OUTPUT_LENS[*]})"
    echo "CONCURRENCIES=(${CONCURRENCIES[*]})"
    echo "NUM_PROMPTS=${NUM_PROMPTS}"
    echo "REPEATS=${REPEATS}"
    echo "COMPLETE_REPEATS=${COMPLETE_REPEATS}"
    echo "DATASET_NAME=\"${DATASET_NAME}\""
    echo "BACKEND=\"${BACKEND}\""
    echo ""
    echo "HOST=\"${HOST}\""
    echo "PORT=${PORT}"
    echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
    echo "MAX_MODEL_LEN=\"${MAX_MODEL_LEN}\"  # empty = auto from model config"
    echo "EXTRA_SERVER_ARGS=\"${EXTRA_SERVER_ARGS}\""
    echo "EXTRA_BENCH_ARGS=\"${EXTRA_BENCH_ARGS}\""
    echo "ENABLE_COMPILATION=${ENABLE_COMPILATION}"
    # Escape double quotes so JSON in COMPILATION_CONFIG doesn't break the file
    local cc_escaped="${COMPILATION_CONFIG//\\/\\\\}"
    cc_escaped="${cc_escaped//\"/\\\"}"
    echo "COMPILATION_CONFIG=\"${cc_escaped}\""
    echo ""
    echo "DOWNLOAD_START_TIMEOUT=${DOWNLOAD_START_TIMEOUT}"
    echo "DOWNLOAD_IDLE_TIMEOUT=${DOWNLOAD_IDLE_TIMEOUT}"
    echo "SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT}"
    echo "SERVER_STOP_TIMEOUT=${SERVER_STOP_TIMEOUT}"
    echo "BENCHMARK_TIMEOUT=${BENCHMARK_TIMEOUT}"
    echo ""
    echo "BASE_OUT_DIR=\"${BASE_OUT_DIR}\""
  } > "$outfile"
}

# ====================================================================
# SECTION 10 -- Validation
# ====================================================================

validate_config() {
  local errors=0

  if [[ ${#MODEL_CONFIGS[@]} -eq 0 ]]; then
    err "No models configured. Add at least one MODEL_CONFIGS entry."
    errors=$((errors + 1))
  fi

  for entry in "${MODEL_CONFIGS[@]}"; do
    local mpath
    mpath=$(parse_model_path "$entry")
    local precs
    precs=$(parse_precisions "$entry")
    local tps
    tps=$(parse_tp_sizes "$entry")

    if [[ -z "$mpath" ]]; then
      err "Empty model path in config entry: $entry"
      errors=$((errors + 1))
    elif is_local_path "$mpath"; then
      if [[ ! -d "$mpath" ]]; then
        err "Local model path does not exist: $mpath"
        errors=$((errors + 1))
      elif [[ ! -f "$mpath/config.json" ]]; then
        warn "No config.json found in $mpath -- are you sure this is a valid model directory?"
      else
        info "Local model validated: ${mpath}"
      fi
    else
      detail "HuggingFace model ID: ${mpath} (will be downloaded if not cached)"
    fi
    if [[ -z "$precs" ]]; then
      err "No precisions defined for model: $mpath"
      errors=$((errors + 1))
    fi
    if [[ -z "$tps" ]]; then
      err "No TP sizes defined for model: $mpath"
      errors=$((errors + 1))
    fi
    for tp in $tps; do
      if ! [[ "$tp" =~ ^[0-9]+$ ]] || (( tp < 1 )); then
        err "Invalid TP size '$tp' for model $mpath (must be positive integer)"
        errors=$((errors + 1))
      fi
    done
  done

  if [[ ${#INPUT_LENS[@]} -eq 0 ]]; then
    err "INPUT_LENS cannot be empty."
    errors=$((errors + 1))
  fi
  if [[ ${#OUTPUT_LENS[@]} -eq 0 ]]; then
    err "OUTPUT_LENS cannot be empty."
    errors=$((errors + 1))
  fi
  if [[ ${#CONCURRENCIES[@]} -eq 0 ]]; then
    err "CONCURRENCIES cannot be empty."
    errors=$((errors + 1))
  fi

  if ! (( COMPLETE_REPEATS >= 1 )) 2>/dev/null; then
    err "COMPLETE_REPEATS must be a positive integer (got: '${COMPLETE_REPEATS}')."
    errors=$((errors + 1))
  fi

  if ! command -v vllm &>/dev/null; then
    warn "vllm command not found in PATH -- make sure it's installed."
  fi
  if ! command -v curl &>/dev/null; then
    err "curl is required but not found."
    errors=$((errors + 1))
  fi

  if (( errors > 0 )); then
    err "Configuration has $errors error(s). Please fix and retry."
    exit 1
  fi
  success "Configuration validated."
}

# ====================================================================
# SECTION 11 -- Pre-Run Summary
# ====================================================================

compute_total_runs() {
  local total=0
  for entry in "${MODEL_CONFIGS[@]}"; do
    local precs tp_sizes
    read -ra precs <<< "$(parse_precisions "$entry")"
    read -ra tp_sizes <<< "$(parse_tp_sizes "$entry")"
    local combos=$(( ${#precs[@]} * ${#tp_sizes[@]} ))
    local bench_per_server=$(( ${#INPUT_LENS[@]} * ${#OUTPUT_LENS[@]} * ${#CONCURRENCIES[@]} * REPEATS ))
    total=$(( total + combos * bench_per_server ))
  done
  echo $(( total * COMPLETE_REPEATS ))
}

compute_server_starts() {
  local total=0
  for entry in "${MODEL_CONFIGS[@]}"; do
    local precs tp_sizes
    read -ra precs <<< "$(parse_precisions "$entry")"
    read -ra tp_sizes <<< "$(parse_tp_sizes "$entry")"
    total=$(( total + ${#precs[@]} * ${#tp_sizes[@]} ))
  done
  echo $(( total * COMPLETE_REPEATS ))
}

show_summary() {
  local total_bench total_servers
  total_bench=$(compute_total_runs)
  total_servers=$(compute_server_starts)
  TOTAL_RUNS=$total_bench

  banner "Benchmark Run Plan"

  # Environment info
  echo -e "  ${C_DIM}Date        :${C_RESET} $(date)"
  echo -e "  ${C_DIM}Host        :${C_RESET} $(hostname)"
  echo -e "  ${C_DIM}vLLM        :${C_RESET} $(vllm --version 2>/dev/null || echo 'not found')"
  echo -e "  ${C_DIM}Output dir  :${C_RESET} ${C_WHITE}${RUN_DIR}${C_RESET}"
  echo ""

  # Models
  section "Models"
  local model_idx=0
  for entry in "${MODEL_CONFIGS[@]}"; do
    model_idx=$((model_idx + 1))
    local mpath precs_str tps_str mextra
    mpath=$(parse_model_path "$entry")
    precs_str=$(parse_precisions "$entry")
    tps_str=$(parse_tp_sizes "$entry")
    mextra=$(parse_model_extra "$entry")

    echo ""
    local source_label
    if is_local_path "$mpath"; then
      source_label="${C_YELLOW}[local]${C_RESET}"
    else
      source_label="${C_GREEN}[HuggingFace]${C_RESET}"
    fi
    echo -e "  ${C_WHITE}Model $model_idx:${C_RESET} ${C_CYAN}${mpath}${C_RESET}  ${source_label}"

    local prec_display=""
    for p in $precs_str; do
      prec_display+="$(precision_display_name "$p"), "
    done
    prec_display="${prec_display%, }"
    echo -e "    ${C_DIM}Precisions :${C_RESET} ${prec_display}"
    echo -e "    ${C_DIM}TP sizes   :${C_RESET} ${tps_str}"
    if [[ -n "$mextra" ]]; then
      echo -e "    ${C_DIM}Extra args :${C_RESET} ${mextra}"
    fi
  done

  # Benchmark params
  section "Benchmark Sweep Parameters"
  echo ""
  echo -e "  ${C_DIM}Input lengths   :${C_RESET} ${INPUT_LENS[*]}"
  echo -e "  ${C_DIM}Output lengths  :${C_RESET} ${OUTPUT_LENS[*]}"
  echo -e "  ${C_DIM}Concurrencies   :${C_RESET} ${CONCURRENCIES[*]}"
  echo -e "  ${C_DIM}Num prompts     :${C_RESET} ${NUM_PROMPTS}"
  echo -e "  ${C_DIM}Repetitions     :${C_RESET} ${REPEATS}"
  if (( COMPLETE_REPEATS > 1 )); then
    echo -e "  ${C_DIM}Complete repeats :${C_RESET} ${COMPLETE_REPEATS}"
  fi
  echo -e "  ${C_DIM}Dataset         :${C_RESET} ${DATASET_NAME}"
  echo -e "  ${C_DIM}Backend         :${C_RESET} ${BACKEND}"

  # Server
  section "Server Configuration"
  echo ""
  echo -e "  ${C_DIM}Host            :${C_RESET} ${HOST}"
  echo -e "  ${C_DIM}Port            :${C_RESET} ${PORT}"
  echo -e "  ${C_DIM}GPU mem util    :${C_RESET} ${GPU_MEMORY_UTILIZATION}"
  echo -e "  ${C_DIM}Max model len   :${C_RESET} ${MAX_MODEL_LEN:-auto (from model config)}"
  if [[ -n "$EXTRA_SERVER_ARGS" ]]; then
    echo -e "  ${C_DIM}Server args     :${C_RESET} ${EXTRA_SERVER_ARGS}"
  fi
  if [[ -n "$EXTRA_BENCH_ARGS" ]]; then
    echo -e "  ${C_DIM}Bench args      :${C_RESET} ${EXTRA_BENCH_ARGS}"
  fi
  if [[ "$ENABLE_COMPILATION" == "true" || "$ENABLE_COMPILATION" == "1" || "$ENABLE_COMPILATION" == "yes" ]]; then
    echo -e "  ${C_DIM}Compilation     :${C_RESET} ${C_GREEN}ON${C_RESET}"
    if [[ -n "$COMPILATION_CONFIG" ]]; then
      if [[ -f "$COMPILATION_CONFIG" ]]; then
        echo -e "  ${C_DIM}Config source   :${C_RESET} file: ${COMPILATION_CONFIG}"
      else
        echo -e "  ${C_DIM}Config         :${C_RESET} ${COMPILATION_CONFIG:0:60}${COMPILATION_CONFIG:+...}"
      fi
    fi
  else
    echo -e "  ${C_DIM}Compilation     :${C_RESET} off"
  fi

  # Totals
  section "Execution Summary"
  echo ""
  if [[ -n "$RESUME_DIR" && $RESUMED_RUNS -gt 0 ]]; then
    echo -e "  ${C_CYAN}Mode            :${C_RESET} ${C_YELLOW}RESUME${C_RESET} (from ${RUN_DIR})"
    echo -e "  ${C_WHITE}Total planned   :${C_RESET} ${C_YELLOW}${total_bench}${C_RESET}"
    echo -e "  ${C_GREEN}Already done    :${C_RESET} ${RESUMED_RUNS}"
    local remaining=$(( total_bench - RESUMED_RUNS ))
    if (( remaining < 0 )); then remaining=0; fi
    echo -e "  ${C_WHITE}Remaining       :${C_RESET} ${C_YELLOW}${remaining}${C_RESET}"
    echo -e "  ${C_WHITE}Server starts   :${C_RESET} ${C_YELLOW}${total_servers} (max, may be fewer if combos done)${C_RESET}"
  else
    echo -e "  ${C_WHITE}Server starts   :${C_RESET} ${C_YELLOW}${total_servers}${C_RESET}"
    echo -e "  ${C_WHITE}Benchmark runs  :${C_RESET} ${C_YELLOW}${total_bench}${C_RESET}"
  fi
  echo ""

  # Save to file
  save_summary_to_file "$total_bench" "$total_servers"
}

save_summary_to_file() {
  local total_bench=$1 total_servers=$2
  {
    echo "==========================================================="
    echo "  vLLM Benchmark Suite -- Run Plan"
    echo "==========================================================="
    echo ""
    echo "Date          : $(date -Is)"
    echo "Host          : $(hostname)"
    echo "vLLM version  : $(vllm --version 2>/dev/null || echo 'N/A')"
    echo "Script version: ${SCRIPT_VERSION}"
    echo "Output dir    : ${RUN_DIR}"
    echo ""
    echo "-- Models --"
    local midx=0
    for entry in "${MODEL_CONFIGS[@]}"; do
      midx=$((midx + 1))
      local mpath precs_str tps_str mextra
      mpath=$(parse_model_path "$entry")
      precs_str=$(parse_precisions "$entry")
      tps_str=$(parse_tp_sizes "$entry")
      mextra=$(parse_model_extra "$entry")
      echo ""
      echo "  Model $midx: $mpath"
      echo "    Precisions : $precs_str"
      echo "    TP sizes   : $tps_str"
      [[ -n "$mextra" ]] && echo "    Extra args : $mextra"
    done
    echo ""
    echo "-- Benchmark Parameters --"
    echo "  Input lengths  : ${INPUT_LENS[*]}"
    echo "  Output lengths : ${OUTPUT_LENS[*]}"
    echo "  Concurrencies  : ${CONCURRENCIES[*]}"
    echo "  Num prompts    : ${NUM_PROMPTS}"
    echo "  Repetitions    : ${REPEATS}"
    if (( COMPLETE_REPEATS > 1 )); then
      echo "  Complete repeats: ${COMPLETE_REPEATS}"
    fi
    echo "  Dataset        : ${DATASET_NAME}"
    echo "  Backend        : ${BACKEND}"
    echo ""
    echo "-- Server --"
    echo "  Host / Port    : ${HOST}:${PORT}"
    echo "  GPU mem util   : ${GPU_MEMORY_UTILIZATION}"
    echo "  Max model len  : ${MAX_MODEL_LEN:-auto (from model config)}"
    [[ -n "$EXTRA_SERVER_ARGS" ]] && echo "  Server args    : ${EXTRA_SERVER_ARGS}"
    [[ -n "$EXTRA_BENCH_ARGS" ]] && echo "  Bench args     : ${EXTRA_BENCH_ARGS}"
    if [[ "$ENABLE_COMPILATION" == "true" || "$ENABLE_COMPILATION" == "1" || "$ENABLE_COMPILATION" == "yes" ]]; then
      echo "  Compilation    : ON"
      [[ -n "$COMPILATION_CONFIG" ]] && echo "  Compilation config : ${COMPILATION_CONFIG}"
    else
      echo "  Compilation    : off"
    fi
    echo ""
    echo "-- Totals --"
    echo "  Server starts  : ${total_servers}"
    echo "  Benchmark runs : ${total_bench}"
    if [[ -n "$RESUME_DIR" && $RESUMED_RUNS -gt 0 ]]; then
      echo ""
      echo "-- Resume --"
      echo "  Mode           : RESUME"
      echo "  Already done   : ${RESUMED_RUNS}"
      echo "  Remaining      : $(( total_bench - RESUMED_RUNS ))"
    fi
    echo ""
    echo "==========================================================="
  } >> "$SUMMARY_FILE"
  detail "Run plan saved to: ${SUMMARY_FILE}"

}

confirm_proceed() {
  local answer
  echo -en "  ${C_YELLOW}Proceed with this benchmark plan? (y/n)${C_RESET} [y]: "
  read -r answer
  answer="${answer:-y}"
  if [[ ! "$answer" =~ ^[Yy] ]]; then
    info "Aborted by user."
    exit 0
  fi
  echo ""
}

# ====================================================================
# SECTION 12 -- vLLM Server Lifecycle
# ====================================================================

start_vllm_server() {
  local model_path="$1"
  local precision="$2"
  local tp_size="$3"
  local model_extra="$4"
  local server_log="$5"

  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    log_error "server-start" "" "Port ${PORT} is already in use (something responded on /health). Stop the existing server first."
    return 1
  fi

  local prec_args
  prec_args=$(map_precision_to_vllm_args "$precision")

  local -a cmd=(
    vllm serve "$model_path"
    $prec_args
    --tensor-parallel-size "$tp_size"
    --host "$HOST"
    --port "$PORT"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  )

  if [[ -n "$MAX_MODEL_LEN" ]]; then
    cmd+=(--max-model-len "$MAX_MODEL_LEN")
  fi

  if [[ "$ENABLE_COMPILATION" == "true" || "$ENABLE_COMPILATION" == "1" || "$ENABLE_COMPILATION" == "yes" ]]; then
    if [[ -n "$COMPILATION_CONFIG" ]]; then
      local compilation_json
      if [[ -f "$COMPILATION_CONFIG" ]]; then
        compilation_json=$(cat "$COMPILATION_CONFIG")
      else
        compilation_json="$COMPILATION_CONFIG"
      fi
      if [[ -n "$compilation_json" ]]; then
        cmd+=(--compilation-config "$compilation_json")
      fi
    fi
  fi

  if [[ -n "$EXTRA_SERVER_ARGS" ]]; then
    read -ra extra_arr <<< "$EXTRA_SERVER_ARGS"
    cmd+=("${extra_arr[@]}")
  fi

  if [[ -n "$model_extra" ]]; then
    read -ra mextra_arr <<< "$model_extra"
    cmd+=("${mextra_arr[@]}")
  fi

  log_command "vllm-serve [$(model_short_name "$model_path") / ${precision} / tp${tp_size}]" "${cmd[*]}"

  # Save the server command alongside the server log
  echo "${cmd[*]}" > "${server_log%.log}_cmd.txt"

  step "Starting vLLM server..."
  detail "Model     : ${model_path}"
  detail "Precision : $(precision_display_name "$precision")"
  detail "TP size   : ${tp_size}"
  if [[ "$ENABLE_COMPILATION" == "true" || "$ENABLE_COMPILATION" == "1" || "$ENABLE_COMPILATION" == "yes" ]] && [[ -n "$COMPILATION_CONFIG" ]]; then
    detail "Compilation : ON (--compilation-config)"
  fi
  if [[ -n "$MAX_MODEL_LEN" ]]; then
    detail "Max len   : ${MAX_MODEL_LEN}"
  else
    detail "Max len   : auto (from model config)"
  fi
  detail "Command   : ${cmd[*]}"
  detail "Log       : ${server_log}"

  if is_local_path "$model_path"; then
    info "Using local model: ${model_path}"
  else
    info "Model source: HuggingFace -- will download if not cached"
  fi

  # Run vllm serve in the BACKGROUND (trailing &) so this shell stays free to run
  # vllm bench serve later. No second terminal needed -- same process, background job.
  "${cmd[@]}" > "$server_log" 2>&1 &
  VLLM_SERVER_PID=$!
  detail "Server PID: ${VLLM_SERVER_PID}"

  # Wait for health
  wait_for_server "$server_log"
}

wait_for_server() {
  local server_log="$1"
  local url="http://localhost:${PORT}/health"
  local interval=3
  local last_line_count=0

  echo -e "${C_DIM}          -- Live server output --${C_RESET}"

  # -- Phase 1: Download (if needed) --
  # Detect whether a download is happening. If the model is cached/local,
  # this phase completes immediately and we go straight to Phase 2.
  local download_state="waiting"  # waiting | active | done
  local phase1_elapsed=0
  local last_download_activity=0

  step "Phase 1/2: Checking for model download..."

  while [[ "$download_state" != "done" ]]; do
    if ! kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
      stream_new_log_lines "$server_log" "$last_line_count" > /dev/null
      echo ""
      log_error "server-start" "$server_log" "Server process (PID ${VLLM_SERVER_PID}) exited prematurely during download phase."
      VLLM_SERVER_PID=""
      return 1
    fi

    last_line_count=$(stream_new_log_lines "$server_log" "$last_line_count")

    # Check if server is already healthy (model was cached, no download needed)
    if curl -sf "$url" >/dev/null 2>&1; then
      stream_new_log_lines "$server_log" "$last_line_count" > /dev/null
      echo -e "${C_DIM}          -- End server output --${C_RESET}"
      success "Server is healthy after ${phase1_elapsed}s (model was cached -- no download needed)."
      return 0
    fi

    # Check the log for download activity
    local has_download=false
    if [[ -f "$server_log" ]]; then
      if grep -q -i -E 'Downloading|downloading|Fetching|fetching' "$server_log" 2>/dev/null; then
        has_download=true
      fi
    fi

    case "$download_state" in
      waiting)
        if [[ "$has_download" == true ]]; then
          download_state="active"
          last_download_activity=$phase1_elapsed
          info "Model download detected -- waiting for download to complete..."
        fi

        # Also detect "Loading" without any prior download → model was cached
        if [[ "$has_download" == false && -f "$server_log" ]]; then
          if grep -q -i -E 'Loading model|loading weights|Loading checkpoint' "$server_log" 2>/dev/null; then
            download_state="done"
            detail "No download needed -- model is cached. Proceeding to server startup."
            continue
          fi
        fi

        if (( phase1_elapsed >= DOWNLOAD_START_TIMEOUT )); then
          # No download started AND server not healthy AND no loading detected
          # → assume model is cached but server is still initializing
          download_state="done"
          detail "No download activity detected within ${DOWNLOAD_START_TIMEOUT}s -- assuming model is cached."
          continue
        fi
        ;;

      active)
        local recent_download=false
        if [[ -f "$server_log" ]]; then
          if tail -n 5 "$server_log" 2>/dev/null | grep -q -i -E 'Downloading|downloading|%|Fetching|fetching'; then
            recent_download=true
          fi
        fi

        if [[ "$recent_download" == true ]]; then
          last_download_activity=$phase1_elapsed
        fi

        local idle_time=$(( phase1_elapsed - last_download_activity ))

        # Download finished: loading has started
        if grep -q -i -E 'Loading model|loading weights|Loading checkpoint' "$server_log" 2>/dev/null; then
          download_state="done"
          success "Download complete (took ~${phase1_elapsed}s). Proceeding to server startup."
          continue
        fi

        if (( idle_time >= DOWNLOAD_IDLE_TIMEOUT )); then
          stream_new_log_lines "$server_log" "$last_line_count" > /dev/null
          echo ""
          log_error "download" "$server_log" "Download appears stalled -- no progress for ${DOWNLOAD_IDLE_TIMEOUT}s."
          stop_vllm_server "download-stalled"
          return 1
        fi

        printf "\r${C_DIM}          Downloading... %ds elapsed, last activity %ds ago${C_RESET}" \
          "$phase1_elapsed" "$idle_time"
        ;;
    esac

    sleep "$interval"
    phase1_elapsed=$((phase1_elapsed + interval))
  done

  # -- Phase 2: Server startup (model loading + ready) --
  local phase2_elapsed=0

  step "Phase 2/2: Waiting for server to become healthy (timeout: ${SERVER_START_TIMEOUT}s)..."
  detail "Health endpoint: ${url}"

  while (( phase2_elapsed < SERVER_START_TIMEOUT )); do
    if ! kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
      stream_new_log_lines "$server_log" "$last_line_count" > /dev/null
      echo ""
      log_error "server-start" "$server_log" "Server process (PID ${VLLM_SERVER_PID}) exited prematurely during startup."
      VLLM_SERVER_PID=""
      return 1
    fi

    last_line_count=$(stream_new_log_lines "$server_log" "$last_line_count")

    if curl -sf "$url" >/dev/null 2>&1; then
      stream_new_log_lines "$server_log" "$last_line_count" > /dev/null
      echo -e "${C_DIM}          -- End server output --${C_RESET}"
      local total_time=$(( phase1_elapsed + phase2_elapsed ))
      success "Server is healthy! (download: ~${phase1_elapsed}s, startup: ${phase2_elapsed}s, total: ${total_time}s)"
      return 0
    fi

    sleep "$interval"
    phase2_elapsed=$((phase2_elapsed + interval))
    printf "\r${C_DIM}          Loading model... %ds / %ds${C_RESET}" "$phase2_elapsed" "$SERVER_START_TIMEOUT"
  done

  stream_new_log_lines "$server_log" "$last_line_count" > /dev/null
  echo ""
  local total_time=$(( phase1_elapsed + phase2_elapsed ))
  log_error "server-start" "$server_log" "Server did not become healthy within ${SERVER_START_TIMEOUT}s after download (total wait: ${total_time}s)."
  stop_vllm_server "timeout"
  return 1
}

stream_new_log_lines() {
  local log_file="$1"
  local prev_count="$2"

  if [[ ! -f "$log_file" ]]; then
    echo "$prev_count"
    return
  fi

  local current_count
  current_count=$(wc -l < "$log_file" 2>/dev/null | tr -d ' ')
  current_count="${current_count:-0}"

  if (( current_count > prev_count )); then
    local new_lines
    new_lines=$(( current_count - prev_count ))
    # Display lines go to fd 2 (stderr -> terminal); only the count goes to stdout
    tail -n "$new_lines" "$log_file" 2>/dev/null | while IFS= read -r line; do
      if [[ "$line" == *"Downloading"* || "$line" == *"downloading"* ]]; then
        echo -e "${C_YELLOW}    v     ${line}${C_RESET}" >&2
      elif [[ "$line" == *"Loading"* || "$line" == *"loading"* ]]; then
        echo -e "${C_CYAN}    *     ${line}${C_RESET}" >&2
      elif [[ "$line" == *"Error"* || "$line" == *"ERROR"* || "$line" == *"error:"* || "$line" == *"error["* || "$line" == *"Traceback"* || "$line" == *"Exception"* ]]; then
        echo -e "${C_RED}    x     ${line}${C_RESET}" >&2
      elif [[ "$line" == *"WARNING"* || "$line" == *"warning"* ]]; then
        echo -e "${C_YELLOW}    !     ${line}${C_RESET}" >&2
      elif [[ "$line" == *"Uvicorn"* || "$line" == *"Started"* || "$line" == *"ready"* ]]; then
        echo -e "${C_GREEN}    +     ${line}${C_RESET}" >&2
      elif [[ "$line" == *"%"* ]]; then
        printf "\r${C_DIM}    ...   %s${C_RESET}" "$line" >&2
      else
        echo -e "${C_DIM}          ${line}${C_RESET}" >&2
      fi
    done
  fi

  # Return the new count via stdout (only value captured by command substitution)
  echo "$current_count"
}

stop_vllm_server() {
  local reason="${1:-normal}"

  if [[ -z "$VLLM_SERVER_PID" ]]; then
    return 0
  fi

  if ! kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
    detail "Server (PID ${VLLM_SERVER_PID}) already stopped."
    VLLM_SERVER_PID=""
    return 0
  fi

  step "Stopping vLLM server (PID ${VLLM_SERVER_PID}, reason: ${reason})..."

  kill "$VLLM_SERVER_PID" 2>/dev/null || true

  local waited=0
  while (( waited < SERVER_STOP_TIMEOUT )); do
    if ! kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
      success "Server stopped gracefully after ${waited}s."
      VLLM_SERVER_PID=""
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done

  warn "Server did not stop gracefully -- sending SIGKILL..."
  kill -9 "$VLLM_SERVER_PID" 2>/dev/null || true
  sleep 2
  VLLM_SERVER_PID=""
  detail "Server killed."
}

# ====================================================================
# SECTION 13 -- Single Benchmark Run
# ====================================================================

bench_live_filter() {
  # Reads benchmark stdout line-by-line; shows result/summary lines on terminal,
  # suppresses the noisy per-request lines to keep output readable.
  while IFS= read -r line; do
    case "$line" in
      *"Throughput"*|*"throughput"*)
        echo -e "${C_GREEN}    >     ${line}${C_RESET}" ;;
      *"Total time"*|*"total time"*)
        echo -e "${C_CYAN}    >     ${line}${C_RESET}" ;;
      *"Request"*"completed"*|*"request"*"completed"*)
        echo -e "${C_DIM}    >     ${line}${C_RESET}" ;;
      *"ITL"*|*"TTFT"*|*"TPOT"*|*"latency"*|*"Latency"*)
        echo -e "${C_CYAN}    >     ${line}${C_RESET}" ;;
      *"token"*"per"*"second"*|*"tokens/s"*|*"tok/s"*)
        echo -e "${C_GREEN}    >     ${line}${C_RESET}" ;;
      *"Generating"*|*"Sending"*|*"Warmup"*|*"warmup"*)
        echo -e "${C_DIM}    ...   ${line}${C_RESET}" ;;
      *"Error"*|*"ERROR"*|*"error:"*|*"error["*|*"FAILED"*|*"Failed:"*|*"RuntimeError"*|*"Exception"*|*"Traceback"*)
        echo -e "${C_RED}    x     ${line}${C_RESET}" ;;
      *"====="*|*"-----"*|*"Result"*|*"result"*|*"Summary"*|*"summary"*)
        echo -e "${C_WHITE}    >     ${line}${C_RESET}" ;;
      *"%|"*|*"%|#"*)
        # Progress bars -- overwrite in place
        printf "\r${C_DIM}    ...   %s${C_RESET}" "$line" ;;
      *)
        # Suppress noisy per-request output
        ;;
    esac
  done
  echo ""  # newline after any \r progress lines
}

run_single_benchmark() {
  local model_path="$1"
  local concurrency="$2"
  local input_len="$3"
  local output_len="$4"
  local rep="$5"
  local bench_dir="$6"
  local precision_label="${7:-}"
  local tp_label="${8:-}"

  local iter="${9:-1}"
  local tag="c${concurrency}_in${input_len}_out${output_len}_rep${rep}"
  local full_tag
  if (( COMPLETE_REPEATS > 1 )); then
    full_tag="iter${iter}/$(model_short_name "$model_path")/${precision_label}_tp${tp_label}/${tag}"
  else
    full_tag="$(model_short_name "$model_path")/${precision_label}_tp${tp_label}/${tag}"
  fi

  # Resume: skip if this run already completed successfully
  if is_run_completed "$full_tag"; then
    CURRENT_RUN=$((CURRENT_RUN + 1))
    PASSED_RUNS=$((PASSED_RUNS + 1))
    detail "SKIP (already passed): ${full_tag}"
    return 0
  fi

  local bench_log="${bench_dir}/bench_${tag}.log"
  local bench_cmd_file="${bench_dir}/bench_${tag}_cmd.txt"

  local -a cmd=(
    vllm bench serve
    --backend "$BACKEND"
    --base-url "http://localhost:${PORT}"
    --model "$model_path"
    --dataset-name "$DATASET_NAME"
    --num-prompts "$NUM_PROMPTS"
    --max-concurrency "$concurrency"
    --random-input-len "$input_len"
    --random-output-len "$output_len"
  )

  if [[ -n "$EXTRA_BENCH_ARGS" ]]; then
    read -ra bench_extra_arr <<< "$EXTRA_BENCH_ARGS"
    cmd+=("${bench_extra_arr[@]}")
  fi

  log_command "bench-run [${tag}]" "${cmd[*]}"
  echo "${cmd[*]}" > "$bench_cmd_file"

  # Pre-flight: verify server is still alive before starting the benchmark
  if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    CURRENT_RUN=$((CURRENT_RUN + 1))
    FAILED_RUNS=$((FAILED_RUNS + 1))
    fail "${tag} -- server is not responding (crashed?). Skipping."
    log_error "benchmark:${tag}" "" "Server health check failed before benchmark start."
    printf "%-70s  rc=%-3s  dur=%5ss  %s\n" \
      "$full_tag" "N/A" "0" "$(date -Is)  [server-down]" >> "${RUN_DIR}/results_tracker.txt"
    return 0
  fi

  CURRENT_RUN=$((CURRENT_RUN + 1))
  progress_bar "$CURRENT_RUN" "$TOTAL_RUNS"
  echo ""
  detail "Run ${CURRENT_RUN}/${TOTAL_RUNS}: ${tag}"
  detail "Command: ${cmd[*]}"

  local start_s end_s rc dur
  start_s=$(date +%s)

  {
    echo "===================================================="
    echo "BENCHMARK: ${tag}"
    echo "COMMAND  : ${cmd[*]}"
    echo "STARTED  : $(date -Is)"
    echo "===================================================="
    echo ""
  } > "$bench_log"

  # Run benchmark: tee to log file, filter key lines to terminal.
  # pipefail ensures $? reflects the benchmark command's exit code, not tee/filter.
  if command -v stdbuf &>/dev/null; then
    timeout "$BENCHMARK_TIMEOUT" stdbuf -oL -eL "${cmd[@]}" 2>&1 \
      | tee -a "$bench_log" \
      | bench_live_filter
    rc=$?
  else
    timeout "$BENCHMARK_TIMEOUT" "${cmd[@]}" 2>&1 \
      | tee -a "$bench_log" \
      | bench_live_filter
    rc=$?
  fi

  end_s=$(date +%s)
  dur=$(( end_s - start_s ))

  {
    echo ""
    echo "===================================================="
    echo "FINISHED : $(date -Is)"
    echo "EXIT CODE: $rc"
    echo "DURATION : ${dur}s"
    echo "===================================================="
  } >> "$bench_log"

  if (( rc == 0 )); then
    PASSED_RUNS=$((PASSED_RUNS + 1))
    success "${tag} -- completed in ${dur}s"
  else
    FAILED_RUNS=$((FAILED_RUNS + 1))
    fail "${tag} -- exit code $rc after ${dur}s"
    log_error "benchmark:${tag}" "$bench_log" "Exit code $rc after ${dur}s."
  fi

  printf "%-70s  rc=%-3s  dur=%5ss  %s\n" \
    "$full_tag" "$rc" "$dur" "$(date -Is)" >> "${RUN_DIR}/results_tracker.txt"

  return 0  # never abort the loop
}

# ====================================================================
# SECTION 14 -- Main Orchestration Loop
# ====================================================================

run_benchmarks() {
  banner "Starting Benchmark Runs"

  for iter in $(seq 1 "$COMPLETE_REPEATS"); do

    if (( COMPLETE_REPEATS > 1 )); then
      banner "Complete Repeat Iteration ${iter}/${COMPLETE_REPEATS}"
    fi

    local model_idx=0

    for entry in "${MODEL_CONFIGS[@]}"; do
      model_idx=$((model_idx + 1))
      local model_path precs_str tps_str model_extra
      model_path=$(parse_model_path "$entry")
      read -ra precs <<< "$(parse_precisions "$entry")"
      read -ra tp_sizes <<< "$(parse_tp_sizes "$entry")"
      model_extra=$(parse_model_extra "$entry")

      local model_name
      model_name=$(model_short_name "$model_path")

      local base_dir
      if (( COMPLETE_REPEATS > 1 )); then
        base_dir="${RUN_DIR}/iter_${iter}"
        mkdir -p "$base_dir"
      else
        base_dir="${RUN_DIR}"
      fi

      local model_dir="${base_dir}/${model_name}"
      if [[ -d "$model_dir" ]]; then
        model_name="${model_name}_${model_idx}"
        model_dir="${base_dir}/${model_name}"
      fi
      mkdir -p "$model_dir"

      section "Model ${model_idx}/${#MODEL_CONFIGS[@]}: ${model_path}"

      for precision in "${precs[@]}"; do
        for tp in "${tp_sizes[@]}"; do
          local combo_name="${precision}_tp${tp}"
          local combo_dir="${model_dir}/${combo_name}"
          mkdir -p "$combo_dir"

          echo ""
          echo -e "  ${C_WHITE}>> Configuration: ${C_CYAN}${model_name}${C_WHITE} / ${C_MAGENTA}$(precision_display_name "$precision")${C_WHITE} / ${C_YELLOW}TP=${tp}${C_RESET}"
          echo ""

          # Resume: skip entire server start if all benchmarks in this combo are done
          if [[ -n "$RESUME_DIR" ]] && all_benchmarks_completed_for_combo "$model_path" "$precision" "$tp" "$iter"; then
            local bench_count=$(( ${#INPUT_LENS[@]} * ${#OUTPUT_LENS[@]} * ${#CONCURRENCIES[@]} * REPEATS ))
            PASSED_RUNS=$((PASSED_RUNS + bench_count))
            CURRENT_RUN=$((CURRENT_RUN + bench_count))
            info "All ${bench_count} benchmarks already completed for ${model_name}/${combo_name} -- skipping server start."
            continue
          fi

          local server_log="${combo_dir}/server.log"

          if ! start_vllm_server "$model_path" "$precision" "$tp" "$model_extra" "$server_log"; then
            local bench_count=$(( ${#INPUT_LENS[@]} * ${#OUTPUT_LENS[@]} * ${#CONCURRENCIES[@]} * REPEATS ))
            SKIPPED_RUNS=$((SKIPPED_RUNS + bench_count))
            CURRENT_RUN=$((CURRENT_RUN + bench_count))
            log_error "server:${model_name}/${combo_name}" "$server_log" \
              "Server failed to start. Skipping ${bench_count} benchmark run(s)."
            warn "Skipping ${bench_count} benchmarks for ${model_name}/${combo_name} (server failed)."
            continue
          fi

          # Run all benchmark combinations for this server instance
          for concurrency in "${CONCURRENCIES[@]}"; do
            for input_len in "${INPUT_LENS[@]}"; do
              for output_len in "${OUTPUT_LENS[@]}"; do
                for rep in $(seq 1 "$REPEATS"); do
                  run_single_benchmark \
                    "$model_path" "$concurrency" "$input_len" "$output_len" "$rep" "$combo_dir" \
                    "$precision" "$tp" "$iter"
                done
              done
            done
          done

          stop_vllm_server "combo-complete"
          echo ""

        done
      done
    done

  done
}

# ====================================================================
# SECTION 15 -- Final Report
# ====================================================================

generate_final_report() {
  local report_file="${RUN_DIR}/final_report.txt"
  local end_time
  end_time=$(date -Is)

  {
    echo "+=========================================================+"
    echo "|         vLLM Benchmark Suite -- Final Report              |"
    echo "+=========================================================+"
    echo ""
    echo "Completed at : ${end_time}"
    echo "Results dir  : ${RUN_DIR}"
    echo ""
    echo "-- Results --"
    echo ""
    echo "  Total benchmark runs : ${TOTAL_RUNS}"
    echo "  Passed               : ${PASSED_RUNS}"
    echo "  Failed               : ${FAILED_RUNS}"
    echo "  Skipped (server err) : ${SKIPPED_RUNS}"
    echo ""
    if [[ -f "${RUN_DIR}/results_tracker.txt" ]]; then
      echo "-- Per-Run Results --"
      echo ""
      cat "${RUN_DIR}/results_tracker.txt"
      echo ""
    fi
    if [[ -s "$ERROR_LOG" ]]; then
      echo "-- Errors --"
      echo ""
      cat "$ERROR_LOG"
    else
      echo "No errors recorded."
    fi
    echo ""
    echo "-- Directory Structure --"
    echo ""
    find "$RUN_DIR" -type f | sort | sed "s|${RUN_DIR}/||"
    echo ""
    echo "==========================================================="
  } > "$report_file"

  banner "Benchmark Complete"

  if (( FAILED_RUNS == 0 && SKIPPED_RUNS == 0 )); then
    echo -e "  ${C_GREEN}All ${TOTAL_RUNS} benchmark runs passed!${C_RESET}"
  else
    echo -e "  ${C_WHITE}Total runs  :${C_RESET} ${TOTAL_RUNS}"
    echo -e "  ${C_GREEN}Passed      :${C_RESET} ${PASSED_RUNS}"
    if (( FAILED_RUNS > 0 )); then
      echo -e "  ${C_RED}Failed      :${C_RESET} ${FAILED_RUNS}"
    fi
    if (( SKIPPED_RUNS > 0 )); then
      echo -e "  ${C_YELLOW}Skipped     :${C_RESET} ${SKIPPED_RUNS}"
    fi
  fi

  echo ""
  echo -e "  ${C_WHITE}Output directory :${C_RESET} ${RUN_DIR}"
  echo -e "  ${C_WHITE}Final report     :${C_RESET} ${report_file}"
  echo -e "  ${C_WHITE}Error log        :${C_RESET} ${ERROR_LOG}"
  echo -e "  ${C_WHITE}All commands     :${C_RESET} ${COMMANDS_LOG}"
  echo -e "  ${C_WHITE}Run plan         :${C_RESET} ${SUMMARY_FILE}"

  if [[ -f "${RUN_DIR}/results_tracker.txt" ]]; then
    echo ""
    echo -e "  ${C_CYAN}Quick results:${C_RESET}"
    echo ""
    while IFS= read -r line; do
      # Skip the header and separator lines
      [[ "$line" == RUN* || "$line" == -* ]] && continue
      [[ -z "$line" ]] && continue
      if [[ "$line" == *"rc=0  "* ]]; then
        echo -e "    ${C_GREEN}[PASS]${C_RESET} $line"
      else
        echo -e "    ${C_RED}[FAIL]${C_RESET} $line"
      fi
    done < "${RUN_DIR}/results_tracker.txt"
  fi

  echo ""
}

# ====================================================================
# SECTION 16 -- Help
# ====================================================================

show_help() {
  cat <<'HELPEOF'

  vLLM Benchmark Suite v1.0.0
  ---------------------------

  All-in-one tool for benchmarking vLLM across multiple models,
  precisions, tensor-parallel sizes, and load configurations.

  USAGE
    ./vllm_benchmark_suite.sh [OPTIONS]

  OPTIONS
    -c, --config FILE       Load configuration from FILE
    -r, --resume [DIR]      Resume an interrupted run (reuses output dir)
                            If DIR omitted, auto-detects the latest run_* dir
    -g, --generate-config   Generate a sample benchmark.conf
    -i, --interactive       Force interactive setup (default when no config)
    -y, --yes               Skip confirmation prompt
    -h, --help              Show this help message

  EXAMPLES
    # Interactive mode (guided setup)
    ./vllm_benchmark_suite.sh

    # Generate and edit a config file
    ./vllm_benchmark_suite.sh --generate-config
    vim benchmark.conf
    ./vllm_benchmark_suite.sh -c benchmark.conf

    # Run without confirmation
    ./vllm_benchmark_suite.sh -c benchmark.conf -y

    # Resume the most recent interrupted run
    ./vllm_benchmark_suite.sh -c benchmark.conf --resume

    # Resume a specific run directory
    ./vllm_benchmark_suite.sh -c benchmark.conf --resume ./benchmark_results/run_20260312_143000

  RESUME
    If a run is interrupted (Ctrl+C, crash, timeout), use --resume to pick
    up where you left off. The script reads results_tracker.txt from the
    previous run directory, identifies which benchmarks completed with rc=0,
    and skips them. Server starts are also skipped when all benchmarks in
    a model/precision/TP combo are already done.

    Requires the same config (-c) used for the original run so the full
    set of planned benchmarks is known. New results append to the same
    results_tracker.txt and logs directory.

  MODEL PATHS
    Both HuggingFace model IDs and local filesystem paths are supported:

      HuggingFace:  "meta-llama/Llama-3.1-8B-Instruct | bf16 | 1"
      Absolute:     "/data/models/my-model             | bf16 | 2"
      Relative:     "./models/finetuned-llama           | fp8  | 1"

    Local paths are validated at startup (directory exists + config.json check).
    HuggingFace IDs are downloaded/cached automatically by vLLM.

  CONFIG FILE FORMAT
    Run --generate-config to see a fully documented example.

    MODEL_CONFIGS entries use pipe-delimited fields:
      "model_path | precisions | tp_sizes | extra_args"

    Example:
      MODEL_CONFIGS=(
        "meta-llama/Llama-3.1-8B-Instruct | bf16,fp8 | 1,2"
        "/data/models/my-finetuned-model   | bf16     | 1"
        "mistralai/Mistral-7B-v0.3         | bf16     | 1   | --enforce-eager"
      )

  COMPLETE REPEATS (COMPLETE_REPEATS)
    Set COMPLETE_REPEATS=N in your config to repeat the entire benchmark
    suite N times end-to-end. Each iteration restarts servers and re-runs
    all benchmark combinations. Results are stored under iter_1/, iter_2/,
    etc. subdirectories. Default is 1 (no repetition).

    This differs from REPEATS, which repeats each individual benchmark
    combination N times before moving to the next one.

  SUPPORTED PRECISIONS
    bf16        BFloat16 (--dtype bfloat16)
    fp16        Float16  (--dtype float16)
    fp8         FP8 quantization (--quantization fp8)
    awq         AWQ int4 quantization
    gptq        GPTQ int4 quantization
    auto        Auto-detect (--dtype auto)

  OUTPUT STRUCTURE
    benchmark_results/
    +-- run_YYYYMMDD_HHMMSS/
        |-- summary.txt               Pre-run plan
        |-- final_report.txt           Post-run report
        |-- error.log                  All errors (separate)
        |-- commands.log               Every command executed
        |-- results_tracker.txt        One-line-per-run results
        +-- Model-Name/
            +-- bf16_tp1/
                |-- server.log         vLLM server output
                |-- server_cmd.txt     Server command used
                |-- bench_c128_in1024_out512_rep1.log
                |-- bench_c128_in1024_out512_rep1_cmd.txt
                +-- ...

HELPEOF
}

# ====================================================================
# SECTION 17 -- Main Entry Point
# ====================================================================

main() {
  local config_file=""
  local force_interactive=false
  local skip_confirm=false

  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -c|--config)
        if [[ -z "${2:-}" ]]; then
          err "Option $1 requires a config file argument."
          echo "  Example: $0 -c benchmark.conf"
          exit 1
        fi
        config_file="$2"
        shift 2
        ;;
      -r|--resume)
        if [[ -n "${2:-}" && "${2:-}" != -* ]]; then
          RESUME_DIR="$2"
          shift 2
        else
          RESUME_DIR="__auto__"
          shift
        fi
        ;;
      -g|--generate-config)
        local gen_file="benchmark.conf"
        if [[ -n "${2:-}" && "${2:-}" != -* ]]; then
          gen_file="$2"
        fi
        generate_sample_config "$gen_file"
        exit 0
        ;;
      -i|--interactive)
        force_interactive=true
        shift
        ;;
      -y|--yes)
        skip_confirm=true
        shift
        ;;
      -h|--help)
        show_help
        exit 0
        ;;
      *)
        err "Unknown option: $1"
        echo "  Run with --help for usage."
        exit 1
        ;;
    esac
  done

  # Welcome banner
  banner "vLLM Benchmark Suite v${SCRIPT_VERSION}"

  # Load configuration
  if [[ -n "$config_file" && "$force_interactive" == false ]]; then
    if [[ ! -f "$config_file" ]]; then
      err "Config file not found: ${config_file}"
      exit 1
    fi
    info "Loading config from: ${C_WHITE}${config_file}${C_RESET}"
    # shellcheck source=/dev/null
    source "$config_file"
  elif [[ "$force_interactive" == true || -z "$config_file" ]]; then
    interactive_setup
  fi

  # Validate
  section "Validation"
  validate_config

  # -- Resume or fresh run? --
  if [[ -n "$RESUME_DIR" ]]; then
    # Auto-detect latest run directory if no explicit path given
    if [[ "$RESUME_DIR" == "__auto__" ]]; then
      RESUME_DIR=$(find_latest_run_dir "$BASE_OUT_DIR")
      if [[ -z "$RESUME_DIR" ]]; then
        err "No previous run directory found in ${BASE_OUT_DIR}. Cannot resume."
        exit 1
      fi
    fi

    if [[ ! -d "$RESUME_DIR" ]]; then
      err "Resume directory does not exist: ${RESUME_DIR}"
      exit 1
    fi

    RUN_DIR="$RESUME_DIR"
    ERROR_LOG="${RUN_DIR}/error.log"
    COMMANDS_LOG="${RUN_DIR}/commands.log"
    SUMMARY_FILE="${RUN_DIR}/summary.txt"
    touch "$ERROR_LOG" "$COMMANDS_LOG"

    info "Resuming from: ${C_WHITE}${RUN_DIR}${C_RESET}"

    # Load previously completed runs from results_tracker.txt
    load_completed_runs "${RUN_DIR}/results_tracker.txt"

    # Append a resume marker to the commands log
    {
      echo ""
      echo "==========================================================="
      echo "  RESUMED: $(date -Is)"
      echo "  Previously completed runs: ${RESUMED_RUNS}"
      echo "==========================================================="
      echo ""
    } >> "$COMMANDS_LOG"

  else
    # Fresh run
    RUN_DIR="${BASE_OUT_DIR}/run_${TIMESTAMP}"
    mkdir -p "$RUN_DIR"
    ERROR_LOG="${RUN_DIR}/error.log"
    COMMANDS_LOG="${RUN_DIR}/commands.log"
    SUMMARY_FILE="${RUN_DIR}/summary.txt"
    touch "$ERROR_LOG" "$COMMANDS_LOG" "$SUMMARY_FILE"
    {
      printf "%-70s  %-5s  %-9s  %s\n" "RUN" "RC" "DURATION" "TIMESTAMP"
      printf '%.0s-' {1..110}
      echo ""
    } > "${RUN_DIR}/results_tracker.txt"

    # Record environment in commands log
    {
      echo "==========================================================="
      echo "  vLLM Benchmark Suite -- Command Log"
      echo "  Started: $(date -Is)"
      echo "  Host: $(hostname)"
      echo "==========================================================="
      echo ""
    } > "$COMMANDS_LOG"
  fi

  # Show summary
  show_summary

  # Confirm
  if [[ "$skip_confirm" == false ]]; then
    confirm_proceed
  fi

  # Run
  local suite_start
  suite_start=$(date +%s)

  run_benchmarks

  local suite_end suite_dur
  suite_end=$(date +%s)
  suite_dur=$(( suite_end - suite_start ))

  info "Total wall-clock time: ${suite_dur}s ($(( suite_dur / 60 ))m $(( suite_dur % 60 ))s)"

  # Append to summary
  {
    echo ""
    echo "-- Execution Results --"
    echo "  Wall-clock time : ${suite_dur}s"
    echo "  Passed          : ${PASSED_RUNS}"
    echo "  Failed          : ${FAILED_RUNS}"
    echo "  Skipped         : ${SKIPPED_RUNS}"
  } >> "$SUMMARY_FILE"

  generate_final_report
}

main "$@"
