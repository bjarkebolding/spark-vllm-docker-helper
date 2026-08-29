#!/usr/bin/env bash
# Run Qwen3.8-Flash-Next (NVFP4) on a single DGX Spark via an UNMODIFIED
# eugr/spark-vllm-docker checkout. https://github.com/bjarkebolding/spark-vllm-docker-helper
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$HERE/recipes/qwen3.8-flash-next-nvfp4.yaml"
UPSTREAM="https://github.com/eugr/spark-vllm-docker.git"
SPARK_DIR="${SPARK_VLLM_DOCKER:-$HERE/.spark-vllm-docker}"
CONTAINER="${SPARK_VLLM_CONTAINER:-vllm_node}"
PORT="${PORT:-8000}"

# Run-time overrides (env). Unset = recipe default.
GPU_MEM="${GPU_MEM:-}"          # gpu_memory_utilization
MTP="${MTP:-}"                  # num_speculative_tokens
CTX="${CTX:-}"                  # max_model_len
MAX_SEQS="${MAX_SEQS:-}"        # max_num_seqs
PLE_WORKERS="${PLE_WORKERS:-}"  # VLLM_PLE_MMAP_WORKERS
PLE_READAHEAD="${PLE_READAHEAD:-}"
MOE_BACKEND="${MOE_BACKEND:-}"  # -> --moe-backend
SPEC_CONFIG="${SPEC_CONFIG:-}"  # full --speculative-config JSON override
PLE_MOD="${PLE_MOD:-0}"   # 1 = apply mods/qwen4-exp-ple-cache (opt-in; see RESULTS.md)
DROP_CACHES="${DROP_CACHES:-1}" # 1 = one-shot drop_caches before launch (fills PLE prewarm)
EXTRA="${EXTRA:-}"              # extra vllm flags, appended verbatim

upstream() {
  if [ ! -d "$SPARK_DIR/.git" ]; then
    echo ">> cloning $UPSTREAM -> $SPARK_DIR"
    git clone --depth 1 "$UPSTREAM" "$SPARK_DIR"
  fi
  echo ">> spark-vllm-docker: $(git -C "$SPARK_DIR" rev-parse --short HEAD) (unmodified)"
}

# Build an effective recipe with env overrides applied (sed on a temp copy).
build_recipe() {
  local out; out="$(mktemp /tmp/flash-next-recipe.XXXXXX.yaml)"
  cp "$RECIPE" "$out"
  [ -n "$MTP" ]           && sed -i "s/^  num_speculative_tokens: .*/  num_speculative_tokens: $MTP/" "$out"
  [ -n "$MAX_SEQS" ]      && sed -i "s/^  max_num_seqs: .*/  max_num_seqs: $MAX_SEQS/" "$out"
  [ -n "$PLE_WORKERS" ]   && sed -i "s/^  VLLM_PLE_MMAP_WORKERS: .*/  VLLM_PLE_MMAP_WORKERS: \"$PLE_WORKERS\"/" "$out"
  [ -n "$PLE_READAHEAD" ] && sed -i "s/^  VLLM_PLE_MMAP_READAHEAD: .*/  VLLM_PLE_MMAP_READAHEAD: \"$PLE_READAHEAD\"/" "$out"
  if [ -n "$SPEC_CONFIG" ]; then
    # run-recipe.py str.format()s the command, so literal JSON braces must be doubled.
    local sc="${SPEC_CONFIG//\{/\{\{}"; sc="${sc//\}/\}\}}"
    sed -i "s|--speculative-config '.*' \\\\|--speculative-config '$sc' \\\\|" "$out"
  fi
  echo "$out"
}

launch() {
  upstream
  [ "$DROP_CACHES" = "1" ] && { echo ">> dropping page cache (one-shot)"; sync; \
    (echo 3 > /proc/sys/vm/drop_caches) 2>/dev/null || sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "   (skipped, needs root)"; }
  local rp; rp="$(build_recipe)"
  local args=("$rp" --solo)
  [ -n "$GPU_MEM" ] && args+=(--gpu-mem "$GPU_MEM")
  [ -n "$CTX" ]     && args+=(--max-model-len "$CTX")
  [ "$PLE_MOD" = "1" ] && [ -d "$HERE/mods/qwen4-exp-ple-cache" ] && args+=(--apply-mod "$HERE/mods/qwen4-exp-ple-cache")
  args+=("$@")
  local extra=()
  [ -n "$MOE_BACKEND" ] && extra+=(--moe-backend "$MOE_BACKEND")
  [ -n "$EXTRA" ] && extra+=($EXTRA)
  [ ${#extra[@]} -gt 0 ] && args+=(-- "${extra[@]}")
  echo ">> run-recipe.sh ${args[*]}"
  "$SPARK_DIR/run-recipe.sh" "${args[@]}"
  rm -f "$rp"
}

case "${1:-help}" in
  setup)    shift; launch --setup --earlyoom -d "$@" ;;
  serve)    shift; launch --earlyoom -d "$@" ;;
  build)    shift; upstream; exec "$SPARK_DIR/run-recipe.sh" "$RECIPE" --solo --build-only "$@" ;;
  download) shift; upstream; exec "$SPARK_DIR/run-recipe.sh" "$RECIPE" --solo --download-only "$@" ;;
  dry-run)  shift; DROP_CACHES=0 launch --dry-run "$@" ;;
  logs)     exec docker logs -f "$CONTAINER" ;;
  stop)     exec docker rm -f "$CONTAINER" ;;
  test)     exec python3 "$HERE/tools/bench.py" --port "$PORT" --mode smoke ;;
  bench)    shift; exec python3 "$HERE/tools/bench.py" --port "$PORT" "$@" ;;
  warmup)   shift; exec python3 "$HERE/tools/bench.py" --port "$PORT" --mode warmup "$@" ;;
  stats)    docker logs "$CONTAINER" 2>&1 | tr '\r' '\n' | grep -E "ple_mmap.py.*copy_ms|SpecDecoding|GPU KV cache size|generation throughput" | tail -20 ;;
  *)
    cat <<EOF
flash-next.sh <command>

  setup       build image + download model (~126 GiB) + serve
  serve       serve (image + model already present)
  build       build the image only
  download    download the model only
  warmup      drive synthetic requests to compile the QSA/indexer kernels
  bench       benchmark a running server (--mode prose|random|needle|all)
  test        one smoke request
  stats       recent PLE / spec-decode / KV metrics from the container log
  logs        follow the vLLM container logs
  dry-run     print the launch command, run nothing
  stop        remove the vLLM container

run-time env overrides:
  GPU_MEM=0.88   MTP=2   CTX=262144   MAX_SEQS=8
  PLE_WORKERS=1  PLE_READAHEAD=2048   MOE_BACKEND=            SPEC_CONFIG='{...}'
  PLE_MOD=1   DROP_CACHES=1        EXTRA='--flag ...'
  SPARK_VLLM_DOCKER=<path>   PORT=<n>
EOF
    ;;
esac
