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

upstream() {
  if [ ! -d "$SPARK_DIR/.git" ]; then
    echo ">> cloning $UPSTREAM -> $SPARK_DIR"
    git clone --depth 1 "$UPSTREAM" "$SPARK_DIR"
  fi
  echo ">> spark-vllm-docker: $(git -C "$SPARK_DIR" rev-parse --short HEAD) (unmodified)"
}

recipe() { upstream; exec "$SPARK_DIR/run-recipe.sh" "$RECIPE" --solo "$@"; }

case "${1:-help}" in
  setup)    shift; recipe --setup --earlyoom -d "$@" ;;   # build image + download model + serve
  serve)    shift; recipe --earlyoom -d "$@" ;;           # serve (image + model already present)
  build)    shift; recipe --build-only "$@" ;;
  download) shift; recipe --download-only "$@" ;;
  dry-run)  shift; recipe --dry-run "$@" ;;
  logs)     exec docker logs -f "$CONTAINER" ;;
  stop)     exec docker rm -f "$CONTAINER" ;;
  test)
    curl -fsS -m 120 "http://localhost:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"List the first 10 prime numbers."}],"max_tokens":80,"chat_template_kwargs":{"enable_thinking":false}}' \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
    ;;
  *)
    cat <<EOF
flash-next.sh <command>

  setup       build the image + download the model (~126 GiB) + serve
  serve       serve (when the image and model are already present)
  build       build the image only
  download    download the model only
  dry-run     print the generated launch command, run nothing
  logs        follow the vLLM container logs
  test        send one chat request to a running server
  stop        remove the vLLM container

env: SPARK_VLLM_DOCKER=<path>  reuse an existing spark-vllm-docker checkout
     PORT=<n>                  test/serve port (default 8000)
EOF
    ;;
esac
