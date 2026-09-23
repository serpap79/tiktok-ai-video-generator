#!/usr/bin/env bash
#
# Sobe o renderizador (MoneyPrinterTurbo) SEM atualizar nem reinstalar nada.
#
# E o que o servico do systemd chama. O setup_renderer.sh --serve faz git pull,
# uv sync e reescreve chaves no .env a cada subida: certo para instalar, errado
# para um servico que reinicia sozinho -- um pull do upstream as 3h da manha
# poderia quebrar o render das 9h, e o .env tambem e escrito pela renovacao do
# token do TikTok. Aqui so se le o .env e se executa.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RENDERER_DIR="$ROOT/.renderer"
ENV_FILE="$ROOT/.env"

[[ -d "$RENDERER_DIR" ]] || { echo "renderizador ausente: rode ./scripts/setup_renderer.sh" >&2; exit 1; }

# So a variavel do ffmpeg interessa ao MPT; o resto do .env fica fora do
# ambiente dele.
if [[ -f "$ENV_FILE" ]]; then
  linha="$(grep -m1 '^IMAGEIO_FFMPEG_EXE=' "$ENV_FILE" || true)"
  [[ -n "$linha" ]] && export IMAGEIO_FFMPEG_EXE="${linha#IMAGEIO_FFMPEG_EXE=}"
fi

cd "$RENDERER_DIR"
exec uv run --python 3.12 python main.py
