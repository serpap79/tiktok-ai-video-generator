#!/usr/bin/env bash
#
# Gera material de teste sintetico em fixtures/material/.
#
# Sirve para probar la cadena de producción (narración, subtítulos y montaje) sin
# chave de API e sem rede. Os arquivos sao gerados em vez de versionados: sao
# ~9 MB de gradiente que o ffmpeg reproduz em segundos.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/fixtures/material"

command -v ffmpeg >/dev/null || { echo "ffmpeg no está en el PATH" >&2; exit 1; }

# O Fedora distribui ffmpeg sem libx264 (ver README). libopenh264 esta presente
# nas duas variantes e basta para material de teste.
ENCODERS="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"
if [[ "$ENCODERS" == *" libx264 "* ]]; then
  CODEC=(-c:v libx264 -preset veryfast -crf 28)
elif [[ "$ENCODERS" == *" libopenh264 "* ]]; then
  CODEC=(-c:v libopenh264 -b:v 2M)
else
  echo "nenhum encoder H.264 disponivel no ffmpeg do sistema" >&2
  exit 1
fi

mkdir -p "$OUT"

# Cores distintas por clipe para dar para ver o corte entre eles no video final.
gerar() {
  local nome="$1" c0="$2" c1="$3"
  ffmpeg -y -v error \
    -f lavfi -i "gradients=s=1080x1920:c0=$c0:c1=$c1:type=radial:speed=0.05:d=12" \
    -vf "noise=alls=6:allf=t,format=yuv420p" -r 30 -t 12 \
    "${CODEC[@]}" "$OUT/$nome"
  echo "  $nome"
}

echo "gerando material de teste em fixtures/material/"
gerar placeholder-1.mp4 0x1a1a2e 0x16213e
gerar placeholder-2.mp4 0x0f3460 0x533483
gerar placeholder-3.mp4 0x2d1b4e 0x0f3460

cat <<'EOF'

pronto. Para renderizar sem nenhuma chave de API:

  AGENT_VIDEO_SOURCE=local \
  AGENT_LOCAL_MATERIALS='["fixtures/material/placeholder-1.mp4","fixtures/material/placeholder-2.mp4","fixtures/material/placeholder-3.mp4"]' \
  uv run agent render --script fixtures/guion_manual.json
EOF
