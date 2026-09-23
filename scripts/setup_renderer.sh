#!/usr/bin/env bash
#
# Prepara e sobe o renderizador.
#
# O renderizador e o MoneyPrinterTurbo (MIT, github.com/harry0703/MoneyPrinterTurbo)
# rodando como servico local. Nao versionamos o codigo dele: e dependencia externa,
# clonada em .renderer/ (git-ignored). Nosso agente fala com ele por HTTP, atras da
# porta Renderer, entao trocar de renderizador depois nao toca em nenhum estagio.
#
#   ./scripts/setup_renderer.sh           instala e configura
#   ./scripts/setup_renderer.sh --serve   instala, configura e sobe o servidor
#
set -euo pipefail

REPO_URL="https://github.com/harry0703/MoneyPrinterTurbo.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RENDERER_DIR="$ROOT/.renderer"
ENV_FILE="$ROOT/.env"
PYTHON_VERSION="3.12"
SERVE=0

[[ "${1:-}" == "--serve" ]] && SERVE=1

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merro:\033[0m %s\n' "$*" >&2; exit 1; }

# Segredos moram no .env (git-ignored), nunca neste arquivo: ele e versionado num
# repo publico. O `source` sobrescreveria uma variavel ja exportada, entao o valor
# vindo do ambiente e guardado antes e restaurado depois -- assim
# `PEXELS_API_KEY=outra ./scripts/setup_renderer.sh` continua valendo como override.
PEXELS_FROM_ENV="${PEXELS_API_KEY:-}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
[[ -n "$PEXELS_FROM_ENV" ]] && PEXELS_API_KEY="$PEXELS_FROM_ENV"
PEXELS_API_KEY="${PEXELS_API_KEY:-}"

command -v uv >/dev/null || die "uv nao encontrado. Instale com: sudo dnf install uv"
command -v ffmpeg >/dev/null || die "ffmpeg nao encontrado no PATH"

# ---------------------------------------------------------------- clone/update

if [[ -d "$RENDERER_DIR/.git" ]]; then
  log "atualizando renderizador em .renderer/"
  git -C "$RENDERER_DIR" pull --ff-only --quiet || log "pull falhou; seguindo com a copia local"
else
  log "clonando MoneyPrinterTurbo em .renderer/"
  git clone --depth 1 --quiet "$REPO_URL" "$RENDERER_DIR"
fi

# ---------------------------------------------------------------- dependencias

# Python 3.12 e deliberado: o sistema traz 3.14, e as dependencias pinadas do MPT
# (ctranslate2, moviepy, azure-speech) nem sempre publicam wheel para a versao mais
# nova. O uv baixa o 3.12 sem precisar de root.
log "instalando dependencias do renderizador (Python $PYTHON_VERSION)"
( cd "$RENDERER_DIR" && uv sync --python "$PYTHON_VERSION" --quiet )

# ------------------------------------------------------------------ ffmpeg

# O Fedora distribui `ffmpeg-free`, compilado sem os codecs sob patente: nao tem
# libx264, que e o encoder padrao do MoviePy e do passo de concatenacao do MPT.
# Sem isso o render morre no fim, depois de gastar todo o TTS e a montagem.
#
# O imageio-ffmpeg, que ja vem como dependencia do MoviePy, traz um binario
# estatico com libx264. `utils.get_ffmpeg_binary()` do MPT honra
# IMAGEIO_FFMPEG_EXE antes do PATH, entao apontar a variavel resolve os dois
# caminhos (MoviePy e concatenacao) sem tocar numa linha do codigo dele.
# `grep -q` fecha o pipe cedo e mata o ffmpeg com SIGPIPE; com `set -o pipefail`
# isso reprovaria um binario que na verdade tem o encoder. Por isso a saida e
# capturada numa variavel antes de ser testada.
has_libx264() {
  local encoders
  encoders="$("$1" -hide_banner -encoders 2>/dev/null || true)"
  [[ "$encoders" == *" libx264 "* ]]
}

if has_libx264 ffmpeg; then
  BUNDLED_FFMPEG=""
  log "ffmpeg do sistema tem libx264; usando ele"
else
  BUNDLED_FFMPEG="$(find "$RENDERER_DIR/.venv" -path '*imageio_ffmpeg/binaries/ffmpeg*' -type f 2>/dev/null | head -1)"
  [[ -n "$BUNDLED_FFMPEG" ]] \
    || die "ffmpeg do sistema nao tem libx264 e o binario do imageio-ffmpeg nao foi encontrado"
  has_libx264 "$BUNDLED_FFMPEG" \
    || die "nem o ffmpeg do sistema nem o do imageio-ffmpeg tem libx264"
  log "ffmpeg do sistema sem libx264; usando o binario do imageio-ffmpeg"
fi

# ---------------------------------------------------------------- configuracao

CONFIG="$RENDERER_DIR/config.toml"
if [[ -f "$CONFIG" ]]; then
  log "config.toml ja existe; preservando (apague para regerar)"
  API_KEY="$(grep -m1 '^api_key' "$CONFIG" | sed 's/.*= *"\(.*\)"/\1/')"
else
  log "gerando config.toml"
  API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  cp "$RENDERER_DIR/config.example.toml" "$CONFIG"

  # 0.0.0.0 expoe o renderizador para a rede local inteira. Ele nao precisa
  # disso: quem fala com ele e o agente, nesta mesma maquina.
  sed -i 's/^listen_host = .*/listen_host = "127.0.0.1"/' "$CONFIG"

  # Autenticacao ligada mesmo em localhost: custa nada e evita que qualquer
  # processo local dispare renderizacao.
  sed -i "0,/^api_key = .*/s||api_key = \"$API_KEY\"|" "$CONFIG"

  # Legenda pelo edge-tts: os tempos por palavra vem do proprio TTS, o que
  # dispensa baixar e rodar o Whisper large-v3 (lento em CPU).
  sed -i 's/^subtitle_provider = .*/subtitle_provider = "edge"/' "$CONFIG"
fi

# A chave do Pexels e aplicada sempre, e nao so quando o config.toml nasce: quem ja
# rodou o setup antes de ter a chave precisa conseguir adiciona-la depois.
if [[ -n "$PEXELS_API_KEY" ]]; then
  sed -i "s|^pexels_api_keys = .*|pexels_api_keys = [\"$PEXELS_API_KEY\"]|" "$CONFIG"
  log "chave do Pexels gravada em config.toml"
else
  printf '\033[1;33maviso:\033[0m PEXELS_API_KEY nao definida.\n'
  printf '  Cadastre-se de graca em https://www.pexels.com/api/ e rode:\n'
  printf '    PEXELS_API_KEY=xxx ./scripts/setup_renderer.sh\n'
  printf '  A chave fica no .env (git-ignored), nunca neste script.\n'
fi

# ---------------------------------------------------------------- .env do agente

touch "$ENV_FILE"
if grep -q '^AGENT_RENDERER_API_KEY=' "$ENV_FILE"; then
  sed -i "s|^AGENT_RENDERER_API_KEY=.*|AGENT_RENDERER_API_KEY=$API_KEY|" "$ENV_FILE"
else
  printf 'AGENT_RENDERER_API_KEY=%s\n' "$API_KEY" >> "$ENV_FILE"
fi
log "chave do renderizador gravada em .env"

if [[ -n "$PEXELS_API_KEY" ]]; then
  if grep -q '^PEXELS_API_KEY=' "$ENV_FILE"; then
    sed -i "s|^PEXELS_API_KEY=.*|PEXELS_API_KEY=$PEXELS_API_KEY|" "$ENV_FILE"
  else
    printf 'PEXELS_API_KEY=%s\n' "$PEXELS_API_KEY" >> "$ENV_FILE"
  fi
  log "chave do Pexels persistida no .env; nas proximas vezes nao precisa passar"
fi

# ---------------------------------------------------------------- servir

if [[ -n "$BUNDLED_FFMPEG" ]]; then
  if grep -q '^IMAGEIO_FFMPEG_EXE=' "$ENV_FILE"; then
    sed -i "s|^IMAGEIO_FFMPEG_EXE=.*|IMAGEIO_FFMPEG_EXE=$BUNDLED_FFMPEG|" "$ENV_FILE"
  else
    printf 'IMAGEIO_FFMPEG_EXE=%s\n' "$BUNDLED_FFMPEG" >> "$ENV_FILE"
  fi
fi

if [[ "$SERVE" -eq 1 ]]; then
  log "subindo renderizador em http://127.0.0.1:8080 (Ctrl-C para parar)"
  cd "$RENDERER_DIR"
  [[ -n "$BUNDLED_FFMPEG" ]] && export IMAGEIO_FFMPEG_EXE="$BUNDLED_FFMPEG"
  exec uv run --python "$PYTHON_VERSION" python main.py
fi

log "pronto. Para subir o servidor:"
printf '    ./scripts/setup_renderer.sh --serve\n'
printf '  Depois, em outro terminal:\n'
printf '    uv run agent render --script fixtures/roteiro_manual.json\n'
