#!/usr/bin/env bash
#
# Instala el piloto automático como servicios de usuario del systemd.
#
#   ./scripts/install_autopilot.sh            instala y enciende (renderizador + 4 timers)
#   ./scripts/install_autopilot.sh --remove   apaga y elimina
#
# Lo que queda encendido:
#   circuitocero-renderer.service   MoneyPrinterTurbo en 127.0.0.1:8080, se reinicia solo
#   circuitocero-1000.timer         09:25 -> produce y publica a las 10:00 (Madrid), corto
#   circuitocero-1300.timer         12:25 -> 13:00, largo
#   circuitocero-1700.timer         16:25 -> 17:00, corto
#   circuitocero-2000.timer         19:25 -> 20:00, largo
#
# Logs: journalctl --user -u 'circuitocero-*' -f
# Día:  uv run agent autopilot-status --detail
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNITS=(circuitocero-renderer.service circuitocero-slot@.service
       circuitocero-1000.timer circuitocero-1300.timer
       circuitocero-1700.timer circuitocero-2000.timer)
TIMERS=(circuitocero-1000.timer circuitocero-1300.timer
        circuitocero-1700.timer circuitocero-2000.timer)
# Timers de la parrilla vieja (3 posts: 09/15/20) y de las unidades con el nombre
# antiguo. Quedan en la lista de remoción para que reinstalar no deje unidades
# viejas disparando en paralelo con la parrilla nueva -- viven en ~/.config y no
# desaparecen solas.
ANTIGUOS=(circuitocero-0900.timer circuitocero-1200.timer circuitocero-1500.timer
          circuitocero-1900.timer
          seucanal-renderer.service seucanal-slot@.service
          seucanal-0900.timer seucanal-1200.timer
          seucanal-1500.timer seucanal-1900.timer seucanal-2000.timer)

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33maviso:\033[0m %s\n' "$*"; }

if [[ "${1:-}" == "--remove" ]]; then
  systemctl --user disable --now "${TIMERS[@]}" "${ANTIGUOS[@]}" \
    circuitocero-renderer.service 2>/dev/null || true
  for u in "${UNITS[@]}" "${ANTIGUOS[@]}"; do rm -f "$UNIT_DIR/$u"; done
  systemctl --user daemon-reload
  log "piloto automático eliminado"
  exit 0
fi

mkdir -p "$UNIT_DIR"
# Apaga la parrilla vieja ANTES de instalar la nueva: un timer huérfano en ~/.config
# sigue disparando y produciría un post fuera de parrilla cada día.
for u in "${ANTIGUOS[@]}"; do
  if [[ -f "$UNIT_DIR/$u" ]]; then
    systemctl --user disable --now "$u" 2>/dev/null || true
    rm -f "$UNIT_DIR/$u"
    log "unidad vieja eliminada: $u"
  fi
done
for u in "${UNITS[@]}"; do
  sed -e "s|@ROOT@|$ROOT|g" -e "s|@HOME@|$HOME|g" "$ROOT/scripts/systemd/$u" > "$UNIT_DIR/$u"
done
systemctl --user daemon-reload
log "unidades instaladas en $UNIT_DIR"

# Renderizador: si ya hay uno corriendo a mano en la 8080, el servicio solo se
# habilita (asume en el próximo boot) -- levantarlo ahora pelearía por el puerto.
puerto_ocupado=0
if curl -s -o /dev/null --max-time 3 http://127.0.0.1:8080/ping; then puerto_ocupado=1; fi
if systemctl --user is-active --quiet circuitocero-renderer.service; then puerto_ocupado=0; fi
if [[ "$puerto_ocupado" -eq 1 ]]; then
  systemctl --user enable circuitocero-renderer.service
  warn "ya hay un renderizador corriendo a mano en la 8080: el servicio fue habilitado y"
  warn "asume en el próximo boot. Para cambiarlo ahora: para el manual (Ctrl-C) y corre"
  warn "  systemctl --user start circuitocero-renderer.service"
else
  systemctl --user enable --now circuitocero-renderer.service
  log "renderizador encendido como servicio"
fi

systemctl --user enable --now "${TIMERS[@]}"
log "timers encendidos"

# Sin linger, los servicios de usuario solo corren con la sesión abierta: tras un
# reboot sin login, ningún slot dispararía.
if loginctl enable-linger "$USER" 2>/dev/null; then
  log "linger encendido: los timers corren incluso sin sesión abierta"
else
  warn "no pude encender el linger; corre: sudo loginctl enable-linger $USER"
fi

systemctl --user list-timers 'circuitocero-*' --no-pager
