#!/usr/bin/env bash
#
# Instala o piloto automatico como servicos de usuario do systemd.
#
#   ./scripts/install_autopilot.sh            instala e liga (renderizador + 4 timers)
#   ./scripts/install_autopilot.sh --remove   desliga e remove
#
# O que fica ligado:
#   seucanal-renderer.service   MoneyPrinterTurbo em 127.0.0.1:8080, reinicia sozinho
#   seucanal-0900.timer         08:25 -> produz e publica as 09:00 (Brasilia), curto
#   seucanal-1200.timer         11:25 -> 12:00, longo
#   seucanal-1600.timer         15:25 -> 16:00, curto
#   seucanal-1900.timer         18:25 -> 19:00, longo
#
# Logs: journalctl --user -u 'seucanal-*' -f
# Dia:  uv run agent autopilot-status --detail
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNITS=(seucanal-renderer.service seucanal-slot@.service
       seucanal-0900.timer seucanal-1200.timer
       seucanal-1600.timer seucanal-1900.timer)
TIMERS=(seucanal-0900.timer seucanal-1200.timer
        seucanal-1600.timer seucanal-1900.timer)
# Timers da grade antiga (3 posts: 09/15/20). Ficam na lista de remocao para
# que reinstalar nao deixe o slot das 15h e o das 20h disparando em paralelo
# com a grade nova -- eles vivem em ~/.config e nao somem sozinhos.
ANTIGOS=(seucanal-1500.timer seucanal-2000.timer)

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33maviso:\033[0m %s\n' "$*"; }

if [[ "${1:-}" == "--remove" ]]; then
  systemctl --user disable --now "${TIMERS[@]}" "${ANTIGOS[@]}" \
    seucanal-renderer.service 2>/dev/null || true
  for u in "${UNITS[@]}" "${ANTIGOS[@]}"; do rm -f "$UNIT_DIR/$u"; done
  systemctl --user daemon-reload
  log "piloto automatico removido"
  exit 0
fi

mkdir -p "$UNIT_DIR"
# Apaga a grade antiga ANTES de instalar a nova: um timer orfao em ~/.config
# continua disparando e produziria um post fora da grade todo dia.
for u in "${ANTIGOS[@]}"; do
  if [[ -f "$UNIT_DIR/$u" ]]; then
    systemctl --user disable --now "$u" 2>/dev/null || true
    rm -f "$UNIT_DIR/$u"
    log "timer da grade antiga removido: $u"
  fi
done
for u in "${UNITS[@]}"; do
  sed -e "s|@ROOT@|$ROOT|g" -e "s|@HOME@|$HOME|g" "$ROOT/scripts/systemd/$u" > "$UNIT_DIR/$u"
done
systemctl --user daemon-reload
log "unidades instaladas em $UNIT_DIR"

# Renderizador: se ja ha um rodando a mao na porta 8080, o servico so e
# habilitado (assume no proximo boot) -- subir agora brigaria pela porta.
porta_ocupada=0
if curl -s -o /dev/null --max-time 3 http://127.0.0.1:8080/ping; then porta_ocupada=1; fi
if systemctl --user is-active --quiet seucanal-renderer.service; then porta_ocupada=0; fi
if [[ "$porta_ocupada" -eq 1 ]]; then
  systemctl --user enable seucanal-renderer.service
  warn "ja ha um renderizador rodando a mao na 8080: o servico foi habilitado e"
  warn "assume no proximo boot. Para trocar agora: pare o manual (Ctrl-C) e rode"
  warn "  systemctl --user start seucanal-renderer.service"
else
  systemctl --user enable --now seucanal-renderer.service
  log "renderizador ligado como servico"
fi

systemctl --user enable --now "${TIMERS[@]}"
log "timers ligados"

# Sem linger, os servicos de usuario so rodam com sessao aberta: depois de um
# reboot sem login, nenhum slot dispararia.
if loginctl enable-linger "$USER" 2>/dev/null; then
  log "linger ligado: os timers rodam mesmo sem sessao aberta"
else
  warn "nao consegui ligar o linger; rode: sudo loginctl enable-linger $USER"
fi

systemctl --user list-timers 'seucanal-*' --no-pager
