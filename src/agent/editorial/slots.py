"""Os quatro horarios do dia, em horario de Brasilia.

09h, 12h, 16h e 19h foram escolhidos pelo autor em 20/09/2026, subindo de
tres para quatro posts por dia. A grade alterna **curto e longo**: curto de
manha e no meio da tarde (alcance), longo no almoco e a noite (monetizacao --
o Creator Rewards exige 60s). O carrossel saiu da grade; ele continua
implementado e alcancavel por `agent slot-extra --format carrossel`.

O horario do slot e quando o post fica **pronto**: o timer do systemd dispara
35 min antes para dar tempo de produzir, e o runner espera a hora certa.

O que cada slot "quer" abaixo e hipotese editorial declarada, nao medida do
canal. Ela entra no planejador como PRIOR -- e perde peso quando as metricas
do proprio canal (tabela `metrics`) tiverem amostra para dizer outra coisa.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Sao_Paulo")


@dataclass(frozen=True)
class Slot:
    id: str
    at: time
    label: str
    intent: str

    def when(self, day: date) -> datetime:
        """O instante do slot naquele dia, com fuso de Brasilia."""
        return datetime.combine(day, self.at, tzinfo=TZ)


SLOTS: dict[str, Slot] = {
    "0900": Slot("0900", time(9, 0), "manha",
                 "noticia do dia em consumo rapido: o que aconteceu e por que importa"),
    "1200": Slot("1200", time(12, 0), "almoco",
                 "pausa do almoco, atencao mais longa: o assunto com desenvolvimento"),
    "1600": Slot("1600", time(16, 0), "tarde",
                 "quebra da tarde: uma ideia so, curta, para alcance"),
    "1900": Slot("1900", time(19, 0), "noite",
                 "fim do dia, o post que sustenta a historia inteira"),
}


def today(now: datetime | None = None) -> date:
    return (now or datetime.now(TZ)).astimezone(TZ).date()


def next_slot(now: datetime | None = None) -> tuple[Slot, date]:
    """O proximo slot a partir de agora (o de hoje, ou o primeiro de amanha)."""
    agora = (now or datetime.now(TZ)).astimezone(TZ)
    for slot in SLOTS.values():
        if slot.when(agora.date()) > agora:
            return slot, agora.date()
    amanha = agora.date() + timedelta(days=1)
    return next(iter(SLOTS.values())), amanha


def parse_slot(texto: str) -> Slot:
    """'0900', '09', '9h', '09:00' -> Slot das 9h."""
    limpo = texto.strip().lower().replace(":", "").replace("h", "")
    if len(limpo) <= 2:
        limpo = limpo.zfill(2) + "00"
    if limpo not in SLOTS:
        raise ValueError(f"slot {texto!r} desconhecido; use um de {', '.join(SLOTS)}")
    return SLOTS[limpo]


__all__ = ["SLOTS", "TZ", "Slot", "next_slot", "parse_slot", "today"]
