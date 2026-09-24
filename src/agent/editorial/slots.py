"""Las cuatro horas del dia, en horario de Espana (peninsular).

10h, 13h, 17h y 20h siguen el consumo espanol: media manana y sobremesa con
consumo rapido (corto), comida y noche con atencion larga (largo -- el Creator
Rewards exige 60s). La parrilla alterna **corto y largo**: corto a media manana
y en la sobremesa (alcance), largo en la comida y de noche (monetizacion).
El carrusel salio de la parrilla; sigue implementado y alcanzable por
`agent slot-extra --format carousel`.

La hora del slot es cuando el post queda **listo**: el timer de systemd dispara
35 min antes para dar tiempo a producir, y el runner espera la hora exacta.

Lo que cada slot "quiere" abajo es hipotesis editorial declarada, no medida del
canal. Entra en el planificador como PRIOR -- y pierde peso cuando las metricas
del propio canal (tabla `metrics`) tengan muestra para decir otra cosa.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Madrid")


@dataclass(frozen=True)
class Slot:
    id: str
    at: time
    label: str
    intent: str

    def when(self, day: date) -> datetime:
        """El instante del slot en ese dia, con huso de Espana."""
        return datetime.combine(day, self.at, tzinfo=TZ)


SLOTS: dict[str, Slot] = {
    "1000": Slot("1000", time(10, 0), "manana",
                 "noticia del dia en consumo rapido: que ha pasado y por que importa"),
    "1300": Slot("1300", time(13, 0), "comida",
                 "pausa de la comida, atencion mas larga: el asunto con desarrollo"),
    "1700": Slot("1700", time(17, 0), "sobremesa",
                 "rotura de la sobremesa: una sola idea, corta, para alcance"),
    "2000": Slot("2000", time(20, 0), "noche",
                 "fin del dia, el post que sostiene la historia entera"),
}


def today(now: datetime | None = None) -> date:
    return (now or datetime.now(TZ)).astimezone(TZ).date()


def next_slot(now: datetime | None = None) -> tuple[Slot, date]:
    """El proximo slot a partir de ahora (el de hoy, o el primero de manana)."""
    ahora = (now or datetime.now(TZ)).astimezone(TZ)
    for slot in SLOTS.values():
        if slot.when(ahora.date()) > ahora:
            return slot, ahora.date()
    manana = ahora.date() + timedelta(days=1)
    return next(iter(SLOTS.values())), manana


def parse_slot(texto: str) -> Slot:
    """'1000', '10', '10h', '10:00' -> Slot de las 10h."""
    limpio = texto.strip().lower().replace(":", "").replace("h", "")
    if len(limpio) <= 2:
        limpio = limpio.zfill(2) + "00"
    if limpio not in SLOTS:
        raise ValueError(f"slot {texto!r} desconocido; usa uno de {', '.join(SLOTS)}")
    return SLOTS[limpio]


__all__ = ["SLOTS", "TZ", "Slot", "next_slot", "parse_slot", "today"]
