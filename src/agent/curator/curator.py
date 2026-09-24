"""Curador: convierte senales crudas en una decision justificada.

El orden de las etapas es deliberado y barato primero:

  1. politica    -- bloquea antes de cualquier calculo; tema vetado no puede
                    ganar en el ranking por tener velocidad alta
  2. nicho       -- puerta, no condimento
  3. duplicado   -- solo entre los que quedaron, porque comparar con el ledger
                    cuesta
  4. score       -- rankea lo que paso las tres puertas

Toda decision, incluidas las rechazadas, se devuelve con motivo. Lo que no fue
elegido hoy es material para calibrar el score manana.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from agent.curator import niche, policy
from agent.curator.dedup import LexicalDeduplicator
from agent.models import Decision, Signal, Verdict
from agent.ports.dedup import Deduplicator

# Pesos del score. La velocidad pesa mas que el volumen porque lo que interesa
# es el asunto que esta subiendo, no el que ya es grande desde hace tiempo --
# asunto grande y estable suele estar saturado de contenido.
PESO_VELOCIDAD = 0.55
PESO_VOLUMEN = 0.20
PESO_NICHO = 0.25


@dataclass
class CurationReport:
    decisions: list[Decision]

    @property
    def selected(self) -> Decision | None:
        return next((d for d in self.decisions if d.approved), None)

    @property
    def eligible(self) -> list[Decision]:
        return [d for d in self.decisions if d.verdict in (Verdict.selected, Verdict.not_selected)]

    def top(self, n: int) -> list[Decision]:
        """Los n mejores elegibles, el elegido primero.

        La rutina publica 3 piezas/dia en temas diferentes: el top-3 de la
        misma recolecta ya viene deduplicado entre si (el lazo anade el
        elegido al ledger), asi que investigar los 3 no repite asunto en el
        mismo dia.
        """
        ordenados = sorted(
            self.eligible,
            key=lambda d: (d.verdict is Verdict.selected, d.score),
            reverse=True,
        )
        return ordenados[:max(1, n)]

    def by_verdict(self, verdict: Verdict) -> list[Decision]:
        return [d for d in self.decisions if d.verdict is verdict]

    def tally(self) -> dict[str, int]:
        return {v.value: len(self.by_verdict(v)) for v in Verdict}


class Curator:
    def __init__(
        self,
        deduplicator: Deduplicator | None = None,
        niche_threshold: float = niche.UMBRAL_DEFECTO,
    ):
        self._dedup = deduplicator or LexicalDeduplicator()
        self._niche_threshold = niche_threshold

    def curate(self, signals: list[Signal], ledger: list[str] | None = None) -> CurationReport:
        ahora = datetime.now(UTC)
        ledger = list(ledger or [])
        percentiles = _percentiles_por_fuente(signals)

        decisions: list[Decision] = []
        supervivientes: list[tuple[Signal, float]] = []

        for s in signals:
            veredicto = policy.check(s.term)
            if not veredicto.allowed:
                decisions.append(_decision(
                    s, Verdict.rejected_policy, ahora,
                    reason=f"politica/{veredicto.rule}: '{veredicto.matched}' — {veredicto.reason}",
                    score=0.0, niche_fit=niche.fit(s.term, s.source),
                ))
                continue

            encaje = niche.fit(s.term, s.source)
            if encaje < self._niche_threshold:
                decisions.append(_decision(
                    s, Verdict.rejected_niche, ahora,
                    reason=f"fuera del nicho tech/IA/ciencia (encaje {encaje:.2f} < "
                           f"{self._niche_threshold:.2f})",
                    score=0.0, niche_fit=encaje,
                ))
                continue

            duplicado = self._dedup.find_duplicate(s.term, ledger)
            if duplicado is not None:
                original, sim = duplicado
                decisions.append(_decision(
                    s, Verdict.rejected_duplicate, ahora,
                    reason=f"ya cubierto (similaridad {sim:.2f} con '{original[:60]}')",
                    score=0.0, niche_fit=encaje, duplicate_of=original,
                ))
                continue

            supervivientes.append((s, encaje))

        # El score solo se calcula entre quien paso las tres puertas: rankear
        # candidatos vetados seria trabajo tirado a la basura.
        puntuados = sorted(
            ((s, e, self._score(s, e, percentiles)) for s, e in supervivientes),
            key=lambda t: t[2],
            reverse=True,
        )

        for posicion, (s, encaje, score) in enumerate(puntuados):
            # Reconfirma contra el ledger corrido (originales + ya rankeados):
            # dos fuentes pueden traer la misma historia, y el top-3 del dia
            # necesita 3 asuntos diferentes, no 3 titulos del mismo.
            duplicado = self._dedup.find_duplicate(s.term, ledger)
            if duplicado is not None:
                original, sim = duplicado
                decisions.append(_decision(
                    s, Verdict.rejected_duplicate, ahora,
                    reason=f"ya cubierto (similaridad {sim:.2f} con '{original[:60]}')",
                    score=0.0, niche_fit=encaje, duplicate_of=original,
                ))
                continue
            ledger.append(s.term)
            decisions.append(_decision(
                s,
                Verdict.selected if posicion == 0 else Verdict.not_selected,
                ahora,
                reason=(_justificacion(s, encaixe=encaje, score=score) if posicion == 0
                        else f"paso las puertas, quedo en {posicion + 1}º (score {score:.3f})"),
                score=score,
                niche_fit=encaje,
            ))

        return CurationReport(decisions)

    def _score(self, s: Signal, encaje: float, percentiles: _Percentiles) -> float:
        """Combina velocidad, volumen y encaje en una nota de 0 a 1.

        Velocidad y volumen entran como **percentil dentro de la propia
        fuente**, y no como valor bruto. Punto de Hacker News y pageview de
        Wikipedia no comparten escala: sumar los numeros crudos haria que
        Wikipedia ganara siempre, por tener unidad mayor, y no por tener
        asunto mejor.

        Velocidad desconocida recibe 0.5 -- el medio de la escala. No es 0
        porque "no medi" no es evidencia contra el tema; asi compite por el
        volumen y por el nicho hasta que la segunda recolecta de la tasa.
        """
        vel_pct = 0.5 if s.velocity is None else percentiles.velocity(s)
        vol_pct = percentiles.volume(s)
        return round(
            PESO_VELOCIDAD * vel_pct + PESO_VOLUMEN * vol_pct + PESO_NICHO * encaje, 4
        )


def _justificacion(s: Signal, *, encaixe: float, score: float) -> str:
    nucleo, _ = niche.matched_terms(s.term)
    caso = ", ".join(sorted(nucleo)[:4]) or "sin termino del nucleo"
    vel = (f"{s.velocity:,.1f} {s.unit}/h" if s.velocity is not None
           else "velocidad no medida")
    return (f"score {score:.3f}; {vel}; volumen {s.volume:,.0f} {s.unit}; "
            f"nicho {encaixe:.2f} ({caso}); fuente {s.source}")


def _decision(s: Signal, verdict: Verdict, ahora: datetime, *, reason: str,
              score: float, niche_fit: float, duplicate_of: str | None = None) -> Decision:
    return Decision(
        term=s.term, source=s.source, verdict=verdict, reason=reason,
        score=score, niche_fit=niche_fit, velocity=s.velocity, volume=s.volume,
        url=s.url, duplicate_of=duplicate_of, decided_at=ahora, news_items=s.news_items,
    )


class _Percentiles:
    """Posicion relativa dentro de la propia fuente, en [0, 1]."""

    def __init__(self, signals: list[Signal]):
        self._vel: dict[str, list[float]] = {}
        self._vol: dict[str, list[float]] = {}
        for s in signals:
            if s.velocity is not None:
                self._vel.setdefault(s.source, []).append(s.velocity)
            self._vol.setdefault(s.source, []).append(s.volume)
        for d in (self._vel, self._vol):
            for valores in d.values():
                valores.sort()

    def velocity(self, s: Signal) -> float:
        return _pct(self._vel.get(s.source, []), s.velocity or 0.0)

    def volume(self, s: Signal) -> float:
        return _pct(self._vol.get(s.source, []), s.volume)


def _percentiles_por_fuente(signals: list[Signal]) -> _Percentiles:
    return _Percentiles(signals)


def _pct(ordenados: list[float], valor: float) -> float:
    """Fraccion de los valores de la fuente que `valor` supera o iguala."""
    if not ordenados:
        return 0.5
    if len(ordenados) == 1:
        return 0.5
    debajo = sum(1 for v in ordenados if v < valor)
    iguales = sum(1 for v in ordenados if v == valor)
    return round((debajo + 0.5 * iguales) / len(ordenados), 4)
