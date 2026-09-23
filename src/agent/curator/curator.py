"""Curador: transforma sinais brutos numa decisao justificada.

A ordem dos estagios e deliberada e barata primeiro:

  1. politica   -- bloqueia antes de qualquer calculo; tema vetado nao pode
                   ganhar no ranking por ter velocidade alta
  2. nicho      -- portao, nao tempero
  3. duplicata  -- so entre os que sobraram, porque comparar com o ledger custa
  4. score      -- ranqueia o que passou nos tres portoes

Toda decisao, inclusive as rejeitadas, e devolvida com motivo. O que nao foi
escolhido hoje e material para calibrar o score amanha.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from agent.curator import niche, policy
from agent.curator.dedup import LexicalDeduplicator
from agent.models import Decision, Signal, Verdict
from agent.ports.dedup import Deduplicator

# Pesos do score. Velocidade pesa mais que volume porque o que interessa e o
# assunto que esta subindo, nao o que ja e grande ha tempo -- assunto grande e
# estavel costuma estar saturado de conteudo.
PESO_VELOCIDADE = 0.55
PESO_VOLUME = 0.20
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
        """Os n melhores elegiveis, o escolhido primeiro.

        A rotina publica 3 pecas/dia em temas diferentes: o top-3 da mesma
        coleta ja vem deduplicado entre si (o laco anexa o escolhido ao
        ledger), entao pesquisar os 3 nao repete assunto no mesmo dia.
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
        niche_threshold: float = niche.LIMIAR_PADRAO,
    ):
        self._dedup = deduplicator or LexicalDeduplicator()
        self._niche_threshold = niche_threshold

    def curate(self, signals: list[Signal], ledger: list[str] | None = None) -> CurationReport:
        agora = datetime.now(UTC)
        ledger = list(ledger or [])
        percentis = _percentis_por_fonte(signals)

        decisions: list[Decision] = []
        sobreviventes: list[tuple[Signal, float]] = []

        for s in signals:
            veredito = policy.check(s.term)
            if not veredito.allowed:
                decisions.append(_decisao(
                    s, Verdict.rejected_policy, agora,
                    reason=f"politica/{veredito.rule}: '{veredito.matched}' — {veredito.reason}",
                    score=0.0, niche_fit=niche.fit(s.term, s.source),
                ))
                continue

            encaixe = niche.fit(s.term, s.source)
            if encaixe < self._niche_threshold:
                decisions.append(_decisao(
                    s, Verdict.rejected_niche, agora,
                    reason=f"fora do nicho tech/IA/ciencia (encaixe {encaixe:.2f} < "
                           f"{self._niche_threshold:.2f})",
                    score=0.0, niche_fit=encaixe,
                ))
                continue

            duplicata = self._dedup.find_duplicate(s.term, ledger)
            if duplicata is not None:
                original, sim = duplicata
                decisions.append(_decisao(
                    s, Verdict.rejected_duplicate, agora,
                    reason=f"ja coberto (similaridade {sim:.2f} com '{original[:60]}')",
                    score=0.0, niche_fit=encaixe, duplicate_of=original,
                ))
                continue

            sobreviventes.append((s, encaixe))

        # O score so e calculado entre quem passou nos tres portoes: ranquear
        # candidatos vetados seria trabalho jogado fora.
        pontuados = sorted(
            ((s, e, self._score(s, e, percentis)) for s, e in sobreviventes),
            key=lambda t: t[2],
            reverse=True,
        )

        for posicao, (s, encaixe, score) in enumerate(pontuados):
            # Reconfere contra o ledger corrido (originais + ja ranqueados):
            # duas fontes podem trazer a mesma historia, e o top-3 do dia
            # precisa de 3 assuntos diferentes, nao 3 titulos do mesmo.
            duplicata = self._dedup.find_duplicate(s.term, ledger)
            if duplicata is not None:
                original, sim = duplicata
                decisions.append(_decisao(
                    s, Verdict.rejected_duplicate, agora,
                    reason=f"ja coberto (similaridade {sim:.2f} com '{original[:60]}')",
                    score=0.0, niche_fit=encaixe, duplicate_of=original,
                ))
                continue
            ledger.append(s.term)
            decisions.append(_decisao(
                s,
                Verdict.selected if posicao == 0 else Verdict.not_selected,
                agora,
                reason=(_justificativa(s, encaixe, score) if posicao == 0
                        else f"passou nos portoes, ficou em {posicao + 1}o (score {score:.3f})"),
                score=score,
                niche_fit=encaixe,
            ))

        return CurationReport(decisions)

    def _score(self, s: Signal, encaixe: float, percentis: _Percentis) -> float:
        """Combina velocidade, volume e encaixe numa nota de 0 a 1.

        Velocidade e volume entram como **percentil dentro da propria fonte**, e
        nao como valor bruto. Ponto do Hacker News e pageview da Wikipedia nao
        compartilham escala: somar os numeros crus faria a Wikipedia vencer
        sempre, por ter unidade maior, e nao por ter assunto melhor.

        Velocidade desconhecida recebe 0.5 -- o meio da escala. Nao e 0 porque
        "nao medi" nao e evidencia contra o tema; e assim ele compete pelo volume
        e pelo nicho ate a segunda coleta dar a taxa.
        """
        vel_pct = 0.5 if s.velocity is None else percentis.velocity(s)
        vol_pct = percentis.volume(s)
        return round(
            PESO_VELOCIDADE * vel_pct + PESO_VOLUME * vol_pct + PESO_NICHO * encaixe, 4
        )


def _justificativa(s: Signal, encaixe: float, score: float) -> str:
    nucleo, _ = niche.matched_terms(s.term)
    casou = ", ".join(sorted(nucleo)[:4]) or "sem termo de nucleo"
    vel = f"{s.velocity:,.1f} {s.unit}/h" if s.velocity is not None else "velocidade nao medida"
    return (f"score {score:.3f}; {vel}; volume {s.volume:,.0f} {s.unit}; "
            f"nicho {encaixe:.2f} ({casou}); fonte {s.source}")


def _decisao(s: Signal, verdict: Verdict, agora: datetime, *, reason: str,
             score: float, niche_fit: float, duplicate_of: str | None = None) -> Decision:
    return Decision(
        term=s.term, source=s.source, verdict=verdict, reason=reason,
        score=score, niche_fit=niche_fit, velocity=s.velocity, volume=s.volume,
        url=s.url, duplicate_of=duplicate_of, decided_at=agora, news_items=s.news_items,
    )


class _Percentis:
    """Posicao relativa dentro da propria fonte, em [0, 1]."""

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


def _percentis_por_fonte(signals: list[Signal]) -> _Percentis:
    return _Percentis(signals)


def _pct(ordenados: list[float], valor: float) -> float:
    """Fracao dos valores da fonte que `valor` supera ou iguala."""
    if not ordenados:
        return 0.5
    if len(ordenados) == 1:
        return 0.5
    abaixo = sum(1 for v in ordenados if v < valor)
    iguais = sum(1 for v in ordenados if v == valor)
    return round((abaixo + 0.5 * iguais) / len(ordenados), 4)
