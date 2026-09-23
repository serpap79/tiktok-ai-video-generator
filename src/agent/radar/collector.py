"""Coletor: roda as fontes, preenche velocidade e grava a serie.

Duas responsabilidades, ambas deliberadamente burras -- nenhum julgamento sobre
o que o sinal vale acontece aqui. Ranquear, filtrar por politica e escolher tema
e trabalho do curador (M2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from agent.memory.store import SignalStore
from agent.models import Signal
from agent.ports.radar import RadarSource, SourceUnavailable

# Abaixo disso a diferenca entre duas coletas e ruido de arredondamento dividido
# por um intervalo minusculo, o que produz velocidade absurda.
_INTERVALO_MINIMO_H = 0.25


@dataclass
class CollectionReport:
    """O que aconteceu na coleta. Falha de fonte e dado, nao excecao perdida."""

    signals: list[Signal] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def sources_ok(self) -> list[str]:
        return sorted({s.source for s in self.signals})

    @property
    def with_velocity(self) -> list[Signal]:
        return [s for s in self.signals if s.has_velocity]


class Radar:
    def __init__(self, sources: list[RadarSource], store: SignalStore):
        self._sources = sources
        self._store = store

    def collect(self) -> CollectionReport:
        inicio = datetime.now(UTC)
        report = CollectionReport()

        for source in self._sources:
            try:
                sinais = source.collect()
            except SourceUnavailable as exc:
                # Esperado. A janela de um trend e de horas: nao ha tempo para
                # esperar um servico se recuperar, e uma fonte fora do ar nao
                # pode derrubar a coleta inteira.
                report.failures[source.name] = str(exc)
                continue
            except Exception as exc:  # noqa: BLE001
                # Bug no parser de uma fonte tambem nao derruba as outras, mas
                # aparece no relatorio com o tipo, para nao virar falha silenciosa.
                report.failures[source.name] = f"{type(exc).__name__}: {exc}"
                continue

            report.signals.extend(self._fill_velocity(sinais))

        self._store.record(report.signals)
        report.elapsed_s = round((datetime.now(UTC) - inicio).total_seconds(), 2)
        return report

    def _fill_velocity(self, sinais: list[Signal]) -> list[Signal]:
        """Calcula velocidade por diferenca para quem so reporta nivel.

        Fonte que ja trouxe velocidade (o Hacker News) passa intacta. Sem coleta
        anterior a velocidade continua None -- que significa "desconhecido" e e
        diferente de zero. O curador precisa poder distinguir os dois.
        """
        saida: list[Signal] = []
        for s in sinais:
            if s.has_velocity:
                saida.append(s)
                continue

            anterior = self._store.previous(s.key, before=s.seen_at)
            if anterior is None:
                saida.append(s)
                continue

            volume_ant, visto_em = anterior
            horas = (s.seen_at - visto_em).total_seconds() / 3600.0
            if horas < _INTERVALO_MINIMO_H:
                saida.append(s)
                continue

            saida.append(s.model_copy(
                update={"velocity": round((s.volume - volume_ant) / horas, 2)}
            ))
        return saida


def default_sources() -> list[RadarSource]:
    """As fontes do nicho, na ordem em que valem a pena.

    O arXiv fica de fora de proposito: nao tem sinal de velocidade nenhum, entao
    nao e radar. Serve de profundidade para um tema ja escolhido (M3), nao para
    descobri-lo.
    """
    from agent.radar.sources.arquivo import Arquivo, WikipediaOnThisDay
    from agent.radar.sources.gdelt import Gdelt
    from agent.radar.sources.google_trends import GoogleTrends
    from agent.radar.sources.hacker_news import HackerNews
    from agent.radar.sources.huggingface import HuggingFaceTrending
    from agent.radar.sources.rss import RssFeeds
    from agent.radar.sources.wikipedia import WikipediaPageviews

    return [HackerNews(), RssFeeds(), HuggingFaceTrending(), GoogleTrends(),
            WikipediaPageviews(), WikipediaOnThisDay(), Arquivo(), Gdelt()]
