"""Recolector: ejecuta las fuentes, completa la velocidad y guarda la serie.

Dos responsabilidades, ambas deliberadamente simples: aquí no se juzga
qué vale la señal. Clasificar, filtrar por política y elegir el tema
corresponde al curador (M2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from agent.memory.store import SignalStore
from agent.models import Signal
from agent.ports.radar import RadarSource, SourceUnavailable

# Por debajo, la diferencia entre dos recolecciones es ruido de redondeo dividido
# por un intervalo minúsculo, lo que produce velocidades absurdas.
_INTERVALO_MINIMO_H = 0.25


@dataclass
class CollectionReport:
    """Qué ocurrió durante la recolección. Un fallo de fuente es un dato,
    no una excepción perdida."""

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
                # Es lo esperado. La ventana de una tendencia dura horas: no hay tiempo de
                # esperar a que un servicio se recupere, y una fuente caída no
                # pode derrubar a coleta inteira.
                report.failures[source.name] = str(exc)
                continue
            except Exception as exc:  # noqa: BLE001
                # Un error en el parser de una fuente tampoco tumba las demás, pero
                # aparece en el informe con su tipo para que no sea un fallo silencioso.
                report.failures[source.name] = f"{type(exc).__name__}: {exc}"
                continue

            report.signals.extend(self._fill_velocity(sinais))

        self._store.record(report.signals)
        report.elapsed_s = round((datetime.now(UTC) - inicio).total_seconds(), 2)
        return report

    def _fill_velocity(self, sinais: list[Signal]) -> list[Signal]:
        """Calcula la velocidad por diferencia para quienes solo informan del nivel.

        Una fuente que ya trae velocidad (Hacker News) pasa intacta. Sin recolección
        anterior, la velocidad sigue en None, lo que significa «desconocido» y es
        diferente de cero. El curador debe poder distinguir ambos casos.
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
    """Las fuentes del nicho, en el orden en que valen la pena.

    El arXiv queda fuera a proposito: no tiene senal de velocidad ninguna, asi
    que no es radar. Sirve de profundidad para un tema ya elegido (M3), no para
    descubrirlo.
    """
    from agent.radar.sources.arquivo import Archivo, WikipediaOnThisDay
    from agent.radar.sources.gdelt import Gdelt
    from agent.radar.sources.google_trends import GoogleTrends
    from agent.radar.sources.hacker_news import HackerNews
    from agent.radar.sources.huggingface import HuggingFaceTrending
    from agent.radar.sources.rss import RssFeeds
    from agent.radar.sources.wikipedia import WikipediaPageviews

    return [HackerNews(), RssFeeds(), HuggingFaceTrending(), GoogleTrends(),
            WikipediaPageviews(), WikipediaOnThisDay(), Archivo(), Gdelt()]
