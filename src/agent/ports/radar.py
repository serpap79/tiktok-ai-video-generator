"""Porta RadarSource: uma fonte de sinal de tendencia.

Cada fonte e substituivel e falha de forma isolada. O radar roda todas e segue
com o que voltou: nenhuma fonte pode derrubar a coleta, porque a janela de um
trend e de horas e nao ha tempo para esperar um servico se recuperar.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent.models import Signal


class SourceUnavailable(RuntimeError):
    """A fonte nao respondeu ou respondeu de forma inutilizavel.

    E esperado e nao e excepcional: o GDELT devolve 429 com frequencia e o
    Reddit devolve 403 sem OAuth. O coletor registra e continua.
    """


@runtime_checkable
class RadarSource(Protocol):
    name: str

    def collect(self) -> list[Signal]:
        """Devolve os sinais que a fonte enxerga agora.

        Levanta SourceUnavailable quando nao conseguir falar com a fonte.
        Lista vazia e resposta valida: significa "a fonte respondeu e nao ha
        nada", que e diferente de "a fonte caiu".
        """
        ...
