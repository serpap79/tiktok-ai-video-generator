"""Puerto RadarSource: una fuente de señales de tendencia.

Cada fuente es sustituible y falla de forma aislada. El radar las ejecuta todas y continúa
con lo que ha vuelto: ninguna fuente puede tumbar la recolección, porque la ventana de una
tendencia dura horas y no hay tiempo para esperar a que un servicio se recupere.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent.models import Signal


class SourceUnavailable(RuntimeError):
    """La fuente no respondió o respondió de forma inutilizable.

    Es algo esperado, no excepcional: GDELT devuelve 429 con frecuencia y
    Reddit devolve 403 sem OAuth. O coletor registra e continua.
    """


@runtime_checkable
class RadarSource(Protocol):
    name: str

    def collect(self) -> list[Signal]:
        """Devuelve las señales que la fuente detecta ahora.

        Levanta SourceUnavailable cuando no puede comunicarse con la fuente.
        Una lista vacía es una respuesta válida: significa «la fuente respondió y no hay
        nada», no «la fuente cayó».
        """
        ...
