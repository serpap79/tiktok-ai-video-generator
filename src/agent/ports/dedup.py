"""Porta Deduplicator: "ja falamos disso?".

Es una puerta porque elegir la técnica es un intercambio medible, no una verdad. La
implementacao inicial e lexica; uma baseada em embedding pode substitui-la e ser
comparada na mesma base de temas, que e o tipo de evidencia que o M5 produz.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Deduplicator(Protocol):
    def find_duplicate(self, termo: str, anteriores: list[str]) -> tuple[str, float] | None:
        """Devolve (tema_original, similaridade) se `termo` repete algo anterior.

        None significa «es un tema nuevo». La implementación decide el umbral: quien
        llama no debería necesitar conocer la escala de similitud de la técnica usada.
        """
        ...
