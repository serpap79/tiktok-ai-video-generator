"""Porta Deduplicator: "ja falamos disso?".

E porta porque a escolha da tecnica e uma troca mensuravel, nao uma verdade. A
implementacao inicial e lexica; uma baseada em embedding pode substitui-la e ser
comparada na mesma base de temas, que e o tipo de evidencia que o M5 produz.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Deduplicator(Protocol):
    def find_duplicate(self, termo: str, anteriores: list[str]) -> tuple[str, float] | None:
        """Devolve (tema_original, similaridade) se `termo` repete algo anterior.

        None significa "e assunto novo". A implementacao decide o limiar: quem
        chama nao deve precisar saber a escala de similaridade da tecnica usada.
        """
        ...
