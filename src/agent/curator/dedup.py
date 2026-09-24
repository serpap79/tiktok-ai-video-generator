"""Deduplicacion lexica por solapamiento de tokens.

Eleccion deliberada, y una desviacion del plan original, que preveia embeddings
locales via sentence-transformers. Medido en 18/09/2026: aquel paquete arrastra
el torch con la stack CUDA entera -- cudnn 527 MB, nccl 206 MB, cufft 204 MB,
cusolver 191 MB, entre otros -- mas de 1,5 GB de bibliotecas NVIDIA en una
maquina sin GPU NVIDIA. La variante CPU-only no termino de instalar en 7
minutos.

Lo que la deduplicacion necesita coger aqui es mayoritariamente lexico: la
misma noticia llegando por fuentes diferentes, o el mismo lanzamiento
reformulado. Para eso, Jaccard sobre tokens de contenido resuelve, y resuelve
de forma determinista y comprobable, sin descarga y sin modelo.

Lo que NO coge es parafrasis sin palabra en comun ("Bonsai 2 comprime modelo"
vs "PrismML reduce el footprint 9x"). Ese es el hueco que justificaria
embeddings -- y, por estar detras de la puerta Deduplicator, cambiarla y medir
despues es barato.
"""

from __future__ import annotations

from agent.text import content_tokens

# Calibrado a mano contra titulos reales del radar. Por encima de esto, dos
# titulos casi siempre hablan de lo mismo; por debajo, empiezan a aparecer pares
# que solo comparten vocabulario generico del nicho.
UMBRAL_DEFECTO = 0.45


class LexicalDeduplicator:
    def __init__(self, umbral: float = UMBRAL_DEFECTO):
        self._umbral = umbral

    def find_duplicate(self, termo: str, anteriores: list[str]) -> tuple[str, float] | None:
        objetivo = content_tokens(termo)
        if not objetivo:
            return None

        mejor: tuple[str, float] | None = None
        for anterior in anteriores:
            sim = jaccard(objetivo, content_tokens(anterior))
            if sim >= self._umbral and (mejor is None or sim > mejor[1]):
                mejor = (anterior, round(sim, 3))
        return mejor


def jaccard(a: set[str], b: set[str]) -> float:
    """Interseccion sobre union. 0 cuando cualquiera de los lados es vacio."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
