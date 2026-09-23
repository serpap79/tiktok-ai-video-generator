"""Deduplicacao lexica por sobreposicao de tokens.

Escolha deliberada, e um desvio do plano original, que previa embeddings locais
via sentence-transformers. Medido em 18/09/2026: aquele pacote arrasta o torch
com a stack CUDA inteira -- cudnn 527 MB, nccl 206 MB, cufft 204 MB, cusolver
191 MB, entre outros -- mais de 1,5 GB de bibliotecas NVIDIA numa maquina sem
GPU NVIDIA. A variante CPU-only nao terminou de instalar em 7 minutos.

O que a deduplicacao precisa pegar aqui e majoritariamente lexico: a mesma
materia chegando por fontes diferentes, ou o mesmo lancamento reformulado. Para
isso, Jaccard sobre tokens de conteudo resolve, e resolve de forma deterministica
e testavel, sem download e sem modelo.

O que ele NAO pega e parafrase sem palavra em comum ("Bonsai 2 comprime modelo"
vs "PrismML reduz footprint em 9x"). Essa e a lacuna que justificaria embeddings
-- e, por estar atras da porta Deduplicator, trocar e medir depois e barato.
"""

from __future__ import annotations

from agent.text import content_tokens

# Calibrado na mao contra titulos reais do radar. Acima disso, dois titulos
# praticamente sempre falam da mesma coisa; abaixo, comecam a aparecer pares que
# so compartilham vocabulario generico do nicho.
LIMIAR_PADRAO = 0.45


class LexicalDeduplicator:
    def __init__(self, limiar: float = LIMIAR_PADRAO):
        self._limiar = limiar

    def find_duplicate(self, termo: str, anteriores: list[str]) -> tuple[str, float] | None:
        alvo = content_tokens(termo)
        if not alvo:
            return None

        melhor: tuple[str, float] | None = None
        for anterior in anteriores:
            sim = jaccard(alvo, content_tokens(anterior))
            if sim >= self._limiar and (melhor is None or sim > melhor[1]):
                melhor = (anterior, round(sim, 3))
        return melhor


def jaccard(a: set[str], b: set[str]) -> float:
    """Intersecao sobre uniao. 0 quando qualquer um dos lados e vazio."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
