"""Sujeito obrigatorio: o roteiro precisa dizer DE QUEM fala.

Falha que o piloto acusou: o video falava de "um modelo" sem nunca dizer o
nome, de onde e nem quem construiu -- e sem nome nao ha busca, nem
credibilidade, nem canal. A causa e estrutural: nada no caminho exigia o
sujeito, entao o modelo generalizava.

O portao e lexical e barato: os identificadores saem do proprio topico
(nome com letra+digito tipo "27B", proprio com 6+ letras tipo "Bonsai").
Numero puro ("5,9") nao e identidade -- e quantidade, e ja tem portao
proprio. Quem construiu vem do dossie via o roteirista (regra de prompt:
diga se o dossie disser, nunca invente).
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-zÀ-ÿ0-9]+(?:[.,][A-Za-zÀ-ÿ0-9]+)*")

# Categoria gramatical, nao identidade: some do portao, fica no texto.
_STOP = frozenset({
    "modelo", "modelos", "video", "sobre", "como", "para", "com", "uma",
})


def subject_terms(topic: str) -> list[str]:
    """Identificadores do assunto, em ordem, sem repetir (minusculos).

    Entra nome proprio (maiuscula no titulo: "Bonsai", "Stanford") e
    codinome com letra+digito ("27B"). Fora: minuscula comum ("modelo"),
    numero puro ("5,9") e curto demais. Titulo em ingles com resto em
    portugues nao e problema: nome proprio nao traduz.
    """
    termos: list[str] = []
    for bruto in _TOKEN.findall(topic):
        if not bruto[:1].isupper() and not _tem_nome(bruto):
            continue
        t = bruto.lower()
        if len(t) < 3 or t in _STOP:
            continue
        if t not in termos:
            termos.append(t)
    return termos


def _tem_nome(token: str) -> bool:
    """Letra+digito colados: 27B, GPT-4 (o hifen separa no token)."""
    tem_digito = any(c.isdigit() for c in token)
    tem_letra = any(c.isalpha() for c in token)
    return tem_digito and tem_letra and len(token) >= 2


def missing_subject(text: str, topic: str, minimum: int = 1) -> list[str]:
    """Identificadores ausentes. Basta 1: o bug a matar e o anonimato total
    ("um modelo"), nao a ficha incompleta. A lista de faltantes vai ao modelo
    para ele completar o que couber.
    """
    termos = subject_terms(topic)
    baixa = text.lower()
    ausentes = [t for t in termos if t not in baixa]
    if len(termos) - len(ausentes) >= minimum:
        return []
    return ausentes


__all__ = ["missing_subject", "subject_terms"]
