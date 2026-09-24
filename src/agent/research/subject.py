"""Sujeto obligatorio: el guion debe decir DE QUIÉN habla.

Un fallo detectado por el piloto: el vídeo hablaba de «un modelo» sin decir nunca el
nombre, el origen ni quién lo creó. Sin nombre no hay búsqueda,
credibilidad ni canal. La causa es estructural: nada en el proceso exigía el
sujeto, por lo que el modelo generalizaba.

La comprobación es léxica y barata: los identificadores salen del propio tema
(nombre con letra+dígito como «27B», propio de 6+ letras como «Bonsai»).
Un número puro («5,9») no es identidad: es una cantidad y ya tiene su propia comprobación
. Quien lo creó llega desde el expediente mediante el guionista (regla del prompt:
decirlo si el expediente lo indica, nunca inventarlo).
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-zÀ-ÿ0-9]+(?:[.,][A-Za-zÀ-ÿ0-9]+)*")

# Categoría gramatical, no identidad: se excluye de la comprobación y queda en el texto.
_STOP = frozenset({
    "modelo", "modelos", "video", "sobre", "como", "para", "com", "uma",
})


def subject_terms(topic: str) -> list[str]:
    """Identificadores del asunto, en orden y sin repetir (en minúsculas).

    Entra nome proprio (maiuscula no titulo: "Bonsai", "Stanford") e
    codinome com letra+digito ("27B"). Fora: minuscula comum ("modelo"),
    numero puro ("5,9") e curto demais. Titulo em ingles com resto em
    un título en inglés con el resto en castellano no es un problema: los nombres
    propios no se traducen.
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
    («un modelo»), no una ficha incompleta. La lista de ausentes se envía al modelo
    para que complete lo que sea posible.
    """
    termos = subject_terms(topic)
    baixa = text.lower()
    ausentes = [t for t in termos if t not in baixa]
    if len(termos) - len(ausentes) >= minimum:
        return []
    return ausentes


__all__ = ["missing_subject", "subject_terms"]
