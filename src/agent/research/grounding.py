"""Puerta determinista: el numero que el hecho cita, existe en la pagina leida?

Por que solo numero, y no la afirmacion entera. El camino obvio seria medir
solapamiento de vocabulario entre la afirmacion y la pagina -- y esta mal aqui:
las fuentes de tech son mayoritariamente en ingles y la afirmacion sale en
castellano. "Retiene 98,2% del rendimiento" y "retains 98.2% of performance" no
comparten ninguna palabra, y la puerta reprobaria justamente los hechos bien
traducidos. **El numero sobrevive a la traduccion; la palabra no.**

Y por eso vale la pena tener la puerta. El numero es lo que el guion usa para
convencer, y el numero es exactamente lo que un modelo inventa con mas
confianza. El criterio 2 de la rubrica del juez (M3, porcion 3) pregunta si la
afirmacion tiene fuente; esta puerta pregunta antes, y sin gastar token, si el
numero de la afirmacion esta en **esa** fuente. Son verificaciones distintas y
las dos tienen que existir.

Lo que no hace, a proposito: no comprueba unidad ni contexto. "5,9 GB" casa con
una pagina que dice "5,9 millones de descargas". Es aproximacion, y la
alternativa seria pedirle al propio modelo que se audite -- lo cual no es
verificacion.
"""

from __future__ import annotations

import re

# Numeros como aparecen en texto real: "5,9", "1.500", "98.2", "2026", "5090".
_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")


def canonical_numbers(texto: str) -> list[str]:
    """Numeros del texto, reducidos a digitos, en el orden en que aparecen.

    El separador se descarta en vez de interpretarse: el castellano escribe
    "5,9" e ingles escribe "5.9" para el mismo valor, y "1.500" es mil
    quinientos en castellano y uno y medio en ingles. Decidir cual es cual
    exigiria saber el idioma de la pagina; casar solo los digitos ("59",
    "1500") resuelve los dos sentidos de una vez y no crea falso negativo por
    coma.
    """
    return [re.sub(r"[.,]", "", m.group()) for m in _NUMERO.finditer(texto)]


def missing_numbers(claim: str, source_text: str) -> list[str]:
    """Numeros citados en la afirmacion que no aparecen en la fuente.

    Lista vacia significa "todo numero citado esta en la pagina", que es lo mas
    fuerte que esta puerta puede afirmar. Afirmacion sin ningun numero pasa:
    todavia tiene URL, y juzgar el resto es trabajo del juez.
    """
    en_fuente = set(canonical_numbers(source_text))
    ausentes: list[str] = []
    for numero in canonical_numbers(claim):
        if numero not in en_fuente and numero not in ausentes:
            ausentes.append(numero)
    return ausentes
