"""Portao deterministico: o numero que o fato cita existe na pagina lida?

Por que so numero, e nao a afirmacao inteira. O caminho obvio seria medir
sobreposicao de vocabulario entre a afirmacao e a pagina -- e ele esta errado
aqui: as fontes de tech sao majoritariamente em ingles e a afirmacao sai em
pt-BR. "Retem 98,2% do desempenho" e "retains 98.2% of performance" nao
compartilham nenhuma palavra, e o portao reprovaria justamente os fatos bem
traduzidos. **Numero sobrevive a traducao; palavra nao.**

E por que vale a pena ter o portao. Numero e o que o roteiro usa para convencer,
e numero e exatamente o que um modelo inventa com mais confianca. O criterio 2
da rubrica do juiz (M3, fatia 3) pergunta se a afirmacao tem fonte; este portao
pergunta antes, e sem gastar token, se o numero da afirmacao esta **naquela**
fonte. Sao verificacoes diferentes e as duas precisam existir.

O que ele nao faz, de proposito: nao confere unidade nem contexto. "5,9 GB" casa
com uma pagina que diz "5,9 milhoes de downloads". E aproximacao, e a alternativa
seria pedir ao proprio modelo para se auditar -- o que nao e verificacao.
"""

from __future__ import annotations

import re

# Numeros como aparecem em texto real: "5,9", "1.500", "98.2", "2026", "5090".
_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")


def canonical_numbers(texto: str) -> list[str]:
    """Numeros do texto, reduzidos a digitos, na ordem em que aparecem.

    Separador e descartado em vez de interpretado: pt-BR escreve "5,9" e ingles
    escreve "5.9" para o mesmo valor, e "1.500" e mil e quinhentos em pt e um e
    meio em ingles. Decidir qual e qual exigiria saber o idioma da pagina; casar
    so os digitos ("59", "1500") resolve os dois sentidos de uma vez e nao cria
    falso negativo por virgula.
    """
    return [re.sub(r"[.,]", "", m.group()) for m in _NUMERO.finditer(texto)]


def missing_numbers(claim: str, source_text: str) -> list[str]:
    """Numeros citados na afirmacao que nao aparecem na fonte.

    Lista vazia significa "todo numero citado esta na pagina", que e o mais forte
    que este portao consegue afirmar. Afirmacao sem numero nenhum passa: ela
    ainda tem URL, e julgar o resto e trabalho do juiz.
    """
    na_fonte = set(canonical_numbers(source_text))
    ausentes: list[str] = []
    for numero in canonical_numbers(claim):
        if numero not in na_fonte and numero not in ausentes:
            ausentes.append(numero)
    return ausentes
