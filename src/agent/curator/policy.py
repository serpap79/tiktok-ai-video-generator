"""Filtro de politica: o que o canal nao fala, decidido antes de gastar token.

O nicho tech/IA/ciencia ja exclui a maior parte do risco por construcao. Este
filtro guarda as tres bordas que sobram, e todas apareceram no radar real:

  (a) saude e medicamento com alegacao de eficacia -- "caneta emagrecedora" foi
      o tema com mais trafego no Google Trends BR em 17/09/2026;
  (b) ciencia instrumentalizada por politica partidaria -- os tres artigos mais
      vistos da Wikipedia em pt eram ministros do STF;
  (c) tragedia com vitima real.

O filtro roda ANTES do score, nao depois: tema bloqueado nao deve consumir
requisicao de pesquisa nem token de LLM, e nao deve poder ganhar no ranking por
ter velocidade alta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.text import normalize


@dataclass(frozen=True)
class PolicyRule:
    nome: str
    padrao: re.Pattern[str]
    motivo: str


def _regra(nome: str, termos: list[str], motivo: str) -> PolicyRule:
    # \b nas bordas evita que "ia" case dentro de "midia" ou "sim" dentro de "assim".
    juncao = "|".join(re.escape(t) for t in termos)
    return PolicyRule(nome, re.compile(rf"\b(?:{juncao})\b"), motivo)


# Os termos ficam sem acento porque `normalize` remove os diacriticos antes do match.
REGRAS: tuple[PolicyRule, ...] = (
    _regra(
        "saude",
        [
            "emagrecedor", "emagrecedora", "emagrecimento", "ozempic", "mounjaro",
            "wegovy", "semaglutida", "tirzepatida", "anabolizante", "suplemento",
            "remedio", "medicamento", "farmaco", "posologia", "dosagem",
            "cura para", "tratamento para", "cancer", "quimioterapia", "canabidiol",
            "weight loss", "diet pill", "cure for", "treatment for",
        ],
        "alegacao de saude ou medicamento exige responsabilidade clinica que o canal nao tem",
    ),
    _regra(
        "politica",
        [
            "stf", "supremo tribunal", "tse", "congresso nacional", "senado",
            "camara dos deputados", "eleicao", "eleicoes", "eleitoral", "urna",
            "lula", "bolsonaro", "ministro do stf", "impeachment", "cpi",
            "deputado", "senador", "governador", "prefeito", "partido",
            "election", "senate", "congress", "impeachment", "parliament",
            # Nomes de chefe de governo em campanha permanente: tema de IA com
            # eles vira politica partidaria no primeiro comentario (visto em
            # 19/09/2026: "Trump abre enquete" passou pelo portao).
            "trump", "biden", "kamala", "putin", "zelensky", "netanyahu", "milei",
            "maduro", "casa branca", "white house",
        ],
        "politica partidaria: fora do nicho e transforma qualquer erro em crise",
    ),
    _regra(
        "tragedia",
        [
            "morte", "mortes", "morre", "morreu", "morrem", "morto", "mortos", "obito",
            "faleceu", "falecimento", "falece",
            "vitima", "vitimas", "acidente", "desastre", "tragedia", "queda de aviao",
            "atentado", "tiroteio", "massacre", "assassinato", "homicidio",
            "estupro", "sequestro", "naufragio", "incendio", "terremoto",
            "died", "dies", "killed", "death", "deaths", "shooting", "crash",
            "victims", "massacre", "earthquake", "wildfire",
        ],
        "tragedia com vitima real: nao se faz conteudo viral sobre isso",
    ),
    _regra(
        "comercial",
        [
            # Guia de compra, promocao e produto financeiro. Visto no radar de
            # 19/09/2026: "Seguro para celular em 2026: quais planos cobrem
            # furto de dados e Pix?" passou no nicho (celular, dados) e ganhou
            # nota alta de interesse -- e e recomendacao de seguro.
            "seguro para", "seguro de celular", "planos de seguro", "quais planos",
            "melhores planos", "cupom", "cupons", "desconto", "descontos", "promocao",
            "promocoes", "black friday", "cyber monday", "vale a pena comprar",
            "onde comprar", "menor preco", "apostas", "bets", "cassino", "emprestimo",
            "consorcio", "renda extra", "ganhar dinheiro", "coupon", "discount",
            "deal alert", "best deals", "on sale",
        ],
        "conteudo comercial ou conselho financeiro: o canal explica tecnologia, nao "
        "recomenda compra, promocao nem produto financeiro",
    ),
    _regra(
        "menores",
        ["crianca", "criancas", "menor de idade", "adolescente", "infantil",
         "child", "children", "minor", "teen", "teenager"],
        "envolve menores: exige cuidado que o pipeline automatico nao oferece",
    ),
)


@dataclass(frozen=True)
class PolicyVerdict:
    allowed: bool
    rule: str = ""
    reason: str = ""
    matched: str = ""


def check(termo: str) -> PolicyVerdict:
    """Avalia um termo contra as regras. Primeira regra que casa decide.

    A ordem das regras nao e por gravidade e sim por frequencia observada no
    radar; qualquer casamento bloqueia igual, entao a ordem so afeta qual motivo
    aparece no registro.
    """
    texto = normalize(termo)
    for regra in REGRAS:
        m = regra.padrao.search(texto)
        if m:
            return PolicyVerdict(
                allowed=False, rule=regra.nome, reason=regra.motivo, matched=m.group(0)
            )
    return PolicyVerdict(allowed=True)
