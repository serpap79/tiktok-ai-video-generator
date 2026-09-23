"""Vocabulario visual do canal: 4 pilares esteticos, tags fixas em ingles.

Canal dark vive de identidade repetida: quem assiste tres videos precisa
reconhecer o quarto pelo visual antes da primeira palavra. Busca generica
("server rack blue lights") rende material aleatorio a cada render e nunca
constrói essa assinatura. Por isso `search_terms` nao e texto livre: sai
verbatim de um dos quatro pilares abaixo, todos de um unico pilar por roteiro.

As tags vao direto para o Pexels sem traducao, entao valem as mesmas regras do
`Script.search_terms`: ASCII e cena filmavel concreta. Tag fora do pool e
defeito mecanico -- volta ao modelo com a lista, como contagem de palavra.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pillar:
    id: str
    nome: str
    tags: tuple[str, ...]


PILLARS: dict[str, Pillar] = {
    "A": Pillar(
        id="A",
        nome="IA Sombria / Consciencia de Maquina (Dark AI / Sci-Fi)",
        tags=(
            "cybernetic brain",
            "sentient ai",
            "android activation",
            "dark tech laboratory",
            "humanoid robot close up",
            "bionic eye neon",
        ),
    ),
    "B": Pillar(
        id="B",
        nome="Programacao / Hacking / Dados (Cyberpunk Code)",
        tags=(
            "matrix code rain",
            "cyberpunk hacking terminal",
            "server room blinking lights",
            "cyber security breach",
            "holographic data glitch",
        ),
    ),
    "C": Pillar(
        id="C",
        nome="Redes Neurais / Deep Web / Conectividade (Abstract Tech)",
        tags=(
            "neural network nodes",
            "abstract digital plexus",
            "ai deep learning loop",
            "quantum computing laser",
            "data stream tunnel",
        ),
    ),
    "D": Pillar(
        id="D",
        nome='Estilo "Futuro Proximo" / Corporativo High-Tech (Sleek Tech)',
        tags=(
            "futuristic clean UI",
            "augmented reality hud",
            "smart city wireframe",
            "minimalist tech laboratory",
        ),
    ),
    # Adicionado em 19/09/2026: o primeiro video de ciencia do piloto (Marte)
    # saiu com androides e olho bionico, porque nenhum dos quatro pilares de
    # tech tinha estetica de ciencia e espaco. Mesma regra: escuro, filmavel.
    "E": Pillar(
        id="E",
        nome="Ciencia e Espaco (Cosmos / Lab)",
        tags=(
            "galaxy stars timelapse",
            "nebula deep space",
            "planet surface orbit",
            "telescope observatory night",
            "microscope laboratory dark",
            "dna double helix",
        ),
    ),
}

def normalize(term: str) -> str:
    """Minuscula e espaco simples: '  Bionic  Eye NEON ' casa com o pool."""
    return " ".join(term.lower().split())


_TAG_TO_PILLAR: dict[str, str] = {
    normalize(tag): pid for pid, p in PILLARS.items() for tag in p.tags
}


def pillar_of(terms: list[str]) -> str | None:
    """O pilar com mais tags entre os termos, ou None se nenhum casa.

    Derivavel a qualquer momento dos `search_terms` gravados -- por isso o
    pilar nao e coluna no banco: e funcao do roteiro, nao dado novo.
    """
    contagem: dict[str, int] = {}
    for t in terms:
        pid = _TAG_TO_PILLAR.get(normalize(t))
        if pid is not None:
            contagem[pid] = contagem.get(pid, 0) + 1
    if not contagem:
        return None
    return max(sorted(contagem), key=lambda pid: contagem[pid])


def validate_terms(terms: list[str]) -> list[str]:
    """Violoes de vocabulario, em texto que volta ao modelo como correcao."""
    problemas: list[str] = []
    fora = sorted({t for t in terms if normalize(t) not in _TAG_TO_PILLAR})
    if fora:
        problemas.append(
            "search_terms fora do vocabulario visual: "
            + ", ".join(f"{t!r}" for t in fora) + ". "
            "Use tags EXATAS de um unico pilar da lista de ESTETICA."
        )
    pilares = sorted({_TAG_TO_PILLAR[normalize(t)] for t in terms
                      if normalize(t) in _TAG_TO_PILLAR})
    if len(pilares) > 1:
        problemas.append(
            "search_terms misturam pilares "
            + "/".join(pilares) + ": um roteiro, um pilar. "
            "Identidade visual se constrói por repeticao, nao por variedade."
        )
    return problemas


# Palavras do tema (pt e en) que puxam cada pilar. Ordem importa: A antes de B
# antes de C; o que nao casar cai no D, que e o pilar generico do canal.
_PALAVRAS: dict[str, tuple[str, ...]] = {
    "E": ("marte", "mars", "espaco", "espaço", "space", "nasa", "esa ", "planeta",
          "planet", "galaxia", "galáxia", "galaxy", "estrela", "telescopio",
          "telescópio", "telescope", "astronom", "cosmos", "universo", "universe",
          " lua", "moon", "satelite", "satélite", "orbita", "órbita", "sonda",
          "foguete", "rocket", "cerebro", "cérebro", "brain", "dna", "celula",
          "célula", "genoma", "fisica", "física", "quimica", "química", "biolog",
          "neurocien", "cientistas", "scientists", "fossil", "fóssil", "vulcao"),
    "A": ("robo", "robô", "robot", "android", "humanoide", "humanoid",
          "consciencia", "consciência", "consciousness", "sentient",
          "bionic", "biônic", "cyborg", "ciborgue"),
    "B": ("codigo", "código", "code", "hack", "cyber", "seguranca",
          "segurança", "security", "servidor", "server", "terminal",
          "breach", "vazamento", "malware", "ransomware", "programa",
          "programador", "developer"),
    "C": ("neural", "quantum", "quantic", "quântic", "deep learning",
          "aprendizado", "network", "rede", "conectividade", "plexus",
          "stream", "dados", "data", "modelo", "model", "parametro",
          "parâmetro", "token", "treinamento", "training"),
}

_PILAR_SUGERIDO_POR_OMISSAO = "D"


def suggest_pillar(topic: str) -> str:
    """Pilar sugerido pelo assunto. Orientacao, nao decisao: o modelo escolhe."""
    baixo = f" {topic.lower()} "
    # Robo antes de ciencia ("robo em Marte" e robo); ciencia antes de dados.
    for pid in ("A", "E", "B", "C"):
        if any(p in baixo for p in _PALAVRAS[pid]):
            return pid
    return _PILAR_SUGERIDO_POR_OMISSAO


# Conceito nao e cena: o Pexels devolve qualquer coisa para "innovation". O
# b-roll do assunto precisa ser objeto ou lugar filmavel.
_ABSTRATOS = frozenset("""
innovation innovative future futuristic technology tech concept success idea ideas
growth business progress digital transformation disruption intelligence artificial ai
data information knowledge change revolution power potential solution strategy
""".split())

MAX_BROLL = {"long": 2, "short": 1, "carousel": 2}


def validate_broll(terms: list[str], mode: str = "long") -> list[str]:
    """B-roll do assunto: ate N termos EM INGLES de objeto/lugar filmavel.

    Existe porque a identidade sozinha nao correlaciona: o short do cerebro
    (19/09) saiu com "futuristic clean UI" e uma mao segurando celular em fundo
    bege -- nada a ver com cerebro, e fora da marca. O pilar continua dando a
    assinatura; o b-roll poe na tela a coisa de que o video fala.
    """
    problemas: list[str] = []
    teto = MAX_BROLL.get(mode, 2)
    if len(terms) > teto:
        problemas.append(f"broll tem {len(terms)} termos; o modo {mode} aceita ate {teto}.")
    for t in terms:
        palavras = t.lower().split()
        if not t.isascii():
            problemas.append(f"broll {t!r} nao e ASCII: o Pexels espera ingles.")
        elif not 1 <= len(palavras) <= 5:
            problemas.append(f"broll {t!r} precisa ter de 1 a 5 palavras.")
        elif normalize(t) in _TAG_TO_PILLAR:
            problemas.append(f"broll {t!r} e tag de pilar: ela vai em search_terms, "
                             "o broll e o objeto concreto do assunto.")
        elif all(p in _ABSTRATOS for p in palavras):
            problemas.append(f"broll {t!r} e conceito, nao cena: use um objeto ou "
                             "lugar filmavel ('graphics card', 'human brain model').")
    return problemas


def brief(pilar_sugerido: str) -> str:
    """Bloco de ESTETICA do prompt, com as tags verbatim para copiar."""
    blocos = []
    for pid, p in PILLARS.items():
        marca = " <-- pilar sugerido para este tema" if pid == pilar_sugerido else ""
        tags = "\n".join(f"    - {t}" for t in p.tags)
        blocos.append(f"  Pilar {pid} ({p.nome}){marca}:\n{tags}")
    return (
        "ESTETICA (identidade do canal)\n"
        "O canal tem assinatura visual fixa: todo search_terms sai COPIADO, "
        "letra por letra, da lista de UM unico pilar abaixo, na ordem "
        "cronologica da narracao. Nada de sinonimo, traducao ou invencao -- "
        "termo fora da lista reprova o roteiro.\n"
        + "\n".join(blocos)
    )


def compact_brief(pilar_sugerido: str, mode: str = "long") -> str:
    """ESTETICA em uma linha por pilar + a regra do b-roll do assunto.

    Mesma informacao do `brief`, com metade dos tokens: vai em TODA chamada
    do roteirista, e no free tier token de prompt e cota.
    """
    linhas = []
    for pid, p in PILLARS.items():
        marca = " (sugerido)" if pid == pilar_sugerido else ""
        linhas.append(f"  {pid}{marca}: " + " | ".join(p.tags))
    teto = MAX_BROLL.get(mode, 2)
    return (
        "ESTETICA (identidade visual fixa)\n"
        "- search_terms: tags COPIADAS letra por letra de UM unico pilar -- o sugerido, "
        "salvo se outro combinar claramente melhor com o assunto --, na ordem "
        "cronologica da narracao (termo fora da lista reprova):\n"
        + "\n".join(linhas) + "\n"
        f"- broll: 0 a {teto} termos EM INGLES do assunto concreto, objeto ou lugar "
        "filmavel que a narracao cita ('graphics card', 'human brain model', "
        "'smartphone screen'); o primeiro abre o video. Nada de conceito abstrato."
    )


__all__ = [
    "MAX_BROLL",
    "PILLARS",
    "Pillar",
    "brief",
    "compact_brief",
    "normalize",
    "pillar_of",
    "suggest_pillar",
    "validate_broll",
    "validate_terms",
]
