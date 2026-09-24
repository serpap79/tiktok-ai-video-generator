"""Vocabulario visual del canal: 5 pilares esteticos, tags fijas en ingles.

Un canal dark vive de identidad repetida: quien ve tres videos tiene que
reconocer el cuarto por el visual antes de la primera palabra. Busca generica
("server rack blue lights") da material aleatorio en cada render y nunca
construye esa firma. Por eso `search_terms` no es texto libre: sale verbatim de
uno de los cinco pilares de abajo, todos de un unico pilar por guion.

Las tags van directo a Pexels sin traduccion, asi que valen las mismas reglas
del `Script.search_terms`: ASCII y escena filmable concreta. Tag fuera del pool
es defecto mecanico -- vuelve al modelo con la lista, como el conteo de palabra.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pillar:
    id: str
    nombre: str
    tags: tuple[str, ...]


PILLARS: dict[str, Pillar] = {
    "A": Pillar(
        id="A",
        nombre="IA Oscura / Conciencia de Maquina (Dark AI / Sci-Fi)",
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
        nombre="Programacion / Hacking / Datos (Cyberpunk Code)",
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
        nombre="Redes Neuronales / Deep Web / Conectividad (Abstract Tech)",
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
        nombre='Estilo "Futuro Cercano" / Corporativo High-Tech (Sleek Tech)',
        tags=(
            "futuristic clean UI",
            "augmented reality hud",
            "smart city wireframe",
            "minimalist tech laboratory",
        ),
    ),
    # Anadido en 19/09/2026: el primer video de ciencia del piloto (Marte)
    # salio con androides y ojo bionico, porque ninguno de los cuatro pilares
    # de tech tenia estetica de ciencia y espacio. Misma regla: oscuro, filmable.
    "E": Pillar(
        id="E",
        nombre="Ciencia y Espacio (Cosmos / Lab)",
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
    """Minusculas y espacio simple: '  Bionic  Eye NEON ' casa con el pool."""
    return " ".join(term.lower().split())


_TAG_TO_PILLAR: dict[str, str] = {
    normalize(tag): pid for pid, p in PILLARS.items() for tag in p.tags
}


def pillar_of(terms: list[str]) -> str | None:
    """El pilar con mas tags entre los terminos, o None si ninguno casa.

    Derivable en cualquier momento de los `search_terms` grabados -- por eso el
    pilar no es columna en la base: es funcion del guion, no dato nuevo.
    """
    conteo: dict[str, int] = {}
    for t in terms:
        pid = _TAG_TO_PILLAR.get(normalize(t))
        if pid is not None:
            conteo[pid] = conteo.get(pid, 0) + 1
    if not conteo:
        return None
    return max(sorted(conteo), key=lambda pid: conteo[pid])


def validate_terms(terms: list[str]) -> list[str]:
    """Violaciones de vocabulario, en texto que vuelve al modelo como correccion."""
    problemas: list[str] = []
    fuera = sorted({t for t in terms if normalize(t) not in _TAG_TO_PILLAR})
    if fuera:
        problemas.append(
            "search_terms fuera del vocabulario visual: "
            + ", ".join(f"{t!r}" for t in fuera) + ". "
            "Usa tags EXACTAS de un unico pilar de la lista de ESTETICA."
        )
    pilares = sorted({_TAG_TO_PILLAR[normalize(t)] for t in terms
                      if normalize(t) in _TAG_TO_PILLAR})
    if len(pilares) > 1:
        problemas.append(
            "search_terms mezclan pilares "
            + "/".join(pilares) + ": un guion, un pilar. "
            "La identidad visual se construye por repeticion, no por variedad."
        )
    return problemas


# Palabras del tema (castellano e ingles) que tiran de cada pilar. El orden
# importa: A antes de B antes de C; lo que no case cae en D, el pilar generico
# del canal.
_PALABRAS: dict[str, tuple[str, ...]] = {
    "E": ("marte", "mars", "espacio", "space", "nasa", "esa ", "planeta",
          "planet", "galaxia", "galaxia", "galaxy", "estrella", "telescopio",
          "telescopio", "telescope", "astronom", "cosmos", "universo", "universe",
          " luna", "moon", "satelite", "satelite", "orbita", "orbita", "sonda",
          "cohete", "rocket", "cerebro", "cerebro", "brain", "dna", "celula",
          "celula", "genoma", "fisica", "fisica", "quimica", "quimica", "biolog",
          "neurocien", "cientificos", "scientists", "fosil", "fosil", "volcan"),
    "A": ("robot", "robot", "robot", "android", "humanoide", "humanoid",
          "conciencia", "conciencia", "consciousness", "sentient",
          "bionic", "bionic", "cyborg", "ciborg"),
    "B": ("codigo", "codigo", "code", "hack", "cyber", "seguridad",
          "seguridad", "security", "servidor", "server", "terminal",
          "breach", "fuga", "malware", "ransomware", "programa",
          "programador", "developer"),
    "C": ("neural", "quantum", "quantic", "cuantic", "deep learning",
          "aprendizaje", "network", "red", "conectividad", "plexus",
          "stream", "datos", "data", "modelo", "model", "parametro",
          "parametro", "token", "entrenamiento", "training"),
}

_PILAR_SUGERIDO_POR_DEFECTO = "D"


def suggest_pillar(topic: str) -> str:
    """Pilar sugerido por el asunto. Orientacion, no decision: el modelo elige."""
    bajo = f" {topic.lower()} "
    # Robot antes de ciencia ("robot en Marte" es robot); ciencia antes de datos.
    for pid in ("A", "E", "B", "C"):
        if any(p in bajo for p in _PALABRAS[pid]):
            return pid
    return _PILAR_SUGERIDO_POR_DEFECTO


# Concepto no es escena: Pexels devuelve cualquier cosa para "innovation". El
# b-roll del asunto tiene que ser objeto o lugar filmable.
_ABSTRACTOS = frozenset("""
innovation innovative future futuristic technology tech concept success idea ideas
growth business progress digital transformation disruption intelligence artificial ai
data information knowledge change revolution power potential solution strategy
""".split())

MAX_BROLL = {"long": 2, "short": 1, "carousel": 2}


def validate_broll(terms: list[str], mode: str = "long") -> list[str]:
    """B-roll del asunto: hasta N terminos EN INGLES de objeto/lugar filmable.

    Existe porque la identidad sola no correlaciona: el corto del cerebro
    (19/09) salio con "futuristic clean UI" y una mano sosteniendo un movil en
    fondo beis -- nada que ver con un cerebro, y fuera de marca. El pilar sigue
    dando la firma; el b-roll pone en pantalla la cosa de la que habla el video.
    """
    problemas: list[str] = []
    techo = MAX_BROLL.get(mode, 2)
    if len(terms) > techo:
        problemas.append(f"broll tiene {len(terms)} terminos; el modo {mode} acepta hasta {techo}.")
    for t in terms:
        palabras = t.lower().split()
        if not t.isascii():
            problemas.append(f"broll {t!r} no es ASCII: Pexels espera ingles.")
        elif not 1 <= len(palabras) <= 5:
            problemas.append(f"broll {t!r} necesita tener de 1 a 5 palabras.")
        elif normalize(t) in _TAG_TO_PILLAR:
            problemas.append(f"broll {t!r} es tag de pilar: esa va en search_terms; "
                             "el broll es el objeto concreto del asunto.")
        elif all(p in _ABSTRACTOS for p in palabras):
            problemas.append(f"broll {t!r} es concepto, no escena: usa un objeto o "
                             "lugar filmable ('graphics card', 'human brain model').")
    return problemas


def brief(pilar_sugerido: str) -> str:
    """Bloco de ESTETICA del prompt, con las tags verbatim para copiar."""
    bloques = []
    for pid, p in PILLARS.items():
        marca = " <-- pilar sugerido para este tema" if pid == pilar_sugerido else ""
        tags = "\n".join(f"    - {t}" for t in p.tags)
        bloques.append(f"  Pilar {pid} ({p.nombre}){marca}:\n{tags}")
    return (
        "ESTETICA (identidad del canal)\n"
        "El canal tiene firma visual fija: todo search_terms sale COPIADO, "
        "letra por letra, de la lista de UN unico pilar de abajo, en el orden "
        "cronologico de la narracion. Nada de sinonimo, traduccion ni invencion -- "
        "termino fuera de la lista reprueba el guion.\n"
        + "\n".join(bloques)
    )


def compact_brief(pilar_sugerido: str, mode: str = "long") -> str:
    """ESTETICA en una linea por pilar + la regla del b-roll del asunto.

    Misma informacion del `brief`, con la mitad de los tokens: va en TODA llamada
    del guionista, y en free tier el token de prompt es cuota.
    """
    lineas = []
    for pid, p in PILLARS.items():
        marca = " (sugerido)" if pid == pilar_sugerido else ""
        lineas.append(f"  {pid}{marca}: " + " | ".join(p.tags))
    techo = MAX_BROLL.get(mode, 2)
    return (
        "ESTETICA (identidad visual fija)\n"
        "- search_terms: tags COPIADAS letra por letra de UN unico pilar -- el sugerido, "
        "salvo que otro combine claramente mejor con el asunto --, en el orden "
        "cronologico de la narracion (termino fuera de la lista reprueba):\n"
        + "\n".join(lineas) + "\n"
        f"- broll: 0 a {techo} terminos EN INGLES del asunto concreto, objeto o lugar "
        "filmable que la narracion cita ('graphics card', 'human brain model', "
        "'smartphone screen'); el primero abre el video. Nada de concepto abstracto."
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
