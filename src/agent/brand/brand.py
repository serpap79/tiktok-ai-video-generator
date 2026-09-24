"""Vector de marca: carga `brand/brand.json` y lo reparte a las capas.

Toda capa agentica lee de aqui, nadie copia el valor dentro del codigo: color,
tag, formula de gancho y hashtag existen en UN sitio. Si la guia cambia, cambia
el JSON y los tests de conformidad acusan donde el codigo divergio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from agent.config import PROJECT_ROOT

BRAND_JSON = PROJECT_ROOT / "brand" / "brand.json"


@dataclass(frozen=True)
class ContentPillar:
    id: str
    tag: str
    accent: str
    duration: str
    hook_formula: str
    example: str
    beats: tuple[str, ...]
    visual: str
    cta: str
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class Presenter:
    id: str
    name: str
    role: str
    formats: tuple[str, ...]
    gender: str
    seed: int
    voice_pitch: str
    voice_rate: float
    voice_note: str
    library_voice: str | None
    identity_prompt: str


@dataclass(frozen=True)
class Brand:
    name: str
    handle: str
    tagline: str
    bio_default: str
    background: str
    surface: str
    ink: str
    muted: str
    accent_primary: str
    accent_secondary: str
    hashtags: tuple[str, ...]
    pillars: dict[str, ContentPillar] = field(default_factory=dict)
    banned: tuple[str, ...] = ()
    presenters: dict[str, Presenter] = field(default_factory=dict)
    presenter_formats: tuple[str, ...] = ()
    variation_prompt: str = ""
    variation_slots: dict[str, tuple[str, ...]] = field(default_factory=dict)
    negative_prompt: str = ""

    def accent_for(self, pillar_id: str) -> str:
        """Un acento por pieza: el del pilar, o el verde por defecto."""
        pillar = self.pillars.get(pillar_id)
        return pillar.accent if pillar is not None else self.accent_primary

    def presenter_for(self, pillar_id: str) -> Presenter | None:
        """Quien presenta este pilar. Sin formato declarado: None -- sin avatar.

        Hasta la manana del 20/09/2026 esto era reparto por formato (Nova en la
        noticia, Atlas en el tutorial y el vs, el resto sin avatar). En la noche
        del mismo dia se volvio firma del canal: existe clip base
        fotorrealista del ATLAS -- parpadeo y balanceo de cabeza **humanos**,
        porque se generaron como video -- y no existe el de Nova. Un
        presentador sintetizado al lado de uno filmado seria una diferencia de
        calidad visible en el mismo canal.

        Nova no se elimino: esta en el `brand.json` con la lista de formatos
        vacia, y vuelve sola a disputar el pilar cuando exista su clip base.
        Quien decide sigue siendo el JSON, no esta funcion.
        """
        for p in self.presenters.values():
            if pillar_id in p.formats:
                return p
        return None


@lru_cache(maxsize=1)
def load(path: str | Path = BRAND_JSON) -> Brand:
    """El vector de marca. Falla alto si el JSON desaparece: marca ausente no genera."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pillars = {
        p["id"]: ContentPillar(
            id=p["id"], tag=p["tag"], accent=p["accent"], duration=p["duration"],
            hook_formula=p["hook_formula"], example=p["example"],
            beats=tuple(p["beats"]), visual=p["visual"], cta=p["cta"],
            keywords=tuple(p.get("keywords", ())))
        for p in raw["pillars"]
    }
    pres = raw.get("presenters", {})
    presenters = {
        c["id"]: Presenter(
            id=c["id"], name=c["name"], role=c["role"],
            formats=tuple(c.get("formats", ())), gender=c.get("gender", ""),
            seed=int(c.get("seed", 0)),
            voice_pitch=c.get("voice", {}).get("pitch", ""),
            voice_rate=float(c.get("voice", {}).get("rate", 1.0)),
            voice_note=c.get("voice", {}).get("note", ""),
            library_voice=c.get("voice", {}).get("library_voice"),
            identity_prompt=c.get("identity_prompt", ""))
        for c in pres.get("cast", [])
    }
    return Brand(
        name=raw["identity"]["name"],
        handle=raw["identity"]["handle"],
        tagline=raw["identity"]["tagline"],
        bio_default=raw["identity"]["bio_default"],
        background=raw["palette"]["background"],
        surface=raw["palette"]["surface"],
        ink=raw["palette"]["ink"],
        muted=raw["palette"].get("muted", "#8B93A1"),
        accent_primary=raw["palette"]["accent_primary"],
        accent_secondary=raw["palette"]["accent_secondary"],
        hashtags=tuple(raw["caption"]["hashtags"]),
        pillars=pillars,
        banned=tuple(raw["voice"]["banned"]),
        presenters=presenters,
        presenter_formats=tuple(pres.get("usage", {}).get("formats", ())),
        variation_prompt=pres.get("variation_prompt", ""),
        variation_slots={k: tuple(v) for k, v in
                         pres.get("variation_slots", {}).items()},
        negative_prompt=pres.get("negative_prompt", ""),
    )


def voice_brief() -> str:
    """Bloque de voz de la marca para el system prompt del guionista."""
    return (
        "VOZ CIRCUITO CERO (@circuitocero): informativo y preciso, provocador "
        "sin ser rabioso, enigmatico en el gancho, futurista en el cierre. "
        "Una idea por video. Maximo un numero por frase. Gancho con hasta 12 "
        "palabras. Nunca prometas lo que el video no entrega. "
        "Nunca emoji, nunca 'hola a todos', nunca 'suscribete'.")


def pillar_brief(pillar_id: str, mode: str = "long") -> str:
    """Bloque TIPO DE CONTENIDO del prompt: formula de gancho, batidas y CTA.

    Hasta el 19/09 la formula de cada pilar existia en el `brand.json` y no
    llegaba a prompt ninguno -- "curiosidad", "tutorial" y "noticia" salian con
    la misma estructura. Es lo que separa los tipos de contenido para quien
    mira.
    """
    brand = load()
    p = brand.pillars.get(pillar_id) or brand.pillars["news"]
    batidas = " -> ".join(f"({i}) {b}" for i, b in enumerate(p.beats, start=1))
    if mode == "short":
        estructura = ("gancho + solo la batida (1) en una frase + cierre en bucle. "
                      f"Batidas: {batidas}")
    elif mode == "carousel":
        estructura = f"slide 1 = gancho; slides 2-4 = batidas; slide 5 = conclusion. {batidas}"
    else:
        estructura = f"gancho -> contexto (quien, que, por que importa) -> {batidas} -> cierre"
    return (
        f"TIPO DE CONTENIDO: {p.tag} ({p.id})\n"
        f"- Formula del gancho: {p.hook_formula}. Ejemplo de tono (no copies): \"{p.example}\"\n"
        f"- Estructura: {estructura}\n"
        f"- CTA de la marca para este tipo (adaptalo al tema): \"{p.cta}\""
    )


_SIN_TILDES = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")


def suggest_content_pillar(topic: str) -> str:
    """Pilar de contenido por el asunto. Orientacion; el guionista elige.

    La comparacion ignora tildes: el publico escribe 'cuál gana' o 'cual gana'
    y ambos deben caer en el mismo pilar.
    """
    brand = load()
    bajo = topic.lower().translate(_SIN_TILDES)
    puntos: dict[str, int] = {}
    for pid, p in brand.pillars.items():
        puntos[pid] = sum(1 for kw in p.keywords
                          if kw in bajo or kw.translate(_SIN_TILDES) in bajo)
    mejor = max(sorted(puntos), key=lambda pid: puntos[pid])
    return mejor if puntos[mejor] else "news"


def avatar_prompt(presenter_id: str, *, angulo: str = "frontal",
                  expresion: str = "neutra", gesto: str = "quieta") -> str:
    """Prompt listo de generacion del avatar: identidad fija + 3 variables.

    Solo las 3 variables cambian por video -- cualquier otro ajuste es deriva
    del rostro y reprueba el QA contra los retratos maestros.
    """
    brand = load()
    if presenter_id not in brand.presenters:
        raise ValueError(f"presentador {presenter_id!r} desconocido")
    for nombre, valor in (("angulo", angulo), ("expresion", expresion),
                          ("gesto", gesto)):
        opciones = brand.variation_slots.get(nombre, ())
        if opciones and valor not in opciones:
            raise ValueError(f"{nombre} {valor!r} fuera de {list(opciones)}")
    base = brand.variation_prompt
    p = brand.presenters[presenter_id]
    texto = base.replace("{prompt_de_identidad}", p.identity_prompt)
    texto = texto.replace("{frontal | 3/4 izquierda | 3/4 derecha}", angulo)
    texto = texto.replace("{neutra | concentrada | una ceja levantada}", expresion)
    texto = texto.replace("{quieta | leve inclinacion de cabeza | mano abierta "
                          "a la altura del pecho}", gesto)
    return texto


__all__ = ["BRAND_JSON", "Brand", "ContentPillar", "Presenter", "avatar_prompt",
           "load", "pillar_brief", "suggest_content_pillar", "voice_brief"]
