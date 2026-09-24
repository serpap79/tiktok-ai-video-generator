"""Fuentes de marca con cobertura comprobada, compartidas por diapositiva y vídeo.

Nasceu de um defeito que passou em todos os portoes: o carrossel das 13h56 de
19/09/2026 saiu com titulo e texto em ~10px, ilegiveis. A cadeia era
`_cobre()` -> sin fontTools -> `except: return False` -> ninguna fuente
«cubre» -> `ImageFont.load_default()`, la fuente de mapa de bits diminuta de Pillow.
aceite so conferia 1080x1920, e o PNG passava.

Duas regras saem disso:

1. **«no sé si lo cubre» no se convierte en «no lo cubre».** Sin leer la cmap, la fuente
   da marca e usada; tofu num acento e defeito visivel, texto de 10px e slide
   perdido.
2. **nunca recurrible a una fuente de mapa de bits.** Sin ninguna TTF cargable,
   falla claramente: una
   diapositiva ilegible publicada es peor que un carrusel que no salió.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageDraw, ImageFont

from agent.config import PROJECT_ROOT

FONTS_DIR = PROJECT_ROOT / "brand" / "assets" / "fonts"
# La única fuente de MPT que cubre los acentos del castellano de España (459 glifos comprobados).
FALLBACK_TTF = PROJECT_ROOT / ".renderer" / "resource" / "fonts" / "BeVietnamPro-Bold.ttf"

DISPLAY = "SpaceGrotesk-Bold.ttf"
MONO = "IBMPlexMono-SemiBold.ttf"
MONO_MEDIUM = "IBMPlexMono-Medium.ttf"

# El archivo «SpaceGrotesk-Bold.ttf» de la marca es la fuente VARIABLE (eje wght
# 300-700) com o Light como padrao -- conferido na tabela fvar em 20/09/2026.
# Pillow e libass carregam o padrao, e os titulos saiam em Light. A instancia
# estatica em 700 e gerada uma vez (fontTools) com familia propria, para o
# libass da legenda achar pelo nome sem ambiguidade.
DISPLAY_FAMILY = "CircuitoCero Display"
STATIC_DIR = PROJECT_ROOT / "data" / "fonts"


class FontUnavailable(RuntimeError):
    """No se cargó ninguna fuente escalable: el texto sería ilegible."""


@lru_cache(maxsize=16)
def _codepoints(path: str) -> frozenset[int] | None:
    """Glifos de la fuente, o None cuando no se puede saber."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None
    try:
        return frozenset(TTFont(path, lazy=True).getBestCmap())
    except Exception:  # noqa: BLE001 -- una fuente dañada cuenta como «no sé»
        return None


def covers(path: str | Path, text: str) -> bool:
    """True si la fuente tiene glifo para todos los caracteres visibles del texto."""
    pontos = _codepoints(str(path))
    if pontos is None:
        return True
    return all(c.isspace() or ord(c) in pontos for c in text)


def display_bold_path() -> Path:
    """Space Grotesk Bold (700) estatica, generada de la variable de la marca si falta."""
    destino = STATIC_DIR / "CircuitoCeroDisplay-Bold.ttf"
    if destino.exists():
        return destino
    origem = FONTS_DIR / DISPLAY
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
    except ImportError:
        return origem
    fonte = TTFont(str(origem))
    if "fvar" not in fonte:
        return origem
    estatica = instantiateVariableFont(fonte, {"wght": 700})
    nomes = estatica["name"]
    for nid, valor in ((1, DISPLAY_FAMILY), (2, "Bold"), (4, f"{DISPLAY_FAMILY} Bold"),
                       (6, "CircuitoCeroDisplay-Bold"), (16, DISPLAY_FAMILY), (17, "Bold")):
        nomes.setName(valor, nid, 3, 1, 0x409)
        nomes.setName(valor, nid, 1, 0, 0)
    estatica["OS/2"].usWeightClass = 700
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    estatica.save(str(destino))
    return destino


def _caminho(name: str) -> Path:
    if name == DISPLAY:
        return display_bold_path()
    return FONTS_DIR / name


def font(name: str, size: int, text: str = "") -> ImageFont.FreeTypeFont:
    """Fuente de marca solicitada; BeVietnamPro si falta un glifo; nunca un mapa de bits."""
    candidatos = [_caminho(name), FALLBACK_TTF]
    carregavel: Path | None = None
    for cand in candidatos:
        if not cand.is_file():
            continue
        carregavel = carregavel or cand
        if not text or covers(cand, text):
            try:
                return ImageFont.truetype(str(cand), size)
            except OSError:
                continue
    if carregavel is not None:
        # Nenhuma cobre tudo: a primeira que carrega, com tofu no glifo que
        # falta un glifo, aún es legible; la fuente de mapa de bits no.
        return ImageFont.truetype(str(carregavel), size)
    raise FontUnavailable(
        f"no hay ninguna fuente TTF en {FONTS_DIR} ni {FALLBACK_TTF}; "
        "el texto saldría con la fuente de mapa de bits de 10 px")


def wrap(draw: ImageDraw.ImageDraw, text: str, fonte: ImageFont.FreeTypeFont,
         width: int) -> list[str]:
    """Parte por palabras en el límite de ancho medido con la propia fuente."""
    linhas: list[str] = []
    atual = ""
    for palavra in text.split():
        teste = f"{atual} {palavra}".strip()
        if draw.textlength(teste, font=fonte) <= width:
            atual = teste
        else:
            if atual:
                linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return linhas


def hex_rgb(cor: str) -> tuple[int, int, int]:
    cor = cor.lstrip("#")
    return (int(cor[0:2], 16), int(cor[2:4], 16), int(cor[4:6], 16))


__all__ = ["DISPLAY", "DISPLAY_FAMILY", "FALLBACK_TTF", "FONTS_DIR", "FontUnavailable",
           "MONO", "MONO_MEDIUM", "covers", "display_bold_path", "font", "hex_rgb", "wrap"]
