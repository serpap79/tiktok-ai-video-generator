"""Pronuncia: el texto que la VOZ lee no es el texto que la LEYENDA muestra.

Reportado en el primer video del piloto (20/09/2026): la voz es-ES lee
"Gemini" como "Jemini" -- regla del castellano, G antes de E suena J -- y dice
"Discord?" con acento de otra lengua. Son dos defectos distintos:

1. **nombre extranjero leido con la fonetica del castellano**. No hay SSML en
   el endpoint gratuito del edge-tts (el `<phoneme>` es rechazado), asi que el
   unico control es la grafia: la voz recibe "Djemini"; la leyenda, "Gemini".
   Este modulo hace el cambio palabra a palabra y devuelve el alineamiento,
   para que la leyenda vuelva a la grafia original con el tiempo correcto.
2. **acento cambiado en medio de la frase**: es la voz *Multilingual*, que
   decide el idioma por trozo y a veces falla. Se resuelve en la eleccion de
   voz (monolingue es-ES), no aqui.

El lexico cubre el vocabulario del nicho (marcas, siglas, modelos). Sigla fuera
de el se deletrea solo cuando es sabidamente sigla tecnica -- deletrear "NASA"
seria peor que el error que se quiere evitar.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Grafia fonetica para voz es-ES. Clave en minuscula y sin acento; el valor es
# lo que la voz lee. Nombres de mas de una palabra entran enteros
# ("hugging face").
LEXICO: dict[str, str] = {
    # IA y empresas
    "gemini": "Yemini",
    "google": "Gugol",
    "openai": "Oupen EI Ei",
    "chatgpt": "Chat YipiTi",
    "gpt": "YipiTi",
    "claude": "Clod",
    "anthropic": "Antropik",
    "deepseek": "Dip Sik",
    "deepmind": "Dip Maind",
    "qwen": "Cuuen",
    "llama": "Lama",
    "copilot": "Cópailot",
    # Hallado por la comprobacion del Whisper en el slot de test de las 15h (20/09).
    "cowork": "Cóu Uerk",
    "midjourney": "Mid Yorni",
    "perplexity": "Perplesiti",
    "hugging face": "Yaguin Fis",
    "huggingface": "Yaguin Fis",
    "nvidia": "Envidia",
    "microsoft": "Máicrosof",
    "apple": "Ápou",
    "iphone": "Áifoun",
    "ipad": "Áiped",
    "youtube": "Yutú",
    "tiktok": "Tiktók",
    "github": "Guítjáb",
    "linux": "Línuc",
    "windows": "Uíndous",
    "android": "Ándroid",
    "python": "Páiton",
    "javascript": "Yávascrito",
    "samsung": "Sámsun",
    "intel": "Íntel",
    "spacex": "Espéi Ecs",
    "starlink": "Star Linc",
    "elon": "Ílon",
    "musk": "Másc",
    "altman": "Áltman",
    "stanford": "Stanford",
    "harvard": "Yárvard",
    "wall street journal": "Uól Strit Yórnal",
    "the verge": "De Vérdy",
    "techcrunch": "Tec Crántch",
    "hacker news": "Yáquer Nius",
    # Apps y servicios que aparecen en tutorial (hallado: "Zapier" y "n8n" en
    # el slot de test de las 15h del 20/09).
    "zapier": "Zéipier",
    "n8n": "ene ocho ene",
    "notion": "Nóchon",
    "slack": "Sléqui",
    "whatsapp": "Guatsáp",
    "facebook": "Féisbuc",
    "twitter": "Tuíter",
    "netflix": "Nétflics",
    "spotify": "Espótifai",
    "amazon": "Ámazon",
    "azure": "Ásiur",
    "reddit": "Rédit",
    "discord": "Díscord",
    "gmail": "Yíméil",
    "excel": "Écsel",
    "chrome": "Cróum",
    "firefox": "Fáierfocs",
    "playstation": "Pléistéichon",
    "xbox": "Equis Bocs",
    "copilot+": "Cópailot Plás",
    "sora": "Sóra",
    "grok": "Gróc",
    "xai": "Equis Ei Ei",
    # Hallado por la comprobacion en el slot de las 20h del 19/09 (video de Marte).
    "high-resolution": "Yái Rezolútion",
    "mars express": "Márs Exprés",
    "wi-fi": "Uái Fái",
    "wifi": "Uái Fái",
    "bluetooth": "Blutú",
    # Terminos del nicho que el sintetizador lee "en castellano"
    "software": "Sófuer",
    "hardware": "Yárduer",
    "startup": "Estártaap",
    "startups": "Estártaaps",
    "deepfake": "Dip Fik",
    "deepfakes": "Dip Fics",
    "prompt": "Prómpt",
    "prompts": "Prómpts",
    "online": "onláin",
    "benchmark": "Béntchmar",
    "benchmarks": "Béntchmars",
    "machine learning": "Máchin Lérnin",
    "deep learning": "Dip Lérnin",
    "open source": "Óupen Sors",
    "chatbot": "Chátbot",
    "chatbots": "Chátbots",
    "agents.md": "Éidchents punto éme dé",
    "claude.md": "Clód punto éme dé",
    "readme": "Rid Mi",
    "token": "Tóquen",
    "tokens": "Tóquens",
    "streaming": "Estrímin",
    "podcast": "Pódcást",
    "feed": "Fid",
    "cloud": "Cláud",
    "firmware": "Fírmer",
    "notebook": "Nótbuc",
    "smartphone": "Esmártfoun",
    "smartphones": "Esmártfouns",
}

# Siglas tecnicas deletreadas letra a letra. Solo las conocidas: sigla leida
# como palabra (NASA, OTAN) deletreada sonaria mal.
SIGLAS = frozenset({
    "api", "apis", "gpu", "gpus", "cpu", "cpus", "npu", "tpu", "llm", "llms", "rtx",
    "amd", "ibm", "aws", "sdk", "cli", "url", "html", "css", "ssd", "hd", "ram",
    "ai", "agi", "usb", "vpn", "pdf", "ceo", "cto", "mit", "sql", "ux", "ui",
})

LETRAS = {
    "a": "a", "b": "bé", "c": "cé", "d": "dé", "e": "e", "f": "éfe", "g": "ge",
    "h": "aché", "i": "i", "j": "yota", "k": "ca", "l": "éle", "m": "éme", "n": "ene",
    "o": "o", "p": "pé", "q": "cu", "r": "erre", "s": "ese", "t": "te", "u": "u",
    "v": "uvé", "w": "doble uvé", "x": "equis", "y": "i griega", "z": "zeta",
}

_PUNTUACION_BORDE = re.compile(
    r"^([\"'\u201c\u201d\u2018\u2019(\[]*)(.*?)([\"'\u201c\u201d\u2018\u2019)\].,;:!?…]*)$"
)


def _clave(texto: str) -> str:
    sin = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in sin if not unicodedata.combining(c))


def spell(sigla: str) -> str:
    """'GPU' -> 'ge pé u' (el plural 's' final se vuelve 'ese' solo si forma parte)."""
    base = sigla.lower()
    plural = base.endswith("s") and base[:-1] in SIGLAS
    letras = base[:-1] if plural else base
    hablado = " ".join(LETRAS.get(c, c) for c in letras)
    return hablado + ("s" if plural else "")


@dataclass
class Respelled:
    """Texto para la voz + alineamiento con las palabras originales.

    `groups[i]` = cuantas palabras del texto hablado nacieron de la palabra
    original i. La leyenda usa esto para devolver la grafia original con el
    tiempo de las palabras habladas.
    """

    original: list[str]
    spoken: list[str]
    groups: list[int] = field(default_factory=list)
    changes: list[tuple[str, str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.spoken)


def respell(texto: str) -> Respelled:
    palabras = texto.split()
    salida = Respelled(original=palabras, spoken=[])
    multi = sorted((k for k in LEXICO if " " in k), key=lambda k: -len(k.split()))
    i = 0
    while i < len(palabras):
        # Nombre de varias palabras primero ("wall street journal").
        caso = False
        for k in multi:
            n = len(k.split())
            trozo = palabras[i:i + n]
            if len(trozo) < n:
                continue
            centro = " ".join(_PUNTUACION_BORDE.match(p).group(2) for p in trozo)
            if _clave(centro) == k:
                pre = _PUNTUACION_BORDE.match(trozo[0]).group(1)
                pos = _PUNTUACION_BORDE.match(trozo[-1]).group(3)
                hablado = (pre + LEXICO[k] + pos).split()
                # La leyenda muestra las n palabras originales: la primera se
                # lleva el grupo hablado entero, las otras quedan con cero.
                salida.spoken.extend(hablado)
                salida.groups.append(len(hablado))
                salida.groups.extend([0] * (n - 1))
                salida.changes.append((" ".join(trozo), LEXICO[k]))
                i += n
                caso = True
                break
        if caso:
            continue
        palabra = palabras[i]
        pre, centro, pos = _PUNTUACION_BORDE.match(palabra).groups()
        hablado_txt = _hablar(centro)
        if hablado_txt != centro:
            salida.changes.append((centro, hablado_txt))
        hablado = (pre + hablado_txt + pos).split() or [palabra]
        salida.spoken.extend(hablado)
        salida.groups.append(len(hablado))
        i += 1
    return salida


def _hablar(token: str) -> str:
    """Una palabra (sin puntuacion de borde) como la voz debe leerla."""
    if not token:
        return token
    clave = _clave(token)
    if clave in LEXICO:
        return LEXICO[clave]
    # Con guion ("GPT-5", "Qwen3.8-27B"): cada parte por el lexico, sin guion.
    if "-" in token:
        partes = [p for p in token.split("-") if p]
        habladas = [_hablar(p) for p in partes]
        if habladas != partes:
            return " ".join(habladas)
    # Letras pegadas a numero ("RTX5090", "GPT5"): separa y resuelve las letras.
    m = re.fullmatch(r"([A-Za-z]+)(\d[\d.,]*)", token)
    if m and (_clave(m.group(1)) in LEXICO or _clave(m.group(1)) in SIGLAS):
        return f"{_hablar(m.group(1))} {m.group(2)}"
    if clave in SIGLAS and token.upper() == token or clave in SIGLAS and len(token) <= 4 \
            and token[:1].isupper():
        return spell(token)
    return token


__all__ = ["LEXICO", "Respelled", "SIGLAS", "respell", "spell"]
