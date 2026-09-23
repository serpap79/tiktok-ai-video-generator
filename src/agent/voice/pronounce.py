"""Pronuncia: o texto que a VOZ le nao e o texto que a LEGENDA mostra.

Reportado pelo autor no primeiro video do piloto (20/09/2026): a voz
pt-BR leu "Gemini" como "Zemini" -- regra do portugues, G antes de E soa J --
e disse "Discorda?" com sotaque de outra lingua. Sao dois defeitos diferentes:

1. **nome estrangeiro lido com a fonetica do portugues**. Nao ha SSML no
   endpoint gratuito do edge-tts (o `<phoneme>` e recusado), entao o unico
   controle e a grafia: a voz recebe "Djemini"; a legenda, "Gemini". Este
   modulo faz a troca palavra a palavra e devolve o alinhamento, para a
   legenda voltar a grafia original com o tempo certo.
2. **sotaque trocado no meio da frase**: e a voz *Multilingual*, que decide
   o idioma por trecho e as vezes erra ("Discorda?" parece espanhol). Isso
   se resolve na escolha da voz (monolingue pt-BR), nao aqui.

O lexico cobre o vocabulario do nicho (marcas, siglas, modelos). Sigla fora
dele e soletrada so quando e sabidamente sigla tecnica -- soletrar "NASA"
seria pior que o erro que se quer evitar.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Grafia fonetica para voz pt-BR. Chave em minuscula e sem acento; valor e o
# que a voz le. Nomes com mais de uma palavra entram inteiros ("hugging face").
LEXICO: dict[str, str] = {
    # IA e empresas
    "gemini": "Djémini",
    "google": "Gúgou",
    "openai": "Ôupen Ei Ái",
    "chatgpt": "Tchat Gê Pê Tê",
    "gpt": "Gê Pê Tê",
    "claude": "Clôd",
    "anthropic": "Ântropic",
    "deepseek": "Dip Síik",
    "deepmind": "Dip Máind",
    "qwen": "Tchuén",
    "llama": "Lhama",
    "copilot": "Cópailot",
    # Achado pela conferencia do Whisper no slot de teste das 15h (20/09).
    "cowork": "Côu Uôrk",
    "midjourney": "Mid Djârni",
    "perplexity": "Perpléxiti",
    "hugging face": "Râguin Fêis",
    "huggingface": "Râguin Fêis",
    "nvidia": "Envídia",
    "microsoft": "Máicrosoft",
    "apple": "Épou",
    "iphone": "Ái Fôun",
    "ipad": "Ái Péd",
    "youtube": "Iutúbi",
    "tiktok": "Tíqui Tóqui",
    "github": "Guít Râb",
    "linux": "Línux",
    "windows": "Uíndous",
    "android": "Ândroid",
    "python": "Páiton",
    "javascript": "Djava Scrípt",
    "samsung": "Sâmsung",
    "intel": "Íntel",
    "spacex": "Spêis Éx",
    "starlink": "Stár Link",
    "elon": "Ílon",
    "musk": "Mâsk",
    "altman": "Áltman",
    "stanford": "Stênford",
    "harvard": "Rárvard",
    "wall street journal": "Uól Strít Djôrnal",
    "the verge": "De Vérdj",
    "techcrunch": "Tec Crântch",
    "hacker news": "Réquer Níus",
    # Apps e servicos que aparecem em tutorial (achado: "Zapier" e "n8n" no
    # slot de teste das 15h de 20/09).
    "zapier": "Zêipier",
    "n8n": "êne oito êne",
    "notion": "Nôuxan",
    "slack": "Sléqui",
    "whatsapp": "Uatsáp",
    "facebook": "Feicibúqui",
    "twitter": "Tuíter",
    "netflix": "Nétflix",
    "spotify": "Spótifai",
    "amazon": "Ámazon",
    "azure": "Ájur",
    "reddit": "Rédit",
    "discord": "Díscord",
    "gmail": "Gê Mêil",
    "excel": "Écsel",
    "chrome": "Crôum",
    "firefox": "Fáier Fóx",
    "playstation": "Plêi Stêixan",
    "xbox": "Équis Bóx",
    "copilot+": "Cópailot Plâs",
    "sora": "Sóra",
    "grok": "Grók",
    "xai": "Éx Ei Ái",
    # Achado pela conferencia no slot das 20h de 19/09 (video de Marte).
    "high-resolution": "Rái Rezolúchan",
    "mars express": "Márs Ecsprés",
    "wi-fi": "Uai Fai",
    "wifi": "Uai Fai",
    "bluetooth": "Blutúf",
    # Termos do nicho que o sintetizador le "em portugues"
    "software": "Sóftuer",
    "hardware": "Rárduer",
    "startup": "Startâp",
    "startups": "Startâps",
    "deepfake": "Dip Fêik",
    "deepfakes": "Dip Fêiks",
    "prompt": "Prômpt",
    "prompts": "Prômpts",
    "online": "Onláin",
    "benchmark": "Bêntchmark",
    "benchmarks": "Bêntchmarks",
    "machine learning": "Machín Lêrning",
    "deep learning": "Dip Lêrning",
    "open source": "Ôupen Sórs",
    "chatbot": "Tchat Bót",
    "chatbots": "Tchat Bóts",
    "agents.md": "Êidjents ponto ême dê",
    "claude.md": "Clôd ponto ême dê",
    "readme": "Rid Mi",
    "token": "Tôken",
    "tokens": "Tôkens",
    "streaming": "Strímin",
    "podcast": "Pódquést",
    "feed": "Fid",
    "cloud": "Cláud",
    "firmware": "Fãrmuer",
    "notebook": "Nôutbuk",
    "smartphone": "Smárt Fôun",
    "smartphones": "Smárt Fôuns",
}

# Siglas tecnicas soletradas letra a letra. So as conhecidas: sigla lida
# como palavra (NASA, OTAN) soletrada soaria errada.
SIGLAS = frozenset({
    "api", "apis", "gpu", "gpus", "cpu", "cpus", "npu", "tpu", "llm", "llms", "rtx",
    "amd", "ibm", "aws", "sdk", "cli", "url", "html", "css", "ssd", "hd", "ram",
    "ai", "agi", "usb", "vpn", "pdf", "ceo", "cto", "mit", "sql", "ux", "ui",
})

LETRAS = {
    "a": "á", "b": "bê", "c": "cê", "d": "dê", "e": "é", "f": "éfe", "g": "gê",
    "h": "agá", "i": "í", "j": "jota", "k": "cá", "l": "éle", "m": "ême", "n": "êne",
    "o": "ó", "p": "pê", "q": "quê", "r": "érre", "s": "ésse", "t": "tê", "u": "ú",
    "v": "vê", "w": "dáblio", "x": "xis", "y": "ípsilon", "z": "zê",
}

_PONTUACAO_BORDA = re.compile(r"^([\"'“”‘’(\[]*)(.*?)([\"'“”‘’)\].,;:!?…]*)$")


def _chave(texto: str) -> str:
    sem = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in sem if not unicodedata.combining(c))


def spell(sigla: str) -> str:
    """'GPU' -> 'gê pê ú' (plural 's' final vira 'ésse' so se fizer parte)."""
    base = sigla.lower()
    plural = base.endswith("s") and base[:-1] in SIGLAS
    letras = base[:-1] if plural else base
    falado = " ".join(LETRAS.get(c, c) for c in letras)
    return falado + ("s" if plural else "")


@dataclass
class Respelled:
    """Texto para a voz + alinhamento com as palavras originais.

    `groups[i]` = quantas palavras do texto falado nasceram da palavra
    original i. A legenda usa isso para devolver a grafia original com o
    tempo das palavras faladas.
    """

    original: list[str]
    spoken: list[str]
    groups: list[int] = field(default_factory=list)
    changes: list[tuple[str, str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.spoken)


def respell(texto: str) -> Respelled:
    palavras = texto.split()
    saida = Respelled(original=palavras, spoken=[])
    multi = sorted((k for k in LEXICO if " " in k), key=lambda k: -len(k.split()))
    i = 0
    while i < len(palavras):
        # Nome de varias palavras primeiro ("wall street journal").
        casou = False
        for k in multi:
            n = len(k.split())
            trecho = palavras[i:i + n]
            if len(trecho) < n:
                continue
            miolo = " ".join(_PONTUACAO_BORDA.match(p).group(2) for p in trecho)
            if _chave(miolo) == k:
                pre = _PONTUACAO_BORDA.match(trecho[0]).group(1)
                pos = _PONTUACAO_BORDA.match(trecho[-1]).group(3)
                falado = (pre + LEXICO[k] + pos).split()
                # A legenda mostra as n palavras originais: a primeira leva o
                # grupo falado inteiro, as outras ficam com zero.
                saida.spoken.extend(falado)
                saida.groups.append(len(falado))
                saida.groups.extend([0] * (n - 1))
                saida.changes.append((" ".join(trecho), LEXICO[k]))
                i += n
                casou = True
                break
        if casou:
            continue
        palavra = palavras[i]
        pre, miolo, pos = _PONTUACAO_BORDA.match(palavra).groups()
        falado_txt = _falar(miolo)
        if falado_txt != miolo:
            saida.changes.append((miolo, falado_txt))
        falado = (pre + falado_txt + pos).split() or [palavra]
        saida.spoken.extend(falado)
        saida.groups.append(len(falado))
        i += 1
    return saida


def _falar(token: str) -> str:
    """Uma palavra (sem pontuacao de borda) como a voz deve ler."""
    if not token:
        return token
    chave = _chave(token)
    if chave in LEXICO:
        return LEXICO[chave]
    # Hifenizado ("GPT-5", "Qwen3.8-27B"): cada parte pelo lexico, sem hifen.
    if "-" in token:
        partes = [p for p in token.split("-") if p]
        faladas = [_falar(p) for p in partes]
        if faladas != partes:
            return " ".join(faladas)
    # Letras coladas a numero ("RTX5090", "GPT5"): separa e resolve as letras.
    m = re.fullmatch(r"([A-Za-z]+)(\d[\d.,]*)", token)
    if m and (_chave(m.group(1)) in LEXICO or _chave(m.group(1)) in SIGLAS):
        return f"{_falar(m.group(1))} {m.group(2)}"
    if chave in SIGLAS and token.upper() == token or chave in SIGLAS and len(token) <= 4 \
            and token[:1].isupper():
        return spell(token)
    return token


__all__ = ["LEXICO", "Respelled", "SIGLAS", "respell", "spell"]
