"""Boca por fonema, nao por volume: visemas de pt-BR sobre o tempo do TTS.

O apresentador anterior abria a boca pela envoltoria de energia da narracao.
Numa silhueta escura isso passava -- o limite estava declarado em CLAUDE.md e a
escolha de polir amplitude primeiro foi do autor. Num rosto fotorrealista
nao passa: energia nao distingue "mamae" de "papai", e a boca fica **aberta no
/m/ e no /b/**, que e o gatilho classico de vale da estranheza. Quem le labios
sem saber que le labios e todo mundo; o erro nao precisa ser nomeado para
incomodar.

Aqui a forma da boca vem da letra e o tamanho vem do audio:

- **forma** -- cada palavra falada vira fonemas por regra de grafema de pt-BR,
  cada fonema vira um visema (abertura + largura), e os visemas se espalham na
  duracao que o edge-tts devolveu para aquela palavra. Determinista e $0: nao
  ha modelo, nao ha chamada, o mesmo texto da sempre a mesma boca.
- **tamanho** -- a envoltoria RMS modula a abertura. Sem ela, a boca declama
  com a mesma intensidade a frase inteira; com ela sozinha, a boca nao sabe
  fechar. Uma governa a outra: `presenter_video` multiplica as duas.

O fonema e do texto **falado** (o respelling de `voice/pronounce.py`, que
escreve "Djemini"), nunca do texto da legenda -- e a voz que a boca acompanha,
e a voz diz "Djemini".

Onde isto para, declarado: sao visemas de grafema, sem dicionario de excecao e
sem silaba tonica de verdade. "Exceto" ou "sublinhar" saem aproximados. A
diferenca que importa num feed vertical -- labio colado na consoante fechada,
boca redonda no /o/, boca espalhada no /i/ -- essa sai certa.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- visemas
# (abertura 0..1, largura -1 arredondada .. +1 espalhada, peso de duracao).
# A abertura e relativa: 1,0 e a silaba mais aberta que a boca faz, e quem
# converte isso em pixel e o `presenter_video` com a altura do rosto medida.


@dataclass(frozen=True)
class Visema:
    abertura: float
    largura: float
    peso: float
    fechado: bool = False   # bilabial: labio colado, custe o que custar


REPOUSO = Visema(0.0, 0.0, 1.0)

VISEMAS: dict[str, Visema] = {
    # vogais orais
    "a": Visema(1.00, 0.15, 1.00),
    "e": Visema(0.62, 0.45, 1.00),
    "i": Visema(0.34, 0.85, 0.90),
    "o": Visema(0.62, -0.55, 1.00),
    "u": Visema(0.30, -0.95, 0.90),
    # vogais nasais: a mesma boca, um pouco menos aberta (o ar sai pelo nariz)
    "a~": Visema(0.80, 0.10, 1.05),
    "e~": Visema(0.50, 0.35, 1.05),
    "i~": Visema(0.28, 0.70, 0.95),
    "o~": Visema(0.50, -0.50, 1.05),
    "u~": Visema(0.26, -0.85, 0.95),
    # consoantes
    "P": Visema(0.00, 0.00, 0.42, fechado=True),   # p, b, m
    "F": Visema(0.16, 0.25, 0.75),                 # f, v
    "T": Visema(0.28, 0.10, 0.45),                 # t, d, n, l, r simples
    "s": Visema(0.18, 0.50, 0.75),                 # s, z, c cedilha
    "S": Visema(0.30, -0.60, 0.80),                # ch, j, x, g antes de e/i
    "K": Visema(0.35, 0.05, 0.45),                 # c, g, qu, r forte, h
    "N": Visema(0.24, 0.10, 0.55),                 # nh
    "L": Visema(0.30, 0.15, 0.55),                 # lh
    ".": REPOUSO,                                  # silencio
}

VOGAIS = set("aeiouáéíóúâêôàãõ")
# Vogal anterior, com ou sem acento: e o que amolece `c` e `g` ("voce" e [v-o-s-e],
# nao [v-o-k-e]). Testar contra a string "ei" crua deixava "voce" e "inteligencia"
# com o [k] duro -- e a boca de [k] e visivelmente mais aberta que a de [s].
_ANTERIOR = frozenset("eiéêíî")
_ACENTO_ABRE = {"á": "a", "à": "a", "â": "a", "é": "e", "ê": "e",
                "í": "i", "ó": "o", "ô": "o", "ú": "u"}
_NASAL = {"ã": "a~", "õ": "o~"}


def _limpa(palavra: str) -> str:
    """Minusculas, sem pontuacao, acento preservado (ele muda o fonema)."""
    return re.sub(r"[^a-záéíóúâêôàãõçñ]", "", palavra.lower())


def fonemas(palavra: str) -> list[str]:
    """Grafema -> fonema de pt-BR, no nivel que a boca enxerga.

    Regras que valem a pena, porque sao as que mudam a forma do labio:

    - `m`/`n` **antes de consoante ou no fim** nasalizam a vogal e NAO colam o
      labio ("bom" nao fecha a boca; "boi**m**a" fecha). Tratar todo `m` como
      bilabial poria um labio colado onde o audio nao tem nenhum -- erro pior
      que o que este modulo veio corrigir.
    - `e`/`o` **atonos no fim** reduzem para [i]/[u]: "bonito" e "bonitu". E
      assim que o edge-tts fala, e a boca tem de concordar com a voz.
    - `l` **no fim de silaba** vira semivogal [u] ("sal" -> "sau"): labio
      arredondado, nao lingua no dente.
    """
    p = _limpa(palavra)
    if not p:
        return []
    saida: list[str] = []
    i, n = 0, len(p)
    while i < n:
        c = p[i]
        prox = p[i + 1] if i + 1 < n else ""
        depois = p[i + 2] if i + 2 < n else ""

        # --- digrafos
        if c == "c" and prox == "h":
            saida.append("S")
            i += 2
            continue
        if c == "l" and prox == "h":
            saida.append("L")
            i += 2
            continue
        if c == "n" and prox == "h":
            saida.append("N")
            i += 2
            continue
        if c == "r" and prox == "r":
            saida.append("K")
            i += 2
            continue
        if c == "s" and prox == "s":
            saida.append("s")
            i += 2
            continue
        if c in "sx" and prox == "c" and depois in _ANTERIOR:
            saida.append("s")
            i += 2
            continue
        if c == "q" and prox == "u":
            saida.append("K")
            if depois in ("a", "o", "á", "ó", "ô"):   # "quatro": sobra o [w]
                saida.append("u")
            i += 2
            continue
        if c == "g" and prox == "u" and depois in _ANTERIOR:
            saida.append("K")
            i += 2
            continue

        # --- vogais
        if c in VOGAIS:
            base = _NASAL.get(c) or _ACENTO_ABRE.get(c, c)
            # vogal + m/n travando silaba = vogal nasal, e o m/n some
            if prox in ("m", "n") and depois not in VOGAIS:
                base = base.rstrip("~") + "~"
                i += 1                    # consome tambem o m/n
            elif c not in _NASAL and _fim_atono(p, i, c):
                base = "i" if base == "e" else "u"
            saida.append(base)
            i += 1
            continue

        # --- consoantes
        if c in "pb":
            saida.append("P")
        elif c == "m":
            # so chega aqui o m que a vogal NAO consumiu como nasal: bilabial.
            saida.append("P")
        elif c in "fv":
            saida.append("F")
        elif c in "tdn":
            saida.append("T")
        elif c == "l":
            # final de silaba (sem vogal em seguida) vira [u] velarizado
            saida.append("T" if prox in VOGAIS else "u")
        elif c == "r":
            # inicio de palavra ou depois de n/l/s = r forte; entre vogais = tap
            forte = i == 0 or (i > 0 and p[i - 1] in "nls")
            saida.append("K" if forte else "T")
        elif c in "zç":
            saida.append("s")
        elif c == "s":
            # entre vogais o som e [z], mas o labio faz a mesma coisa nos dois.
            saida.append("s")
        elif c == "c":
            saida.append("s" if prox in _ANTERIOR else "K")
        elif c == "g":
            saida.append("S" if prox in _ANTERIOR else "K")
        elif c in "jx":
            saida.append("S")
        elif c in "kw":
            saida.append("K")
        elif c == "y":
            saida.append("i")
        elif c == "ñ":
            saida.append("N")
        # 'h' mudo cai fora sozinho
        i += 1
    return saida


def _fim_atono(p: str, i: int, c: str) -> bool:
    """`e`/`o` sem acento na ultima letra da palavra (reduz para [i]/[u])."""
    return c in "eo" and i == len(p) - 1 and len(p) > 2


def visemas(palavra: str) -> list[Visema]:
    return [VISEMAS.get(f, VISEMAS["T"]) for f in fonemas(palavra)]


# ------------------------------------------------------------------ trilha
# Amostras por quadro no calculo da dinamica. Nao e capricho: e a mesma licao
# ja paga na envoltoria do apresentador parado -- uma constante de tempo de 40
# ms e MAIS CURTA que o quadro de 33,3 ms, entao rodar o filtro na taxa de
# quadro nao filtra nada. A 4x (120 Hz) ela vale 5 amostras e de fato suaviza;
# so depois a curva desce para o quadro, por media.
SUB = 4
# Constante de tempo do maxilar e do labio. Sao diferentes de proposito: o
# maxilar e osso com massa e chega devagar; o labio (espalhar no /i/,
# arredondar no /u/) e leve e chega antes. Usar um numero so para os dois
# deixava a boca inteira com a mesma cadencia, que e metade do que denuncia
# marionete.
TAU_MAXILAR = 0.035
TAU_LABIO = 0.024
# A consoante fechada tem de CHEGAR a zero, senao nao le como labio colado.
# Reimposta na taxa de SUB (nao na de quadro) e com janela de cosseno elevado:
# em 30 fps um "V" de um quadro so lia como piscada da boca, um defeito no
# lugar do outro. No sub-quadro o fecho e continuo, e a media do quadro vira
# sozinha o borrao de um gesto rapido -- que e o que a camera faria.
FECHO_S = 0.055
# Buraco entre palavras que ja conta como pausa: abaixo disso a boca segue de
# uma palavra para a outra sem passar pelo repouso, que e como se fala.
PAUSA_S = 0.12


def _inercia(alvo: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Resposta criticamente amortecida de `alvo`: boca com massa.

    O interpolador anterior segurava o visema parado e saltava para o proximo
    em 70 ms. Medido na narracao real de 16,5 s: salto medio de 0,109 entre
    quadros e 0,381 no p95 -- a boca andava 38% do curso em 33 ms. Isso le como
    estalo, nao como fala, e nenhum ajuste de forma do visema conserta, porque
    o defeito esta na DINAMICA e nao na pose.

    Massa-mola criticamente amortecida chega rapido, nao oscila, e sobretudo
    **nao alcanca o alvo quando o alvo muda depressa**. Essa falta de alcance e
    a coarticulacao: na fala corrida ninguem articula cada fonema por inteiro,
    e era exatamente isso que a escada fazia -- pronunciava tudo com a mesma
    perfeicao, que e o jeito mais rapido de soar robo.
    """
    y = np.empty_like(alvo)
    pos = float(alvo[0])
    vel = 0.0
    k = 1.0 / (tau * tau)
    c = 2.0 / tau
    for i in range(len(alvo)):
        vel += dt * (k * (float(alvo[i]) - pos) - c * vel)
        pos += dt * vel
        y[i] = pos
    return y


def _degraus(alvos: list[tuple[float, float, Visema]], campo: str,
             t: np.ndarray) -> np.ndarray:
    """Alvo constante por fonema: cada visema vale a DURACAO dele, nao um ponto.

    Segurar o visema so no centro do fonema e interpolar entre centros dava uma
    rampa permanente, sem patamar nenhum; com o patamar, quem decide se o
    fonema e alcancado ou nao passa a ser a inercia -- que e quem decide na
    boca de verdade.
    """
    fora = np.zeros(len(t), dtype=np.float32)
    for ini, fim, v in alvos:
        fora[(t >= ini) & (t < fim)] = getattr(v, campo)
    return fora


def trilha(palavras, n: int, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """Abertura e largura da boca em cada um dos `n` quadros.

    `palavras` sao os tempos do TTS (`voice/edge.WordTiming`) do texto FALADO.
    Entre duas palavras a boca volta ao repouso se a pausa passar de
    `PAUSA_S` -- e o que separa palavra de palavra visualmente.
    """
    if n <= 0:
        return (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))

    # --- 1. o que a boca deveria fazer, fonema a fonema, com duracao
    alvos: list[tuple[float, float, Visema]] = []
    fechos: list[tuple[float, float]] = []
    fim_anterior = 0.0
    for w in palavras:
        ini = float(getattr(w, "start_s", getattr(w, "start", 0.0)))
        fim = float(getattr(w, "end_s", getattr(w, "end", 0.0)))
        vs = visemas(getattr(w, "text", ""))
        if not vs or fim <= ini:
            continue
        # Descanso no FIM da palavra anterior, e nao so um pouco antes da
        # proxima: com o descanso so na entrada, o ultimo visema da palavra
        # valia ate a palavra seguinte comecar -- numa pausa de 2,4 s entre
        # frases a boca ficava escancarada no /a/ final o tempo todo.
        if fim_anterior > 0 and ini - fim_anterior > PAUSA_S:
            alvos.append((fim_anterior + 0.04, ini - 0.04, REPOUSO))
        total = sum(v.peso for v in vs)
        t = ini
        for v in vs:
            dur = (fim - ini) * v.peso / total
            alvos.append((t, t + dur, v))
            if v.fechado:
                fechos.append((t, t + dur))
            t += dur
        fim_anterior = fim
    if not alvos:
        return (np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32))

    # --- 2. a dinamica, no sub-quadro
    m = n * SUB
    dt = 1.0 / (fps * SUB)
    t_sub = np.arange(m, dtype=np.float32) * dt
    abertura = _inercia(_degraus(alvos, "abertura", t_sub), dt, TAU_MAXILAR)
    largura = _inercia(_degraus(alvos, "largura", t_sub), dt, TAU_LABIO)

    # --- 3. o labio colado, reimposto depois da dinamica
    # Entre duas vogais abertas a inercia nunca chega a zero num /p/ de 40 ms:
    # sobraria um respiro de abertura justamente onde o espectador confere o
    # labio sem saber que confere.
    for ini, fim in fechos:
        centro = (ini + fim) / 2
        k0 = max(0, int((centro - FECHO_S) / dt))
        k1 = min(m, int((centro + FECHO_S) / dt) + 1)
        if k1 <= k0:
            continue
        d = (t_sub[k0:k1] - centro) / FECHO_S
        # cosseno elevado: vale 0 no centro e volta a 1 nas pontas, sem quina
        abertura[k0:k1] *= (1 - np.cos(np.pi * np.clip(np.abs(d), 0, 1))) / 2

    # --- 4. de volta para o quadro, por media (o borrao de quem filma)
    abertura = np.clip(abertura, 0.0, 1.0).reshape(n, SUB).mean(axis=1)
    largura = np.clip(largura, -1.0, 1.0).reshape(n, SUB).mean(axis=1)
    return abertura.astype(np.float32), largura.astype(np.float32)


def texto_falado(respelled) -> str:
    """O texto que a voz diz, para conferencia em teste e no log."""
    return " ".join(getattr(respelled, "spoken", []) or [])


def _sem_acento(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    return "".join(c for c in d if not unicodedata.combining(c))


__all__ = ["FECHO_S", "PAUSA_S", "REPOUSO", "SUB", "TAU_LABIO", "TAU_MAXILAR",
           "VISEMAS", "Visema",
           "fonemas", "texto_falado", "trilha", "visemas"]
