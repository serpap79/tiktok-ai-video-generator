"""Apresentador animado: um retrato parado vira alguem falando, sem modelo.

O desenho anterior pegava o PNG do apresentador e passeava uma **janela
retangular** de 340x640 por cima dele. Duas coisas davam errado ao mesmo tempo:
a janela cortava a silhueta (o recorte alfa nao servia para nada, o que se via
era um retangulo de rosto) e o unico movimento era o da janela -- a pessoa
dentro dela ficava imovel. Na tela isso le como figurinha colada, nao como
apresentador.

Aqui a silhueta e sempre inteira e quem se mexe e ela. Quatro camadas de
movimento, todas deterministicas e de graca (nenhum modelo, nenhuma API):

1. **Boca pela voz.** A envoltoria de energia da narracao (RMS por quadro, com
   ataque rapido e relaxamento lento, como um compressor) abre o maxilar: um
   campo de deslocamento vertical que nasce em zero na base do nariz, chega ao
   maximo no queixo e volta a zero no pescoco. Por cima, a abertura da boca e
   desenhada entre os labios, com dente aparecendo so nas silabas mais abertas.
   Nao e viseme -- e amplitude. A 3 metros de distancia de um feed vertical, a
   diferenca entre os dois e invisivel; a diferenca entre boca parada e boca
   que acompanha o audio e o video inteiro.
2. **Piscada.** A palpebra desce de verdade: a faixa entre a sobrancelha e a
   palpebra inferior e comprimida contra a linha do cilio, entao a pele de cima
   cobre o olho. Intervalo sorteado com semente fixa (2,6 a 6 s) -- o mesmo
   roteiro pisca sempre igual, o que torna o quadro conferivel em teste.
3. **Cabeca.** A cabeca e o tronco sao duas camadas com mascara complementar
   (a soma das duas da exatamente o alfa original, entao com angulo zero o
   quadro e identico ao retrato). A cabeca gira em torno de um pivo dentro do
   peito, com balanco lento + acento nas silabas fortes; o tronco respira.
4. **Encenacao pelo roteiro.** Tamanho e lugar do apresentador mudam com o que
   ele esta dizendo: grande e ao centro na chamada, pequeno no canto enquanto o
   material de apoio conta a historia, e de volta para o fechamento. E isso que
   `plan` calcula a partir dos tempos de palavra -- movimento *em relacao ao
   contexto*, nao movimento por movimento.

A camada sai num arquivo so, com **duas trilhas de video**: a cor (ja
pre-multiplicada pelo alfa) e a mascara em cinza. Nao e capricho -- e o que
cabe em disco. Medido nesta maquina, 67s de camada RGBA sem perda em qtrle dao
1569 MB; as duas trilhas em h264 dao 8,7 MB, e o composto final difere em media
0,7 de 255 por pixel. O VP9 com alfa deste ffmpeg entrega o alfa opaco (testado
antes de escolher), e RGB comum comprimido ganharia franja escura na silhueta --
por isso pre-multiplicado, que faz a cor cair a zero junto com a mascara.

Ficar num arquivo separado e de proposito: da para abrir o artefato e olhar o
apresentador sozinho, sem o video atras.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from agent.render.typography import MONO, font

W, H = 1080, 1920
FPS = 30

# ---------------------------------------------------------------- encenacao
# Alturas em pixel na tela (1920 de altura). O guia da marca pede 30-36% para o
# apresentador de canto; a chamada e o fechamento passam disso de proposito --
# ali ele e o assunto, nao a assinatura.
ALTURA_CHAMADA = 980
ALTURA_CORPO = 600
ALTURA_FECHAMENTO = 640
BASE_CHAMADA = 1740
BASE_CORPO = 1700
CX_CHAMADA = 540
CX_CORPO = 320
CX_FECHAMENTO = 340
# Tempo de travessia entre duas encenacoes (suavizado, sem corte seco).
TRANSICAO_S = 0.7
# Area util da camada: tudo o que a encenacao alcanca, com folga para a
# rotacao e para a plaquinha de nome. Menor que o quadro inteiro porque cada
# pixel a mais e um pixel copiado em cada um dos ~2000 quadros.
AREA = (0, 660, 1080, 1160)  # x, y, largura, altura
# Legenda sobe quando ha apresentador. Conferido nas duas encenacoes de canto:
# texto centrado em 940 (fonte 84, topo em ~898) contra topo do avatar em 1100
# (corpo) e 1060 (fechamento) -- sobra mais de 60px nas duas.
Y_LEGENDA_COM_APRESENTADOR = 940

# ------------------------------------------------------------------ boca
# Queda maxima do queixo, como fracao da altura do rosto. Fala normal abre a
# mandibula entre 10% e 15% da altura do rosto; 10,5% porque acima disso o
# rosto visivelmente alonga nas silabas abertas (olhado quadro a quadro).
MAXILAR = 0.105
# Abertura entre os labios, como fracao da queda do queixo naquele quadro.
ABERTURA = 0.80
# So aparece dente acima desta abertura (senao todo "a" vira sorriso).
DENTE_MIN = 0.52
# Super-amostragem do ladrilho da boca. O `ImageDraw` do Pillow nao tem
# anti-aliasing nenhum -- medido em 20/09/2026, a elipse da boca saia com ZERO
# pixel de borda parcial em 12, 26 e 44 px de altura. Desenhar 4x maior e
# reduzir por media de area da a borda suave por 16 amostras de um ladrilho de
# ~150x70 px, o que nao aparece no relogio do quadro.
SS = 4
# Abaixo disso a boca entra em degrade (ease cubico) em vez de aparecer de uma
# vez com 2 px de altura, que era a unica descontinuidade que sobrava.
BOCA_MIN_PX = 3.5
# Borrao do labio. Era 1,0 no piso para disfarcar o serrilhado; com a borda ja
# suave ele pode ser menor, e a boca fica menos empastada.
BOCA_BORRAO_MIN = 0.6
ATAQUE_S = 0.035
RELAXAMENTO_S = 0.085
# Quantas amostras de envoltoria por quadro. O ataque de 35 ms e MAIS CURTO que
# um quadro de 33,3 ms: rodando o filtro na taxa de quadro ele nao filtrava
# nada e a boca abria de um quadro para o outro. A 4x (120 Hz) o mesmo ataque
# sao 4 amostras e ele de fato suaviza; so depois a curva desce para o quadro.
ENV_SUB = 4
# A curva da envoltoria, calibrada na narracao real de 67s de 20/09/2026. A
# primeira versao normalizava pelo p92 com gamma 0,7 e a boca ficava em 0,68 de
# abertura na mediana -- ou seja, escancarada o video inteiro. Descontar o piso
# de ruido (p20) e puxar o meio para baixo (gamma 1,3) devolve a distribuicao de
# fala de verdade: mediana 0,37, 31% dos quadros de boca quase fechada, 10%
# aberta de par em par.
# Recalibrados em 20/09/2026 junto com a janela de Hann. A janela com
# sobreposicao preenche os silencios curtos entre silabas -- que e justamente
# o que tira a tremedeira --, mas com isso ela levanta o piso: nos mesmos 20s
# de narracao a mediana subia de 0,39 para 0,44 e a boca passava a descansar
# mais aberta. p30/p92 devolve a distribuicao antiga: os picos batem na mosca
# (16% dos quadros de boca aberta de par em par, identico) e a mediana fica
# em 0,43 contra 0,39 -- 4 centesimos, que numa abertura maxima de 34 px sao
# 1,4 px. Troca aceita de propostio: o pico e o que le como expressivo.
ENV_PISO_PCT = 30
ENV_TETO_PCT = 92
ENV_GAMA = 1.3

# ---------------------------------------------------------------- piscada
# Piscada assimetrica: fecha rapido, segura um instante, abre devagar. A
# versao anterior era `sin(pi*u)**0.6` numa janela de 0,13 s -- medido, 2
# quadros fechando e 2 abrindo, com o primeiro quadro saltando de 0,00 para
# 0,82. Simetrica e quase um corte: e dai que vinha o ar de mecanismo.
PISCADA_FECHA_S = 0.055    # ~1,7 quadros a 30 fps
PISCADA_SEGURA_S = 0.035   # ~1 quadro com o olho fechado
PISCADA_ABRE_S = 0.115     # ~3,5 quadros -- o dobro do fechamento
PISCADA_S = PISCADA_FECHA_S + PISCADA_SEGURA_S + PISCADA_ABRE_S
PISCADA_MIN_S = 2.6
PISCADA_MAX_S = 6.0
# O quanto uma piscada pode ser puxada para a fronteira de frase mais proxima.
PISCADA_ANCORA_S = 0.9
# Quanto a palpebra inferior e a bochecha sobem no fecho total, em fracao da
# altura do olho. Piscar nao e so a palpebra de cima: o orbicular puxa o que
# esta em volta, e e a ausencia disso que faz o olho parecer uma persiana.
SOBE_VIZINHO = 0.20
# Ate onde a caixa do olho desce, em alturas de olho, para alcancar a bochecha.
CAIXA_BOCHECHA = 2.4
# Meia-largura da sombra do cilio, em pixel.
CILIO_PX = 2.2
# Teto da compressao abaixo do cilio. Espremer uma faixa de altura fixa numa
# faixa que vai a zero da inclinacao infinita: com o olho quase fechado ela
# chegava a 38x e saturava a caixa inteira (medido: 47 linhas grudadas na
# ultima). Acima deste teto o que sobra do olho ja esta coberto pela palpebra
# de qualquer jeito -- nao ha o que espremer.
COMPRESSAO_MAX = 3.0
# Em quantas alturas de olho a compressao volta a ser identidade, contadas a
# partir do cilio de baixo. Curto de proposito: abaixo do olho e bochecha, e
# bochecha nao comprime, so sobe.
SOLTA_ALTO = 0.6
# Folga lateral da caixa do olho, em alturas de olho. E tambem a largura da
# pena: a deformacao vale 1 no canto do olho e 0 na borda da caixa, entao a
# coluna de dentro e a de fora encostam sem salto.
PENA_OLHO = 0.55

# ----------------------------------------------------------------- cabeca
GIRO_CABECA = 1.7   # graus, balanco lento
GIRO_TRONCO = 0.8   # graus
ACENTO_CABECA = 1.1  # graus a mais nas silabas fortes
RESPIRO = 0.005     # fracao da altura


# ===========================================================================
# retrato medido
# ===========================================================================


@dataclass(frozen=True)
class Retrato:
    """O recorte do apresentador com os pontos do rosto ja medidos.

    Vem de `scripts/make_presenter_cutouts.py`; aqui nada e adivinhado.
    """

    id: str
    imagem: Image.Image
    pontos: dict[str, tuple[float, float]]
    head_top: int
    neck_y: int
    face_height: float
    pivot: tuple[float, float]

    @property
    def size(self) -> tuple[int, int]:
        return self.imagem.size

    def p(self, nome: str) -> tuple[float, float]:
        return self.pontos[nome]


def carregar(png: Path) -> Retrato:
    meta_path = png.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} nao existe -- rode scripts/make_presenter_cutouts.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return Retrato(
        id=meta.get("id", png.stem),
        imagem=Image.open(png).convert("RGBA"),
        pontos={k: (v[0], v[1]) for k, v in meta["points"].items()},
        head_top=int(meta["head_top"]),
        neck_y=int(meta["neck_y"]),
        face_height=float(meta["face_height"]),
        pivot=(float(meta["pivot"][0]), float(meta["pivot"][1])),
    )


# ===========================================================================
# tempo: envoltoria da voz e batidas do roteiro
# ===========================================================================


def envoltoria(audio: Path, n: int, fps: int = FPS, *, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Energia da narracao por quadro, de 0 (silencio) a 1 (silaba mais aberta).

    Ataque rapido e relaxamento lento porque boca tem inercia: fecha mais
    devagar do que abre, e sem isso a animacao treme a cada consoante.

    Medido em 20/09/2026: a versao anterior lia o RMS em janelas retangulares
    de um quadro inteiro (33,3 ms) **sem sobreposicao e sem janelamento**, e o
    ataque de 25 ms -- mais curto que um quadro -- nao filtrava coisa nenhuma.
    O sinal de controle chegava a geometria ja serrilhado no tempo, e a boca
    abria de um quadro para o outro: e dai que vinha metade do ar mecanico.

    Agora sao tres passos, na ordem que importa: RMS com janela de Hann a
    `ENV_SUB` vezes a taxa de quadro, ataque/relaxamento **nessa** taxa (onde
    35 ms sao 4 amostras e o filtro tem o que fazer), e so no fim a media do
    grupo desce a curva para o quadro. Filtrar antes de decimar e a mesma
    ideia da super-amostragem da boca, aplicada ao eixo do tempo.
    """
    bruto = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(audio), "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"], capture_output=True, timeout=300).stdout
    x = np.frombuffer(bruto, dtype="<i2").astype(np.float32) / 32768.0
    if x.size == 0:
        return np.zeros(n, dtype=np.float32)
    taxa = fps * ENV_SUB
    m = n * ENV_SUB
    passo = 16000 / taxa
    # Janela de Hann com 50% de sobreposicao, numa convolucao so do sinal ao
    # quadrado: o RMS de todas as posicoes de uma vez, sem laco em Python.
    largura = max(int(round(passo * 2)), 3)
    peso = np.hanning(largura + 2)[1:-1].astype(np.float32)
    peso /= float(peso.sum())
    energia = np.convolve(np.square(x), peso, mode="same")
    centros = np.clip(((np.arange(m) + 0.5) * passo).astype(np.int64), 0, x.size - 1)
    rms = np.sqrt(np.maximum(energia[centros], 0.0)).astype(np.float32)
    fala = rms[rms > 1e-4]
    if not fala.size:
        return np.zeros(n, dtype=np.float32)
    piso = float(np.percentile(fala, ENV_PISO_PCT))
    teto = float(np.percentile(fala, ENV_TETO_PCT))
    nivel = np.clip((rms - piso) / max(teto - piso, 1e-6), 0.0, 1.0) ** ENV_GAMA
    return _ataque_relaxamento(nivel, taxa).reshape(n, ENV_SUB).mean(axis=1)


def _ataque_relaxamento(nivel: np.ndarray, taxa: int) -> np.ndarray:
    """Um polo por sentido, na taxa em que o sinal vive (nao na do quadro)."""
    sobe = math.exp(-1.0 / max(ATAQUE_S * taxa, 1e-6))
    desce = math.exp(-1.0 / max(RELAXAMENTO_S * taxa, 1e-6))
    saida = np.empty_like(nivel)
    y = 0.0
    for i, alvo in enumerate(nivel):
        a = sobe if alvo > y else desce
        y = a * y + (1 - a) * float(alvo)
        saida[i] = y
    return saida


@dataclass(frozen=True)
class Batidas:
    """Onde o roteiro muda de assunto, em segundos."""

    hook_end: float
    closing_start: float
    duration: float
    sentencas: tuple[float, ...] = ()


def batidas_por_tempo(script_hook: str, script_closing: str, palavras, duracao: float
                      ) -> Batidas:
    """Batidas exatas a partir do tempo de cada palavra que o TTS devolveu."""
    n_hook = len(script_hook.split())
    n_close = len(script_closing.split())
    total = len(palavras)
    if total == 0:
        return batidas_estimadas(n_hook, n_close, total, duracao)
    fim_gancho = palavras[min(n_hook, total) - 1].end
    inicio_fecho = palavras[max(0, total - n_close)].start
    sentencas = [p.start for i, p in enumerate(palavras)
                 if i and palavras[i - 1].text.endswith((".", "!", "?"))]
    return Batidas(hook_end=min(fim_gancho, duracao),
                   closing_start=min(max(inicio_fecho, fim_gancho), duracao),
                   duration=duracao, sentencas=tuple(sentencas))


def batidas_estimadas(n_hook: int, n_close: int, total: int, duracao: float) -> Batidas:
    """Reserva quando nao ha tempo de palavra (caminho do MPT): por proporcao."""
    total = max(total, n_hook + n_close, 1)
    return Batidas(hook_end=duracao * n_hook / total,
                   closing_start=duracao * (total - n_close) / total,
                   duration=duracao)


# ===========================================================================
# encenacao
# ===========================================================================


@dataclass(frozen=True)
class Marca:
    """Uma pose de encenacao: quando, de que tamanho e onde."""

    t: float
    altura: float
    cx: float
    base: float
    alfa: float = 1.0


@dataclass(frozen=True)
class Encenacao:
    """O plano de cena inteiro: o apresentador e o que ele empurra na tela.

    A legenda e o cartao do gancho nao sao independentes do apresentador --
    disputam o mesmo espaco. Quem decide e a encenacao, num lugar so.
    """

    marcas: list[Marca]
    legenda_y: int
    legenda_inicio: float
    cartao_s: float


def plan(batidas: Batidas) -> Encenacao:
    """Chamada grande ao centro -> canto durante o corpo -> volta no fechamento.

    O apresentador sai do centro assim que o material de apoio passa a contar a
    historia, e volta quando o texto volta a falar com quem assiste (o CTA).
    Nao e enfeite: e a regra de para onde o olho deve ir em cada trecho.

    Enquanto ele esta grande, quem escreve o gancho e o cartao -- a legenda
    karaoke so entra quando ele encolhe. Sem isso o mesmo texto apareceria
    duas vezes na tela e a segunda cairia em cima do rosto dele.
    """
    fim_gancho = max(1.2, min(batidas.hook_end, batidas.duration - 0.5))
    inicio_fecho = max(fim_gancho + TRANSICAO_S,
                       min(batidas.closing_start, batidas.duration - 0.4))
    marcas = [
        Marca(t=0.0, altura=ALTURA_CHAMADA, cx=CX_CHAMADA, base=BASE_CHAMADA, alfa=0.0),
        Marca(t=0.45, altura=ALTURA_CHAMADA, cx=CX_CHAMADA, base=BASE_CHAMADA),
        Marca(t=fim_gancho, altura=ALTURA_CHAMADA, cx=CX_CHAMADA, base=BASE_CHAMADA),
        Marca(t=fim_gancho + TRANSICAO_S, altura=ALTURA_CORPO, cx=CX_CORPO,
              base=BASE_CORPO),
        Marca(t=inicio_fecho, altura=ALTURA_CORPO, cx=CX_CORPO, base=BASE_CORPO),
        Marca(t=inicio_fecho + TRANSICAO_S, altura=ALTURA_FECHAMENTO,
              cx=CX_FECHAMENTO, base=BASE_CORPO),
        Marca(t=batidas.duration, altura=ALTURA_FECHAMENTO, cx=CX_FECHAMENTO,
              base=BASE_CORPO),
    ]
    marcas = [m for i, m in enumerate(marcas) if i == 0 or m.t > marcas[i - 1].t]
    troca = fim_gancho + TRANSICAO_S
    return Encenacao(marcas=marcas, legenda_y=Y_LEGENDA_COM_APRESENTADOR,
                     legenda_inicio=troca, cartao_s=troca)


def _suave(u: float) -> float:
    """Aceleracao e freada (smoothstep): encenacao nao troca de lugar em corte."""
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _ease_cubico(u: float) -> float:
    """Ease-in-out cubico: sai de zero e chega a um com velocidade zero.

    Mais lento no comeco e no fim que o smoothstep, e e isso que se quer numa
    palpebra: o que denuncia mecanismo e a quina na partida e na chegada.
    """
    u = min(1.0, max(0.0, u))
    return 4 * u * u * u if u < 0.5 else 1 - ((-2 * u + 2) ** 3) / 2


def pose(marcas: list[Marca], t: float) -> Marca:
    if t <= marcas[0].t:
        return marcas[0]
    for a, b in zip(marcas, marcas[1:], strict=False):
        if t <= b.t:
            u = _suave((t - a.t) / max(b.t - a.t, 1e-6))
            return Marca(t=t,
                         altura=a.altura + (b.altura - a.altura) * u,
                         cx=a.cx + (b.cx - a.cx) * u,
                         base=a.base + (b.base - a.base) * u,
                         alfa=a.alfa + (b.alfa - a.alfa) * u)
    return marcas[-1]


# ===========================================================================
# deformacoes no retrato
# ===========================================================================


def _reamostra_y(bloco: np.ndarray, desloc: np.ndarray) -> np.ndarray:
    """`saida[y] = bloco[y - desloc[y]]`, com interpolacao linear entre linhas."""
    h = bloco.shape[0]
    ys = np.clip(np.arange(h, dtype=np.float32) - desloc, 0, h - 1.001)
    lo = ys.astype(np.int32)
    fr = (ys - lo).astype(np.float32)[:, None, None]
    return (bloco[lo] * (1 - fr) + bloco[lo + 1] * fr)


@dataclass
class Maxilar:
    """Regiao do rosto que a fala move, pre-calculada uma vez."""

    caixa: tuple[int, int, int, int]
    perfil: np.ndarray          # deslocamento por linha, para abertura 1.0
    peso_x: np.ndarray          # 1 na mandibula, 0 fora dela (bordas suaves)
    boca_cx: float
    boca_y: float
    boca_w: float
    cor_boca: tuple[int, int, int]
    cor_dente: tuple[int, int, int]
    queda_labio: float          # deslocamento da linha dos labios, abertura 1.0


def preparar_maxilar(r: Retrato) -> Maxilar:
    """Onde a fala mexe o rosto, medido nos pontos -- nada e chute.

    O perfil nao e uma rampa do nariz ao queixo: na fala o labio de cima fica
    parado e o de baixo desce **junto com o queixo**. Entao o deslocamento sobe
    de zero a cheio num trecho curto logo abaixo da linha dos labios (e esse
    trecho esticado e justamente a boca abrindo), segue cheio ate o queixo e
    volta a zero no pescoco. A primeira versao rampava do subnasal ao queixo e
    o labio inferior descia so um terco do que devia: a boca virava um risco.
    """
    sub_y = r.p("subnasal")[1]
    labio_y = r.p("labio_sup")[1]
    queixo_y = r.p("queixo")[1]
    fim = queixo_y + 0.55 * (queixo_y - sub_y)
    x0 = int(max(0, r.p("face_esq")[0] - 30))
    x1 = int(min(r.size[0], r.p("face_dir")[0] + 30))
    y0 = int(max(0, sub_y - 6))
    y1 = int(min(r.size[1], fim + 8))

    linhas = np.arange(y0, y1, dtype=np.float32)
    amp = MAXILAR * r.face_height
    abre = max(0.30 * (queixo_y - labio_y), 8.0)
    sobe = np.clip((linhas - labio_y) / abre, 0, 1)
    sobe = sobe * sobe * (3 - 2 * sobe)      # sem aresta na linha dos labios
    volta = np.clip(1 - (linhas - queixo_y) / max(fim - queixo_y, 1e-6), 0, 1)
    perfil = (amp * np.minimum(sobe, volta)).astype(np.float32)

    colunas = np.arange(x0, x1, dtype=np.float32)
    m_esq, m_dir = r.p("mand_esq")[0], r.p("mand_dir")[0]
    pena = 46.0
    peso = np.minimum(np.clip((colunas - (m_esq - pena)) / pena, 0, 1),
                      np.clip(((m_dir + pena) - colunas) / pena, 0, 1))

    px = np.asarray(r.imagem.convert("RGB")).astype(np.float32)
    cx, cy = r.p("labio_sup")
    amostra = px[int(cy) - 4:int(cy) + 5, int(cx) - 14:int(cx) + 15].reshape(-1, 3)
    labio = amostra.mean(axis=0) if amostra.size else np.array([120.0, 80.0, 80.0])
    # Dente sai do tom claro do proprio rosto (a bochecha iluminada), nao de uma
    # cor fixa: cada apresentador tem sua exposicao.
    face = px[int(r.p("olho_esq_baixo")[1]):int(queixo_y),
              int(r.p("mand_esq")[0]):int(r.p("mand_dir")[0])].reshape(-1, 3)
    claro = (np.percentile(face, 88, axis=0) if face.size
             else np.array([200.0, 190.0, 185.0]))
    dente = claro * 0.55 + float(claro.mean()) * 0.52
    return Maxilar(
        caixa=(x0, y0, x1, y1), perfil=perfil,
        peso_x=peso.astype(np.float32)[None, :, None],
        boca_cx=(r.p("boca_esq")[0] + r.p("boca_dir")[0]) / 2,
        boca_y=labio_y,
        boca_w=r.p("boca_dir")[0] - r.p("boca_esq")[0],
        cor_boca=tuple(int(c) for c in np.clip(labio * 0.22, 6, 70)),
        cor_dente=tuple(int(c) for c in np.clip(dente, 60, 244)),
        queda_labio=float(amp),
    )


def aplicar_maxilar(quadro: np.ndarray, m: Maxilar, abertura: float) -> None:
    """Abre o maxilar no lugar (quadro e modificado)."""
    if abertura <= 0.01:
        return
    x0, y0, x1, y1 = m.caixa
    bloco = quadro[y0:y1, x0:x1].astype(np.float32)
    movido = _reamostra_y(bloco, m.perfil * abertura)
    quadro[y0:y1, x0:x1] = (bloco + (movido - bloco) * m.peso_x).astype(np.uint8)


def tile_boca(m: Maxilar, abertura: float) -> tuple[Image.Image, tuple[int, int]] | None:
    """A abertura entre os labios, desenhada num ladrilho pequeno.

    Num quadro de 1072x1053 o borrao gaussiano da imagem inteira custaria mais
    que todo o resto do quadro somado; aqui ele roda numa caixa de ~300x200.

    Dois defeitos morreram aqui em 20/09/2026, os dois medidos antes:

    1. **O `ImageDraw` do Pillow nao tem anti-aliasing.** A elipse saia com
       zero pixel de borda parcial (conferido em 12, 26 e 44 px de altura) --
       degrau duro, que era o serrilhado que se via no labio. O borrao
       gaussiano depois disso nao resolve: o degrau ja esta assado.
    2. **O canto da colagem era truncado em inteiro.** Entre abertura 0,36 e
       0,39 o canto saltava 1 px de lado enquanto a boca crescia 1 px de
       altura: ela tremia no lugar de crescer.

    Os dois somem na mesma conta. As mascaras sao desenhadas em `SS` vezes o
    tamanho e reduzidas por media de area (`Image.BOX`, que para fator inteiro
    e a media exata do bloco); e a **parte fracionaria** da posicao entra nas
    coordenadas do desenho grande, entao o ladrilho ja nasce deslocado de
    um quarto de pixel e a colagem segue em inteiro.

    So a mascara e super-amostrada, nunca a cor: reduzir RGBA sobre fundo
    transparente puxaria a cor para o preto na borda e devolveria a mesma
    franja escura que `encode_command` existe para evitar. A cor do ladrilho e
    chapada em todo pixel -- inclusive onde o alfa e zero.
    """
    altura = m.queda_labio * abertura * ABERTURA
    if altura < BOCA_MIN_PX * 0.5:
        return None
    # Entrada suave: a boca cresce de zero em vez de surgir com 2 px de uma
    # vez. E a ultima descontinuidade que sobrava no caminho da abertura.
    surge = _ease_cubico(min(1.0, altura / BOCA_MIN_PX))
    # Boca aberta e mais estreita que boca fechada: o labio se recolhe.
    largura = m.boca_w * (0.78 - 0.12 * abertura)
    pad = 30

    # O buraco comeca na linha do labio de cima (ponto 13 da malha e o labio
    # INTERNO), com 8% de sobreposicao para nao deixar fresta. A posicao exata
    # e fracionaria: o inteiro vai para a colagem, o resto para o desenho.
    fx = m.boca_cx - largura / 2 - pad
    fy = m.boca_y - altura * 0.08 - pad
    canto = (math.floor(fx), math.floor(fy))
    sx, sy = (fx - canto[0]) * SS, (fy - canto[1]) * SS
    w, h = int(largura + 2 * pad) + 1, int(altura + 2 * pad) + 1

    caixa = (pad * SS + sx, pad * SS + sy,
             pad * SS + sx + largura * SS, pad * SS + sy + altura * SS)
    grande = Image.new("L", (w * SS, h * SS), 0)
    ImageDraw.Draw(grande).ellipse(caixa, fill=255)
    alfa = np.asarray(grande.resize((w, h), Image.BOX), dtype=np.float32) / 255.0

    cor = np.empty((h, w, 3), dtype=np.float32)
    cor[:] = m.cor_boca
    if abertura > DENTE_MIN:
        forca = min(1.0, (abertura - DENTE_MIN) / (1 - DENTE_MIN))
        dente = (caixa[0] + largura * SS * 0.09, caixa[1] + altura * SS * 0.02,
                 caixa[2] - largura * SS * 0.09,
                 caixa[1] + altura * SS * (0.13 + 0.11 * forca))
        if dente[3] > dente[1] + 1.5 * SS:
            md = Image.new("L", (w * SS, h * SS), 0)
            ImageDraw.Draw(md).rounded_rectangle(
                dente, radius=max(1, int(altura * SS * 0.10)), fill=255)
            # O dente e translucido sobre a boca (190/255), como antes.
            peso = (np.asarray(md.resize((w, h), Image.BOX), dtype=np.float32)
                    / 255.0 * (190 / 255) * forca)[:, :, None]
            cor += (np.asarray(m.cor_dente, dtype=np.float32) - cor) * peso

    buf = np.empty((h, w, 4), dtype=np.uint8)
    buf[:, :, :3] = cor.astype(np.uint8)
    buf[:, :, 3] = (alfa * (255.0 * surge)).astype(np.uint8)
    tile = Image.fromarray(buf).filter(
        ImageFilter.GaussianBlur(max(BOCA_BORRAO_MIN, altura * 0.055)))
    return tile, canto


@dataclass
class Olho:
    caixa: tuple[int, int, int, int]
    lid_y: float
    base_y: float
    alto: float = 8.0


def preparar_olhos(r: Retrato) -> list[Olho]:
    """A regiao que a piscada move, medida nos pontos do olho.

    A caixa desce ate a bochecha (`CAIXA_BOCHECHA` alturas de olho) e nao mais
    so ate um pouco abaixo do cilio: piscar puxa a palpebra inferior e a
    bochecha junto, e nao havia pixel onde desenhar isso. Ela para antes do
    subnasal para nunca encostar na regiao que o maxilar ja mexe -- duas
    deformacoes no mesmo pixel se somam e viram careta.
    """
    olhos = []
    limite = r.p("subnasal")[1] - 4
    for lado in ("esq", "dir"):
        cima = r.p(f"olho_{lado}_cima")[1]
        baixo = r.p(f"olho_{lado}_baixo")[1]
        xs = sorted((r.p(f"olho_{lado}_out")[0], r.p(f"olho_{lado}_in")[0]))
        alto = max(baixo - cima, 4.0)
        # A bochecha e mais larga que o olho; a folga lateral acompanha, e e
        # nela que a pena desliga a deformacao antes da borda da caixa.
        olhos.append(Olho(
            caixa=(int(max(0, xs[0] - PENA_OLHO * alto)),
                   int(max(0, cima - 2.4 * alto)),
                   int(min(r.size[0], xs[1] + PENA_OLHO * alto)),
                   int(min(r.size[1], limite, baixo + CAIXA_BOCHECHA * alto))),
            lid_y=cima, base_y=baixo + 0.45 * alto, alto=alto))
    return olhos


def aplicar_piscada(rgb: np.ndarray, olhos: list[Olho], fecho: float) -> None:
    """Comprime a faixa entre a sobrancelha e o cilio: a pele de cima cobre o olho.

    Dois trechos lineares com a linha da palpebra andando: acima dela a pele
    estica para baixo, abaixo dela o que sobra do olho se espreme. Em fecho=1 o
    olho sumiu e sobrou palpebra -- que e o que uma piscada e.

    Tres defeitos sairam daqui em 20/09/2026, os tres medidos antes:

    1. **A sombra do cilio andava de linha inteira.** Era `int(round(linha))`
       e uma faixa dura de 2 px: entre o quadro 60 e o 61 ela saltava 31 px de
       uma vez. Agora e um peso por linha centrado na posicao **fracionaria**,
       entao ela desce continuamente -- fim do degrau na palpebra.
    2. **O trecho abaixo do cilio extrapolava.** Com o olho quase fechado o
       divisor `base_y - linha` ia a zero, a inclinacao explodia e as ultimas
       linhas da caixa grudavam todas na mesma linha de origem: um borrao
       vertical de 5 px bem na palpebra inferior (medido em fecho 0,9 e 1,0).
       Agora a compressao se desfaz ate a borda da caixa.
    3. **Nada em volta reagia.** Piscar nao e so a palpebra de cima -- o
       orbicular puxa a palpebra inferior e a bochecha junto. Um segundo campo
       sobe o que esta abaixo do olho e morre no fim da caixa.
    4. **A caixa tinha borda dura nas laterais.** A deformacao valia cheia ate
       a ultima coluna e zero na seguinte: medido, um salto de 9,4/255 contra
       2,8 tipico -- 3,4x, que na pele lisa da tempora le como risco vertical.
       O maxilar ja resolvia isso com `peso_x`; o olho nao tinha pena nenhuma.
       Agora tem, e ela cabe exatamente na folga lateral da caixa.
    """
    if fecho <= 0.02:
        return
    for o in olhos:
        x0, y0, x1, y1 = o.caixa
        if y1 <= y0 + 2 or x1 <= x0:
            continue
        bloco = rgb[y0:y1, x0:x1].astype(np.float32)
        h = bloco.shape[0]
        linha = o.lid_y + (o.base_y - o.lid_y) * fecho
        ys = np.arange(y0, y1, dtype=np.float32)

        acima = y0 + (ys - y0) * (o.lid_y - y0) / max(linha - y0, 1e-6)
        # Inclinacao com teto: sem ele ela vai a infinito quando o cilio
        # encosta em `base_y`, e a caixa inteira satura na ultima linha.
        taxa = min((o.base_y - o.lid_y) / max(o.base_y - linha, 1e-6), COMPRESSAO_MAX)
        comprime = o.lid_y + (ys - linha) * taxa
        # E a compressao volta a ser identidade logo abaixo do cilio: dali
        # para baixo e bochecha, e bochecha nao comprime.
        solta = np.clip((ys - o.base_y) / max(SOLTA_ALTO * o.alto, 1e-6), 0, 1)
        solta = solta * solta * (3 - 2 * solta)
        abaixo = comprime + (ys - comprime) * solta

        # Palpebra inferior e bochecha sobem junto, com o pico no cilio de
        # baixo e o efeito morrendo (quadratico) na borda da caixa.
        reacao = np.clip((ys - o.lid_y) / max(o.base_y - o.lid_y, 1e-6), 0, 1)
        decai = np.clip(1 - (ys - o.base_y) / max(y1 - o.base_y, 1e-6), 0, 1)
        sobe = SOBE_VIZINHO * o.alto * fecho * reacao * decai * decai

        # Monotona a forca. A soma de tres perfis (estica, comprime, sobe) pode
        # recuar de fracao de pixel onde eles se cruzam -- medido em 0,12 a
        # 0,20 px, invisivel hoje. Mas recuo e dobra: a imagem volta atras e
        # vira vinco. Uma linha fecha a porta para qualquer calibracao futura
        # de SOBE_VIZINHO transformar isso num defeito visivel.
        origem = np.maximum.accumulate(
            np.where(ys <= linha, acima, abaixo) + sobe)
        origem = np.clip(origem - y0, 0, h - 1.001)
        lo = origem.astype(np.int32)
        fr = (origem - lo).astype(np.float32)[:, None, None]
        saida = bloco[lo] * (1 - fr) + bloco[lo + 1] * fr

        # Sombra do cilio com centro fracionario -- sem salto de linha.
        dist = np.abs(ys - linha) / CILIO_PX
        peso = np.clip(1.0 - dist * dist, 0.0, 1.0) ** 2
        saida *= (1 - 0.42 * fecho * peso)[:, None, None]

        # Pena lateral: a deformacao desliga antes da borda da caixa, senao a
        # ultima coluna mexida e a primeira parada encostam e viram risco.
        pena = max(PENA_OLHO * o.alto, 1.0)
        cols = np.arange(x0, x1, dtype=np.float32)
        lat = np.minimum(np.clip((cols - x0) / pena, 0, 1),
                         np.clip((x1 - 1 - cols) / pena, 0, 1))
        lat = (lat * lat * (3 - 2 * lat))[None, :, None]
        rgb[y0:y1, x0:x1] = np.clip(
            bloco + (saida - bloco) * lat, 0, 255).astype(np.uint8)


def _ancora(t: float, marcos: list[float]) -> float:
    """Puxa `t` para a fronteira de frase mais proxima, se houver uma ao alcance.

    A piscada comeca um pouco ANTES da pausa: o olho fecha ao terminar a
    frase, nao depois de a proxima ja ter comecado.
    """
    if not marcos:
        return t
    i = bisect.bisect_left(marcos, t)
    perto = [marcos[j] for j in (i - 1, i) if 0 <= j < len(marcos)]
    if not perto:
        return t
    alvo = min(perto, key=lambda m: abs(m - t)) - PISCADA_FECHA_S
    return alvo if abs(alvo - t) <= PISCADA_ANCORA_S else t


def piscadas(duracao: float, semente: str,
             sentencas: tuple[float, ...] = ()) -> list[tuple[float, float]]:
    """Janelas de piscada, sorteadas com semente fixa (quadro reprodutivel).

    O sorteio da o **ritmo**; as fronteiras de frase dao o **lugar**. Gente
    pisca na pontuacao, no respiro entre uma ideia e a proxima, e nao no meio
    de uma palavra -- que e onde o sorteio puro caia. Isso e de graca: o
    `batidas.sentencas` ja sai do tempo de palavra que o TTS devolve.

    Sem tempo de palavra (caminho do MPT) `sentencas` vem vazio e o
    comportamento e o antigo, sorteado do inicio ao fim.
    """
    rng = random.Random(semente)
    marcos = sorted(sentencas)
    janelas: list[tuple[float, float]] = []
    t = rng.uniform(0.8, 2.2)
    while t < duracao:
        a = _ancora(t, marcos)
        janelas.append((a, a + PISCADA_S))
        if rng.random() < 0.18:      # piscada dupla, como gente
            t = a + PISCADA_S + rng.uniform(0.12, 0.22)
            janelas.append((t, t + PISCADA_S))
        t += rng.uniform(PISCADA_MIN_S, PISCADA_MAX_S)
    return janelas


def fecho_em(janelas: list[tuple[float, float]], t: float) -> float:
    """Quanto o olho esta fechado em `t`, de 0 (aberto) a 1 (fechado).

    Assimetrica de proposito. A curva anterior era `sin(pi*u)**0.6` numa
    janela de 0,13 s: medido, 2 quadros fechando e 2 abrindo, com o primeiro
    quadro ja saltando de 0,00 para 0,82. Piscada humana nao e simetrica nem
    entra em degrau -- o fechamento e balistico e a abertura leva o dobro do
    tempo. Aqui: ~1,7 quadros fechando, ~1 segurando, ~3,5 abrindo, cada
    trecho com ease cubico para nao deixar quina na partida nem na chegada.
    """
    for ini, _fim in janelas:
        u = t - ini
        if u < 0.0 or u > PISCADA_S:
            continue
        if u <= PISCADA_FECHA_S:
            return _ease_cubico(u / PISCADA_FECHA_S)
        if u <= PISCADA_FECHA_S + PISCADA_SEGURA_S:
            return 1.0
        resta = (u - PISCADA_FECHA_S - PISCADA_SEGURA_S) / PISCADA_ABRE_S
        return 1.0 - _ease_cubico(resta)
    return 0.0


# ===========================================================================
# camadas cabeca / tronco
# ===========================================================================


def mascara_cabeca(r: Retrato) -> np.ndarray:
    """Peso de 1 (cabeca) a 0 (tronco), com a costura na linha dos ombros.

    Complementar de proposito: cabeca + tronco somam o alfa original, entao com
    angulo zero o composto e identico ao retrato -- e o teste mede isso. A
    costura fica no peito, nao no pescoco: cabeca que gira sobre pescoco parado
    e o efeito de boneco de ventriloquo.
    """
    # Medido no recorte da Iris: queixo em 648, pescoco em 682, ombro em ~880.
    # A costura tem de cair entre o pescoco e o ombro -- mais acima e cabeca
    # girando sobre pescoco parado (boneco de ventriloquo); mais abaixo e o
    # tronco que some, porque o recorte acaba em 1053 e ja vem esmaecido.
    centro = r.neck_y + 0.35 * r.face_height
    meia = 0.28 * r.face_height
    ys = np.arange(r.size[1], dtype=np.float32)
    u = np.clip((centro + meia - ys) / max(2 * meia, 1e-6), 0, 1)
    return (u * u * (3 - 2 * u)).astype(np.float32)[:, None]


def realce(img: Image.Image, accent: str) -> Image.Image:
    """Contraluz na cor do pilar + sombra atras: separa o apresentador do fundo.

    O guia de identidade ja pedia "recorte #39FF88 no ombro". Aqui ele tambem
    resolve um problema medido: sobre clipe claro, silhueta escura sem contorno
    some; sobre clipe escuro, silhueta escura vira borrao. Calculado uma vez --
    depende so do alfa, que a fala nao muda (boca e olho sao interior).
    """
    alfa = img.getchannel("A")
    dentro = alfa.filter(ImageFilter.MinFilter(5))
    anel = Image.fromarray(
        np.clip(np.asarray(alfa, dtype=np.int16) - np.asarray(dentro, dtype=np.int16),
                0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3.0))
    rgb = tuple(int(accent.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    luz = Image.new("RGBA", img.size, (*rgb, 0))
    luz.putalpha(Image.fromarray(
        (np.asarray(anel, dtype=np.float32) * 0.8).clip(0, 255).astype(np.uint8)))

    sombra = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sombra.putalpha(Image.fromarray(
        (np.asarray(alfa.filter(ImageFilter.GaussianBlur(13)), dtype=np.float32) * 0.38)
        .clip(0, 255).astype(np.uint8)))

    saida = Image.new("RGBA", img.size, (0, 0, 0, 0))
    saida.alpha_composite(sombra)
    saida.alpha_composite(img)
    saida.alpha_composite(luz)
    return saida


def _afim(origem: Image.Image, destino: tuple[int, int], *, escala: float,
          graus: float, pivo: tuple[float, float], alvo: tuple[float, float],
          ) -> Image.Image:
    """Leva `pivo` da origem ao `alvo` do destino, com escala e giro na mesma conta.

    Uma transformacao so em vez de resize + rotate: medido em 11 ms contra 25 ms
    por quadro neste retrato, e sao milhares de quadros por video.
    """
    th = math.radians(graus)
    k = 1.0 / max(escala, 1e-6)
    cos, sin = math.cos(th), math.sin(th)
    a, b = k * cos, k * sin
    d, e = -k * sin, k * cos
    c = pivo[0] - (a * alvo[0] + b * alvo[1])
    f = pivo[1] - (d * alvo[0] + e * alvo[1])
    return origem.transform(destino, Image.AFFINE, (a, b, c, d, e, f),
                            resample=Image.BILINEAR)


def plaquinha(nome: str, accent: str) -> Image.Image:
    """Chip com o nome do apresentador -- entra e sai junto com a chamada."""
    texto = nome.upper()
    f = font(MONO, 34)
    medida = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), texto, font=f)
    w = medida[2] - medida[0] + 54
    h = 62
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=14, fill=(10, 10, 12, 228))
    rgb = tuple(int(accent.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    d.rectangle((0, 12, 5, h - 13), fill=(*rgb, 255))
    d.text((22, (h - (medida[3] - medida[1])) / 2 - medida[1]), texto, font=f,
           fill=(233, 238, 241, 255))
    return img


# ===========================================================================
# o animador
# ===========================================================================


@dataclass(frozen=True)
class Camada:
    """O que `render/post.py` precisa saber para sobrepor a camada pronta."""

    path: Path
    x: int
    y: int
    width: int
    height: int
    subtitle_y: int
    subtitle_start: float
    card_s: float
    frames: int
    fps: int
    seconds: float


class Animador:
    """Sintetiza os quadros do apresentador. Tudo o que nao muda e pre-calculado."""

    def __init__(self, retrato: Retrato, accent: str, nome: str = "",
                 area: tuple[int, int, int, int] = AREA) -> None:
        self.retrato = retrato
        self.accent = accent
        self.area = area
        self.maxilar = preparar_maxilar(retrato)
        self.olhos = preparar_olhos(retrato)
        decorado = np.asarray(realce(retrato.imagem, accent)).copy()
        self._rgb = np.ascontiguousarray(decorado[:, :, :3])
        peso = mascara_cabeca(retrato)
        alfa = decorado[:, :, 3].astype(np.float32)
        self._alfa_cab = (alfa * peso).astype(np.uint8)
        self._alfa_tro = (alfa * (1 - peso)).astype(np.uint8)
        # Cada camada carrega so o retangulo em que tem tinta. Nao e detalhe:
        # a transformacao custa por pixel de SAIDA, e sao dois mil quadros --
        # medido, recortar tirou 40% do tempo de cada quadro.
        self._cx_cab = _tinta(self._alfa_cab)
        self._cx_tro = _tinta(self._alfa_tro)
        self._chip = plaquinha(nome, accent) if nome else None

    # ------------------------------------------------------------- quadro

    def _camada(self, rgb: np.ndarray, alfa: np.ndarray,
                caixa: tuple[int, int, int, int]) -> Image.Image:
        x0, y0, x1, y1 = caixa
        buf = np.empty((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        buf[:, :, :3] = rgb[y0:y1, x0:x1]
        buf[:, :, 3] = alfa[y0:y1, x0:x1]
        return Image.fromarray(buf)

    def quadro(self, t: float, abertura: float, fecho: float,
               marcas: list[Marca]) -> Image.Image:
        """Um quadro RGBA do tamanho da area util da camada."""
        ax, ay, aw, ah = self.area
        tela = Image.new("RGBA", (aw, ah), (0, 0, 0, 0))
        p = pose(marcas, t)
        if p.alfa <= 0.004 or p.altura <= 1:
            return tela

        rgb = self._rgb.copy()
        aplicar_maxilar(rgb, self.maxilar, abertura)
        aplicar_piscada(rgb, self.olhos, fecho)
        boca = tile_boca(self.maxilar, abertura)

        rw, rh = self.retrato.size
        escala = (p.altura / rh) * (1 + RESPIRO * math.sin(2 * math.pi * t / 4.4))
        giro_t = GIRO_TRONCO * math.sin(2 * math.pi * t / 7.3 + 0.7)
        giro_c = (GIRO_CABECA * math.sin(2 * math.pi * t / 5.7 + 2.1)
                  + ACENTO_CABECA * (abertura - 0.45))
        entrada = (1 - p.alfa) * 90     # desliza de fora enquanto aparece

        # Onde o pivo do retrato cai na tela: a base do recorte encosta em
        # p.base e o centro horizontal em p.cx, girando em torno do pivo.
        pivo = self.retrato.pivot
        base = (p.cx - ax - entrada, p.base - ay)
        # Um pivo so na tela para as duas camadas: quem define onde o corpo
        # pousa e o angulo do tronco. Calcular um alvo por camada as separaria
        # -- pouco, mas separaria, e a costura e justamente ali.
        alvo = _pivo_na_tela(base, pivo, (rw / 2, rh), escala, giro_t)
        for alfa_camada, cx_fonte, graus in (
                (self._alfa_tro, self._cx_tro, giro_t),
                (self._alfa_cab, self._cx_cab, giro_t + giro_c)):
            if cx_fonte is None:
                continue
            saida = _caixa_saida(cx_fonte, pivo, alvo, escala, graus)
            if saida is None:
                continue
            ox, oy, ow, oh = saida
            x0, y0 = cx_fonte[0], cx_fonte[1]
            img = self._camada(rgb, alfa_camada, cx_fonte)
            if boca is not None and y0 <= self.maxilar.boca_y < cx_fonte[3]:
                tile, canto = boca
                _colar(img, tile, canto[0] - x0, canto[1] - y0)
            girada = _afim(img, (ow, oh), escala=escala, graus=graus,
                           pivo=(pivo[0] - x0, pivo[1] - y0),
                           alvo=(alvo[0] - ox, alvo[1] - oy))
            if p.alfa < 0.999:
                girada.putalpha(girada.getchannel("A").point(
                    lambda v, a=p.alfa: int(v * a)))
            _colar(tela, girada, ox, oy)
        self._plaquinha(tela, p, base[0], base[1])
        return tela

    def _plaquinha(self, tela: Image.Image, p: Marca, base_x: float,
                   base_y: float) -> None:
        """So na chamada, onde ele se apresenta -- some junto com o cartao.

        No fechamento ele ja foi apresentado; a plaquinha reaparecendo a 10% de
        opacidade parecia defeito de composicao, nao assinatura.
        """
        if self._chip is None:
            return
        faixa = max(ALTURA_CHAMADA - ALTURA_FECHAMENTO, 1.0)
        visivel = min(1.0, max(0.0, (p.altura - ALTURA_FECHAMENTO - 60) / faixa)) * p.alfa
        if visivel <= 0.02:
            return
        chip = self._chip
        if visivel < 0.999:
            chip = chip.copy()
            chip.putalpha(chip.getchannel("A").point(lambda v, a=visivel: int(v * a)))
        _colar(tela, chip, int(base_x - chip.width / 2),
               int(base_y - p.altura * 0.115))


def _giro(dx: float, dy: float, graus: float, escala: float) -> tuple[float, float]:
    th = math.radians(graus)
    cos, sin = math.cos(th), math.sin(th)
    return (escala * (dx * cos - dy * sin), escala * (dx * sin + dy * cos))


def _pivo_na_tela(base: tuple[float, float], pivo: tuple[float, float],
                  ancora: tuple[float, float], escala: float,
                  graus: float) -> tuple[float, float]:
    """Posicao do pivo na tela, dado que a ancora (base do recorte) vai em `base`."""
    dx, dy = _giro(ancora[0] - pivo[0], ancora[1] - pivo[1], graus, escala)
    return (base[0] - dx, base[1] - dy)


def _caixa_saida(fonte: tuple[int, int, int, int], pivo: tuple[float, float],
                 alvo: tuple[float, float], escala: float, graus: float,
                 folga: int = 3) -> tuple[int, int, int, int] | None:
    """Retangulo que a camada ocupa na tela depois de girada e escalada."""
    xs, ys = [], []
    for px in (fonte[0], fonte[2]):
        for py in (fonte[1], fonte[3]):
            dx, dy = _giro(px - pivo[0], py - pivo[1], graus, escala)
            xs.append(alvo[0] + dx)
            ys.append(alvo[1] + dy)
    x0 = int(math.floor(min(xs))) - folga
    y0 = int(math.floor(min(ys))) - folga
    w = int(math.ceil(max(xs))) + folga - x0
    h = int(math.ceil(max(ys))) + folga - y0
    if w < 2 or h < 2:
        return None
    return (x0, y0, w, h)


def _tinta(alfa: np.ndarray) -> tuple[int, int, int, int] | None:
    """Retangulo com tinta (folga de 2 px para o filtro bilinear nao cortar)."""
    linhas = np.flatnonzero(alfa.max(axis=1) > 0)
    colunas = np.flatnonzero(alfa.max(axis=0) > 0)
    if not linhas.size or not colunas.size:
        return None
    return (max(0, int(colunas[0]) - 2), max(0, int(linhas[0]) - 2),
            min(alfa.shape[1], int(colunas[-1]) + 3),
            min(alfa.shape[0], int(linhas[-1]) + 3))


def _colar(tela: Image.Image, peca: Image.Image, x: int, y: int) -> None:
    """Compoe recortando no que cabe -- `alpha_composite` nao aceita sair da tela."""
    sx0, sy0 = max(0, -x), max(0, -y)
    sx1 = min(peca.width, tela.width - x)
    sy1 = min(peca.height, tela.height - y)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    if (sx0, sy0, sx1, sy1) != (0, 0, peca.width, peca.height):
        peca = peca.crop((sx0, sy0, sx1, sy1))
    tela.alpha_composite(peca, dest=(x + sx0, y + sy0))


def encode_command(out: Path, w: int, h: int, fps: int, *,
                   ffmpeg: str = "ffmpeg") -> list[str]:
    """Quadros RGBA crus na entrada -> cor pre-multiplicada + mascara, num arquivo.

    Pre-multiplicar antes de comprimir e o que evita franja: com alfa direto, a
    cor salta do rosto para o preto na borda da silhueta, o h264 borra esse
    salto e sobra um contorno escuro. Pre-multiplicada, a cor desce a zero na
    mesma rampa da mascara, e `render/post.py` desfaz a conta na volta.
    """
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
            "-r", str(fps), "-i", "-",
            "-filter_complex",
            "[0:v]split=2[pm][al];"
            "[pm]premultiply=inplace=1,format=yuv420p[c];"
            "[al]alphaextract,format=gray[m]",
            "-map", "[c]", "-map", "[m]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]


def render_layer(retrato: Retrato, audio: Path, out: Path, *, duracao: float,
                 batidas: Batidas, accent: str, nome: str = "",
                 fps: int = FPS, ffmpeg: str = "ffmpeg",
                 semente: str = "ai-central") -> Camada:
    """Gera o `.mov` RGBA do apresentador e devolve onde sobrepor."""
    n = max(1, int(round(duracao * fps)))
    env = envoltoria(audio, n, fps, ffmpeg=ffmpeg)
    janelas = piscadas(duracao, semente, batidas.sentencas)
    enc = plan(batidas)
    anim = Animador(retrato, accent, nome)
    ax, ay, aw, ah = anim.area

    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(encode_command(out, aw, ah, fps, ffmpeg=ffmpeg),
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for k in range(n):
            t = k / fps
            anim_quadro = anim.quadro(t, float(env[k]), fecho_em(janelas, t), enc.marcas)
            proc.stdin.write(anim_quadro.tobytes())
    except BrokenPipeError:  # pragma: no cover - so com ffmpeg quebrado
        pass
    finally:
        proc.stdin.close()
    erro = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg falhou ao gravar o apresentador: {erro[-400:]}")
    return Camada(path=out, x=ax, y=ay, width=aw, height=ah,
                  subtitle_y=enc.legenda_y, subtitle_start=enc.legenda_inicio,
                  card_s=enc.cartao_s, frames=n, fps=fps, seconds=n / fps)


def conferir_apresentador(final: Path, camada: Camada, trabalho: Path,
                          *, ffmpeg: str = "ffmpeg") -> str:
    """Confere no MP4 **final** que o apresentador pousou mesmo no quadro.

    A camada pode ser gerada certinha e nao chegar a lugar nenhum: basta um
    `overlay` errado, um rotulo de fluxo trocado ou um `alphamerge` que o
    ffmpeg descarta em silencio. Nada disso levanta erro -- sai um MP4 valido,
    com a duracao certa, sem apresentador. E o mesmo defeito do MP4 mudo, e a
    regra do projeto e medir o artefato.

    **A medida mudou em 20/09/2026, porque a antiga passou a mentir.** Ela
    comparava o quadro final com o quadro *bruto* e pedia que a diferenca
    dentro da mascara fosse o dobro da diferenca num anel em volta. Isso mede
    "alguma coisa mudou aqui", e o que mudou pode ser qualquer coisa: com o
    apresentador fotorrealista e o cartao do gancho logo acima dele, o anel
    passou a mudar tanto quanto a mascara e o veredito virou NAO CHEGOU em
    video que tinha o apresentador no quadro, conferido a olho. Alarme que
    dispara sempre e alarme que ninguem le -- e ai o dia em que ele tiver
    razao passa batido junto.

    A medida nova compara o final com a **propria camada**: dentro da mascara,
    o quadro final tem de ser o apresentador, e nao so "diferente do bruto".
    Correlacao (e nao diferenca absoluta) porque a pos-producao aplica eq e
    vinheta por cima -- as duas mudam o nivel do pixel e nenhuma desmancha a
    estrutura do rosto. Avatar ausente deixa b-roll ali, que nao correlaciona
    com rosto nenhum.

    So os pixels de alfa quase cheio entram: a camada e gravada com a cor
    pre-multiplicada, entao e onde a mascara esta cheia que a cor gravada
    equivale a cor de verdade.

    Devolve a frase para o log; nao levanta -- video sem avatar ainda e
    publicavel, e o motivo tem de ficar registrado.
    """
    t = max(0.3, camada.card_s * 0.5)
    try:
        quadro = _um_quadro(final, t, trabalho / "_aceite_final.png", ffmpeg=ffmpeg)
        cor = _um_quadro(camada.path, t, trabalho / "_aceite_cor.png",
                         ffmpeg=ffmpeg, stream="0:v:0")
        masc = _um_quadro(camada.path, t, trabalho / "_aceite_masc.png",
                          ffmpeg=ffmpeg, stream="0:v:1")
    except (OSError, ValueError) as exc:
        return f"apresentador: conferencia no quadro indisponivel ({exc})"
    if quadro is None or cor is None or masc is None:
        return "apresentador: conferencia no quadro indisponivel (quadro vazio)"

    plano_m = np.zeros(quadro.shape, dtype=np.uint8)
    plano_c = np.zeros(quadro.shape, dtype=np.uint8)
    h = min(masc.shape[0], quadro.shape[0] - camada.y)
    w = min(masc.shape[1], quadro.shape[1] - camada.x)
    if h <= 0 or w <= 0:
        return "apresentador: camada fora do quadro"
    plano_m[camada.y:camada.y + h, camada.x:camada.x + w] = masc[:h, :w]
    plano_c[camada.y:camada.y + h, camada.x:camada.x + w] = cor[:h, :w]

    solido = plano_m > 230
    if solido.sum() < 4000:
        return "apresentador: mascara vazia no instante da chamada"
    a = quadro[solido].astype(np.float64)
    b = plano_c[solido].astype(np.float64)
    if a.std() < 4 or b.std() < 4:
        return "apresentador: sem estrutura para comparar na mascara"
    corr = float(np.corrcoef(a, b)[0, 1])
    veredito = "OK" if corr > 0.55 else "NAO CHEGOU AO QUADRO"
    return (f"apresentador no quadro: {veredito} "
            f"(t={t:.1f}s, correlacao com a camada {corr:.2f}, "
            f"{int(solido.sum())} px)")


def _um_quadro(video: Path, t: float, destino: Path, *, ffmpeg: str = "ffmpeg",
               stream: str = "0:v:0") -> np.ndarray | None:
    r = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}",
         "-i", str(video), "-map", stream, "-frames:v", "1", str(destino)],
        capture_output=True, timeout=120)
    if r.returncode != 0 or not destino.exists():
        return None
    with Image.open(destino) as img:
        return np.asarray(img.convert("L")).copy()


__all__ = ["AREA", "Animador", "Batidas", "Camada", "Encenacao", "Marca",
           "Maxilar", "Olho", "Retrato", "Y_LEGENDA_COM_APRESENTADOR",
           "aplicar_maxilar", "aplicar_piscada", "batidas_estimadas",
           "batidas_por_tempo", "carregar", "conferir_apresentador",
           "encode_command", "envoltoria", "fecho_em", "mascara_cabeca",
           "piscadas", "plan", "pose", "preparar_maxilar", "preparar_olhos",
           "realce", "render_layer", "tile_boca"]
