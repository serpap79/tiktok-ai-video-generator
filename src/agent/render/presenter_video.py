"""Apresentador de video: o clipe humano da a pessoa, nos damos a boca.

`render/presenter.py` sintetiza tudo a partir de um retrato parado -- piscada,
balanco de cabeca, respiracao. Funciona e o limite esta declarado la: e uma
silhueta convincente, nao uma pessoa. Desde 20/09/2026 existe um clipe
fotorrealista do THEO (Hailuo, identidade travada do `brand.json`) em que a
piscada e o balanco **ja sao humanos**, porque foram gerados como video. O que
falta nele e a unica coisa que nao pode vir pronta: a boca, que depende da
narracao daquele video especifico.

Entao a divisao de trabalho aqui e o oposto da do modulo antigo:

| o que                       | de onde vem                                  |
|-----------------------------|----------------------------------------------|
| piscada, cabeca, respiracao | do clipe base, quadro a quadro (humano)      |
| boca                        | sintetizada aqui, no tempo da narracao       |
| tamanho e lugar na tela     | da encenacao do roteiro (`presenter.plan`)   |

Nada de piscada sorteada nem giro por seno: seria movimento nosso brigando com
o movimento que ja esta no quadro.

**A boca segue o rosto.** `scripts/make_presenter_video.py` gravou os pontos da
malha em *cada* quadro do clipe; a cabeca se mexe, e uma boca desenhada em
coordenada fixa desgruda do rosto no primeiro balanco. Aqui cada quadro le os
pontos do seu proprio quadro.

**A boca tem forma, nao so tamanho.** A abertura sai de `render/visemes.py`
(fonema de pt-BR sobre o tempo de palavra do TTS) modulada pela envoltoria de
energia da narracao: a letra diz que forma, o audio diz com quanta forca. Num
rosto fotorrealista isso deixou de ser refinamento -- boca aberta no /m/ e
gatilho de vale da estranheza, e num close ninguem perdoa.

O clipe base e curto e a narracao nao: a leitura e em **vai-e-vem** (para
frente ate o fim, para tras ate o comeco). Emenda por corte foi medida e
descartada -- o melhor par de quadros nao vizinhos do clipe do THEO difere
6,74/255, contra 1,13 entre vizinhos, ou seja, seis vezes o movimento normal:
daria um solavanco visivel a cada volta. No vai-e-vem a virada usa quadros
vizinhos e e invisivel. O preco, declarado: a piscada toca ao contrario uma vez
por ciclo (fecha devagar, abre rapido -- o inverso do humano). Com um clipe
base mais longo que a narracao o vai-e-vem nunca chega a virar.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from agent.render.presenter import (
    AREA,
    FPS,
    Batidas,
    Camada,
    Marca,
    _afim,
    _colar,
    _ease_cubico,
    encode_command,
    envoltoria,
    plan,
    plaquinha,
    pose,
)
from agent.render.visemes import trilha

# Queda maxima do queixo como fracao da altura do rosto -- a mesma medida do
# apresentador parado, que saiu de olhar quadro a quadro: acima de ~10% o rosto
# visivelmente alonga nas silabas abertas.
MAXILAR = 0.105
# Comprimento da rampa do maxilar no centro da boca. Tem de ser um degrau: e
# ali que a boca abre. Com os 18 px da primeira versao o labio de baixo quase
# nao descia e a boca saia achatada, um risco horizontal em vez de um vao.
DEGRAU_PX = 3.0
# Acima desta abertura aparece dente. Mais baixo que no apresentador parado
# (0,52): num rosto real o dente e parte da boca aberta, e a falta dele le como
# buraco preto. Abaixo disso a boca esta apenas entreaberta e dente nao aparece.
DENTE_MIN = 0.30
# Super-amostragem das mascaras. O `ImageDraw` do Pillow nao tem anti-aliasing
# nenhum (medido em 20/09/2026: zero pixel de borda parcial); desenhar 4x maior
# e reduzir por media de area (`Image.BOX`) e o que da a borda suave.
SS = 4
# Abaixo desta altura em pixel a boca entra por degrade em vez de surgir.
BOCA_MIN_PX = 3.0
# Folga em volta do poligono do vao, para o borrao ter onde cair.
PAD_VAO = 14
# Quanto a largura do visema estica ou recolhe a boca. 0,12 saiu de olhar: em
# 0,20 o /u/ vira bico de desenho animado, em 0,06 o /i/ nao se distingue.
LARGURA_MAX = 0.12
# Pena lateral do campo do maxilar: sem ela a deformacao morre de uma linha
# para a outra na borda da caixa e aparece uma costura vertical na bochecha.
PENA_X = 46.0
# Piso da modulacao pela energia. A forma do visema nunca e zerada pelo audio:
# a envoltoria pode estar baixa num fonema surdo que a boca faz do mesmo jeito.
ENERGIA_PISO = 0.55


# ===========================================================================
# o clipe base assado
# ===========================================================================


@dataclass(frozen=True)
class Base:
    """O clipe do apresentador com os pontos do rosto ja medidos por quadro."""

    id: str
    video: Path
    size: tuple[int, int]
    fps: float
    frames: int
    head_top: int
    neck_y: int
    face_height: float
    pivot: tuple[float, float]
    tracks: dict[str, np.ndarray] = field(default_factory=dict)
    n_baixo: int = 0
    n_cima: int = 0

    def pontos(self, k: int) -> dict[str, tuple[float, float]]:
        k = min(max(k, 0), self.frames - 1)
        return {nome: (float(v[k, 0]), float(v[k, 1])) for nome, v in self.tracks.items()}

    def contorno(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Os dois arcos do labio interno neste quadro: (baixo, cima).

        E o que substituiu a elipse desenhada a mao. Boca tem canto em bico e
        elipse tem tangente vertical no canto: a elipse cobria a borda do labio
        de cima e deixava um contorno fantasma por baixo dela -- visto na
        ampliacao 2x do artefato de 20/09/2026.
        """
        k = min(max(k, 0), self.frames - 1)
        baixo = np.array([self.tracks[f"labio_in_baixo_{j:02d}"][k]
                          for j in range(self.n_baixo)], dtype=np.float32)
        cima = np.array([self.tracks[f"labio_in_cima_{j:02d}"][k]
                         for j in range(self.n_cima)], dtype=np.float32)
        return baixo, cima


def carregar_base(meta_json: Path) -> Base:
    """Le `<id>_base.json` e confere que o mp4 irmao existe."""
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    video = meta_json.with_name(meta_json.stem + ".mp4")
    if not video.exists():
        raise FileNotFoundError(
            f"{video} nao existe -- rode scripts/make_presenter_video.py")
    if not meta.get("contorno"):
        raise ValueError(
            f"{meta_json.name} foi assado sem o contorno do labio -- "
            "rode scripts/make_presenter_video.py de novo")
    return Base(
        id=meta.get("id", meta_json.stem),
        video=video,
        size=(int(meta["size"][0]), int(meta["size"][1])),
        fps=float(meta["fps"]),
        frames=int(meta["frames"]),
        head_top=int(meta["head_top"]),
        neck_y=int(meta["neck_y"]),
        face_height=float(meta["face_height"]),
        pivot=(float(meta["pivot"][0]), float(meta["pivot"][1])),
        tracks={nome: np.asarray(v, dtype=np.float32)
                for nome, v in meta["tracks"].items()},
        n_baixo=int(meta.get("contorno", {}).get("baixo", 0)),
        n_cima=int(meta.get("contorno", {}).get("cima", 0)),
    )


def indice_vaivem(t: float, base: Base) -> int:
    """Qual quadro do clipe toca no instante `t`, indo e voltando.

    Fora do vai-e-vem nao ha emenda invisivel: medido no clipe do THEO, o
    melhor corte entre quadros nao vizinhos custa 6x o movimento normal.
    """
    if base.frames <= 1:
        return 0
    pos = t * base.fps
    ciclo = 2 * (base.frames - 1)
    r = pos % ciclo
    if r > base.frames - 1:
        r = ciclo - r
    return int(round(r))


class Quadros:
    """Acesso aleatorio aos quadros do clipe assado.

    O mp4 vem em duas trilhas (cor pre-multiplicada + mascara). Decodificar o
    clipe inteiro uma vez para um arquivo mapeado em memoria custa segundos e
    troca CPU por disco; decodificar sob demanda custaria um processo de ffmpeg
    por quadro de saida, que sao milhares.
    """

    def __init__(self, base: Base, trabalho: Path, *, ffmpeg: str = "ffmpeg") -> None:
        w, h = base.size
        self.base = base
        trabalho.mkdir(parents=True, exist_ok=True)
        destino = trabalho / f"{base.id}_base_rgba.raw"
        cru = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(base.video),
             "-filter_complex",
             "[0:v:0][0:v:1]alphamerge,unpremultiply=inplace=1,format=rgba[o]",
             "-map", "[o]", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
            capture_output=True, check=True, timeout=600).stdout
        esperado = base.frames * h * w * 4
        if len(cru) < esperado:
            raise RuntimeError(
                f"clipe base incompleto: {len(cru)} bytes, esperava {esperado}")
        destino.write_bytes(cru[:esperado])
        self._dados = np.memmap(destino, dtype=np.uint8, mode="r",
                                shape=(base.frames, h, w, 4))

    def __getitem__(self, k: int) -> np.ndarray:
        return np.asarray(self._dados[min(max(k, 0), self.base.frames - 1)])


# ===========================================================================
# a boca
# ===========================================================================


@dataclass(frozen=True)
class Tons:
    """Cores da boca, amostradas uma vez no clipe.

    Uma vez, e nao por quadro, de proposito: a luz do clipe e travada, e cor
    reamostrada a cada quadro tremeria com o ruido do grao -- boca piscando de
    tom e pior que boca de tom levemente errado.
    """

    interior: tuple[float, float, float]
    dente: tuple[float, float, float]
    lingua: tuple[float, float, float]


def medir_tons(quadro: np.ndarray, pts: dict[str, tuple[float, float]]) -> Tons:
    px = quadro[:, :, :3].astype(np.float32)
    cx, cy = pts["labio_sup"]
    amostra = px[int(cy) - 4:int(cy) + 5, int(cx) - 14:int(cx) + 15].reshape(-1, 3)
    labio = amostra.mean(axis=0) if amostra.size else np.array([120.0, 80.0, 80.0])
    rosto = px[int(pts["olho_esq_baixo"][1]):int(pts["queixo"][1]),
               int(pts["mand_esq"][0]):int(pts["mand_dir"][0])].reshape(-1, 3)
    # p97 e nao p88: o recorte do rosto pega barba e sombra do queixo, e a
    # media alta demais puxava o "claro" para abaixo do tom do labio -- o
    # dente saia mais escuro que a boca, que e o oposto de um dente.
    claro = (np.percentile(rosto, 97, axis=0) if rosto.size
             else np.array([200.0, 190.0, 185.0]))
    # O dente nao e branco nem e pele: e quase neutro, puxado do brilho do
    # proprio rosto (cada apresentador tem sua exposicao) e rebaixado porque
    # esta dentro da boca, na sombra. Tomar a cor da pele direto deixava o
    # dente rosado -- lia como plastico, nao como dente.
    luz = float(np.mean(claro))
    dente = np.clip(luz * 1.18 * np.array([1.0, 0.985, 0.95]), 60, 242)
    # Boca por dentro e escura, nao preta: em 0,20 o vao virava um buraco
    # chapado e o dente nao tinha contra o que contrastar.
    # A lingua existe para o vao deixar de ser um buraco preto. Ela nao
    # precisa de forma -- basta um volume fosco e avermelhado no fundo, que e
    # o que se ve de relance numa boca falando.
    return Tons(interior=tuple(np.clip(labio * 0.38, 20, 78)),
                dente=tuple(dente),
                lingua=tuple(np.clip(labio * 0.62, 40, 150)))


def _reamostra_y2(bloco: np.ndarray, desloc: np.ndarray) -> np.ndarray:
    """`saida[y,x] = bloco[y - desloc[y,x], x]`, linear entre linhas.

    Deslocamento por PIXEL, nao por linha: a mandibula nao desce em bloco (ver
    `abrir_maxilar`), entao cada coluna tem o seu proprio perfil.
    """
    h = bloco.shape[0]
    ys = np.clip(np.arange(h, dtype=np.float32)[:, None] - desloc, 0, h - 1.001)
    lo = ys.astype(np.int32)
    fr = (ys - lo)[:, :, None].astype(np.float32)
    colunas = np.arange(bloco.shape[1])[None, :]
    return bloco[lo, colunas] * (1 - fr) + bloco[lo + 1, colunas] * fr


def campo_maxilar(pts: dict[str, tuple[float, float]], face_h: float,
                  abertura: float, shape: tuple[int, int],
                  ) -> tuple[tuple[int, int, int, int], np.ndarray, float] | None:
    """O deslocamento vertical, em pixel, de cada ponto da metade de baixo.

    Sai como funcao propria porque **o campo e o artefato**: olhar a imagem
    deformada e achar que esta boa foi como o risco horizontal na bochecha
    passou despercebido. Com o campo na mao da para medir o salto entre linhas
    vizinhas e travar isso em teste.

    O modelo tem tres fatores, e cada um veio de um defeito visto na ampliacao:

    - `S(y)` -- a mandibula e osso: da linha dos labios ao queixo ela desce
      inteira (degrau curto), e so o pescoco absorve a diferenca. Rampa longa
      aqui espremia o labio de baixo e a boca virava um risco.
    - `amp(x)` -- o osso GIRA em torno da articulacao perto da orelha, entao o
      queixo desce tudo e o canto do maxilar quase nada. Sem isso sobrava um
      degrau de 61 px atravessando a bochecha.
    - `abre(x, y)` -- **so a boca abre na linha dos labios.** Na altura do
      labio o deslocamento vale o perfil de abertura (zero nas comissuras);
      descendo para o queixo ele vira 1 em toda a largura. E o que faz o canto
      da boca ficar colado enquanto o meio abre, e ao mesmo tempo nao deixa
      nenhum degrau na bochecha, porque ali o campo comeca em zero e cresce.
    """
    h, w = shape
    labio_y = (pts["labio_sup"][1] + pts["labio_inf"][1]) / 2
    queixo_y = pts["queixo"][1]
    fim = queixo_y + 0.5 * face_h
    y0 = int(max(0, labio_y - 6))
    y1 = int(min(h, fim + 8))
    x0 = int(max(0, pts["face_esq"][0] - PENA_X))
    x1 = int(min(w, pts["face_dir"][0] + PENA_X))
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None

    queda = MAXILAR * face_h * abertura
    colunas = np.arange(x0, x1, dtype=np.float32)
    linhas = np.arange(y0, y1, dtype=np.float32)[:, None]

    sobe = np.clip((linhas - labio_y) / DEGRAU_PX, 0, 1)
    sobe = sobe * sobe * (3 - 2 * sobe)
    volta = np.clip(1 - (linhas - queixo_y) / max(fim - queixo_y, 1e-6), 0, 1)
    volta = volta * volta * (3 - 2 * volta)
    campo = queda * np.minimum(sobe, volta) * _amplitude_x(pts, colunas)[None, :]

    # Profundidade: 0 na linha dos labios, 1 no queixo. Na linha do labio quem
    # manda e o perfil de abertura; no queixo, a mandibula inteira.
    prof = np.clip((linhas - labio_y) / max(queixo_y - labio_y, 1e-6), 0, 1)
    prof = prof * prof * (3 - 2 * prof)
    abre = _abre_x(pts, colunas)[None, :]
    campo *= abre + (1.0 - abre) * prof

    # Pena lateral: o campo ja morre no rosto, mas a caixa e mais larga que ele
    # e uma costura vertical na borda da caixa foi defeito pago no apresentador
    # parado -- aqui ela nao volta.
    peso = np.minimum(
        np.clip((colunas - (pts["mand_esq"][0] - PENA_X)) / PENA_X, 0, 1),
        np.clip(((pts["mand_dir"][0] + PENA_X) - colunas) / PENA_X, 0, 1))
    campo *= (peso * peso * (3 - 2 * peso))[None, :]
    return (x0, y0, x1, y1), campo, float(queda)


def abrir_maxilar(quadro: np.ndarray, pts: dict[str, tuple[float, float]],
                  face_h: float, abertura: float) -> float:
    """Desce a mandibula no lugar. Devolve a queda em pixel no queixo."""
    if abertura <= 0.01:
        return 0.0
    feito = campo_maxilar(pts, face_h, abertura, quadro.shape[:2])
    if feito is None:
        return 0.0
    (x0, y0, x1, y1), campo, queda = feito
    bloco = quadro[y0:y1, x0:x1].astype(np.float32)
    quadro[y0:y1, x0:x1] = _reamostra_y2(bloco, campo).astype(np.uint8)
    return queda


def espalhar_labios(quadro: np.ndarray, pts: dict[str, tuple[float, float]],
                    largura: float) -> None:
    """Estica (/i/) ou recolhe (/u/) a boca na horizontal, no lugar.

    O campo morre nas bordas da caixa de proposito: sem isso a bochecha inteira
    andaria junto e o rosto mudaria de largura a cada silaba.
    """
    if abs(largura) < 0.02:
        return
    h, w = quadro.shape[:2]
    cx = (pts["boca_esq"][0] + pts["boca_dir"][0]) / 2
    cy = (pts["labio_sup"][1] + pts["labio_inf"][1]) / 2
    boca_w = max(pts["boca_dir"][0] - pts["boca_esq"][0], 8.0)
    # A caixa vai ate onde o LABIO vai, e nao um tanto arbitrario da largura da
    # boca. Com `boca_w * 0,62` ela media 121 px de altura -- do nariz ao
    # queixo -- e o esticar ondulava a barba e a bochecha a cada silaba, bem
    # visivel na ampliacao 2x. Lábio e o que estica; pele em volta, nao.
    labio_h = abs(pts["labio_inf_out"][1] - pts["labio_sup_out"][1])
    rx, ry = boca_w * 0.85, max(labio_h * 0.65, 20.0)
    x0, x1 = int(max(0, cx - rx)), int(min(w, cx + rx))
    y0, y1 = int(max(0, cy - ry)), int(min(h, cy + ry))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return

    escala = 1.0 + LARGURA_MAX * largura
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    # Deslocamento que levaria a escala exata, apagado nas bordas da caixa.
    desloc = (xs - cx) * (1.0 / max(escala, 1e-6) - 1.0)
    tx = np.clip(1 - np.abs(xs - cx) / max(rx, 1e-6), 0, 1)
    ty = np.clip(1 - np.abs(ys - cy) / max(ry, 1e-6), 0, 1)
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    campo = desloc[None, :] * tx[None, :] * ty[:, None]

    bloco = quadro[y0:y1, x0:x1].astype(np.float32)
    src = np.clip(np.arange(x1 - x0, dtype=np.float32)[None, :] + campo,
                  0, (x1 - x0) - 1.001)
    lo = src.astype(np.int32)
    fr = (src - lo)[:, :, None].astype(np.float32)
    linhas = np.arange(y1 - y0)[:, None]
    quadro[y0:y1, x0:x1] = (bloco[linhas, lo] * (1 - fr)
                            + bloco[linhas, lo + 1] * fr).astype(np.uint8)


def _abre_x(pts: dict[str, tuple[float, float]], xs: np.ndarray) -> np.ndarray:
    """Quanto a boca ABRE em cada coluna: cheio no meio, zero nas comissuras.

    Canto de boca nao abre -- e onde o labio de cima encontra o de baixo. Sem
    este perfil o vao saia como uma **barra retangular** de canto em esquadro,
    porque o labio de baixo descia o mesmo tanto no meio e na ponta. Com ele o
    vao vira lente: cheio no meio, fechando em bico nos dois cantos.

    O expoente abaixo de 1 alarga o meio: boca aberta e quase igual ao longo do
    centro e so fecha perto da ponta, que e diferente de um seno puro.
    """
    esq, dir_ = pts["boca_esq"][0], pts["boca_dir"][0]
    u = np.clip((xs - esq) / max(dir_ - esq, 1e-6), 0.0, 1.0)
    # O clip no seno nao e paranoia: `sin(pi)` devolve -8,7e-17 em ponto
    # flutuante, e base negativa com expoente fracionario vira NaN -- que
    # desce inteiro ate o indice da reamostragem e derruba o quadro.
    return np.clip(np.sin(np.pi * u), 0.0, 1.0) ** 0.65


def _amplitude_x(pts: dict[str, tuple[float, float]], xs: np.ndarray) -> np.ndarray:
    """Quanto do giro do maxilar chega a cada coluna (o mesmo perfil do campo).

    O labio de baixo e osso: ele desce com a mandibula. Para o vao pintado
    casar com o pixel que a deformacao moveu, os dois tem de usar este mesmo
    perfil -- se divergirem, sobra fresta de um lado e cobre labio do outro.
    """
    cx = (pts["boca_esq"][0] + pts["boca_dir"][0]) / 2
    boca_meia = max((pts["boca_dir"][0] - pts["boca_esq"][0]) / 2 * 0.84, 6.0)
    face_meia = max((pts["face_dir"][0] - pts["face_esq"][0]) / 2, boca_meia + 8.0)
    fora = np.clip((np.abs(xs - cx) - boca_meia) / (face_meia - boca_meia), 0, 1)
    fora = fora * fora * (3 - 2 * fora)
    return 1.0 - 0.68 * fora


def vao_boca(pts: dict[str, tuple[float, float]],
             contorno: tuple[np.ndarray, np.ndarray], queda: float,
             abertura: float, largura: float, tons: Tons
             ) -> tuple[Image.Image, tuple[int, int]] | None:
    """O vao entre os labios, recortado pelo contorno REAL da boca.

    A versao anterior desenhava uma elipse entre os labios. Na ampliacao 2x do
    artefato de 20/09/2026 dava para ver os tres defeitos que isso custa:

    1. a elipse tem tangente vertical no canto e boca tem canto em bico, entao
       ela avancava por cima da borda do labio de cima;
    2. onde ela nao alcancava sobrava o labio de cima **duplicado** pela
       deformacao, um contorno fantasma logo abaixo do verdadeiro;
    3. a boca ficava com a mesma forma em toda silaba, porque a elipse so
       mudava de tamanho.

    Aqui o vao e o poligono entre o arco interno de CIMA (que fica parado, e do
    cranio) e o arco interno de BAIXO deslocado pela queda do maxilar (que e
    osso e desce). Ou seja: o vao e exatamente a area que o labio de baixo
    desocupou, e a forma vem da boca do THEO, nao de uma elipse nossa.

    O dente pende do topo do poligono coluna a coluna, entao ele nasce com a
    curva do labio de cima de graca.
    """
    baixo, cima = contorno
    if baixo.size < 3 or cima.size < 3 or queda <= 0:
        return None
    cx = (pts["boca_esq"][0] + pts["boca_dir"][0]) / 2
    # Boca aberta recolhe nos cantos, e o visema ainda estica ou arredonda.
    escala_x = (1.0 - 0.10 * abertura) * (1.0 + LARGURA_MAX * largura)
    # Os dois arcos correm em sentidos opostos; alinhados, a media deles e a
    # **linha de costura** dos labios -- onde a boca de fato se abre.
    #
    # O vao nasce dessa costura, e nao do arco de baixo cru, porque na malha os
    # dois arcos ficam 1 a 3 px separados mesmo com a boca fechada (espessura
    # do labio e ruido de medida). Usando o arco cru, uma boca fechada ja
    # comecava com area e o resultado era um risco escuro permanente entre os
    # labios. Da costura, area zero em repouso, por construcao.
    cima_a = cima[::-1]
    costura = np.column_stack([cx + ((cima_a[:, 0] + baixo[:, 0]) / 2 - cx) * escala_x,
                               (cima_a[:, 1] + baixo[:, 1]) / 2])
    # O labio de baixo desce com o osso, pelo MESMO perfil que o campo usa na
    # linha dos labios -- se os dois divergirem, sobra fresta de um lado e o
    # vao cobre labio do outro. Afina um pouco ao abrir, porque labio estica:
    # sem isso ele desliza como uma laje rigida.
    desloc = (queda * _abre_x(pts, costura[:, 0])
              * _amplitude_x(pts, costura[:, 0]) * (1.0 - 0.12 * abertura))
    if float(desloc.max()) < 0.6:
        # Menos de meio pixel de vao: nao ha boca aberta, e desenhar aqui so
        # gastaria um ladrilho invisivel em todo quadro de consoante fechada.
        return None
    cima_e = costura
    baixo_e = np.column_stack([costura[:, 0], costura[:, 1] + desloc])

    # A borda de baixo da esquerda para a direita e a de cima na volta: e o
    # laco fechado. Empilhar os dois no mesmo sentido faz o poligono se cruzar
    # no meio e a boca sai com um X dentro -- aconteceu, da para ver.
    poli = np.vstack([baixo_e, cima_e[::-1]])
    x0 = math.floor(poli[:, 0].min()) - PAD_VAO
    y0 = math.floor(poli[:, 1].min()) - PAD_VAO
    w = int(math.ceil(poli[:, 0].max()) - x0) + PAD_VAO
    h = int(math.ceil(poli[:, 1].max()) - y0) + PAD_VAO
    if w < 4 or h < 4 or w > 4000 or h > 4000:
        return None

    # Mascara em SS vezes o tamanho: o `ImageDraw` do Pillow nao tem
    # anti-aliasing nenhum (medido: zero pixel de borda parcial). A parte
    # fracionaria da posicao entra nas coordenadas do desenho grande, entao o
    # vao cresce em passos menores que um pixel em vez de saltar.
    grande = Image.new("L", (w * SS, h * SS), 0)
    ImageDraw.Draw(grande).polygon(
        [((px - x0) * SS, (py - y0) * SS) for px, py in poli], fill=255)
    alfa = np.asarray(grande.resize((w, h), Image.BOX), dtype=np.float32) / 255.0
    if alfa.max() <= 0.02:
        return None
    # Entrada suave: o vao nasce de zero em vez de aparecer com 2 px de uma vez.
    altura = float((alfa > 0.5).sum(axis=0).max())
    alfa *= _ease_cubico(min(1.0, max(altura, 0.1) / BOCA_MIN_PX))

    # --- profundidade: onde o vao comeca e acaba, coluna a coluna
    dentro = alfa > 0.35
    tem = dentro.any(axis=0)
    topo = np.where(tem, dentro.argmax(axis=0), 0).astype(np.float32)
    fundo = np.where(tem, h - 1 - dentro[::-1].argmax(axis=0), 0).astype(np.float32)
    vao_h = np.maximum(fundo - topo, 1.0)
    linhas = np.arange(h, dtype=np.float32)[:, None]
    prof = np.clip((linhas - topo[None, :]) / vao_h[None, :], 0, 1)

    cor = np.empty((h, w, 3), dtype=np.float32)
    cor[:] = tons.interior
    # Fundo da boca mais escuro que a frente, e canto mais fundo que o meio.
    cor *= (1.0 - 0.34 * prof)[:, :, None]
    borda = np.clip(np.abs(np.arange(w, dtype=np.float32) - (cx - x0))
                    / max(w * 0.5, 1e-6), 0, 1)
    cor *= (1.0 - 0.30 * borda * borda)[None, :, None]

    # --- lingua: o fundo do vao nao e preto
    lingua = np.clip((prof - 0.46) / 0.22, 0, 1)
    lingua *= np.clip(1 - borda[None, :] / 0.82, 0, 1)
    lingua = lingua * lingua * (3 - 2 * lingua) * 0.75
    cor += (np.asarray(tons.lingua, dtype=np.float32) - cor) * lingua[:, :, None]

    if abertura > DENTE_MIN:
        forca = min(1.0, (abertura - DENTE_MIN) / (1 - DENTE_MIN))
        # O dente pende do labio de CIMA (ele e do cranio, nao desce com o
        # osso): nasce colado no topo do vao, com uma linha de sombra antes, e
        # termina numa ARESTA. Com a descida suave que estava aqui antes ele
        # lia como uma barra de metal polido dentro de um buraco -- o que
        # identifica dente e a borda de corte reta embaixo, nao o brilho.
        sobe = np.clip((prof - 0.03) / 0.09, 0, 1)
        desce = 1.0 - np.clip((prof - (0.30 + 0.10 * forca)) / 0.07, 0, 1)
        vy = np.clip(sobe, 0, 1) * np.clip(desce, 0, 1)
        vy = vy * vy * (3 - 2 * vy)
        vx = np.clip(1 - borda / 0.90, 0, 1)
        vx = vx * vx * (3 - 2 * vx)
        # Separacao entre os dentes: pouca, so para quebrar o gradiente liso.
        # Um dente do THEO tem ~14 px na tela, entao o periodo sai dai.
        fase = (np.arange(w, dtype=np.float32) - (cx - x0)) / 14.0
        sulco = 1.0 - 0.10 * (0.5 + 0.5 * np.cos(2 * np.pi * fase))
        peso = (alfa * vy * (vx * sulco)[None, :] * (0.55 + 0.45 * forca))[:, :, None]
        cor += (np.asarray(tons.dente, dtype=np.float32) - cor) * peso

    buf = np.empty((h, w, 4), dtype=np.uint8)
    buf[:, :, :3] = np.clip(cor, 0, 255).astype(np.uint8)
    buf[:, :, 3] = np.clip(alfa * 255.0, 0, 255).astype(np.uint8)
    tile = Image.fromarray(buf).filter(
        ImageFilter.GaussianBlur(max(0.6, altura * 0.045)))
    return tile, (x0, y0)


def falar(quadro: np.ndarray, pts: dict[str, tuple[float, float]],
          contorno: tuple[np.ndarray, np.ndarray], face_h: float,
          abertura: float, largura: float, tons: Tons) -> None:
    """Uma silaba no rosto: largura do labio, maxilar e o vao por cima.

    Nesta ordem de proposito: o esticar mexe no labio fechado, o maxilar desce
    o que esta abaixo dele, e o vao e pintado por ultimo, no buraco que os dois
    deixaram.
    """
    espalhar_labios(quadro, pts, largura)
    queda = abrir_maxilar(quadro, pts, face_h, abertura)
    if queda <= 0:
        return
    feito = vao_boca(pts, contorno, queda, abertura, largura, tons)
    if feito is None:
        return
    tile, canto = feito
    _compor(quadro, tile, canto[0], canto[1])


def _compor(quadro: np.ndarray, peca: Image.Image, x: int, y: int) -> None:
    """Sobrepoe o ladrilho na cor do quadro, preservando o alfa da silhueta.

    O alfa do quadro e o recorte do apresentador e nao pode ser tocado: a boca
    esta *dentro* da silhueta, e mexer no alfa ali abriria um buraco por onde o
    video de fundo apareceria no meio do rosto.
    """
    h, w = quadro.shape[:2]
    px = np.asarray(peca, dtype=np.float32)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + peca.width), min(h, y + peca.height)
    if x1 <= x0 or y1 <= y0:
        return
    sub = px[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (sub[:, :, 3:4] / 255.0)
    alvo = quadro[y0:y1, x0:x1, :3].astype(np.float32)
    quadro[y0:y1, x0:x1, :3] = (alvo + (sub[:, :, :3] - alvo) * a).astype(np.uint8)


# ===========================================================================
# o animador
# ===========================================================================


class AnimadorVideo:
    """Monta cada quadro: o clipe base com boca, posto na encenacao."""

    def __init__(self, base: Base, quadros: Quadros, accent: str, nome: str = "",
                 area: tuple[int, int, int, int] = AREA) -> None:
        self.base = base
        self.quadros = quadros
        self.area = area
        self.tons = medir_tons(quadros[base.frames // 2],
                               base.pontos(base.frames // 2))
        self._chip = plaquinha(nome, accent) if nome else None

    def quadro(self, t: float, abertura: float, largura: float,
               marcas: list[Marca]) -> Image.Image:
        ax, ay, aw, ah = self.area
        tela = Image.new("RGBA", (aw, ah), (0, 0, 0, 0))
        p = pose(marcas, t)
        if p.alfa <= 0.004 or p.altura <= 1:
            return tela

        k = indice_vaivem(t, self.base)
        quadro = self.quadros[k].copy()
        falar(quadro, self.base.pontos(k), self.base.contorno(k),
              self.base.face_height, abertura, largura, self.tons)

        bw, bh = self.base.size
        escala = p.altura / bh
        entrada = (1 - p.alfa) * 90        # desliza de fora enquanto aparece
        # A base do recorte encosta em p.base e o centro horizontal em p.cx.
        alvo_x = p.cx - ax - entrada
        alvo_y = p.base - ay
        ox = int(math.floor(alvo_x - bw * escala / 2)) - 2
        oy = int(math.floor(alvo_y - bh * escala)) - 2
        ow = int(math.ceil(bw * escala)) + 5
        oh = int(math.ceil(bh * escala)) + 5

        img = Image.fromarray(quadro)
        girada = _afim(img, (ow, oh), escala=escala, graus=0.0,
                       pivo=(bw / 2, bh), alvo=(alvo_x - ox, alvo_y - oy))
        if p.alfa < 0.999:
            girada.putalpha(girada.getchannel("A").point(
                lambda v, a=p.alfa: int(v * a)))
        _colar(tela, girada, ox, oy)
        self._plaquinha(tela, p, alvo_x, alvo_y)
        return tela

    def _plaquinha(self, tela: Image.Image, p: Marca, base_x: float,
                   base_y: float) -> None:
        """So enquanto ele esta grande, onde ele se apresenta."""
        from agent.render.presenter import ALTURA_CHAMADA, ALTURA_FECHAMENTO

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


def modular(abertura_visema: np.ndarray, energia: np.ndarray) -> np.ndarray:
    """Forma do fonema x forca do audio.

    A letra sabe *que* boca fazer, o audio sabe *com quanta forca*. Sozinha, a
    trilha de visema declama a frase inteira na mesma intensidade; sozinha, a
    envoltoria nao sabe fechar o labio no /m/. O piso existe porque fonema
    surdo tem pouca energia e a boca o faz do mesmo jeito.
    """
    n = min(len(abertura_visema), len(energia))
    ganho = ENERGIA_PISO + (1.0 - ENERGIA_PISO) * energia[:n]
    return np.clip(abertura_visema[:n] * ganho, 0.0, 1.0)


def render_layer(base: Base, audio: Path, out: Path, *, duracao: float,
                 batidas: Batidas, accent: str, palavras_faladas,
                 nome: str = "", fps: int = FPS, ffmpeg: str = "ffmpeg",
                 trabalho: Path | None = None) -> Camada:
    """Gera a camada do apresentador a partir do clipe base e devolve onde pousar."""
    n = max(1, int(round(duracao * fps)))
    energia = envoltoria(audio, n, fps, ffmpeg=ffmpeg)
    abertura, largura = trilha(palavras_faladas, n, fps)
    abertura = modular(abertura, energia)
    enc = plan(batidas)

    quadros = Quadros(base, trabalho or out.parent, ffmpeg=ffmpeg)
    anim = AnimadorVideo(base, quadros, accent, nome)
    ax, ay, aw, ah = anim.area

    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(encode_command(out, aw, ah, fps, ffmpeg=ffmpeg),
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for k in range(n):
            t = k / fps
            img = anim.quadro(t, float(abertura[k]), float(largura[k]), enc.marcas)
            proc.stdin.write(img.tobytes())
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


__all__ = ["AnimadorVideo", "Base", "DENTE_MIN", "MAXILAR", "PAD_VAO",
           "campo_maxilar", "vao_boca",
           "Quadros", "Tons", "abrir_maxilar", "carregar_base", "espalhar_labios",
           "falar", "indice_vaivem", "medir_tons", "modular", "render_layer"]
