"""Slides 1080x1920 do carrossel, no vetor de marca Seu Canal.

Uma peca, um acento: o pilar de conteudo decide (verde padrao, ciano nas
pecas de ruptura fato/futuro). Fundo #0A0A0C sempre, metade do quadro
respirando. Tipografia da marca (Space Grotesk nos titulos, Plex Mono nos
numeros) com fallback para a BeVietnamPro quando o glifo nao existir -- tofu
nunca, e fonte bitmap de 10px menos ainda (ver `render/typography.py`).

A distribuicao do conteudo veio de uma referencia que o autor trouxe em
20/09/2026 (um kit de post de IA no Envato). O layout anterior era honesto e
vazio: contador solto no topo, titulo e apoio no mesmo peso, um terco do quadro
sem nada. O que a referencia faz e dar **hierarquia em cinco camadas**, e e
isso que esta aqui:

1. chip da marca no topo (barra de acento + nome) e o contador do outro lado --
   o leitor sabe de quem e a peca e quanto falta antes de ler o titulo;
2. banho de acento na diagonal, que amarra foto e cor do pilar;
3. titulo grande com a **ultima linha no acento** -- na referencia e o que
   separa a promessa do assunto, e e o unico lugar onde a cor toca o texto;
4. apoio marcado por um quadrado de acento, nao solto sob o titulo;
5. rodape com regua, arroba e a acao (deslizar, ou salvar no ultimo).

O aceite mede o texto no PNG (`ink_height`), nao so as dimensoes: foi um
slide 1080x1920 com titulo ilegivel que passou no portao antigo.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from agent.brand.brand import load as load_brand
from agent.models import Carousel
from agent.render.typography import DISPLAY, MONO, font, hex_rgb, wrap

W, H = 1080, 1920
MARGEM = 80
LARGURA_TEXTO = W - 2 * MARGEM

# Faixa do titulo: o aceite mede tinta clara aqui. Um titulo legivel em
# Space Grotesk 70-96px ocupa bem mais que 50px de altura nessa faixa.
# O texto vive na metade escura de baixo: foto em cima, texto sobre fundo
# da marca -- texto sobre foto movimentada foi o que se viu na rodada de
# 19/09 (microscopio atras de "Identidade ancorada ao OS").
TITULO_Y = 960
ALTURA_MINIMA_TITULO_PX = 50

# Titulo longo desce de tamanho antes de passar de 4 linhas: 5 linhas de 96px
# empurram o texto de apoio para a zona que a interface do TikTok cobre.
TAMANHOS_TITULO = (96, 88, 78, 70)
MAX_LINHAS_TITULO = 4

TOPO_CHIP = 110
BARRA_ESQ = 12          # fita de acento na borda esquerda
FOTO_FIM = 820          # onde a foto ja virou fundo da marca
RODAPE_REGUA = 1640
RODAPE_TEXTO = 1690


def _fundo(foto: Path | None, fundo: tuple[int, int, int],
            acento: tuple[int, int, int]) -> Image.Image:
    """Foto em 9:16 escurecida ate virar o fundo da marca, + banho de acento.

    O banho na diagonal do topo e o que a referencia usa para a peca nao
    parecer um print de artigo: amarra a foto a cor do pilar. So entra **com
    foto**: sobre o quase-preto puro ele nao vira banho, vira dominante verde
    no quadro inteiro -- foi o que aconteceu na primeira tentativa, e a regra
    da marca e fundo quase-preto com 50% de vazio.
    """
    base = Image.new("RGB", (W, H), fundo)
    if foto is not None:
        try:
            img = _cobrir(Image.open(foto).convert("RGB"), W, H)
            img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), 0.5)
            mask = Image.new("L", (1, H))
            for y in range(H):
                # Foto inteira ate 26% da altura; do fim do degrade em diante,
                # so fundo da marca -- e ali que o titulo comeca.
                t = min(1.0, max(0.0, (y - H * 0.26) / (FOTO_FIM - H * 0.26)))
                mask.putpixel((0, y), int(255 * t))
            return _banho(Image.composite(base, img, mask.resize((W, H))), acento)
        except (OSError, ValueError):
            pass
    return base


def _banho(img: Image.Image, acento: tuple[int, int, int], forca: float = 0.16
           ) -> Image.Image:
    """Degrade diagonal do acento, so no canto de cima a direita.

    A rampa nasce zerada e so acende no ultimo terco da diagonal -- o resto da
    foto continua neutro, senao o acento vira filtro em vez de acento.
    """
    peq = 96
    rampa = Image.new("L", (peq, peq))
    px = rampa.load()
    for y in range(peq):
        for x in range(peq):
            d = (x / peq) * 0.6 + (1 - y / peq) * 0.4
            px[x, y] = int(255 * forca * max(0.0, d - 0.62) / 0.38)
    camada = Image.new("RGB", img.size, acento)
    return Image.composite(camada, img, rampa.resize(img.size, Image.BILINEAR))


def _cobrir(img: Image.Image, w: int, h: int) -> Image.Image:
    """Redimensiona cortando o excesso (cover), sem distorcer a foto."""
    escala = max(w / img.width, h / img.height)
    novo = img.resize((max(w, round(img.width * escala)),
                       max(h, round(img.height * escala))))
    x = (novo.width - w) // 2
    y = (novo.height - h) // 2
    return novo.crop((x, y, x + w, y + h))


def _titulo(draw: ImageDraw.ImageDraw, texto: str):
    """Maior tamanho da escala que cabe em MAX_LINHAS_TITULO linhas."""
    for tamanho in TAMANHOS_TITULO:
        f = font(DISPLAY, tamanho, texto)
        linhas = wrap(draw, texto, f, LARGURA_TEXTO)
        if len(linhas) <= MAX_LINHAS_TITULO:
            return f, linhas, tamanho
    return f, linhas, tamanho


def _destaque(linhas: list[str]) -> tuple[int, int]:
    """Onde o acento toca o titulo: (indice da linha, palavras coloridas do fim).

    A referencia sempre acende alguma coisa no titulo -- e o que separa a
    promessa do assunto. Com duas linhas ou mais, acende a ultima inteira; com
    uma linha so, acende a ultima palavra, desde que ela se sustente (palavra
    de tres letras colorida parece erro de digitacao, nao enfase).
    """
    if not linhas:
        return (-1, 0)
    if len(linhas) >= 2:
        ultima = linhas[-1]
        if len(ultima.split()) >= 2 or len(ultima) >= 6:
            return (len(linhas) - 1, 0)
        return (len(linhas) - 2, 0)
    palavras = linhas[0].split()
    if len(palavras) >= 2 and len(palavras[-1].strip(".,:;!?")) >= 5:
        return (0, 1)
    return (-1, 0)


def _chip(draw: ImageDraw.ImageDraw, brand, acento, apagada, slide_n: int,
          total: int) -> None:
    """Barra de acento + nome da marca a esquerda; contador a direita."""
    f = font(MONO, 36)
    draw.rectangle((MARGEM, TOPO_CHIP, MARGEM + 7, TOPO_CHIP + 40), fill=acento)
    draw.text((MARGEM + 24, TOPO_CHIP + 2), brand.name.upper(), font=f, fill=apagada)
    contador = f"{slide_n:02d}/{total:02d}"
    largura = draw.textbbox((0, 0), contador, font=f)[2]
    draw.text((W - MARGEM - largura, TOPO_CHIP + 2), contador, font=f, fill=acento)


def render_slide(carrossel: Carousel, n: int, out: Path,
                 pillar: str = "news", photo: Path | None = None) -> Path:
    """Um slide em PNG, no acento do pilar de conteudo."""
    brand = load_brand()
    slide = next(s for s in carrossel.slides if s.n == n)
    total = len(carrossel.slides)
    fundo = hex_rgb(brand.background)
    tinta = hex_rgb(brand.ink)
    apagada = hex_rgb(brand.muted)
    acento = hex_rgb(brand.accent_for(pillar))
    tag = brand.pillars[pillar].tag if pillar in brand.pillars else ""

    img = _fundo(photo, fundo, acento)
    draw = ImageDraw.Draw(img, "RGBA")
    # Fita de acento na borda esquerda, desbotando para baixo: a assinatura que
    # o canal ja tinha, agora estreita, porque o chip assumiu o papel de dizer
    # de quem e a peca.
    for i in range(H):
        draw.line([(0, i), (BARRA_ESQ, i)],
                  fill=tuple(int(fundo[k] + (acento[k] - fundo[k]) * (1 - i / H) * 0.6)
                             for k in range(3)))

    _chip(draw, brand, acento, apagada, slide.n, total)
    if n == 1 and tag:
        f_tag = font(MONO, 44)
        draw.text((MARGEM, TOPO_CHIP + 74), tag, font=f_tag, fill=acento)

    f_head, linhas_titulo, tamanho = _titulo(draw, slide.headline)
    linha_acesa, palavras_acesas = _destaque(linhas_titulo)
    y = TITULO_Y
    for i, linha in enumerate(linhas_titulo):
        if i == linha_acesa and palavras_acesas:
            partes = linha.rsplit(" ", palavras_acesas)
            inicio = " ".join(partes[:-palavras_acesas]) + " "
            draw.text((MARGEM, y), inicio, font=f_head, fill=tinta)
            draw.text((MARGEM + draw.textlength(inicio, font=f_head), y),
                      " ".join(partes[-palavras_acesas:]), font=f_head, fill=acento)
        else:
            draw.text((MARGEM, y), linha, font=f_head,
                      fill=acento if i == linha_acesa else tinta)
        y += int(tamanho * 1.18)

    # Apoio marcado: o quadrado de acento amarra a frase ao titulo em vez de
    # deixa-la boiando, e e o que a referencia faz em cada item da lista.
    f_text = font(DISPLAY, 52, slide.text)
    y += 34
    draw.rectangle((MARGEM, y + 14, MARGEM + 18, y + 32), fill=acento)
    for linha in wrap(draw, slide.text, f_text, LARGURA_TEXTO - 48):
        draw.text((MARGEM + 48, y), linha, font=f_text, fill=apagada)
        y += 68

    f_rodape = font(MONO, 34)
    if n == total:
        # Credito da fonte na propria peca: conteudo assistido por IA so e
        # elegivel ao Rewards quando original e verificavel, e o dominio na
        # tela e o que da para verificar sem link.
        dominios = _dominios(carrossel)
        if dominios:
            texto = "Fontes: " + " · ".join(dominios)
            base = RODAPE_REGUA - 40 - 44 * min(2, len(wrap(draw, texto, f_rodape,
                                                            LARGURA_TEXTO)))
            for i, linha in enumerate(wrap(draw, texto, f_rodape, LARGURA_TEXTO)[:2]):
                draw.text((MARGEM, base + i * 44), linha, font=f_rodape, fill=apagada)

    draw.line((MARGEM, RODAPE_REGUA, W - MARGEM, RODAPE_REGUA), fill=(*apagada, 90))
    f_pe = font(MONO, 36)
    draw.text((MARGEM, RODAPE_TEXTO), brand.handle, font=f_pe, fill=apagada)
    acao = "DESLIZE →" if n < total else "SALVE ESTE POST"
    largura = draw.textbbox((0, 0), acao, font=f_pe)[2]
    draw.text((W - MARGEM - largura, RODAPE_TEXTO), acao, font=f_pe, fill=acento)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def _dominios(carrossel: Carousel) -> list[str]:
    vistos: list[str] = []
    for f in carrossel.facts:
        d = str(f.source_url).split("://", 1)[-1].split("/", 1)[0].removeprefix("www.")
        if d and d not in vistos:
            vistos.append(d)
    return vistos[:3]


def render_carousel(carrossel: Carousel, out_dir: Path | str,
                    pillar: str = "news",
                    photos: dict[int, Path | None] | None = None) -> list[Path]:
    """Os 5 slides + caption.txt (gancho + contexto + fontes + hashtags)."""
    destino = Path(out_dir)
    fotos = photos or {}
    slides = [render_slide(carrossel, s.n, destino / f"slide-{s.n}.png", pillar,
                           fotos.get(s.n))
              for s in carrossel.slides]
    (destino / "caption.txt").write_text(caption_text(carrossel), encoding="utf-8")
    return slides


def caption_text(carrossel: Carousel) -> str:
    """Legenda do post: a linha do roteirista, as fontes e as 5 hashtags."""
    brand = load_brand()
    partes = [carrossel.caption]
    dominios = _dominios(carrossel)
    if dominios:
        partes.append("Fontes: " + ", ".join(dominios))
    partes.append(" ".join(brand.hashtags))
    return "\n\n".join(partes) + "\n"


def ink_height(png: Path | str, top: int = TITULO_Y, bottom: int = TITULO_Y + 480,
               limiar: int = 170) -> int:
    """Altura, em px, da faixa com tinta clara entre `top` e `bottom`.

    Mede o artefato, nao o codigo: titulo em fonte bitmap de 10px da ~10;
    titulo legivel da centenas. Pixel "claro" e canal medio >= `limiar`
    sobre o fundo escurecido, que e onde o texto da marca vive.
    """
    with Image.open(png) as img:
        cinza = img.convert("L").crop((MARGEM, top, W - MARGEM, bottom))
    largura = cinza.width
    dados = cinza.load()
    linhas_com_tinta = [
        y for y in range(cinza.height)
        if sum(1 for x in range(0, largura, 3) if dados[x, y] >= limiar) >= 3
    ]
    if not linhas_com_tinta:
        return 0
    return linhas_com_tinta[-1] - linhas_com_tinta[0] + 1


def legible(png: Path | str) -> bool:
    return ink_height(png) >= ALTURA_MINIMA_TITULO_PX


__all__ = ["caption_text", "ink_height", "legible", "render_carousel", "render_slide"]
