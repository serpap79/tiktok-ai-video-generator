"""Diapositivas 1080x1920 del carrusel, con el vector de marca de Circuito Cero.

Una pieza, un acento: el pilar de contenido decide (verde predeterminado, cian para
pecas de ruptura fato/futuro). Fundo #0A0A0C sempre, metade do quadro
respirando. Tipografia da marca (Space Grotesk nos titulos, Plex Mono nos
números), con BeVietnamPro como alternativa cuando falte un glifo. Nunca tofu
y menos aún una fuente de mapa de bits de 10 px (consulta `render/typography.py`).

La distribución del contenido procede de una referencia aportada por el autor
20/09/2026 (um kit de post de IA no Envato). O layout anterior era honesto e
vazio: contador solto no topo, titulo e apoio no mesmo peso, um terco do quadro
sin nada más. La referencia aporta **jerarquía en cinco capas**, que es
isso que esta aqui:

1. chip da marca no topo (barra de acento + nome) e o contador do outro lado --
   o leitor sabe de quem e a peca e quanto falta antes de ler o titulo;
2. banho de acento na diagonal, que amarra foto e cor do pilar;
3. título grande con la **última línea en el acento**; en la referencia es lo que
   separa la promesa del asunto y el único lugar donde el color toca el texto;
4. apoyo marcado con un cuadro de acento, no suelto bajo el título;
5. rodape com regua, arroba e a accion (deslizar, ou salvar no ultimo).

La aceptación mide el texto en el PNG (`ink_height`), no solo las dimensiones: fue un
slide 1080x1920 com titulo ilegivel que passou no portao antigo.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from agent.brand.brand import load as load_brand
from agent.models import Carousel
from agent.render.typography import DISPLAY, MONO, font, hex_rgb, wrap

W, H = 1080, 1920
MARGEN = 80
LARGURA_TEXTO = W - 2 * MARGEN

# Faixa do titulo: o aceite mede tinta clara aqui. Um titulo legivel em
# Space Grotesk 70-96px ocupa bem mais que 50px de altura nessa faixa.
# O texto vive na metade escura de baixo: foto em cima, texto sobre fundo
# de la marca. El texto sobre una foto en movimiento fue lo observado en la ronda de
# 19/09 (microscopio atras de "Identidade ancorada ao OS").
TITULO_Y = 960
ALTURA_MINIMA_TITULO_PX = 50

# Titulo longo desce de tamanho antes de passar de 4 linhas: 5 linhas de 96px
# empurram o texto de apoio para a zona que a interface do TikTok cobre.
TAMANHOS_TITULO = (96, 88, 78, 70)
MAX_LINHAS_TITULO = 4

TOPO_CHIP = 110
BARRA_IZQ = 12          # fita de acento na borda esquerda
FOTO_FIM = 820          # onde a foto ja virou fundo da marca
RODAPIE_REGLA = 1640
RODAPIE_TEXTO = 1690


def _fundo(foto: Path | None, fundo: tuple[int, int, int],
            acento: tuple[int, int, int]) -> Image.Image:
    """Foto em 9:16 escurecida ate virar o fundo da marca, + banho de acento.

    El degradado diagonal superior es lo que usa la referencia para que la pieza no
    parecer um print de artigo: amarra a foto a cor do pilar. So entra **com
    sea una foto**: sobre un casi negro puro no se vuelve degradado, sino verde dominante
    en todo el cuadro. Sucedió en el primer intento y la regla
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

    La referencia siempre resalta algo en el título; es lo que separa la
    promesa del asunto. Con dos líneas o más, resalta la última completa; con
    una sola línea, resalta la última palabra si se sostiene (una palabra
    de tres letras coloreada parece un error de escritura, no un énfasis).
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
    draw.rectangle((MARGEN, TOPO_CHIP, MARGEN + 7, TOPO_CHIP + 40), fill=acento)
    draw.text((MARGEN + 24, TOPO_CHIP + 2), brand.name.upper(), font=f, fill=apagada)
    contador = f"{slide_n:02d}/{total:02d}"
    largura = draw.textbbox((0, 0), contador, font=f)[2]
    draw.text((W - MARGEN - largura, TOPO_CHIP + 2), contador, font=f, fill=acento)


def render_slide(carrossel: Carousel, n: int, out: Path,
                 pillar: str = "news", photo: Path | None = None) -> Path:
    """Una diapositiva en PNG, con el acento del pilar de contenido."""
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
        draw.line([(0, i), (BARRA_IZQ, i)],
                  fill=tuple(int(fundo[k] + (acento[k] - fundo[k]) * (1 - i / H) * 0.6)
                             for k in range(3)))

    _chip(draw, brand, acento, apagada, slide.n, total)
    if n == 1 and tag:
        f_tag = font(MONO, 44)
        draw.text((MARGEN, TOPO_CHIP + 74), tag, font=f_tag, fill=acento)

    f_head, linhas_titulo, tamanho = _titulo(draw, slide.headline)
    linha_acesa, palavras_acesas = _destaque(linhas_titulo)
    y = TITULO_Y
    for i, linha in enumerate(linhas_titulo):
        if i == linha_acesa and palavras_acesas:
            partes = linha.rsplit(" ", palavras_acesas)
            inicio = " ".join(partes[:-palavras_acesas]) + " "
            draw.text((MARGEN, y), inicio, font=f_head, fill=tinta)
            draw.text((MARGEN + draw.textlength(inicio, font=f_head), y),
                      " ".join(partes[-palavras_acesas:]), font=f_head, fill=acento)
        else:
            draw.text((MARGEN, y), linha, font=f_head,
                      fill=acento if i == linha_acesa else tinta)
        y += int(tamanho * 1.18)

    # Apoio marcado: o quadrado de acento amarra a frase ao titulo em vez de
    # la deja flotar, como hace la referencia con cada elemento de la lista.
    f_text = font(DISPLAY, 52, slide.text)
    y += 34
    draw.rectangle((MARGEN, y + 14, MARGEN + 18, y + 32), fill=acento)
    for linha in wrap(draw, slide.text, f_text, LARGURA_TEXTO - 48):
        draw.text((MARGEN + 48, y), linha, font=f_text, fill=apagada)
        y += 68

    f_rodape = font(MONO, 34)
    if n == total:
        # Crédito de la fuente en la propia pieza: el contenido asistido por IA solo es
        # elegivel ao Rewards quando original e verificavel, e o dominio na
        # una imagen completa y es lo que se puede verificar sin enlace.
        dominios = _dominios(carrossel)
        if dominios:
            texto = "Fuentes: " + " · ".join(dominios)
            base = RODAPIE_REGLA - 40 - 44 * min(2, len(wrap(draw, texto, f_rodape,
                                                            LARGURA_TEXTO)))
            for i, linha in enumerate(wrap(draw, texto, f_rodape, LARGURA_TEXTO)[:2]):
                draw.text((MARGEN, base + i * 44), linha, font=f_rodape, fill=apagada)

    draw.line((MARGEN, RODAPIE_REGLA, W - MARGEN, RODAPIE_REGLA), fill=(*apagada, 90))
    f_pie = font(MONO, 36)
    draw.text((MARGEN, RODAPIE_TEXTO), brand.handle, font=f_pie, fill=apagada)
    accion = "DESLIZE →" if n < total else "SALVE ESTE POST"
    largura = draw.textbbox((0, 0), accion, font=f_pie)[2]
    draw.text((W - MARGEN - largura, RODAPIE_TEXTO), accion, font=f_pie, fill=acento)
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
    """Las 5 diapositivas + caption.txt (gancho + contexto + fuentes + hashtags)."""
    destino = Path(out_dir)
    fotos = photos or {}
    slides = [render_slide(carrossel, s.n, destino / f"slide-{s.n}.png", pillar,
                           fotos.get(s.n))
              for s in carrossel.slides]
    (destino / "caption.txt").write_text(caption_text(carrossel), encoding="utf-8")
    return slides


def caption_text(carrossel: Carousel) -> str:
    """Descripción de la publicación: la línea del guionista, las fuentes y 5 hashtags."""
    brand = load_brand()
    partes = [carrossel.caption]
    dominios = _dominios(carrossel)
    if dominios:
        partes.append("Fuentes: " + ", ".join(dominios))
    partes.append(" ".join(brand.hashtags))
    return "\n\n".join(partes) + "\n"


def ink_height(png: Path | str, top: int = TITULO_Y, bottom: int = TITULO_Y + 480,
               limiar: int = 170) -> int:
    """Altura, em px, da faixa com tinta clara entre `top` e `bottom`.

    Mide el artefacto, no el código: un título con fuente de mapa de bits de 10 px da ~10;
    titulo legivel da centenas. Pixel "claro" e canal medio >= `limiar`
    sobre o fundo escurecido, que e onde o texto da marca vive.
    """
    with Image.open(png) as img:
        cinza = img.convert("L").crop((MARGEN, top, W - MARGEN, bottom))
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
