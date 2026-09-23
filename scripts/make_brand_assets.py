"""Gera os assets da marca a partir do vetor do guia (brand/brand.json).

Simbolo: pulso de dados entre dois pilares, so traco, nas 3 cores
permitidas (verde padrao, branco p/ video, ciano p/ ruptura). Avatar 1080:
fundo #0A0A0C + marca centralizada com area de protecao. Rode de novo se o
guia mudar; os PNG/SVG saem deterministicos.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "brand" / "assets"

# Do guia, secao "O simbolo" (viewBox 0 0 120 56, traco 3.4, ponta quadrada).
MARK_PATH = ("M20 44 L20 12 M100 12 L100 44 M40 40 L48 16 L56 40 "
             "M64 16 L64 40 M72 16 L80 40 L88 16")
VIEW_W, VIEW_H, STROKE = 120.0, 56.0, 3.4
CORES = {"verde": "#39FF88", "branco": "#E9EEF1", "ciano": "#00E0FF"}
BG = (10, 10, 12)


def _segmentos() -> list[list[tuple[float, float]]]:
    segs: list[list[tuple[float, float]]] = []
    atual: list[tuple[float, float]] = []
    for cmd, xs, ys in re.findall(r"([ML])\s*([\d.]+)\s+([\d.]+)", MARK_PATH):
        if cmd == "M":
            if atual:
                segs.append(atual)
            atual = [(float(xs), float(ys))]
        else:
            atual.append((float(xs), float(ys)))
    if atual:
        segs.append(atual)
    return [s for s in segs if len(s) >= 2]


def simbolo_svg(cor: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 56">'
        f'<path d="{MARK_PATH}" fill="none" stroke="{cor}" '
        f'stroke-width="{STROKE}" stroke-linecap="square"/></svg>\n')


def avatar_png(cor_hex: str, destino: Path, tamanho: int = 1080) -> None:
    cor = tuple(int(cor_hex[i:i + 2], 16) for i in (1, 3, 5))
    img = Image.new("RGB", (tamanho, tamanho), BG)
    draw = ImageDraw.Draw(img)
    # Marca ocupando ~55% da largura, centralizada um pouco acima do meio.
    escala = (tamanho * 0.55) / VIEW_W
    ox = (tamanho - VIEW_W * escala) / 2
    oy = tamanho * 0.5 - VIEW_H * escala / 2
    for seg in _segmentos():
        pts = [(ox + x * escala, oy + y * escala) for x, y in seg]
        draw.line(pts, fill=cor, width=max(2, int(STROKE * escala)), joint="curve")
    img.save(destino)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for nome, cor in CORES.items():
        (ASSETS / f"seucanal-simbolo-{nome}.svg").write_text(
            simbolo_svg(cor), encoding="utf-8")
    avatar_png(CORES["verde"], ASSETS / "seucanal-perfil-1080.png")
    print("assets em", ASSETS)


if __name__ == "__main__":
    main()
