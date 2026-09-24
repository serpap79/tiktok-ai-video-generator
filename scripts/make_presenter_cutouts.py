#!/usr/bin/env python
"""Recorta el retrato del presentador y mide los puntos del rostro: una sola vez.

Está fuera del paquete porque `rembg` (matting) y `mediapipe` (malla facial) suman
más de 400 MB de dependencias con onnxruntime y opencv. El agente se ejecuta tres veces
al día y nunca necesita esto: el recorte y las mediciones se hacen al generar el avatar
y el resultado (PNG + JSON) es lo que se versiona. Misma decisión que
`sentence-transformers`: una dependencia grande solo entra si se usa en el bucle principal.

    uv run --no-project --with rembg --with mediapipe --with pillow --with numpy \
        python scripts/make_presenter_cutouts.py

El modelo de malla facial se descarga una vez a `data/models/` (fuera de Git).

Qué erraba el recorte antiguo y por qué existe este: cortaba el sujeto en las
bordes de la imagen. La silueta se convertía en un rectángulo con el pecho cortado:
en la composición parecía una pegatina, no un presentador. Aquí el busto
se corta por encima de donde el hombro toca el borde y la última franja se vuelve
transparente con un degradado, como resuelve el mismo problema la viñeta de un telediario.

Genera un archivo por presentador:

- `<id>.png`: RGBA, sujeto entero con margen y base degradada;
- `<id>.json`: puntos del rostro en píxeles del PNG (ojos, base de la nariz, boca,
  mentón, cuello y pivote de rotación de la cabeza). Es lo que usa `render/presenter.py`
  para animar sin adivinar dónde está la boca.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

RAIZ = Path(__file__).resolve().parent.parent
FONTE = RAIZ / "brand" / "assets" / "presenters" / "source"
SAIDA = RAIZ / "brand" / "assets" / "presenters"
MODELO_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
MODELO = RAIZ / "data" / "models" / "face_landmarker.task"

# Índices de la malla de 468 puntos de MediaPipe (canónicos, no cambian entre versiones).
PONTOS = {
    "olho_esq_out": 33, "olho_esq_in": 133, "olho_esq_cima": 159, "olho_esq_baixo": 145,
    "olho_dir_in": 362, "olho_dir_out": 263, "olho_dir_cima": 386, "olho_dir_baixo": 374,
    "nariz_ponta": 1, "subnasal": 2, "labio_sup": 13, "labio_inf": 14,
    "boca_esq": 61, "boca_dir": 291, "queixo": 152, "testa": 10,
    "mand_esq": 172, "mand_dir": 397, "face_esq": 234, "face_dir": 454,
}
# Margen lateral y superior alrededor del sujeto: la cabeza gira hasta ~3 grados en la
# animación y no debe rozar el borde del PNG.
FOLGA = 60
# Degrade da base: fracao da altura do recorte que vai de opaco a transparente.
DEGRADE = 0.20
ALFA_MIN = 24


def _baixar_modelo() -> Path:
    if MODELO.exists():
        return MODELO
    MODELO.parent.mkdir(parents=True, exist_ok=True)
    print(f"baixando {MODELO_URL} -> {MODELO}")
    urllib.request.urlretrieve(MODELO_URL, MODELO)  # noqa: S310  (URL fixa do Google)
    return MODELO


def medir_rosto(img: Image.Image) -> dict[str, tuple[float, float]]:
    """Pontos do rosto em pixel da imagem de entrada."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opcoes = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_baixar_modelo())), num_faces=1)
    with vision.FaceLandmarker.create_from_options(opcoes) as detector:
        entrada = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.asarray(img.convert("RGB")))
        achado = detector.detect(entrada)
    if not achado.face_landmarks:
        raise SystemExit("nenhum rosto encontrado na imagem de origem")
    malha = achado.face_landmarks[0]
    w, h = img.size
    return {nome: (malha[i].x * w, malha[i].y * h) for nome, i in PONTOS.items()}


def recortar_fundo(img: Image.Image) -> Image.Image:
    from rembg import new_session, remove

    return remove(img, session=new_session("u2net"), post_process_mask=True)


def _linha_do_busto(alfa: np.ndarray) -> int:
    """Primera fila en la que el sujeto toca el borde lateral.

    Acima dela a silhueta e livre (da para cortar sem deixar aresta reta); dali
    para baixo o ombro ja saiu pela lateral e qualquer corte vira retangulo.
    """
    h, w = alfa.shape
    encosta = (alfa[:, :3] > ALFA_MIN).any(axis=1) | (alfa[:, -3:] > ALFA_MIN).any(axis=1)
    linhas = np.flatnonzero(encosta)
    return int(linhas[0]) if linhas.size else h


def _pescoco(alfa: np.ndarray, queixo_y: float, altura_rosto: float) -> int:
    """Linha mais estreita abaixo do queixo -- onde a cabeca encontra o tronco."""
    h = alfa.shape[0]
    ini = min(h - 2, int(queixo_y) + 4)
    fim = min(h - 1, int(queixo_y + 0.55 * altura_rosto))
    if fim <= ini:
        return ini
    larguras = (alfa[ini:fim] > ALFA_MIN).sum(axis=1)
    return ini + int(np.argmin(larguras))


def processar(pid: str, origem: Path) -> dict:
    img = Image.open(origem).convert("RGB")
    pontos = medir_rosto(img)
    recorte = recortar_fundo(img).convert("RGBA")
    alfa = np.array(recorte.getchannel("A"))

    corte = _linha_do_busto(alfa)
    queixo_y = pontos["queixo"][1]
    altura_rosto = queixo_y - pontos["testa"][1]
    # Nunca cortar antes do peito: o busto precisa de pelo menos meia altura de
    # rosto abaixo do queixo, senao sobra uma cabeca decepada no pescoco.
    corte = max(corte, int(queixo_y + 0.85 * altura_rosto))
    corte = min(corte, alfa.shape[0])

    recorte = recorte.crop((0, 0, recorte.width, corte))
    alfa = alfa[:corte]
    colunas = np.flatnonzero((alfa > ALFA_MIN).any(axis=0))
    linhas = np.flatnonzero((alfa > ALFA_MIN).any(axis=1))
    if not colunas.size or not linhas.size:
        raise SystemExit(f"{pid}: recorte vazio")
    x0, x1 = int(colunas[0]), int(colunas[-1]) + 1
    y0, y1 = int(linhas[0]), int(linhas[-1]) + 1

    caixa = (max(0, x0 - FOLGA), max(0, y0 - FOLGA),
             min(recorte.width, x1 + FOLGA), min(recorte.height, y1 + FOLGA))
    corpo = recorte.crop(caixa)
    # El margen que el recorte no consiguió (sujeto pegado al borde original) entra
    # como margem transparente, para a rotacao ter para onde ir.
    esq = FOLGA - (x0 - caixa[0])
    topo = FOLGA - (y0 - caixa[1])
    dir_ = FOLGA - (caixa[2] - x1)
    tela = Image.new("RGBA", (corpo.width + esq + dir_, corpo.height + topo), (0, 0, 0, 0))
    tela.paste(corpo, (esq, topo))

    dx = esq - caixa[0]
    dy = topo - caixa[1]
    pontos = {k: (x + dx, y + dy) for k, (x, y) in pontos.items()}

    # Degrade da base: da opacidade cheia ao zero na ultima faixa. Sem isso a
    # borda reta do busto denuncia o recorte assim que entra sobre o video.
    canal = np.array(tela.getchannel("A")).astype(np.float32)
    altura = canal.shape[0]
    faixa = max(1, int(altura * DEGRADE))
    rampa = np.linspace(1.0, 0.0, faixa, dtype=np.float32) ** 1.4
    canal[altura - faixa:] *= rampa[:, None]
    tela.putalpha(Image.fromarray(canal.clip(0, 255).astype(np.uint8)))
    # Uma passada de suavizacao no alfa tira o serrilhado que o matting deixa
    # no cabelo e no ombro.
    tela.putalpha(tela.getchannel("A").filter(ImageFilter.GaussianBlur(0.7)))

    alfa_final = np.array(tela.getchannel("A"))
    queixo_y = pontos["queixo"][1]
    altura_rosto = queixo_y - pontos["testa"][1]
    topo_cabeca = int(np.flatnonzero((alfa_final > ALFA_MIN).any(axis=1))[0])
    pescoco_y = _pescoco(alfa_final, queixo_y, altura_rosto)

    meta = {
        "id": pid,
        "source": str(origem.relative_to(RAIZ)),
        "size": [tela.width, tela.height],
        "head_top": topo_cabeca,
        "face_height": round(altura_rosto, 1),
        "neck_y": pescoco_y,
        # Pivo da rotacao da cabeca: dentro do tronco, bem abaixo do pescoco.
        # Girar em torno do proprio pescoco abriria fenda; daqui, o deslocamento
        # na costura e de poucos pixels e o topo da cabeca anda ~20.
        "pivot": [round(pontos["queixo"][0], 1),
                  round(pescoco_y + 0.75 * altura_rosto, 1)],
        "points": {k: [round(x, 1), round(y, 1)] for k, (x, y) in pontos.items()},
    }
    SAIDA.mkdir(parents=True, exist_ok=True)
    tela.save(SAIDA / f"{pid}.png")
    (SAIDA / f"{pid}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{pid}: {tela.width}x{tela.height} | rosto {altura_rosto:.0f}px | "
          f"pescoco y={pescoco_y} | corte do busto y={corte}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="so um apresentador (iris|theo)")
    args = ap.parse_args()
    origens = sorted(FONTE.glob("*.jpg"))
    if not origens:
        raise SystemExit(f"sem retratos em {FONTE}")
    for origem in origens:
        if args.only and origem.stem != args.only:
            continue
        processar(origem.stem, origem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
