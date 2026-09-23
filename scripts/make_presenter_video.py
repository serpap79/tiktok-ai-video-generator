#!/usr/bin/env python
"""Assa o video base do apresentador -- uma vez so, por clipe gerado.

O apresentador antigo era sintese pura: um retrato parado que `render/
presenter.py` fazia piscar e girar quadro a quadro. Funcionava, e o limite
estava declarado: silhueta, nao pessoa. Agora existe um clipe fotorrealista do
THEO (Hailuo, identidade travada do `brand.json`) em que a piscada, o balanco
de cabeca e a respiracao **ja sao humanos** -- foram gerados por um modelo de
video, nao por numpy. O que esse clipe nao tem e boca: ele escuta de labios
fechados do primeiro ao ultimo quadro (conferido: 141 quadros, nenhum com os
labios separados). E exatamente a divisao que serve, porque a boca e a unica
coisa que depende da narracao e, portanto, a unica que nao pode vir pronta.

Este script transforma o clipe cru em dois artefatos versionados:

- `<id>_base.mp4` -- o recorte do apresentador em duas trilhas de video: cor
  pre-multiplicada e mascara em cinza. Mesmo formato que `presenter.
  encode_command` ja usa, pela mesma razao: RGBA sem perda nao cabe em disco e
  RGB comum ganha franja escura na borda da silhueta.
- `<id>_base.json` -- os pontos do rosto **em cada quadro**. Sem isso a boca
  sintetizada nao sabe para onde ir: a cabeca se mexe no clipe, e uma boca
  desenhada em coordenada fixa desgruda do rosto no primeiro balanco.

Por que fora do pacote: `rembg` e `mediapipe` somam mais de 400 MB de roda com
onnxruntime e opencv atras. O agente roda tres vezes por dia e nunca precisa
disso -- mesma decisao do `make_presenter_cutouts.py`.

    uv run --no-project --with rembg --with mediapipe --with pillow --with numpy \
        python scripts/make_presenter_video.py --id theo --video brand/<clipe>.mp4

Se o clipe for curto, `render/presenter_video.py` toca em vai-e-vem para cobrir
a narracao; nada aqui depende da duracao.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "brand" / "assets" / "presenters"
MODELO_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
MODELO = RAIZ / "data" / "models" / "face_landmarker.task"

# Os mesmos indices da malha de 468 pontos que o recorte parado ja usa, mais os
# pontos internos do labio: no clipe a boca esta fechada, entao `labio_sup` e
# `labio_inf` caem quase no mesmo pixel -- e justamente a linha onde a sintese
# vai abrir a boca.
PONTOS = {
    "olho_esq_out": 33, "olho_esq_in": 133, "olho_esq_cima": 159, "olho_esq_baixo": 145,
    "olho_dir_in": 362, "olho_dir_out": 263, "olho_dir_cima": 386, "olho_dir_baixo": 374,
    "nariz_ponta": 1, "subnasal": 2, "labio_sup": 13, "labio_inf": 14,
    "boca_esq": 61, "boca_dir": 291, "queixo": 152, "testa": 10,
    "mand_esq": 172, "mand_dir": 397, "face_esq": 234, "face_dir": 454,
    # Borda externa dos labios: onde o labio encontra a pele. A deformacao da
    # boca precisa saber onde parar, senao arrasta bigode e queixo junto.
    "labio_sup_out": 0, "labio_inf_out": 17,
}
# O contorno INTERNO do labio, ponto a ponto. Sem ele a sintese desenhava uma
# elipse inventada no lugar do vao: a elipse tem tangente vertical nos cantos e
# boca tem canto em bico, entao ela cobria a borda do labio de cima e sobrava
# um contorno fantasma por baixo. Com o contorno de verdade o vao e recortado
# exatamente onde o labio acaba -- deixa de ser adesivo e passa a ser boca.
#
# Ordem canonica da malha: do canto esquerdo (78), pela borda de BAIXO ate o
# canto direito (308), e de volta pela borda de CIMA. Guardamos os dois arcos
# separados porque eles se movem por motivos diferentes: o de baixo desce com o
# osso, o de cima fica no cranio.
LABIO_INTERNO_BAIXO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LABIO_INTERNO_CIMA = [308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
FOLGA = 60
DEGRADE = 0.20
ALFA_MIN = 24
# Mediana temporal do alfa. O matting roda quadro a quadro e nao sabe que e um
# video: a borda do cabelo ferve alguns pixels entre quadros vizinhos. Com a
# camera travada, a mediana de 3 quadros mata a fervura sem comer movimento.
JANELA_ALFA = 3


def _baixar_modelo() -> Path:
    if MODELO.exists():
        return MODELO
    MODELO.parent.mkdir(parents=True, exist_ok=True)
    print(f"baixando {MODELO_URL} -> {MODELO}")
    urllib.request.urlretrieve(MODELO_URL, MODELO)  # noqa: S310  (URL fixa do Google)
    return MODELO


def _ffmpeg() -> str:
    import os
    return os.environ.get("IMAGEIO_FFMPEG_EXE", "ffmpeg")


def ler_quadros(video: Path) -> tuple[list[Image.Image], float]:
    """Decodifica o clipe inteiro para a memoria. Sao segundos, nao minutos."""
    sonda = subprocess.run(
        [_ffmpeg().replace("ffmpeg", "ffprobe") if "imageio" not in _ffmpeg() else "ffprobe",
         "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", str(video)], capture_output=True, text=True)
    if sonda.returncode != 0:
        # ffprobe pode nao existir ao lado do binario estatico do imageio.
        info = _sonda_por_ffmpeg(video)
    else:
        s = json.loads(sonda.stdout)["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        info = (int(s["width"]), int(s["height"]), float(num) / float(den))
    w, h, fps = info
    cru = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    n = len(cru) // (w * h * 3)
    dados = np.frombuffer(cru[:n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3)
    return [Image.fromarray(q) for q in dados], fps


def _sonda_por_ffmpeg(video: Path) -> tuple[int, int, float]:
    saida = subprocess.run([_ffmpeg(), "-i", str(video)],
                           capture_output=True, text=True).stderr
    import re
    m = re.search(r"(\d{2,5})x(\d{2,5})", saida)
    f = re.search(r"([\d.]+) fps", saida)
    if not m or not f:
        raise SystemExit(f"nao consegui sondar {video}")
    return int(m.group(1)), int(m.group(2)), float(f.group(1))


def medir_rosto(quadros: list[Image.Image], fps: float) -> list[dict[str, tuple[float, float]]]:
    """Pontos do rosto em cada quadro, no modo VIDEO da malha.

    O modo VIDEO nao e capricho: no modo IMAGE cada quadro e detectado do zero
    e os pontos tremem entre quadros vizinhos mesmo com a cabeca parada. O
    tremor viraria boca tremendo, que e pior que boca parada.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opcoes = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_baixar_modelo())),
        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    saida: list[dict[str, tuple[float, float]]] = []
    with vision.FaceLandmarker.create_from_options(opcoes) as detector:
        for k, img in enumerate(quadros):
            entrada = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=np.asarray(img.convert("RGB")))
            achado = detector.detect_for_video(entrada, int(k * 1000 / fps))
            if not achado.face_landmarks:
                if not saida:
                    raise SystemExit(f"nenhum rosto no quadro {k}")
                saida.append(saida[-1])  # segura o ultimo: melhor que buraco
                print(f"  aviso: sem rosto no quadro {k}, repetindo o anterior")
                continue
            malha = achado.face_landmarks[0]
            w, h = img.size
            pontos = {nome: (malha[i].x * w, malha[i].y * h)
                      for nome, i in PONTOS.items()}
            for rotulo, indices in (("labio_in_baixo", LABIO_INTERNO_BAIXO),
                                    ("labio_in_cima", LABIO_INTERNO_CIMA)):
                for j, i in enumerate(indices):
                    pontos[f"{rotulo}_{j:02d}"] = (malha[i].x * w, malha[i].y * h)
            saida.append(pontos)
    return saida


def recortar(quadros: list[Image.Image]) -> list[np.ndarray]:
    """Alfa de cada quadro, com mediana temporal contra a fervura da borda."""
    from rembg import new_session, remove

    sessao = new_session("u2net_human_seg")
    alfas = []
    for k, img in enumerate(quadros):
        fora = remove(img, session=sessao, post_process_mask=True,
                      only_mask=True)
        alfas.append(np.asarray(fora.convert("L"), dtype=np.uint8))
        if (k + 1) % 20 == 0:
            print(f"  matting {k + 1}/{len(quadros)}")
    pilha = np.stack(alfas)
    meio = JANELA_ALFA // 2
    suave = np.empty_like(pilha)
    for k in range(len(pilha)):
        ini, fim = max(0, k - meio), min(len(pilha), k + meio + 1)
        suave[k] = np.median(pilha[ini:fim], axis=0).astype(np.uint8)
    return list(suave)


def linha_da_marca(quadros: list[Image.Image]) -> int | None:
    """Primeira linha da marca d'agua do gerador, se houver uma.

    O Hailuo carimba "MINIMAX | Hailuo AI" no canto de baixo do clipe, e ela
    sobreviveu ao primeiro assado inteira: marca de outra ferramenta em cima do
    peito do apresentador, num canal que vive de parecer proprio e que precisa
    passar por originalidade para monetizar. Achada por acaso ao ampliar um
    quadro para olhar a boca -- ninguem tinha olhado o peito dele ainda.

    Como achar sem saber o texto: marca d'agua e **clara e imovel**. O
    apresentador respira, pisca e balanca; a marca nao muda um pixel em 15 s.
    Entao e a interseccao de "variancia temporal quase zero" com "bem mais
    claro que a camiseta preta", na metade de baixo do quadro. Medido no clipe
    do THEO: y 1290-1305, x 459-741, em 1344 linhas.

    Devolve None quando nao ha marca -- clipe de conta paga nao tem, e o corte
    nao deve encurtar o busto a toa.
    """
    if len(quadros) < 4:
        return None
    amostra = np.stack([np.asarray(q.convert("L"), dtype=np.float32)
                        for q in quadros[::max(1, len(quadros) // 16)]])
    h = amostra.shape[1]
    estatico = (amostra.var(axis=0) < 2.0) & (amostra.mean(axis=0) > 130)
    linhas = estatico.sum(axis=1)
    ys = np.flatnonzero(linhas > 25)
    ys = ys[ys > h // 2]
    return int(ys[0]) if ys.size else None


def _linha_do_busto(alfa: np.ndarray) -> int:
    h, _ = alfa.shape
    encosta = (alfa[:, :3] > ALFA_MIN).any(axis=1) | (alfa[:, -3:] > ALFA_MIN).any(axis=1)
    linhas = np.flatnonzero(encosta)
    return int(linhas[0]) if linhas.size else h


def _pescoco(alfa: np.ndarray, queixo_y: float, altura_rosto: float) -> int:
    h = alfa.shape[0]
    ini = min(h - 2, int(queixo_y) + 4)
    fim = min(h - 1, int(queixo_y + 0.55 * altura_rosto))
    if fim <= ini:
        return ini
    larguras = (alfa[ini:fim] > ALFA_MIN).sum(axis=1)
    return ini + int(np.argmin(larguras))


def processar(pid: str, video: Path) -> dict:
    print(f"lendo {video}")
    quadros, fps = ler_quadros(video)
    print(f"  {len(quadros)} quadros a {fps:.2f} fps, {quadros[0].width}x{quadros[0].height}")
    pontos = medir_rosto(quadros, fps)
    alfas = recortar(quadros)

    # Um corte so para o clipe inteiro: a camera e travada, e recorte que muda
    # de tamanho entre quadros faria o apresentador pular de escala na tela.
    medio = alfas[len(alfas) // 2]
    queixo_y = float(np.median([p["queixo"][1] for p in pontos]))
    testa_y = float(np.median([p["testa"][1] for p in pontos]))
    altura_rosto = queixo_y - testa_y
    corte = max(_linha_do_busto(medio), int(queixo_y + 0.85 * altura_rosto))
    corte = min(corte, medio.shape[0])
    marca = linha_da_marca(quadros)
    if marca is not None and marca - 8 < corte:
        print(f"  marca d'agua do gerador em y={marca}: cortando o busto acima dela")
        corte = max(int(queixo_y + 0.5 * altura_rosto), marca - 8)

    uniao = np.max(np.stack([a[:corte] for a in alfas]), axis=0)
    colunas = np.flatnonzero((uniao > ALFA_MIN).any(axis=0))
    linhas = np.flatnonzero((uniao > ALFA_MIN).any(axis=1))
    if not colunas.size or not linhas.size:
        raise SystemExit(f"{pid}: recorte vazio")
    x0, x1 = int(colunas[0]), int(colunas[-1]) + 1
    y0, y1 = int(linhas[0]), int(linhas[-1]) + 1
    caixa = (max(0, x0 - FOLGA), max(0, y0 - FOLGA),
             min(uniao.shape[1], x1 + FOLGA), min(corte, y1 + FOLGA))
    lw, lh = caixa[2] - caixa[0], caixa[3] - caixa[1]
    # Dimensao par: o yuv420p subamostra croma pela metade e recusa lado impar.
    lw -= lw % 2
    lh -= lh % 2
    dx, dy = -caixa[0], -caixa[1]

    faixa = max(1, int(lh * DEGRADE))
    rampa = np.linspace(1.0, 0.0, faixa, dtype=np.float32) ** 1.4

    saida_mp4 = SAIDA / f"{pid}_base.mp4"
    SAIDA.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(_encode(saida_mp4, lw, lh, fps),
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    topo_cabeca = lh
    for img, alfa in zip(quadros, alfas, strict=True):
        rgb = np.asarray(img.convert("RGB"))[caixa[1]:caixa[1] + lh,
                                             caixa[0]:caixa[0] + lw]
        a = alfa[caixa[1]:caixa[1] + lh, caixa[0]:caixa[0] + lw].astype(np.float32)
        a[lh - faixa:] *= rampa[:, None]
        suave = np.asarray(Image.fromarray(a.clip(0, 255).astype(np.uint8))
                           .filter(ImageFilter.GaussianBlur(0.7)))
        linhas_com_tinta = np.flatnonzero((suave > ALFA_MIN).any(axis=1))
        if linhas_com_tinta.size:
            topo_cabeca = min(topo_cabeca, int(linhas_com_tinta[0]))
        quadro = np.dstack([rgb, suave])
        proc.stdin.write(np.ascontiguousarray(quadro).tobytes())
    proc.stdin.close()
    erro = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg falhou: {erro[-500:]}")

    pontos = [{k: (x + dx, y + dy) for k, (x, y) in p.items()} for p in pontos]
    pescoco_y = _pescoco(np.asarray(medio)[caixa[1]:caixa[1] + lh,
                                           caixa[0]:caixa[0] + lw],
                         queixo_y + dy, altura_rosto)
    meta = {
        "id": pid,
        "source": str(video.relative_to(RAIZ)),
        "size": [lw, lh],
        "fps": round(fps, 4),
        "frames": len(quadros),
        "head_top": topo_cabeca,
        "face_height": round(altura_rosto, 1),
        "neck_y": pescoco_y,
        "pivot": [round(float(np.median([p["queixo"][0] for p in pontos])), 1),
                  round(pescoco_y + 0.75 * altura_rosto, 1)],
        # Um vetor por ponto, um par por quadro. Fica legivel no diff e nao
        # multiplica chave de dicionario por quadro.
        "tracks": {nome: [[round(p[nome][0], 1), round(p[nome][1], 1)] for p in pontos]
                   for nome in pontos[0]},
        "contorno": {"baixo": len(LABIO_INTERNO_BAIXO),
                     "cima": len(LABIO_INTERNO_CIMA)},
    }
    (SAIDA / f"{pid}_base.json").write_text(json.dumps(meta, indent=1) + "\n",
                                            encoding="utf-8")
    print(f"{pid}: {lw}x{lh} | {len(quadros)} quadros | rosto {altura_rosto:.0f}px | "
          f"pescoco y={pescoco_y} | topo da cabeca y={topo_cabeca}")
    print(f"  -> {saida_mp4}")
    print(f"  -> {SAIDA / f'{pid}_base.json'}")
    return meta


def _encode(out: Path, w: int, h: int, fps: float) -> list[str]:
    """Cor pre-multiplicada + mascara em cinza, as duas num arquivo so."""
    return [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
            "-r", f"{fps:.4f}", "-i", "-",
            "-filter_complex",
            "[0:v]split=2[pm][al];"
            "[pm]premultiply=inplace=1,format=yuv420p[c];"
            "[al]alphaextract,format=gray[m]",
            "-map", "[c]", "-map", "[m]",
            "-c:v", "libx264", "-preset", "slow", "-crf", "17",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", default="theo", help="id do apresentador")
    ap.add_argument("--video", required=True, type=Path, help="clipe base gerado")
    args = ap.parse_args()
    video = args.video if args.video.is_absolute() else RAIZ / args.video
    if not video.exists():
        raise SystemExit(f"{video} nao existe")
    processar(args.id, video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
