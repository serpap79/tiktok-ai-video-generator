"""Posproduccion del video: marca, gancho en pantalla, presentador, pista y loudness.

El renderizador entrega narracion, leyenda y clips. Lo que faltaba para que el
video parezca del canal, y sujetar los primeros segundos, es nuestro:

- **tarjeta del gancho**: la frase del hook escrita en pantalla, con la etiqueta
  del tipo de contenido. Quien hace scroll sin sonido (la mayoria) lee el hueco
  antes de decidir; el audio solo no lo hace. Con presentador sale exactamente
  cuando entra la leyenda karaoke, para que el gancho no aparezca escrito dos
  veces;
- **presentador**: la capa RGBA que `render/presenter.py` sintetizo (silueta
  entera, boca al ritmo de la voz, parpadeo, escenificacion por guion) entra
  aqui como un video mas con alfa -- este archivo solo lo superpone;
- **marca de agua** discreta con el @ del canal, para el video que circula fuera;
- **pista generada** (`render/music.py`) con *ducking*: la musica baja sola
  cuando entra la voz (sidechain), en vez de un volumen fijo que pelea con el
  habla o desaparece en el silencio;
- **loudness a -14 LUFS** (EBU R128): volumen consistente entre videos, y sin
  clipear;
- **reticula de marca**: leve oscurecimiento y vineta, para que un clip claro no
  rompa el "dark" del canal.

La tipografia sale de Pillow con las fuentes de la marca (el mismo camino del
carrusel) y entra al ffmpeg como PNG: el `drawtext` dependeria de fontconfig y
de escapar texto con acento en la linea de comandos.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from agent.config import PROJECT_ROOT
from agent.render.presenter import Capa
from agent.render.typography import DISPLAY, MONO, font, hex_rgb, wrap

W, H = 1080, 1920
# Cuanto tiempo la tarjeta del gancho se queda en pantalla (0,4s de fade al
# final) cuando no hay presentador. Con presentador quien manda es la
# escenificacion: `Capa.card_s`.
GANCHO_S = 3.2
# Cima del panel del gancho. Con presentador sube: en la llamada el pelo del
# presentador empieza en ~815, y un panel de 4 lineas baja hasta ~754.
TOPO_TARJETA = 300
TOPO_TARJETA_CON_PRESENTADOR = 240
# Pista relativa a la voz antes del ducking: ~ -20 dB.
MUSICA_GANANCIA = 0.10
LOUDNESS_LUFS = -14


@dataclass(frozen=True)
class PostSpec:
    hook: str
    tag: str
    accent: str
    handle: str
    presenter: Capa | None = None

    @property
    def card_s(self) -> float:
        """La tarjeta desaparece cuando entra la leyenda -- o en GANCHO_S, sin presentador."""
        return self.presenter.card_s if self.presenter is not None else GANCHO_S


def ffmpeg_bin() -> str:
    """El ffmpeg estatico con libx264 (el de Fedora no lo tiene; ver setup_renderer.sh)."""
    candidato = os.environ.get("IMAGEIO_FFMPEG_EXE", "")
    if not candidato:
        env = PROJECT_ROOT / ".env"
        if env.exists():
            for linea in env.read_text(encoding="utf-8").splitlines():
                if linea.startswith("IMAGEIO_FFMPEG_EXE="):
                    candidato = linea.split("=", 1)[1].strip().strip('"')
                    break
    return candidato if candidato and Path(candidato).exists() else "ffmpeg"


def tarjeta_gancho(spec: PostSpec, out: Path) -> Path:
    """PNG 1080x1920 transparente: panel oscuro con etiqueta + gancho, tercio superior.

    Con presentador el panel sube -- en la llamada ocupa la mitad de abajo del
    cuadro. El chip con el nombre salio de aqui: es del presentador, camina con
    el, y quien lo dibuja es `render/presenter.py`.
    """
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    acento = hex_rgb(spec.accent)
    f_tag = font(MONO, 40)
    anchura = W - 2 * 90
    for tamano in (76, 68, 60, 54):
        f_hook = font(DISPLAY, tamano, spec.hook)
        lineas = wrap(draw, spec.hook, f_hook, anchura - 60)
        if len(lineas) <= 4:
            break
    paso = int(tamano * 1.2)
    cima = TOPO_TARJETA_CON_PRESENTADOR if spec.presenter is not None else TOPO_TARJETA
    altura = 70 + 30 + len(lineas) * paso + 50
    draw.rounded_rectangle((60, cima, W - 60, cima + altura), radius=28,
                           fill=(10, 10, 12, 205))
    draw.rectangle((60, cima + 24, 68, cima + altura - 24), fill=(*acento, 255))
    y = cima + 44
    if spec.tag:
        draw.text((100, y), spec.tag, font=f_tag, fill=(*acento, 255))
        y += 70
    for linea in lineas:
        draw.text((100, y), linea, font=f_hook, fill=(233, 238, 241, 255))
        y += paso
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def marca_agua(spec: PostSpec, out: Path) -> Path:
    """PNG transparente con el @ del canal, pequeño, en la esquina de arriba."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((56, 200), spec.handle, font=font(MONO, 30), fill=(233, 238, 241, 150))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def build_command(video_in: Path, out: Path, *, hook_png: Path, mark_png: Path,
                  music_wav: Path | None, duration_s: float,
                  presenter: Capa | None = None, card_s: float = GANCHO_S,
                  ffmpeg: str = "ffmpeg") -> list[str]:
    """Linea de comandos del ffmpeg. Aparte para que el test compruebe el grafo sin correr."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(video_in),
           "-loop", "1", "-t", f"{card_s:.2f}", "-i", str(hook_png),
           "-loop", "1", "-t", f"{duration_s:.2f}", "-i", str(mark_png)]
    entradas = 3
    if music_wav is not None:
        cmd += ["-i", str(music_wav)]
        musica = entradas
        entradas += 1
    if presenter is not None:
        cmd += ["-i", str(presenter.path)]
        capa = entradas
        entradas += 1

    grafo = [
        "[0:v]eq=brightness=-0.03:contrast=1.06:saturation=0.9,vignette=angle=0.55[base]",
        f"[1:v]format=rgba,fade=t=in:st=0:d=0.3:alpha=1,"
        f"fade=t=out:st={max(card_s - 0.4, 0.1):.2f}:d=0.4:alpha=1[hook]",
        "[base][hook]overlay=0:0:eof_action=pass[v1]",
        "[v1][2:v]overlay=0:0:shortest=1[v2]",
    ]
    ultimo = "v2"
    if presenter is not None:
        # La capa ya viene posicionada y con alfa: aqui es solo donde posa en
        # el cuadro. Nada de crop -- fue exactamente el crop el que recortaba
        # la silueta en rectangulo en el dibujo anterior.
        # La capa viene en dos pistas: color premultiplicado + mascara en gris
        # (ver `render/presenter.encode_command`). Aqui vuelven a ser un RGBA
        # con alfa directo.
        grafo.append(f"[{capa}:v:0][{capa}:v:1]alphamerge,"
                     f"unpremultiply=inplace=1,setpts=PTS-STARTPTS[apr]")
        grafo.append(f"[v2][apr]overlay={presenter.x}:{presenter.y}:"
                     f"eof_action=pass:format=auto[v3]")
        ultimo = "v3"
    if music_wav is not None:
        grafo += [
            "[0:a]asplit=2[voz][clave]",
            f"[{musica}:a]volume={MUSICA_GANANCIA}[mus]",
            "[mus][clave]sidechaincompress=threshold=0.03:ratio=6:attack=15:release=400[fondo]",
            "[voz][fondo]amix=inputs=2:duration=first:normalize=0[mix]",
            f"[mix]loudnorm=I={LOUDNESS_LUFS}:TP=-1.5:LRA=11[aout]",
        ]
    else:
        grafo.append(f"[0:a]loudnorm=I={LOUDNESS_LUFS}:TP=-1.5:LRA=11[aout]")

    cmd += ["-filter_complex", ";".join(grafo),
            "-map", f"[{ultimo}]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-movflags", "+faststart", "-t", f"{duration_s:.2f}", str(out)]
    return cmd


def postprocess(video_in: Path, out: Path, spec: PostSpec, *, duration_s: float,
                music_wav: Path | None, work_dir: Path, timeout_s: float = 900) -> Path:
    """Aplica la posproduccion y devuelve el MP4 final. Levanta RuntimeError con el stderr."""
    work_dir.mkdir(parents=True, exist_ok=True)
    gancho = tarjeta_gancho(spec, work_dir / "hook.png")
    marca = marca_agua(spec, work_dir / "marca.png")
    presentador = (spec.presenter
                   if spec.presenter is not None and spec.presenter.path.exists()
                   else None)
    cmd = build_command(video_in, out, hook_png=gancho, mark_png=marca,
                        music_wav=music_wav, duration_s=duration_s,
                        presenter=presentador, card_s=spec.card_s,
                        ffmpeg=ffmpeg_bin())
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg fallo en la posproduccion: {exc.stderr[-600:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg paso de {timeout_s:.0f}s en la posproduccion") from exc
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("posproduccion no genero archivo")
    return out


__all__ = ["GANCHO_S", "LOUDNESS_LUFS", "TOPO_TARJETA",
           "TOPO_TARJETA_CON_PRESENTADOR",
           "PostSpec", "build_command", "ffmpeg_bin", "tarjeta_gancho", "postprocess",
           "marca_agua"]
