"""Renderizador propio: ffmpeg + edge-tts, detras de la misma puerta `Renderer`.

El MoneyPrinterTurbo sigue disponible (`mpt_renderer.py`), pero dejo de ser el
estandar por un motivo que solo aparecio con el video listo: la pronunciacion.
El hace la narracion solo, a partir del MISMO texto de la leyenda, y por la API
no acepta audio listo (`custom_audio_file` solo vale para archivo subido por su
interfaz web). Entonces "Gemini" salia "Zemini" y no habia donde corregirlo sin
tocar su codigo -- lo que la regla del proyecto prohibe.

Aqui cada pieza es nuestra y ya existia en el agente:

- narracion: `voice/edge.py` sobre el texto de `voice/pronounce.py` (la voz lee
  "Yemini"), con reserva offline en Piper;
- leyenda: `render/subtitles.py`, grafia original al tiempo de la voz;
- material: los clips que `render/footage.py` eligio, cortados en 5s en el
  orden de la narracion, cubriendo 1080x1920 sin distorsionar;
- verificacion: el Whisper de Groq (free tier) transcribe la narracion, y todo
  nombre que el oyente automatico no reconoce se vuelve aviso -- fue asi como
  se confirmo el caso de "Zemini" en las tres voces el 20/09/2026.

La marca, la pista y el loudness siguen en la posproduccion (`render/post.py`).
"""

from __future__ import annotations

import json
import math
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from agent.config import Settings
from agent.config import settings as default_settings
from agent.models import RenderResult, RenderState, Script
from agent.ports.renderer import RendererError
from agent.render.post import ffmpeg_bin
from agent.render.presenter import Capa, cargar, tiempos_por_palabras
from agent.render.presenter import render_capa as render_presentador
from agent.render.subtitles import Y_LEYENDA, align, build_ass
from agent.render.typography import DISPLAY_FAMILY, STATIC_DIR, display_bold_path
from agent.voice.edge import VOZ_FEMENINA, TTSUnavailable, synthesize
from agent.voice.pronounce import respell

CLIP_S = 5.0
FPS = 30
# Respiro despues de la ultima palabra, para que el corte no se trague el fin
# de la frase.
COLA_S = 0.6
GROQ_ASR = "https://api.groq.com/openai/v1/audio/transcriptions"


@dataclass
class NarrationCheck:
    transcript: str = ""
    suspicious: list[str] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class PresenterRequest:
    """Quien presenta este video, y con que material.

    `base` es el clip fotorrealista horneado (`<id>_base.json` + `.mp4`), en el
    que el parpadeo y el balanceo de cabeza ya son humanos; `cutout` es el
    retrato quieto, del que `render/presenter.py` sintetiza todo. Cuando los dos
    existen, el clip gana -- y por eso es un campo aparte y no un cambio de
    camino: la reserva tiene que seguir alcanzable si falta el clip.
    """

    cutout: Path | None = None
    name: str = ""
    base: Path | None = None


class FfmpegRenderer:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self.last_check: NarrationCheck | None = None
        self.last_changes: list[tuple[str, str]] = []
        # La capa del presentador nace aqui porque solo aqui existen las dos
        # cosas que necesita: el audio de la narracion (para la boca) y el
        # tiempo de cada palabra (para saber donde acaba el gancho). El
        # `render/post.py` solo la superpone.
        self.last_presenter: Capa | None = None
        self.presenter_error: str = ""

    def health(self) -> bool:
        try:
            out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-filters"],
                                 capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return False
        return " subtitles " in out and " concat " in out

    def render(self, script: Script, materials: list[Path] | None = None,
               voice: str | None = None, work_dir: Path | None = None,
               presenter: PresenterRequest | None = None) -> RenderResult:
        if not materials:
            raise RendererError("el renderizador ffmpeg necesita los clips elegidos "
                                "(render/footage.py); sin ellos, usa el MPT")
        trabajo = work_dir or (self.settings.output_dir / "_render")
        trabajo.mkdir(parents=True, exist_ok=True)

        self.presenter_error = ""
        hablado = respell(script.narration)
        self.last_changes = hablado.changes
        audio = trabajo / "narracion.mp3"
        try:
            tiempos = synthesize(hablado.text, audio, voice=voice or VOZ_FEMENINA)
        except TTSUnavailable as exc:
            audio, tiempos = self._reserva_local(hablado.text, trabajo, exc)
        duracion = _duracion(audio) + COLA_S
        self.last_check = self._comprobar(audio, script)

        palabras = align(hablado, tiempos)
        acento = self._acento(script)
        self.last_presenter = self._presentador(presenter, script, palabras,
                                                duracion, acento, trabajo, audio,
                                                tiempos)
        capa = self.last_presenter
        leyenda = build_ass(palabras, trabajo / "leyenda.ass",
                            accent=acento, font_family=DISPLAY_FAMILY,
                            duration=duracion,
                            y=capa.subtitle_y if capa else Y_LEYENDA,
                            start_at=capa.subtitle_start if capa else 0.0)
        display_bold_path()  # garantiza la fuente estatica para el libass
        salida = trabajo / "render.mp4"
        cmd = build_command(materials, audio, leyenda, salida, duracion,
                            ffmpeg=ffmpeg_bin(), fonts_dir=STATIC_DIR)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1200)
        except subprocess.CalledProcessError as exc:
            return RenderResult(state=RenderState.failed,
                                error=f"ffmpeg fallo: {exc.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            return RenderResult(state=RenderState.failed, error="ffmpeg paso de 20 min")
        medida = _probe(salida)
        return RenderResult(state=RenderState.complete, video_path=str(salida),
                            duration_s=medida["duration_s"], width=medida["width"],
                            height=medida["height"], has_audio=medida["has_audio"])

    # ------------------------------------------------------------------ apoyo

    def _acento(self, script: Script) -> str:
        from agent.brand.brand import load

        return load().accent_for(script.pillar or "news")

    def _presentador(self, pedido: PresenterRequest | None, script: Script,
                     palabras: list, duracion: float, acento: str,
                     trabajo: Path, audio: Path,
                     tiempos: list | None = None) -> Capa | None:
        """Sintetiza la capa del presentador, o sigue sin ella.

        Dos caminos, en este orden: el **clip base** (persona filmada, boca
        nuestra) y, si no existe, el **retrato quieto** (todo sintetizado).

        La boca del camino de video lee los tiempos del texto HABLADO
        (`tiempos`), no las palabras de la leyenda: el visema es de lo que la
        voz dice. La leyenda muestra "Gemini" y la voz dice "Yemini" -- tomar el
        fonema de la grafia de la leyenda pondria la boca haciendo una /g/
        donde el audio tiene una /y/.

        Fallar aqui no tira el video: el presentador es acabado, y un slot sin
        el sigue siendo publicable. El motivo queda en el log.
        """
        if pedido is None:
            return None
        tiempos_guion = tiempos_por_palabras(script.hook, script.closing,
                                             palabras, duracion)
        salida = trabajo / "presentador.mp4"
        if pedido.base is not None and pedido.base.exists():
            try:
                from agent.render.presenter_video import cargar_base
                from agent.render.presenter_video import render_capa as render_video

                return render_video(
                    cargar_base(pedido.base), audio, salida, duracion=duracion,
                    tiempos=tiempos_guion, accent=acento, nombre=pedido.name,
                    palabras_habladas=tiempos or [], ffmpeg=ffmpeg_bin(),
                    trabajo=trabajo)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                self.presenter_error = f"clip base: {type(exc).__name__}: {exc}"
        if pedido.cutout is None or not pedido.cutout.exists():
            return None
        try:
            retrato = cargar(pedido.cutout)
            return render_presentador(
                retrato, audio, salida, duracion=duracion,
                tiempos=tiempos_guion, accent=acento, nombre=pedido.name,
                ffmpeg=ffmpeg_bin(), semilla=script.topic)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            self.presenter_error = f"{type(exc).__name__}: {exc}"
            return None

    def _reserva_local(self, texto: str, trabajo: Path,
                       exc: Exception) -> tuple[Path, list]:
        from agent.voice.edge import local_fallback
        from agent.voice.library import VOICES

        modelo = self.settings.data_dir / "voices" / VOICES["dave"].archivo
        if not modelo.exists():
            raise RendererError(f"edge-tts caido ({exc}) y sin voz local en {modelo}") from exc
        tiempos = local_fallback(texto, trabajo / "narracion.mp3", voice_model=modelo)
        return trabajo / "narracion.wav", tiempos

    def _comprobar(self, audio: Path, script: Script) -> NarrationCheck:
        """Transcribe la narracion y apunta nombre propio que el oyente no reconocio."""
        clave = self.settings.groq_api_key
        if not clave:
            return NarrationCheck(error="sin clave de Groq: verificacion de pronunciacion saltada")
        try:
            with audio.open("rb") as fh:
                r = httpx.post(GROQ_ASR, headers={"authorization": f"Bearer {clave}"},
                               files={"file": (audio.name, fh, "audio/mpeg")},
                               data={"model": "whisper-large-v3", "language": "es",
                                     "temperature": "0", "response_format": "json"},
                               timeout=90)
            r.raise_for_status()
            texto = str(r.json().get("text", ""))
        except (httpx.HTTPError, OSError, ValueError) as exc:
            return NarrationCheck(error=f"verificacion indisponible: {exc}")
        return NarrationCheck(transcript=texto,
                              suspicious=suspicious_names(script.narration, texto))


def suspicious_names(original: str, transcript: str) -> list[str]:
    """Nombres propios del guion que no aparecen en la transcripcion del audio.

    Nombre con inicial mayuscula fuera del inicio de frase es lo que la voz
    castellana suele errar (grafia extranjera); si el Whisper no lo transcribe
    de vuelta, probablemente el oyente tampoco lo reconocio.
    """
    import re

    def norm(t: str) -> str:
        sin = unicodedata.normalize("NFD", t.lower())
        return "".join(c for c in sin if not unicodedata.combining(c))

    oido = norm(transcript)
    nombres: list[str] = []
    for frase in re.split(r"(?<=[.!?])\s+", original):
        palabras = frase.split()
        for p in palabras[1:]:
            limpio = re.sub(r"[^\wÀ-ÿ.-]", "", p).strip(".-")
            if len(limpio) >= 3 and limpio[:1].isupper() and not limpio.isupper():
                if norm(limpio) not in oido and limpio not in nombres:
                    nombres.append(limpio)
    return nombres


def build_command(clips: list[Path], audio: Path, leyenda: Path, salida: Path,
                  duracion: float, *, ffmpeg: str = "ffmpeg",
                  fonts_dir: Path | None = None) -> list[str]:
    """Clips en rebanadas de 5s en el orden del habla + narracion + leyenda quemada."""
    n = max(1, math.ceil(duracion / CLIP_S))
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    grafo: list[str] = []
    for k in range(n):
        clip = clips[k % len(clips)]
        # Clip reusado muestra otro tramo de el, no los mismos 5 segundos.
        vuelta = k // len(clips)
        dur = min(CLIP_S, duracion - k * CLIP_S)
        cmd += ["-ss", f"{0.3 + vuelta * CLIP_S:.2f}", "-t", f"{dur + 0.5:.2f}",
                "-i", str(clip)]
        grafo.append(
            f"[{k}:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,fps={FPS},setsar=1,format=yuv420p,"
            f"tpad=stop_mode=clone:stop_duration={CLIP_S},"
            f"trim=duration={dur:.3f},setpts=PTS-STARTPTS[s{k}]")
    cmd += ["-i", str(audio)]
    rebanadas = "".join(f"[s{k}]" for k in range(n))
    grafo.append(f"{rebanadas}concat=n={n}:v=1:a=0[vc]")
    fuentes = f":fontsdir='{fonts_dir}'" if fonts_dir else ""
    grafo.append(f"[vc]subtitles=filename='{leyenda}'{fuentes}[vout]")
    cmd += ["-filter_complex", ";".join(grafo),
            "-map", "[vout]", "-map", f"{n}:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-af", "apad", "-t", f"{duracion:.2f}", "-movflags", "+faststart", str(salida)]
    return cmd


def _duracion(audio: Path) -> float:
    medida = _probe(audio)
    if not medida["duration_s"]:
        raise RendererError(f"narracion sin duracion medible: {audio}")
    return float(medida["duration_s"])


def _probe(path: Path) -> dict:
    ffprobe = str(Path(ffmpeg_bin()).with_name("ffprobe"))
    if not Path(ffprobe).exists():
        ffprobe = "ffprobe"
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "stream=codec_type,width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60)
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    dur = (info.get("format") or {}).get("duration")
    return {"width": int(video["width"]) if video.get("width") else None,
            "height": int(video["height"]) if video.get("height") else None,
            "duration_s": round(float(dur), 3) if dur else None,
            "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


__all__ = ["FfmpegRenderer", "NarrationCheck", "PresenterRequest", "build_command",
           "suspicious_names"]
