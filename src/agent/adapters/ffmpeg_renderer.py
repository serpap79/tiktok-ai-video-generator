"""Renderizador proprio: ffmpeg + edge-tts, atras da mesma porta `Renderer`.

O MoneyPrinterTurbo continua disponivel (`mpt_renderer.py`), mas deixou de
ser o padrao por um motivo que so apareceu com o video pronto: a pronuncia.
Ele faz a narracao sozinho, a partir do MESMO texto da legenda, e pela API
nao aceita audio pronto (`custom_audio_file` so vale para arquivo enviado
pela interface web dele). Entao "Gemini" saia "Zemini" e nao havia onde
corrigir sem mexer no codigo dele -- o que a regra do projeto proibe.

Aqui cada peca e nossa e ja existia no agente:

- narracao: `voice/edge.py` sobre o texto de `voice/pronounce.py` (a voz le
  "Djemini"), com reserva offline em Piper;
- legenda: `render/subtitles.py`, grafia original no tempo da voz;
- material: os clipes que `render/footage.py` escolheu, cortados em 5s na
  ordem da narracao, cobrindo 1080x1920 sem distorcer;
- conferencia: o Whisper do Groq (free tier) transcreve a narracao, e todo
  nome que o ouvinte automatico nao reconhece vira aviso -- foi assim que o
  "Zemini" foi confirmado nas tres vozes em 20/09/2026.

A marca, a trilha e o loudness continuam na pos-producao (`render/post.py`).
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
from agent.render.presenter import Camada, batidas_por_tempo, carregar
from agent.render.presenter import render_layer as render_presenter
from agent.render.subtitles import Y_LEGENDA, align, build_ass
from agent.render.typography import DISPLAY_FAMILY, STATIC_DIR, display_bold_path
from agent.voice.edge import VOZ_FEMININA, TTSUnavailable, synthesize
from agent.voice.pronounce import respell

CLIP_S = 5.0
FPS = 30
# Respiro depois da ultima palavra, para o corte nao engolir o fim da frase.
CAUDA_S = 0.6
GROQ_ASR = "https://api.groq.com/openai/v1/audio/transcriptions"


@dataclass
class NarrationCheck:
    transcript: str = ""
    suspicious: list[str] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class PresenterRequest:
    """Quem apresenta este video, e de que material.

    `base` e o clipe fotorrealista assado (`<id>_base.json` + `.mp4`), em que a
    piscada e o balanco de cabeca ja sao humanos; `cutout` e o retrato parado,
    de que `render/presenter.py` sintetiza tudo. Quando os dois existem, o
    clipe ganha -- e por isso ele e um campo a parte e nao uma troca de
    caminho: a reserva tem de continuar alcancavel se o clipe faltar.
    """

    cutout: Path | None = None
    name: str = ""
    base: Path | None = None


class FfmpegRenderer:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self.last_check: NarrationCheck | None = None
        self.last_changes: list[tuple[str, str]] = []
        # A camada do apresentador nasce aqui porque so aqui existem as duas
        # coisas de que ela precisa: o audio da narracao (para a boca) e o
        # tempo de cada palavra (para saber onde o gancho acaba). O
        # `render/post.py` so a sobrepoe.
        self.last_presenter: Camada | None = None
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
            raise RendererError("o renderizador ffmpeg precisa dos clipes escolhidos "
                                "(render/footage.py); sem eles, use o MPT")
        trabalho = work_dir or (self.settings.output_dir / "_render")
        trabalho.mkdir(parents=True, exist_ok=True)

        self.presenter_error = ""
        falado = respell(script.narration)
        self.last_changes = falado.changes
        audio = trabalho / "narracao.mp3"
        try:
            tempos = synthesize(falado.text, audio, voice=voice or VOZ_FEMININA)
        except TTSUnavailable as exc:
            audio, tempos = self._reserva_local(falado.text, trabalho, exc)
        duracao = _duracao(audio) + CAUDA_S
        self.last_check = self._conferir(audio, script)

        palavras = align(falado, tempos)
        acento = self._acento(script)
        self.last_presenter = self._apresentador(presenter, script, palavras,
                                                 duracao, acento, trabalho, audio,
                                                 tempos)
        camada = self.last_presenter
        legenda = build_ass(palavras, trabalho / "legenda.ass",
                            accent=acento, font_family=DISPLAY_FAMILY,
                            duration=duracao,
                            y=camada.subtitle_y if camada else Y_LEGENDA,
                            start_at=camada.subtitle_start if camada else 0.0)
        display_bold_path()  # garante a fonte estatica para o libass
        saida = trabalho / "render.mp4"
        cmd = build_command(materials, audio, legenda, saida, duracao, ffmpeg=ffmpeg_bin(),
                            fonts_dir=STATIC_DIR)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1200)
        except subprocess.CalledProcessError as exc:
            return RenderResult(state=RenderState.failed,
                                error=f"ffmpeg falhou: {exc.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            return RenderResult(state=RenderState.failed, error="ffmpeg passou de 20 min")
        medida = _probe(saida)
        return RenderResult(state=RenderState.complete, video_path=str(saida),
                            duration_s=medida["duration_s"], width=medida["width"],
                            height=medida["height"], has_audio=medida["has_audio"])

    # ------------------------------------------------------------------ apoio

    def _acento(self, script: Script) -> str:
        from agent.brand.brand import load

        return load().accent_for(script.pillar or "news")

    def _apresentador(self, pedido: PresenterRequest | None, script: Script,
                      palavras: list, duracao: float, acento: str,
                      trabalho: Path, audio: Path,
                      tempos: list | None = None) -> Camada | None:
        """Sintetiza a camada do apresentador, ou segue sem ela.

        Dois caminhos, nesta ordem: o **clipe base** (pessoa filmada, boca
        nossa) e, se ele nao existir, o **retrato parado** (tudo sintetizado).

        A boca do caminho de video le os tempos do texto FALADO (`tempos`), nao
        das palavras da legenda: o visema e do que a voz diz. A legenda mostra
        "Gemini" e a voz diz "Djemini" -- tirar fonema da grafia da legenda
        poria a boca fazendo um /g/ onde o audio tem um /dj/.

        Falhar aqui nao derruba o video: o apresentador e acabamento, e um
        slot sem ele ainda e um slot publicavel. O motivo fica no log.
        """
        if pedido is None:
            return None
        batidas = batidas_por_tempo(script.hook, script.closing, palavras, duracao)
        saida = trabalho / "apresentador.mp4"
        if pedido.base is not None and pedido.base.exists():
            try:
                from agent.render.presenter_video import carregar_base
                from agent.render.presenter_video import render_layer as render_video

                return render_video(
                    carregar_base(pedido.base), audio, saida, duracao=duracao,
                    batidas=batidas, accent=acento, nome=pedido.name,
                    palavras_faladas=tempos or [], ffmpeg=ffmpeg_bin(),
                    trabalho=trabalho)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                self.presenter_error = f"clipe base: {type(exc).__name__}: {exc}"
        if pedido.cutout is None or not pedido.cutout.exists():
            return None
        try:
            retrato = carregar(pedido.cutout)
            return render_presenter(
                retrato, audio, saida, duracao=duracao,
                batidas=batidas, accent=acento, nome=pedido.name,
                ffmpeg=ffmpeg_bin(), semente=script.topic)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            self.presenter_error = f"{type(exc).__name__}: {exc}"
            return None

    def _reserva_local(self, texto: str, trabalho: Path,
                       exc: Exception) -> tuple[Path, list]:
        from agent.voice.edge import local_fallback
        from agent.voice.library import VOICES

        modelo = self.settings.data_dir / "voices" / VOICES["razo"].arquivo
        if not modelo.exists():
            raise RendererError(f"edge-tts fora ({exc}) e sem voz local em {modelo}") from exc
        tempos = local_fallback(texto, trabalho / "narracao.mp3", voice_model=modelo)
        return trabalho / "narracao.wav", tempos

    def _conferir(self, audio: Path, script: Script) -> NarrationCheck:
        """Transcreve a narracao e aponta nome proprio que o ouvinte nao reconheceu."""
        chave = self.settings.groq_api_key
        if not chave:
            return NarrationCheck(error="sem chave do Groq: conferencia de pronuncia pulada")
        try:
            with audio.open("rb") as fh:
                r = httpx.post(GROQ_ASR, headers={"authorization": f"Bearer {chave}"},
                               files={"file": (audio.name, fh, "audio/mpeg")},
                               data={"model": "whisper-large-v3", "language": "pt",
                                     "temperature": "0", "response_format": "json"},
                               timeout=90)
            r.raise_for_status()
            texto = str(r.json().get("text", ""))
        except (httpx.HTTPError, OSError, ValueError) as exc:
            return NarrationCheck(error=f"conferencia indisponivel: {exc}")
        return NarrationCheck(transcript=texto,
                              suspicious=suspicious_names(script.narration, texto))


def suspicious_names(original: str, transcript: str) -> list[str]:
    """Nomes proprios do roteiro que nao aparecem na transcricao do audio.

    Nome com inicial maiuscula fora do inicio de frase e o que a voz pt-BR
    erra (grafia estrangeira); se o Whisper nao o transcreve de volta, o
    ouvinte provavelmente tambem nao reconheceu.
    """
    import re

    def norm(t: str) -> str:
        sem = unicodedata.normalize("NFD", t.lower())
        return "".join(c for c in sem if not unicodedata.combining(c))

    ouvido = norm(transcript)
    nomes: list[str] = []
    for frase in re.split(r"(?<=[.!?])\s+", original):
        palavras = frase.split()
        for p in palavras[1:]:
            limpo = re.sub(r"[^\wÀ-ÿ.-]", "", p).strip(".-")
            if len(limpo) >= 3 and limpo[:1].isupper() and not limpo.isupper():
                if norm(limpo) not in ouvido and limpo not in nomes:
                    nomes.append(limpo)
    return nomes


def build_command(clips: list[Path], audio: Path, legenda: Path, saida: Path,
                  duracao: float, *, ffmpeg: str = "ffmpeg",
                  fonts_dir: Path | None = None) -> list[str]:
    """Clipes em fatias de 5s na ordem da fala + narracao + legenda queimada."""
    n = max(1, math.ceil(duracao / CLIP_S))
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    grafo: list[str] = []
    for k in range(n):
        clipe = clips[k % len(clips)]
        # Clipe reusado mostra outro trecho dele, nao os mesmos 5 segundos.
        volta = k // len(clips)
        dur = min(CLIP_S, duracao - k * CLIP_S)
        cmd += ["-ss", f"{0.3 + volta * CLIP_S:.2f}", "-t", f"{dur + 0.5:.2f}",
                "-i", str(clipe)]
        grafo.append(
            f"[{k}:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,fps={FPS},setsar=1,format=yuv420p,"
            f"tpad=stop_mode=clone:stop_duration={CLIP_S},"
            f"trim=duration={dur:.3f},setpts=PTS-STARTPTS[s{k}]")
    cmd += ["-i", str(audio)]
    fatias = "".join(f"[s{k}]" for k in range(n))
    grafo.append(f"{fatias}concat=n={n}:v=1:a=0[vc]")
    fontes = f":fontsdir='{fonts_dir}'" if fonts_dir else ""
    grafo.append(f"[vc]subtitles=filename='{legenda}'{fontes}[vout]")
    cmd += ["-filter_complex", ";".join(grafo),
            "-map", "[vout]", "-map", f"{n}:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-af", "apad", "-t", f"{duracao:.2f}", "-movflags", "+faststart", str(saida)]
    return cmd


def _duracao(audio: Path) -> float:
    medida = _probe(audio)
    if not medida["duration_s"]:
        raise RendererError(f"narracao sem duracao medivel: {audio}")
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
