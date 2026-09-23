"""Pos-producao do video: marca, gancho na tela, apresentador, trilha e loudness.

O renderizador entrega narracao, legenda e clipes. O que faltava para o video
parecer do canal, e segurar os primeiros segundos, e nosso:

- **cartao do gancho**: a frase do hook escrita na tela, com a tag do tipo de
  conteudo. Quem rola o feed sem som (maioria) le a lacuna antes de decidir;
  o audio sozinho nao faz isso. Com apresentador ele sai exatamente quando a
  legenda karaoke entra, para o gancho nunca aparecer escrito duas vezes;
- **apresentador**: a camada RGBA que `render/presenter.py` sintetizou
  (silhueta inteira, boca no tempo da voz, piscada, encenacao pelo roteiro)
  entra aqui como mais um video com alfa -- este arquivo so sobrepoe;
- **marca d'agua** discreta com o @ do canal, para o video que circula fora;
- **trilha gerada** (`render/music.py`) com *ducking*: a musica abaixa
  sozinha quando a voz entra (sidechain), em vez de um volume fixo que
  briga com a fala ou some no silencio;
- **loudness em -14 LUFS** (EBU R128): volume consistente entre videos, e
  sem clipar;
- **grade da marca**: leve escurecimento e vinheta, para clipe claro nao
  quebrar o "dark" do canal.

Tipografia sai do Pillow com as fontes da marca (mesmo caminho do
carrossel) e entra no ffmpeg como PNG: o `drawtext` dependeria de fontconfig
e de escapar texto com acento na linha de comando.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from agent.config import PROJECT_ROOT
from agent.render.presenter import Camada
from agent.render.typography import DISPLAY, MONO, font, hex_rgb, wrap

W, H = 1080, 1920
# Quanto tempo o cartao do gancho fica na tela (0,4s de fade no fim) quando nao
# ha apresentador. Com apresentador quem manda e a encenacao: `Camada.card_s`.
HOOK_S = 3.2
# Topo do painel do gancho. Com apresentador ele sobe: na chamada o cabelo do
# apresentador comeca em ~815, e um painel de 4 linhas desce ate ~754.
TOPO_CARTAO = 300
TOPO_CARTAO_COM_APRESENTADOR = 240
# Trilha relativa a voz antes do ducking: ~ -20 dB.
MUSIC_GAIN = 0.10
LOUDNESS_LUFS = -14


@dataclass(frozen=True)
class PostSpec:
    hook: str
    tag: str
    accent: str
    handle: str
    presenter: Camada | None = None

    @property
    def card_s(self) -> float:
        """O cartao some quando a legenda entra -- ou em HOOK_S, sem apresentador."""
        return self.presenter.card_s if self.presenter is not None else HOOK_S


def ffmpeg_bin() -> str:
    """O ffmpeg estatico com libx264 (o do Fedora nao tem; ver setup_renderer.sh)."""
    candidato = os.environ.get("IMAGEIO_FFMPEG_EXE", "")
    if not candidato:
        env = PROJECT_ROOT / ".env"
        if env.exists():
            for linha in env.read_text(encoding="utf-8").splitlines():
                if linha.startswith("IMAGEIO_FFMPEG_EXE="):
                    candidato = linha.split("=", 1)[1].strip().strip('"')
                    break
    return candidato if candidato and Path(candidato).exists() else "ffmpeg"


def hook_card(spec: PostSpec, out: Path) -> Path:
    """PNG 1080x1920 transparente: painel escuro com tag + gancho, terco de cima.

    Com apresentador o painel sobe -- na chamada ele ocupa a metade de baixo do
    quadro. A plaquinha com o nome saiu daqui: ela e do apresentador, anda com
    ele, e quem desenha e `render/presenter.py`.
    """
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    acento = hex_rgb(spec.accent)
    f_tag = font(MONO, 40)
    largura = W - 2 * 90
    for tamanho in (76, 68, 60, 54):
        f_hook = font(DISPLAY, tamanho, spec.hook)
        linhas = wrap(draw, spec.hook, f_hook, largura - 60)
        if len(linhas) <= 4:
            break
    passo = int(tamanho * 1.2)
    topo = TOPO_CARTAO_COM_APRESENTADOR if spec.presenter is not None else TOPO_CARTAO
    altura = 70 + 30 + len(linhas) * passo + 50
    draw.rounded_rectangle((60, topo, W - 60, topo + altura), radius=28,
                           fill=(10, 10, 12, 205))
    draw.rectangle((60, topo + 24, 68, topo + altura - 24), fill=(*acento, 255))
    y = topo + 44
    if spec.tag:
        draw.text((100, y), spec.tag, font=f_tag, fill=(*acento, 255))
        y += 70
    for linha in linhas:
        draw.text((100, y), linha, font=f_hook, fill=(233, 238, 241, 255))
        y += passo
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def watermark(spec: PostSpec, out: Path) -> Path:
    """PNG transparente com o @ do canal, pequeno, no canto de cima."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((56, 200), spec.handle, font=font(MONO, 30), fill=(233, 238, 241, 150))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def build_command(video_in: Path, out: Path, *, hook_png: Path, mark_png: Path,
                  music_wav: Path | None, duration_s: float,
                  presenter: Camada | None = None, card_s: float = HOOK_S,
                  ffmpeg: str = "ffmpeg") -> list[str]:
    """Linha de comando do ffmpeg. Separada para o teste conferir o grafo sem rodar."""
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
        camada = entradas
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
        # A camada ja vem posicionada e com alfa: aqui e so onde ela pousa no
        # quadro. Nada de crop -- foi exatamente o crop que recortava a
        # silhueta em retangulo no desenho anterior.
        # A camada vem em duas trilhas: cor pre-multiplicada + mascara em cinza
        # (ver `render/presenter.encode_command`). Aqui elas voltam a ser um
        # RGBA com alfa direto.
        grafo.append(f"[{camada}:v:0][{camada}:v:1]alphamerge,"
                     f"unpremultiply=inplace=1,setpts=PTS-STARTPTS[apr]")
        grafo.append(f"[v2][apr]overlay={presenter.x}:{presenter.y}:"
                     f"eof_action=pass:format=auto[v3]")
        ultimo = "v3"
    if music_wav is not None:
        grafo += [
            "[0:a]asplit=2[voz][chave]",
            f"[{musica}:a]volume={MUSIC_GAIN}[mus]",
            "[mus][chave]sidechaincompress=threshold=0.03:ratio=6:attack=15:release=400[fundo]",
            "[voz][fundo]amix=inputs=2:duration=first:normalize=0[mix]",
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
    """Aplica a pos-producao e devolve o MP4 final. Levanta RuntimeError com o stderr."""
    work_dir.mkdir(parents=True, exist_ok=True)
    gancho = hook_card(spec, work_dir / "hook.png")
    marca = watermark(spec, work_dir / "marca.png")
    apresentador = (spec.presenter
                    if spec.presenter is not None and spec.presenter.path.exists()
                    else None)
    cmd = build_command(video_in, out, hook_png=gancho, mark_png=marca,
                        music_wav=music_wav, duration_s=duration_s,
                        presenter=apresentador, card_s=spec.card_s,
                        ffmpeg=ffmpeg_bin())
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg falhou na pos-producao: {exc.stderr[-600:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg passou de {timeout_s:.0f}s na pos-producao") from exc
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("pos-producao nao gerou arquivo")
    return out


__all__ = ["HOOK_S", "LOUDNESS_LUFS", "TOPO_CARTAO", "TOPO_CARTAO_COM_APRESENTADOR",
           "PostSpec", "build_command", "ffmpeg_bin", "hook_card", "postprocess",
           "watermark"]
