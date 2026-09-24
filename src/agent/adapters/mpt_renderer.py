"""Adaptador da porta Renderer para o MoneyPrinterTurbo rodando local.

O MPT (MIT, github.com/harry0703/MoneyPrinterTurbo) sobe um FastAPI em
/api/v1. Pasamos el guion y los términos de búsqueda listos, lo que evita el LLM de MPT
por completo: `task.py:generate_script` so chama o LLM quando `video_script`
chega vazio. Nosso agente e dono do julgamento; o MPT e a grafica.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from agent.config import Settings
from agent.config import settings as default_settings
from agent.models import RenderResult, RenderState, Script
from agent.ports.renderer import RendererError

# app/models/const.py do MPT
MPT_STATE_FAILED = -1
MPT_STATE_COMPLETE = 1
MPT_STATE_PROCESSING = 4

API = "/api/v1"


class MptRenderer:
    """Implementa a porta Renderer via HTTP contra o MPT local."""

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or default_settings
        self._client = client or httpx.Client(
            base_url=self.settings.renderer_url.rstrip("/"),
            timeout=httpx.Timeout(30.0, read=60.0),
            headers=self._auth_headers(),
        )

    def _auth_headers(self) -> dict[str, str]:
        # O MPT so exige x-api-key quando config.app.api_key esta preenchida.
        # Mandar sempre quando temos a chave mantem o adaptador valido nos dois modos.
        key = self.settings.renderer_api_key
        return {"x-api-key": key} if key else {}

    # ---------------------------------------------------------------- health

    def health(self) -> bool:
        try:
            r = self._client.get("/ping", timeout=5.0)
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    # ---------------------------------------------------------------- render

    def build_payload(self, script: Script, materials: list[str] | None = None,
                      voice: str | None = None) -> dict[str, Any]:
        """Monta o VideoParams do MPT. Separado de `render` para ser testavel.

        `materials`: nombres de archivos ya enviados al renderizador, en el orden
        de la narración (elegidos por `render/footage.py`). Con ellos, MPT no
        busca nada: so corta 5s de cada um, em sequencia.
        """
        payload: dict[str, Any] = {
            # Guion y términos listos: no se activa el LLM de MPT.
            "video_subject": script.topic,
            "video_script": script.narration,
            "video_terms": script.search_terms,
            # 9:16 => 1080x1920 (VideoAspect.to_resolution no MPT)
            "video_aspect": "9:16",
            "video_source": self.settings.video_source,
            "video_count": 1,
            "video_clip_duration": 5,
            # Casa cada termo com o trecho correspondente da narracao, em vez de
            # sortear material. Os termos vao em ordem cronologica por contrato.
            "match_materials_to_script": True,
            "voice_name": voice or self.settings.voice_name,
            "voice_rate": 1.0,
            "voice_volume": 1.0,
            # Subtítulo karaoke: los tiempos por palabra proceden del SubMaker de edge-tts,
            # así que no hace falta ejecutar Whisper (large-v3 por defecto, lento en CPU).
            "subtitle_enabled": True,
            "subtitle_display_mode": "word_by_word",
            "subtitle_animation": "pop_spring",
            "subtitle_position": self.settings.subtitle_position,
            "custom_position": self.settings.subtitle_custom_position,
            "font_name": self.settings.font_name,
            "font_size": self.settings.font_size,
            "text_fore_color": "#FFFFFF",
            "stroke_color": "#000000",
            "stroke_width": self.settings.subtitle_stroke_width,
            # Pista desactivada en MPT. Las músicas de `resource/songs` proceden de
            # videos do YouTube ("If there are copyright issues, please delete
            # ellos», en su README): un canal monetizado no usa audio sin
            # licenca. A trilha entra depois, gerada localmente
            # (`render/music.py`), en posproducción.
            "bgm_type": "",
            "bgm_volume": 0.0,
        }

        if materials:
            payload["video_source"] = "local"
            payload["video_materials"] = [
                {"provider": "local", "url": nome, "duration": 0} for nome in materials
            ]
            payload["video_concat_mode"] = "sequential"
            return payload

        if self.settings.video_source == "local":
            # get_video_materials de MPT solo consulta video_materials en este modo;
            # os search_terms passam a ser ignorados. O `url` e o nome devolvido
            # por la subida; no es un camino externo: MPT lo resuelve dentro de
            # storage/local_videos e rejeita qualquer path que escape dali.
            payload["video_materials"] = [
                {"provider": "local", "url": nome, "duration": 0}
                for nome in self._upload_local_materials()
            ]

        return payload

    def _upload_local_materials(self, paths: list[Path] | None = None) -> list[str]:
        """Sube los archivos locales y devuelve los nombres almacenados por el renderizador.

        Usa HTTP en lugar de copiar a su disco de forma intencionada: así
        el puerto sigue siendo válido si el renderizador sale de esta máquina.
        """
        nomes: list[str] = []
        for caminho in (paths if paths is not None else self.settings.local_materials):
            path = Path(caminho)
            if not path.is_file():
                raise RendererError(f"material local inexistente: {path}")
            with path.open("rb") as fh:
                try:
                    r = self._client.post(
                        f"{API}/video_materials",
                        files={"file": (path.name, fh, "video/mp4")},
                        timeout=httpx.Timeout(30.0, write=300.0),
                    )
                except httpx.HTTPError as exc:
                    raise RendererError(f"falha subindo {path.name}: {exc}") from exc
            if r.status_code >= 400:
                raise RendererError(
                    f"el renderizador rechazó {path.name} ({r.status_code}): {r.text[:200]}"
                )
            nome = _envelope(r).get("file")
            if not nome:
                raise RendererError(
                    f"la subida de {path.name} no devolvió ningún nombre de archivo")
            nomes.append(str(nome))
        return nomes

    def render(self, script: Script, materials: list[Path] | None = None,
               voice: str | None = None) -> RenderResult:
        nomes = self._upload_local_materials(materials) if materials else None
        task_id = self._create_task(script, nomes, voice)
        task = self._await_task(task_id)

        state = task.get("state")
        if state == MPT_STATE_FAILED:
            return RenderResult(
                state=RenderState.failed,
                task_id=task_id,
                error=self._failure_reason(task),
            )

        uri = self._pick_video_uri(task)
        if not uri:
            return RenderResult(
                state=RenderState.failed,
                task_id=task_id,
                error=f"task concluida sin video: chaves={sorted(task)}",
            )

        local_path = self._download(uri, task_id)
        probed = probe_video(local_path)
        return RenderResult(
            state=RenderState.complete,
            task_id=task_id,
            video_path=str(local_path),
            duration_s=probed["duration_s"],
            width=probed["width"],
            height=probed["height"],
            has_audio=probed["has_audio"],
        )

    # ------------------------------------------------------------- internos

    def _create_task(self, script: Script, materials: list[str] | None = None,
                     voice: str | None = None) -> str:
        try:
            r = self._client.post(f"{API}/videos",
                                  json=self.build_payload(script, materials, voice))
        except httpx.HTTPError as exc:
            raise RendererError(
                f"renderizador inaccesible en {self._client.base_url}: {exc}"
            ) from exc

        if r.status_code == 429:
            raise RendererError("la cola del renderizador está llena (429)")
        if r.status_code == 401:
            raise RendererError("el renderizador rechazó la x-api-key (401)")
        if r.status_code >= 400:
            raise RendererError(f"el renderizador devolvió {r.status_code}: {r.text[:300]}")

        data = _envelope(r)
        task_id = data.get("task_id")
        if not task_id:
            raise RendererError(f"resposta sem task_id: {data}")
        return str(task_id)

    def _await_task(self, task_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.settings.renderer_timeout_s
        last_progress = -1
        while True:
            try:
                r = self._client.get(f"{API}/tasks/{task_id}")
            except httpx.HTTPError as exc:
                raise RendererError(f"se perdió la conexión con el renderizador: {exc}") from exc

            if r.status_code == 404:
                raise RendererError(f"la tarea {task_id} desapareció del renderizador")
            if r.status_code >= 400:
                raise RendererError(
                    f"la consulta de la tarea devolvió {r.status_code}: {r.text[:300]}")

            task = _envelope(r)
            state = task.get("state")
            progress = int(task.get("progress") or 0)
            if progress != last_progress:
                last_progress = progress
            if state in (MPT_STATE_COMPLETE, MPT_STATE_FAILED):
                return task

            if time.monotonic() >= deadline:
                raise RendererError(
                    f"timeout de {self.settings.renderer_timeout_s:.0f}s renderizando "
                    f"{task_id} (parou em {progress}%)"
                )
            time.sleep(self.settings.renderer_poll_interval_s)

    @staticmethod
    def _pick_video_uri(task: dict[str, Any]) -> str | None:
        # El orden importa y no es intuitivo. En MPT:
        #   combined_videos -> combined-N.mp4, o concat SO DE VIDEO (intermediario)
        #   videos          -> final-N.mp4, o corte com narracao e legenda
        # Preferir "combined" pelo nome entrega um MP4 mudo que passa em qualquer
        # checagem de dimensao e duracao. Verificado numa task real: combined-1.mp4
        # tinha apenas stream h264; final-1.mp4 tinha h264 + aac.
        for key in ("videos", "combined_videos"):
            items = task.get(key)
            if isinstance(items, list) and items:
                first = items[0]
                if isinstance(first, str) and first:
                    return first
        return None

    @staticmethod
    def _failure_reason(task: dict[str, Any]) -> str:
        for key in ("error", "message", "failure_reason", "failed_stage"):
            value = task.get(key)
            if value:
                return str(value)
        return "el renderizador informó de un fallo sin detalles"

    def _download(self, uri: str, task_id: str) -> Path:
        """Baixa o MP4 via HTTP.

        Sem `endpoint` configurado, o MPT devolve caminho relativo `/tasks/<...>`.
        Convertemos para o endpoint de download, que resolve a partir de
        storage/tasks/. Usamos HTTP en vez de leer el archivo directamente para que
        el puerto siga siendo válido si el renderizador sale de esta máquina.
        """
        self.settings.ensure_dirs()
        dest = self.settings.output_dir / f"{task_id}.mp4"

        if uri.startswith(("http://", "https://")):
            url = uri
        else:
            relative = uri.lstrip("/").removeprefix("tasks/")
            url = f"{API}/download/{relative}"

        try:
            with self._client.stream("GET", url, timeout=httpx.Timeout(30.0, read=300.0)) as r:
                if r.status_code >= 400:
                    raise RendererError(f"la descarga de {url} devolvió {r.status_code}")
                with dest.open("wb") as fh:
                    for chunk in r.iter_bytes(chunk_size=1 << 16):
                        fh.write(chunk)
        except httpx.HTTPError as exc:
            raise RendererError(f"falha baixando {url}: {exc}") from exc

        if dest.stat().st_size == 0:
            raise RendererError(f"download de {url} veio vazio")
        return dest


def _envelope(response: httpx.Response) -> dict[str, Any]:
    """Desembrulha `{"status": 200, "data": {...}}` do MPT."""
    try:
        body = response.json()
    except ValueError as exc:
        raise RendererError(
            f"respuesta que no es JSON del renderizador: {response.text[:200]}") from exc
    if not isinstance(body, dict):
        raise RendererError(f"envelope inesperado: {body!r}")
    data = body.get("data")
    if not isinstance(data, dict):
        raise RendererError(f"envelope sem objeto 'data': {body!r}")
    return data


def probe_video(path: Path) -> dict[str, Any]:
    """Mede duracao e dimensoes com ffprobe.

    Medimos em vez de confiar no que pedimos: `video_aspect` e um pedido, e a
    duracao depende do TTS. O aceite do M0 verifica 1080x1920 e 60-90s, e so
    vale se o numero for medido.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,width,height:format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
    except FileNotFoundError as exc:
        raise RendererError("ffprobe no está en el PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RendererError(f"ffprobe falló con {path}: {exc.stderr[:200]}") from exc

    info = json.loads(out.stdout)
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    duration = (info.get("format") or {}).get("duration")
    return {
        "width": int(video["width"]) if video.get("width") else None,
        "height": int(video["height"]) if video.get("height") else None,
        "duration_s": round(float(duration), 3) if duration else None,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }
