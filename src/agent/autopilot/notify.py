"""Avisos del piloto: qué se publicó, qué falta hacer en la app y qué falló.

Dos canales, ambos opcionales y ninguno bloqueante (un aviso que falla no puede
derrubar um post que deu certo):

- `notify-send`: notificación en el área de trabajo cuando hay sesión gráfica;
- ntfy (https://ntfy.sh): push gratuito no celular, sem conta -- so com
  `AGENT_NTFY_TOPIC` no `.env`. O topico funciona como senha: quem souber o
  nombre puede leer los mensajes, así que usa uno largo y aleatorio. No se incluye ningún secreto
  nenhum na mensagem, so tema, formato e legenda.

Siempre queda también `aviso.txt` dentro del paquete: el registro que no
depende de nenhum dos dois.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import httpx


def notify(title: str, body: str, *, ntfy_topic: str = "", ntfy_server: str = "https://ntfy.sh",
           package_dir: Path | None = None, urgent: bool = False) -> list[str]:
    """Envía por los canales disponibles; devuelve los que funcionaron."""
    enviados: list[str] = []
    if package_dir is not None:
        try:
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "aviso.txt").write_text(f"{title}\n\n{body}\n", encoding="utf-8")
            enviados.append("aviso.txt")
        except OSError:
            pass
    if shutil.which("notify-send"):
        try:
            subprocess.run(["notify-send", "-a", "Circuito Cero",
                            "-u", "critical" if urgent else "normal", title, body[:400]],
                           check=True, capture_output=True, timeout=10)
            enviados.append("desktop")
        except (subprocess.SubprocessError, OSError):
            pass
    if ntfy_topic:
        try:
            # Publicación en JSON: un título con acento en una cabecera HTTP no es
            # seguro, y el JSON de ntfy acepta UTF-8 completo.
            r = httpx.post(ntfy_server.rstrip("/"), json={
                "topic": ntfy_topic, "title": title, "message": body[:3500],
                "priority": 4 if urgent else 3,
                "tags": ["warning"] if urgent else ["clapper"],
            }, timeout=15)
            if r.status_code < 300:
                enviados.append("ntfy")
        except httpx.HTTPError:
            pass
    return enviados


__all__ = ["notify"]
