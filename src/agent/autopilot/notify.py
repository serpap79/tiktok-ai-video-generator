"""Avisos do piloto: o que saiu, o que falta fazer no app, o que falhou.

Dois canais, os dois opcionais e nenhum bloqueante (aviso que falha nao pode
derrubar um post que deu certo):

- `notify-send`: notificacao na area de trabalho, quando ha sessao grafica;
- ntfy (https://ntfy.sh): push gratuito no celular, sem conta -- so com
  `AGENT_NTFY_TOPIC` no `.env`. O topico funciona como senha: quem souber o
  nome le as mensagens, entao use um nome longo e aleatorio. Nao vai segredo
  nenhum na mensagem, so tema, formato e legenda.

Sempre fica tambem o `aviso.txt` dentro do pacote: o registro que nao
depende de nenhum dos dois.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import httpx


def notify(title: str, body: str, *, ntfy_topic: str = "", ntfy_server: str = "https://ntfy.sh",
           package_dir: Path | None = None, urgent: bool = False) -> list[str]:
    """Envia pelos canais disponiveis; devolve quais funcionaram."""
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
            subprocess.run(["notify-send", "-a", "Seu Canal",
                            "-u", "critical" if urgent else "normal", title, body[:400]],
                           check=True, capture_output=True, timeout=10)
            enviados.append("desktop")
        except (subprocess.SubprocessError, OSError):
            pass
    if ntfy_topic:
        try:
            # Publicacao em JSON: titulo com acento em cabecalho HTTP nao e
            # seguro, e o JSON do ntfy aceita UTF-8 inteiro.
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
