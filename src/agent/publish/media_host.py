"""Hospedagem das imagens do carrossel num GitHub Pages proprio, a $0.

A API de foto do TikTok nao recebe arquivo: ela BAIXA as imagens de URLs
publicas, e so de um dominio ou prefixo de URL verificado no portal do app
(PULL_FROM_URL). Um repositorio com GitHub Pages resolve os dois lados de
graca: as imagens ficam publicas (o post vai ser publico de todo jeito) e o
prefixo `https://<usuario>.github.io/<repo>/` e verificavel no portal.

Desligado por padrao. Liga com duas variaveis no `.env`:

    AGENT_MEDIA_REPO_DIR=/caminho/do/clone/do/repo-do-pages
    AGENT_MEDIA_BASE_URL=https://usuario.github.io/repo

O clone precisa conseguir `git push` sozinho (credencial ja configurada).
Depois do push, espera o Pages servir cada URL (HTTP 200) antes de chamar o
TikTok: URL que ainda devolve 404 faz o download do lado deles falhar.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import httpx
from PIL import Image


class MediaHostError(RuntimeError):
    """Nao foi possivel publicar ou servir as imagens."""


def to_jpeg(pngs: list[Path], destino: Path) -> list[Path]:
    """PNG -> JPEG (a API de foto aceita JPEG/WebP, nao PNG)."""
    destino.mkdir(parents=True, exist_ok=True)
    saida = []
    for png in pngs:
        jpg = destino / (png.stem + ".jpg")
        with Image.open(png) as img:
            img.convert("RGB").save(jpg, "JPEG", quality=92, optimize=True)
        saida.append(jpg)
    return saida


class GitPagesHost:
    def __init__(self, repo_dir: Path, base_url: str, *, timeout_s: float = 300,
                 runner=subprocess.run, client: httpx.Client | None = None,
                 sleeper=time.sleep):
        self.repo_dir = Path(repo_dir)
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._run = runner
        self._client = client or httpx.Client(timeout=httpx.Timeout(15.0),
                                              follow_redirects=True)
        self._sleep = sleeper

    def publish(self, files: list[Path], subdir: str) -> list[str]:
        """Copia, faz commit e push, e devolve as URLs ja servidas."""
        if not (self.repo_dir / ".git").exists():
            raise MediaHostError(f"{self.repo_dir} nao e um clone git")
        pasta = self.repo_dir / subdir
        pasta.mkdir(parents=True, exist_ok=True)
        nomes = []
        for f in files:
            shutil.copy2(f, pasta / f.name)
            nomes.append(f"{subdir}/{f.name}")
        for cmd in (["git", "add", "--", subdir],
                    ["git", "commit", "-m", f"carrossel {subdir}"],
                    ["git", "push", "--quiet"]):
            r = self._run(cmd, cwd=self.repo_dir, capture_output=True, text=True, timeout=120)
            if r.returncode != 0 and not (cmd[1] == "commit" and "nothing to commit" in r.stdout):
                raise MediaHostError(f"{' '.join(cmd[:2])} falhou: {(r.stderr or r.stdout)[-300:]}")
        urls = [f"{self.base_url}/{n}" for n in nomes]
        self._esperar(urls)
        return urls

    def _esperar(self, urls: list[str]) -> None:
        limite = time.monotonic() + self._timeout
        pendentes = list(urls)
        while pendentes:
            pendentes = [u for u in pendentes if not self._ok(u)]
            if not pendentes:
                return
            if time.monotonic() > limite:
                raise MediaHostError(f"Pages nao serviu {pendentes[0]} em {self._timeout:.0f}s")
            self._sleep(10)

    def _ok(self, url: str) -> bool:
        try:
            return self._client.head(url).status_code == 200
        except httpx.HTTPError:
            return False


__all__ = ["GitPagesHost", "MediaHostError", "to_jpeg"]
