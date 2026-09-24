"""Actualiza claves en `.env` sin tocar el resto del archivo.

Existe por el token de TikTok: el access token dura 24 horas y el
refresh token rota en cada renovación. Un piloto automático que publica a las 9:00,
13:00 y 20:00 necesita guardar el token nuevo en algún lugar. La regla del proyecto
e que segredo so vive no `.env` (git-ignored). Entao o `.env` e reescrito,
com tres cuidados:

- solo cambian las claves solicitadas; los comentarios, el orden y las demás líneas permanecen;
- escritura atómica (archivo temporal + renombrado): una caída a mitad no deja un
  `.env` a medias, lo que invalidaría todas las claves a la vez;
- permisos 600: el archivo contiene credenciales de tres servicios.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def update_env(path: Path | str, valores: dict[str, str]) -> None:
    destino = Path(path)
    linhas = destino.read_text(encoding="utf-8").splitlines() if destino.exists() else []
    pendentes = dict(valores)
    saida: list[str] = []
    for linha in linhas:
        chave = linha.split("=", 1)[0].strip() if "=" in linha else ""
        if chave and not linha.lstrip().startswith("#") and chave in pendentes:
            saida.append(f"{chave}={pendentes.pop(chave)}")
        else:
            saida.append(linha)
    saida.extend(f"{k}={v}" for k, v in pendentes.items())

    fd, tmp = tempfile.mkstemp(dir=str(destino.parent), prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(saida) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, destino)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


__all__ = ["update_env"]
