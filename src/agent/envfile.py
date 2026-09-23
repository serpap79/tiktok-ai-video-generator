"""Atualiza chaves no `.env` sem tocar no resto do arquivo.

Existe por causa do token do TikTok: o access token vale 24 horas e o
refresh token gira a cada renovacao. Um piloto automatico que posta as 9h,
15h e 20h precisa gravar o token novo em algum lugar -- e a regra do projeto
e que segredo so vive no `.env` (git-ignored). Entao o `.env` e reescrito,
com tres cuidados:

- so as chaves pedidas mudam; comentario, ordem e as outras linhas ficam;
- escrita atomica (arquivo temporario + rename): queda no meio nao deixa um
  `.env` pela metade, que derrubaria todas as chaves de uma vez;
- permissao 600: o arquivo tem credencial de tres servicos.
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
