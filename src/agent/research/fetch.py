"""Busca a pagina de uma fonte e devolve o texto que o modelo vai ler.

Separado do extrator porque um tem rede e o outro nao: assim a limpeza de HTML
tem teste puro, e o que sobra aqui e so politica de rede -- limite de bytes,
tipo de conteudo, redirecionamento.

Duas decisoes que parecem detalhe e nao sao:

- **a URL gravada e a final, depois dos redirecionamentos.** O `Fact` precisa
  apontar para o que foi lido de fato. Link de agregador que redireciona para
  outro veiculo, gravado como se fosse a fonte, e citacao que nao confere.
- **o limite de bytes corta o download, nao o texto depois.** Pagina de noticia
  com video embutido passa de dezenas de megabytes, e a janela de um tema e de
  horas.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from agent.research.extract import extract

# Compromisso conhecido: veiculo de noticia costuma recusar cliente sem cara de
# navegador, e sem o prefixo Mozilla a coleta perde metade das fontes. O sufixo
# identifica o agente para quem le log -- e o maximo de honestidade que da para
# ter sem inviabilizar a leitura.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; tiktok-viral-generator/0.1; pesquisador)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}

TIPOS_ACEITOS = ("text/html", "application/xhtml", "text/plain")


class PageUnavailable(RuntimeError):
    """A pagina nao pode ser lida. Esperado: 403, paywall, PDF, timeout."""


@dataclass
class Page:
    """O que foi lido de uma fonte, com a URL que realmente foi lida."""

    url: str
    title: str
    text: str
    source_name: str

    @property
    def usable(self) -> bool:
        # Abaixo disso sobrou menu e cookie banner. Mandar isso para o modelo
        # gasta cota e devolve fato inventado a partir de quase nada.
        return len(self.text) >= 400


class PageFetcher:
    def __init__(
        self,
        client: httpx.Client | None = None,
        timeout_s: float = 20.0,
        max_bytes: int = 1_500_000,
        max_chars: int = 8000,
    ):
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_s), headers=HEADERS, follow_redirects=True
        )
        self._max_bytes = max_bytes
        self._max_chars = max_chars

    def fetch(self, url: str, source_name: str = "") -> Page:
        try:
            with self._client.stream("GET", url, headers=HEADERS) as r:
                if r.status_code != 200:
                    raise PageUnavailable(f"HTTP {r.status_code}")
                tipo = (r.headers.get("content-type") or "").lower()
                if tipo and not tipo.startswith(TIPOS_ACEITOS):
                    raise PageUnavailable(f"conteudo nao textual ({tipo.split(';')[0]})")

                bruto = bytearray()
                for pedaco in r.iter_bytes():
                    bruto += pedaco
                    if len(bruto) >= self._max_bytes:
                        # Para de baixar no teto, e ainda corta: o tamanho do
                        # pedaco e do servidor, e um unico chunk grande faria o
                        # teto valer so no papel.
                        del bruto[self._max_bytes:]
                        break
                final = str(r.url)
                codificacao = r.encoding or "utf-8"
        except httpx.HTTPError as exc:
            raise PageUnavailable(f"{type(exc).__name__}: {exc}") from exc

        html = bytes(bruto).decode(codificacao, errors="replace")
        titulo, texto = extract(html, max_chars=self._max_chars)
        return Page(
            url=final,
            title=titulo,
            text=texto,
            source_name=source_name or _dominio(final),
        )


def _dominio(url: str) -> str:
    """Nome de exibicao quando a fonte nao veio nomeada: o dominio, sem www."""
    resto = url.split("://", 1)[-1]
    return resto.split("/", 1)[0].removeprefix("www.")
