"""HTML -> texto legivel, com a biblioteca padrao e mais nada.

Ha bibliotecas melhores nisso (trafilatura, readability). Ficaram de fora porque
Lo que el investigador necesita es modesto: texto continuo suficiente para que un modelo
extraiga afirmaciones con cifras de páginas de noticias y publicaciones. No necesita
reconstruir el artículo principal ni gestionar los paywalls, y el coste de
errar e baixo -- o portao de ancoragem numerica derruba o fato depois.

Lo que **no** es modesto es la cuenta de tokens: una página de noticias llega con menús,
rodape e newsletter, e no free tier isso e cota gasta em navegacao. Por isso o
parser descarta blocos de estrutura em vez de limpar depois.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Etiquetas cuyo contenido nunca es el asunto de la página.
IGNORADAS = frozenset({
    "script", "style", "noscript", "template", "svg", "canvas", "iframe",
    "nav", "header", "footer", "aside", "form", "button", "select", "label",
})

# Tags que separam frases. Sem isso "...do modelo.Leia mais" vira uma palavra so
# e o modelo passa a citar um numero colado no menu.
BLOCOS = frozenset({
    "p", "br", "div", "section", "article", "li", "tr", "td", "th", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6", "pre", "figcaption", "dd", "dt",
})

_ESPACOS = re.compile(r"[ \t\r\f\v]+")
_LINHAS = re.compile(r"\n{2,}")


class _Coletor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pedacos: list[str] = []
        self.titulo = ""
        self._ignorando = 0
        self._em_titulo = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in IGNORADAS:
            self._ignorando += 1
        elif tag == "title":
            self._em_titulo = True
        elif tag == "meta":
            self._titulo_de_meta(dict(attrs))
        if tag in BLOCOS:
            self.pedacos.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in IGNORADAS and self._ignorando:
            self._ignorando -= 1
        elif tag == "title":
            self._em_titulo = False
        if tag in BLOCOS:
            self.pedacos.append("\n")

    def handle_data(self, data: str) -> None:
        if self._em_titulo and not self.titulo:
            self.titulo = " ".join(data.split())
            return
        if self._ignorando:
            return
        if data.strip():
            self.pedacos.append(data)

    def _titulo_de_meta(self, attrs: dict[str, str | None]) -> None:
        # og:title costuma ser a manchete limpa, enquanto <title> vem com o nome
        # do veiculo grudado ("... | Veiculo"). Quando os dois existem, o og
        # gana, porque es lo que se convertirá en `Fact.claim` de contexto.
        propriedade = (attrs.get("property") or attrs.get("name") or "").lower()
        contenido = attrs.get("content") or ""
        if propriedade in {"og:title", "twitter:title"} and contenido.strip():
            self.titulo = " ".join(contenido.split())


def extract(html: str, max_chars: int = 8000) -> tuple[str, str]:
    """Devuelve (título, texto). Texto cortado en `max_chars` sin romper palabras.

    El HTML mal formado no genera errores: HTMLParser de la stdlib es tolerante por defecto y
    una página parcialmente rota con las cifras correctas aún sirve. Una respuesta vacía es válida
    y quien llama decide qué hacer con una página sin texto.
    """
    coletor = _Coletor()
    coletor.feed(html)
    coletor.close()

    bruto = "".join(coletor.pedacos)
    linhas = [_ESPACOS.sub(" ", linha).strip() for linha in bruto.split("\n")]
    texto = _LINHAS.sub("\n\n", "\n".join(linha for linha in linhas if linha))
    return coletor.titulo, _cortar(texto.strip(), max_chars)


def _cortar(texto: str, limite: int) -> str:
    if len(texto) <= limite:
        return texto
    corte = texto.rfind(" ", 0, limite)
    return texto[: corte if corte > limite // 2 else limite].rstrip()
