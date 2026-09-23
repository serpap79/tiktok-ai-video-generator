"""HTML -> texto legivel, com a biblioteca padrao e mais nada.

Ha bibliotecas melhores nisso (trafilatura, readability). Ficaram de fora porque
o que o pesquisador precisa e modesto: texto corrido suficiente para um modelo
extrair afirmacoes com numero, de paginas de noticia e de release. Nao precisa
reconstruir o artigo principal, nao precisa lidar com paywall, e o custo de
errar e baixo -- o portao de ancoragem numerica derruba o fato depois.

O que **nao** e modesto e a conta de token: pagina de noticia vem com menu,
rodape e newsletter, e no free tier isso e cota gasta em navegacao. Por isso o
parser descarta blocos de estrutura em vez de limpar depois.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Tags cujo conteudo nunca e o assunto da pagina.
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
        # ganha, porque e o que vai virar `Fact.claim` de contexto.
        propriedade = (attrs.get("property") or attrs.get("name") or "").lower()
        conteudo = attrs.get("content") or ""
        if propriedade in {"og:title", "twitter:title"} and conteudo.strip():
            self.titulo = " ".join(conteudo.split())


def extract(html: str, max_chars: int = 8000) -> tuple[str, str]:
    """Devolve (titulo, texto). Texto cortado em `max_chars` sem quebrar palavra.

    HTML malformado nao levanta: o HTMLParser da stdlib e tolerante por padrao, e
    uma pagina meio quebrada com o numero certo ainda serve. Vazio e resposta
    valida, e quem chama decide o que fazer com pagina sem texto.
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
