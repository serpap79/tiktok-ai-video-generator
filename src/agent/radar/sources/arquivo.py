"""Historia e curiosidade: "neste dia" da Wikipedia e o arquivo atemporal.

O radar so enxergava o que esta em alta AGORA -- e o autor pediu
historia, curiosidade e tutorial alem de noticia. Duas fontes, as duas
ancoradas em pagina da Wikipedia (que o pesquisador le como qualquer fonte):

1. **Neste dia** (`api.wikimedia.org/feed/v1/wikipedia/pt/onthisday`): os
   eventos da data de hoje. Aniversario e o gancho de tempo da historia --
   "ha 55 anos, neste dia" -- e o portao de nicho do curador descarta o que
   nao e tech/ciencia (a maior parte da lista e politica e tragedia, que o
   portao de politica ja barra).
2. **Arquivo** (`ARQUIVO` abaixo): temas atemporais de historia da
   computacao e curiosidade cientifica, com a pagina de referencia. E a
   reserva: dia de radar magro ainda tem pauta, e o cooldown do ledger (30
   dias) impede repetir.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import httpx

from agent.models import Signal
from agent.ports.radar import SourceUnavailable

ONTHISDAY = "https://api.wikimedia.org/feed/v1/wikipedia/pt/onthisday/events/{mm}/{dd}"
HEADERS = {"User-Agent": "tiktok-viral-generator/0.1 (github.com/guilhermehrsilva)"}

# (tema, pagina de referencia). Tema em pt-BR, escrito como pauta.
ARQUIVO: tuple[tuple[str, str], ...] = (
    ("ENIAC: o computador de 30 toneladas que calculava trajetorias",
     "https://pt.wikipedia.org/wiki/ENIAC"),
    ("Transistor: a invencao de 1947 que esta dentro de todo chip",
     "https://pt.wikipedia.org/wiki/Transistor"),
    ("ARPANET: a rede militar que virou a internet",
     "https://pt.wikipedia.org/wiki/ARPANET"),
    ("Deep Blue contra Kasparov: quando a maquina venceu o campeao de xadrez",
     "https://pt.wikipedia.org/wiki/Deep_Blue_versus_Kasparov"),
    ("AlphaGo: a jogada 37 que surpreendeu os mestres do Go",
     "https://pt.wikipedia.org/wiki/AlphaGo"),
    ("Teste de Turing: a pergunta de 1950 que ainda divide a IA",
     "https://pt.wikipedia.org/wiki/Teste_de_Turing"),
    ("Ada Lovelace: o primeiro algoritmo escrito antes do computador existir",
     "https://pt.wikipedia.org/wiki/Ada_Lovelace"),
    ("Lei de Moore: a previsao de 1965 que ditou o ritmo dos chips",
     "https://pt.wikipedia.org/wiki/Lei_de_Moore"),
    ("World Wide Web: o projeto do CERN que virou a web",
     "https://pt.wikipedia.org/wiki/World_Wide_Web"),
    ("Linux: o kernel que comecou como hobby de um estudante",
     "https://pt.wikipedia.org/wiki/Linux"),
    ("ELIZA: o chatbot de 1966 que ja enganava pessoas",
     "https://pt.wikipedia.org/wiki/ELIZA"),
    ("Perceptron: a primeira rede neural, de 1958",
     "https://pt.wikipedia.org/wiki/Perceptron"),
    ("Bug do milenio: o erro de dois digitos que custou bilhoes",
     "https://pt.wikipedia.org/wiki/Problema_do_ano_2000"),
    ("Voyager 1: a sonda que saiu do sistema solar com 69 KB de memoria",
     "https://pt.wikipedia.org/wiki/Voyager_1"),
    ("Sputnik 1: o satelite que comecou a corrida espacial",
     "https://pt.wikipedia.org/wiki/Sputnik_1"),
    ("Enigma: a maquina de criptografia quebrada por Turing",
     "https://pt.wikipedia.org/wiki/Enigma_(m%C3%A1quina)"),
    ("Colossus: o computador secreto da Segunda Guerra",
     "https://pt.wikipedia.org/wiki/Colossus"),
    ("Apollo Guidance Computer: o computador que levou o homem a Lua",
     "https://pt.wikipedia.org/wiki/Apollo_Guidance_Computer"),
    ("Unix: o sistema de 1969 por tras do seu celular",
     "https://pt.wikipedia.org/wiki/Unix"),
    ("Intel 4004: o primeiro microprocessador comercial, de 1971",
     "https://pt.wikipedia.org/wiki/Intel_4004"),
    ("Telescopio James Webb: o espelho dourado a 1,5 milhao de km",
     "https://pt.wikipedia.org/wiki/Telesc%C3%B3pio_Espacial_James_Webb"),
    ("Computacao quantica: o qubit explicado sem misterio",
     "https://pt.wikipedia.org/wiki/Qubit"),
    ("Blockchain: o registro que ninguem consegue apagar",
     "https://pt.wikipedia.org/wiki/Blockchain"),
    ("Algoritmo de busca do Google: o PageRank de dois estudantes",
     "https://pt.wikipedia.org/wiki/PageRank"),
    ("Redes neurais convolucionais: como a IA aprendeu a enxergar",
     "https://pt.wikipedia.org/wiki/Rede_neural_convolucional"),
    ("Transformer: o artigo de 2017 por tras do ChatGPT",
     "https://pt.wikipedia.org/wiki/Transformer_(aprendizado_de_m%C3%A1quina)"),
    ("Hubble: o telescopio que nasceu miope e foi consertado no espaco",
     "https://pt.wikipedia.org/wiki/Telesc%C3%B3pio_espacial_Hubble"),
    ("Criptografia RSA: a matematica que protege o seu cartao",
     "https://pt.wikipedia.org/wiki/RSA_(sistema_criptogr%C3%A1fico)"),
)

# Quantos itens do arquivo entram por coleta. Poucos: sao reserva, e o
# ledger ja impede repetir por 30 dias.
POR_COLETA = 3


class WikipediaOnThisDay:
    name = "wikipedia_onthisday"

    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=httpx.Timeout(20.0), headers=HEADERS)

    def collect(self) -> list[Signal]:
        agora = datetime.now(UTC)
        url = ONTHISDAY.format(mm=f"{agora:%m}", dd=f"{agora:%d}")
        try:
            r = self._client.get(url, headers=HEADERS)
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"wikipedia onthisday inacessivel: {exc}") from exc
        if r.status_code != 200:
            raise SourceUnavailable(f"wikipedia onthisday devolveu {r.status_code}")
        try:
            eventos = (r.json() or {}).get("events") or []
        except ValueError as exc:
            raise SourceUnavailable("wikipedia onthisday devolveu resposta nao-JSON") from exc
        return self.parse(eventos, now=agora)

    @staticmethod
    def parse(eventos: list[dict], now: datetime) -> list[Signal]:
        sinais: list[Signal] = []
        for e in eventos:
            texto = " ".join(str(e.get("text") or "").split())
            ano = e.get("year")
            paginas = e.get("pages") or []
            if not texto or not isinstance(ano, int) or not paginas:
                continue
            url = ((paginas[0].get("content_urls") or {}).get("desktop") or {}).get("page")
            if not url:
                continue
            anos = now.year - ano
            if anos <= 0:
                continue
            try:
                sinais.append(Signal(
                    term=f"Ha {anos} anos: {texto}"[:280],
                    source=WikipediaOnThisDay.name,
                    volume=float(anos), unit="anos",
                    velocity=None, seen_at=now, url=url,
                ))
            except ValueError:
                continue
        return sinais


class Arquivo:
    """Temas atemporais, uma fatia deterministica por dia."""

    name = "arquivo"

    def __init__(self, per_collection: int = POR_COLETA,
                 items: tuple[tuple[str, str], ...] = ARQUIVO):
        self._n = per_collection
        self._items = items

    def collect(self) -> list[Signal]:
        agora = datetime.now(UTC)
        return self.pick(agora)

    def pick(self, now: datetime) -> list[Signal]:
        """A mesma data sempre escolhe os mesmos itens (reproduzivel), e datas
        diferentes giram o arquivo inteiro."""
        semente = int(hashlib.sha256(f"{now:%Y-%m-%d}".encode()).hexdigest(), 16)
        inicio = semente % len(self._items)
        escolhidos = [self._items[(inicio + i * 7) % len(self._items)] for i in range(self._n)]
        return [Signal(term=tema, source=self.name, volume=1.0, unit="arquivo",
                       velocity=None, seen_at=now, url=url)
                for tema, url in dict(escolhidos).items()]


__all__ = ["ARQUIVO", "Arquivo", "WikipediaOnThisDay"]
