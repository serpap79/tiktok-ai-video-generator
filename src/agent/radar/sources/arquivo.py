"""Historia y curiosidad: "un dia como hoy" de Wikipedia y el archivo atemporal.

El radar solo veia lo que esta en alta AHORA -- y se pidio historia,
curiosidad y tutorial ademas de noticia. Dos fuentes, las dos ancladas en
pagina de Wikipedia (que el investigador lee como cualquier fuente):

1. **Un dia como hoy** (`api.wikimedia.org/feed/v1/wikipedia/es/onthisday`): los
   eventos de la fecha de hoy. El aniversario es el gancho temporal de la
   historia -- "hace 55 anos, un dia como hoy" -- y la puerta de nicho del
   curador descarta lo que no es tech/ciencia (la mayor parte de la lista es
   politica y tragedia, que la puerta de politica ya bloquea).
2. **Archivo** (`ARCHIVO` abajo): temas atemporales de historia de la
   computacion y curiosidad cientifica, con la pagina de referencia. Es la
   reserva: un dia de radar pobre todavia tiene pauta, y el cooldown del ledger
   (30 dias) impide repetir.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import httpx

from agent.models import Signal
from agent.ports.radar import SourceUnavailable

ONTHISDAY = "https://api.wikimedia.org/feed/v1/wikipedia/es/onthisday/events/{mm}/{dd}"
HEADERS = {"User-Agent": "tiktok-viral-generator/0.1"}

# (tema, pagina de referencia). Tema en castellano, escrito como pauta.
ARCHIVO: tuple[tuple[str, str], ...] = (
    ("ENIAC: el ordenador de 30 toneladas que calculaba trayectorias",
     "https://es.wikipedia.org/wiki/ENIAC"),
    ("Transistor: la invencion de 1947 que esta dentro de cada chip",
     "https://es.wikipedia.org/wiki/Transistor"),
    ("ARPANET: la red militar que se convirtio en internet",
     "https://es.wikipedia.org/wiki/ARPANET"),
    ("Deep Blue contra Kasparov: cuando la maquina vencio al campeon de ajedrez",
     "https://es.wikipedia.org/wiki/Deep_Blue_versus_Kasparov"),
    ("AlphaGo: la jugada 37 que sorprendio a los maestros del Go",
     "https://es.wikipedia.org/wiki/AlphaGo"),
    ("Test de Turing: la pregunta de 1950 que todavia divide a la IA",
     "https://es.wikipedia.org/wiki/Test_de_Turing"),
    ("Ada Lovelace: el primer algoritmo escrito antes de que existiera el ordenador",
     "https://es.wikipedia.org/wiki/Ada_Lovelace"),
    ("Ley de Moore: la prediccion de 1965 que marco el ritmo de los chips",
     "https://es.wikipedia.org/wiki/Ley_de_Moore"),
    ("World Wide Web: el proyecto del CERN que se convirtio en la web",
     "https://es.wikipedia.org/wiki/World_Wide_Web"),
    ("Linux: el nucleo que empezo como hobby de un estudiante",
     "https://es.wikipedia.org/wiki/Linux"),
    ("ELIZA: el chatbot de 1966 que ya enganaba a la gente",
     "https://es.wikipedia.org/wiki/ELIZA"),
    ("Perceptron: la primera red neuronal, de 1958",
     "https://es.wikipedia.org/wiki/Perceptr%C3%B3n"),
    ("El error del milenio: el fallo de dos digitos que costo miles de millones",
     "https://es.wikipedia.org/wiki/Problema_del_a%C3%B1o_2000"),
    ("Voyager 1: la sonda que salio del sistema solar con 69 KB de memoria",
     "https://es.wikipedia.org/wiki/Voyager_1"),
    ("Sputnik 1: el satelite que inicio la carrera espacial",
     "https://es.wikipedia.org/wiki/Sputnik_1"),
    ("Enigma: la maquina de cifrado rota por Turing",
     "https://es.wikipedia.org/wiki/M%C3%A1quina_Enigma"),
    ("Colossus: el ordenador secreto de la Segunda Guerra Mundial",
     "https://es.wikipedia.org/wiki/Colossus"),
    ("Apolo Guidance Computer: el ordenador que llevo al hombre a la Luna",
     "https://es.wikipedia.org/wiki/Ordenador_de_guidance_de_Apollo"),
    ("Unix: el sistema de 1969 detras de tu movil",
     "https://es.wikipedia.org/wiki/Unix"),
    ("Intel 4004: el primer microprocesador comercial, de 1971",
     "https://es.wikipedia.org/wiki/Intel_4004"),
    ("Telescopio James Webb: el espejo dorado a 1,5 millones de km",
     "https://es.wikipedia.org/wiki/Telescopio_espacial_James_Webb"),
    ("Computacion cuantica: el qubit explicado sin misterio",
     "https://es.wikipedia.org/wiki/Qubit"),
    ("Blockchain: el registro que nadie consigue borrar",
     "https://es.wikipedia.org/wiki/Cadena_de_bloques"),
    ("El algoritmo de busqueda de Google: el PageRank de dos estudiantes",
     "https://es.wikipedia.org/wiki/PageRank"),
    ("Redes neuronales convolucionales: como la IA aprendio a ver",
     "https://es.wikipedia.org/wiki/Red_neuronal_convolucional"),
    ("Transformer: el articulo de 2017 detras del ChatGPT",
     "https://es.wikipedia.org/wiki/Transformer"),
    ("Hubble: el telescopio que nacio miope y fue reparado en el espacio",
     "https://es.wikipedia.org/wiki/Telescopio_espacial_Hubble"),
    ("Criptografia RSA: la matematica que protege tu tarjeta",
     "https://es.wikipedia.org/wiki/RSA"),
)

# Cuantos items del archivo entran por recolecta. Pocos: son reserva, y el
# ledger ya impide repetir durante 30 dias.
POR_COLETA = 3


class WikipediaOnThisDay:
    name = "wikipedia_onthisday"

    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=httpx.Timeout(20.0), headers=HEADERS)

    def collect(self) -> list[Signal]:
        ahora = datetime.now(UTC)
        url = ONTHISDAY.format(mm=f"{ahora:%m}", dd=f"{ahora:%d}")
        try:
            r = self._client.get(url, headers=HEADERS)
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"wikipedia onthisday inaccesible: {exc}") from exc
        if r.status_code != 200:
            raise SourceUnavailable(f"wikipedia onthisday devolvio {r.status_code}")
        try:
            eventos = (r.json() or {}).get("events") or []
        except ValueError as exc:
            raise SourceUnavailable("wikipedia onthisday devolvio respuesta no-JSON") from exc
        return self.parse(eventos, now=ahora)

    @staticmethod
    def parse(eventos: list[dict], now: datetime) -> list[Signal]:
        senales: list[Signal] = []
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
                senales.append(Signal(
                    term=f"Hace {anos} anos: {texto}"[:280],
                    source=WikipediaOnThisDay.name,
                    volume=float(anos), unit="anos",
                    velocity=None, seen_at=now, url=url,
                ))
            except ValueError:
                continue
        return senales


class Archivo:
    """Temas atemporales, una porcion determinista por dia."""

    name = "archivo"

    def __init__(self, per_collection: int = POR_COLETA,
                 items: tuple[tuple[str, str], ...] = ARCHIVO):
        self._n = per_collection
        self._items = items

    def collect(self) -> list[Signal]:
        ahora = datetime.now(UTC)
        return self.pick(ahora)

    def pick(self, now: datetime) -> list[Signal]:
        """La misma fecha siempre elige los mismos items (reproducible), y
        fechas distintas rotan el archivo entero."""
        semilla = int(hashlib.sha256(f"{now:%Y-%m-%d}".encode()).hexdigest(), 16)
        inicio = semilla % len(self._items)
        elegidos = [self._items[(inicio + i * 7) % len(self._items)] for i in range(self._n)]
        return [Signal(term=tema, source=self.name, volume=1.0, unit="archivo",
                       velocity=None, seen_at=now, url=url)
                for tema, url in dict(elegidos).items()]


__all__ = ["ARCHIVO", "Archivo", "WikipediaOnThisDay"]
