"""Filtro de politica: de lo que el canal no habla, decidido antes de gastar token.

El nicho tech/IA/ciencia ya excluye la mayor parte del riesgo por construccion.
Este filtro guarda las tres bordas que quedan, y todas aparecieron en el radar
real:

  (a) salud y medicamento con alegacion de eficacia -- "pluma de adelgazamiento"
      fue el tema con mas trafico en Google Trends ES en 17/09/2026;
  (b) ciencia instrumentalizada por politica partidista -- los tres articulos
      mas vistos de la Wikipedia en espanol eran magistrados del Tribunal
      Constitucional;
  (c) tragedia con victima real.

El filtro corre ANTES del score, no despues: un tema bloqueado no debe consumir
peticion de investigacion ni token de LLM, y no debe poder ganar en el ranking
por tener velocidad alta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.text import normalize


@dataclass(frozen=True)
class PolicyRule:
    nombre: str
    patron: re.Pattern[str]
    motivo: str


def _regla(nombre: str, terminos: list[str], motivo: str) -> PolicyRule:
    # \b en los bordes evita que "ia" case dentro de "media" o "si" dentro de "asis".
    union = "|".join(re.escape(t) for t in terminos)
    return PolicyRule(nombre, re.compile(rf"\b(?:{union})\b"), motivo)


# Los terminos quedan sin acento porque `normalize` quita los diacriticos antes
# del match.
REGLAS: tuple[PolicyRule, ...] = (
    _regla(
        "salud",
        [
            "adelgazante", "adelgazamiento", "perder peso", "ozempic", "mounjaro",
            "wegovy", "semaglutida", "tirzepatida", "anabolizante", "suplemento",
            "medicamento", "medicina", "farmaco", "posologia", "dosificacion",
            "cura para", "tratamiento para", "cancer", "quimioterapia", "canabidiol",
            "weight loss", "diet pill", "cure for", "treatment for",
        ],
        "una alegacion de salud o medicamento exige responsabilidad clinica que el canal no tiene",
    ),
    _regla(
        "politica",
        [
            "tribunal constitucional", "constitucional", "congreso de los diputados",
            "senado", "congreso nacional", "eleccion", "elecciones", "electoral",
            "urna", "votacion", "referendum", "mocion de censura", "diputado",
            "senador", "presidente del gobierno", "ministro", "ministra",
            "partido", "podemos", "vox", "sumar",
            "election", "senate", "congress", "parliament",
            # Nombres de jefe de gobierno en campana permanente: tema de IA con
            # ellos se vuelve politica partidista en el primer comentario
            # (visto en 19/09/2026: "Trump abre encuesta" paso la puerta).
            "trump", "biden", "kamala", "sanchez", "feijoo", "ayuso", "putin",
            "zelensky", "netanyahu", "milei", "maduro", "casa blanca", "moncloa",
            "white house", "rey felipe", "abascal", "yolanda diaz",
        ],
        "politica partidista: fuera del nicho y convierte cualquier error en crisis",
    ),
    _regla(
        "tragedia",
        [
            "muerte", "muertes", "muere", "murio", "mueren", "muerto", "muertos",
            "fallecido", "fallecimiento", "fallece", "fallece",
            "victima", "victimas", "accidente", "desastre", "tragedia",
            "caida de avion", "atentado", "tiroteo", "massacre", "masacre",
            "asesinato", "homicidio", "violacion", "secuestro", "naufragio",
            "incendio", "terremoto",
            "died", "dies", "killed", "death", "deaths", "shooting", "crash",
            "victims", "massacre", "earthquake", "wildfire",
        ],
        "tragedia con victima real: no se hace contenido viral sobre esto",
    ),
    _regla(
        "comercial",
        [
            # Guia de compra, promocion y producto financiero. Visto en el radar
            # de 19/09/2026: "Seguro para movil en 2026: que planes cubren el
            # robo de datos y Bizum?" paso el nicho (movil, datos) y gano nota
            # alta de interes -- y es recomendacion de seguro.
            "seguro para", "seguro de movil", "planes de seguro", "que planes",
            "mejores planes", "cupon", "cupones", "descuento", "descuentos",
            "promocion", "promociones", "black friday", "cyber monday",
            "vale la pena comprar", "donde comprar", "mejor precio", "apuestas",
            "bets", "casino", "prestamo", "credito", "ingreso extra",
            "ganar dinero", "bizum", "coupon", "discount", "deal alert",
            "best deals", "on sale",
        ],
        "contenido comercial o consejo financiero: el canal explica tecnologia, no "
        "recomienda compra, promocion ni producto financiero",
    ),
    _regla(
        "menores",
        ["nino", "ninos", "menor de edad", "adolescente", "infantil",
         "child", "children", "minor", "teen", "teenager"],
        "involucra menores: exige cuidado que el pipeline automatico no ofrece",
    ),
)


@dataclass(frozen=True)
class PolicyVerdict:
    allowed: bool
    rule: str = ""
    reason: str = ""
    matched: str = ""


def check(termo: str) -> PolicyVerdict:
    """Evalua un termino contra las reglas. La primera regla que casa decide.

    El orden de las reglas no es por gravedad sino por frecuencia observada en
    el radar; cualquier casamiento bloquea igual, asi que el orden solo afecta
    a que motivo aparece en el registro.
    """
    texto = normalize(termo)
    for regla in REGLAS:
        m = regla.patron.search(texto)
        if m:
            return PolicyVerdict(
                allowed=False, rule=regla.nombre, reason=regla.motivo, matched=m.group(0)
            )
    return PolicyVerdict(allowed=True)
