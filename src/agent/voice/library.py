"""Biblioteca de voces: cada entrada dice que es, de donde viene y que puede.

Regla de honestidad de este archivo: campo no verificado queda marcado, nunca
inventado. Genero aparente, por ejemplo, solo entra cuando la fuente (model
card o escucha) lo dice -- las muestras en `output/voces/` existen justamente
para que la escucha calibre estos metadatos.

Lagunas declaradas (19/09/2026):
- **Voz de Nova**: la presentadora pide es-ES femenino medio-grave, y voz
  femenina es-ES abierta no existe en los catalogos Piper. Sin ella, Nova
  tiene rostro y prompt, pero no voz -- y sintetizarla en timbre masculino
  romperia el personaje. Atlas usa `dave` (estilo `atlas`).
- **Acento regional**: ningun modelo abierto reproduce acento (andaluza,
  gallega, catalana...). Variedad aqui es cambio de locutor, no de acento.
  Fingir acento con pitch/velocidad seria caricatura, no recurso.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    id: str
    nombre: str
    genero: str
    edad_aparente: str
    timbre: str
    acento: str
    usos: tuple[str, ...] = ()
    modelo_url: str = ""
    config_url: str = ""
    archivo: str = ""
    licencia: str = ""
    notas: str = ""
    activo: bool = True


VOICES: dict[str, Voice] = {
    "dave": Voice(
        id="dave",
        nombre="Dave (tecnologia)",
        genero="masculino",
        edad_aparente="30-40",
        timbre="medio, articulado",
        acento="castellano neutro",
        usos=("noticia", "curiosidad", "documental", "historia"),
        modelo_url=("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                    "es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"),
        config_url=("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                    "es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"),
        archivo="dave.onnx",
        licencia="MIT (modelo y codigo)",
        notas=("Voz es-ES del catalogo oficial de Piper; RTF a medir en la "
               "primera pasada del piloto. Sin warnings de fonema."),
    ),
    "carla": Voice(
        id="carla",
        nombre="Carla (neutro)",
        genero="a confirmar por la escucha",
        edad_aparente="a confirmar por la escucha",
        timbre="neutro",
        acento="castellano neutro",
        usos=("documental", "noticia", "calmo"),
        modelo_url=("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                    "es/es_ES/carlfm/medium/es_ES-carlfm-x_low.onnx"),
        archivo="carla.onnx",
        licencia="MIT (repo piper-voices)",
        notas="x_low: calidad menor; solo de reserva hasta haber escucha.",
    ),
    "karen": Voice(
        id="karen",
        nombre="Karen (grave, pausado)",
        genero="a confirmar por la escucha",
        edad_aparente="a confirmar por la escucha",
        timbre="grave, ritmo mas lento",
        acento="castellano neutro",
        usos=("suspense", "misterio", "dramatico", "emocional"),
        modelo_url=("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                    "es/es_ES/karen/medium/es_ES-karen-medium.onnx"),
        archivo="karen.onnx",
        licencia="MIT (repo piper-voices)",
        notas=("RTF por medir; ~2,4 palabras/s (mas lento que las 2,5 "
               "asumidas -- guion rinde audio ~20% mas largo)."),
    ),
}

# Fuera de la biblioteca, con motivo grabado (misma regla del ledger: rechazo
# con motivo calibra, rechazo silencioso se vuelve a re-testear).
RECHAZADAS: dict[str, str] = {
    "es-carlfm-x_low": ("voz x_low: calidad de radio antigua -- no sirve para "
                        "narracion de marca."),
    "F5-TTS-es": "checkpoint CC-BY-NC 4.0: sin uso comercial. El canal monetiza.",
    "XTTS-v2": "CPML: gratis solo para no-comercial. El canal monetiza.",
    "Bark": "MIT, pero RTF 10-20x en CPU: 60s de audio = 10-20min. Inviable.",
    "Kokoro-82M": "Apache-2.0, pero sin voz es-ES verificada hasta 19/09/2026.",
    "edge-tts": ("ya usado en el MPT: gratis sin contrato, puede romperse; voces "
                 "femeninas existen alli, pero externas y sin SLA -- fallback, "
                 "no base."),
}


@dataclass(frozen=True)
class Style:
    id: str
    nombre: str
    voz: str
    velocidad: float
    ruido: float
    pausa_frase_s: float
    descripcion: str = ""


# Estilo es capa interpretativa sobre 3 timbres: cambia ritmo, pausa y
# variacion -- no finge ser locutor nuevo. Velocidad cerca de 1.0 de
# proposito: naturalidad > efecto.
STYLES: dict[str, Style] = {
    "documental": Style("documental", "Narrador documental", "carla", 1.0, 0.5, 0.30,
                        "serio, seguro, informativo"),
    "suspense": Style("suspense", "Suspense", "karen", 0.92, 0.4, 0.55,
                      "bajo, controlado, pausas estrategicas"),
    "misterio": Style("misterio", "Misterio", "karen", 0.95, 0.6, 0.45,
                      "intrigante, descubrimiento"),
    "historia": Style("historia", "Historia", "dave", 0.98, 0.6, 0.35,
                      "natural, progresiva"),
    "curiosidad": Style("curiosidad", "Curiosidad", "dave", 1.05, 0.7, 0.25,
                        "energetico sin exagerar"),
    "noticia": Style("noticia", "Noticia", "dave", 1.08, 0.4, 0.25,
                     "objetivo, claro, profesional"),
    "dramatico": Style("dramatico", "Dramatico", "karen", 0.95, 0.8, 0.40,
                       "variacion de intensidad"),
    "emocional": Style("emocional", "Emocional", "karen", 0.92, 0.8, 0.50,
                       "sensible, ritmo variado"),
    "energetico": Style("energetico", "Energetico", "dave", 1.12, 0.7, 0.20,
                        "rapido, dinamico"),
    "calmo": Style("calmo", "Calmo", "carla", 0.90, 0.4, 0.50,
                   "lento, suave"),
    # Voz del Atlas (guia de marca: grave, ritmo 1.0, enfasis en los numeros).
    # Nova no tiene estilo: sin voz femenina abierta, sintetizarla en voz
    # masculina romperia el personaje -- laguna declarada, no chapuza.
    "atlas": Style("atlas", "Atlas (tutorial/comparacion)", "karen", 1.0, 0.6, 0.35,
                   "didactico, directo; numeros con micro-pausa via *enfasis*"),
}

NAMES = sorted(VOICES)
STYLE_NAMES = sorted(STYLES)

__all__ = ["NAMES", "RECHAZADAS", "STYLES", "STYLE_NAMES", "VOICES", "Style", "Voice"]
