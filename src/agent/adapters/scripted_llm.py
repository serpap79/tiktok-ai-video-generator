"""Adaptador de la puerta LLM que no llama a ningún modelo.

Vive en `src/` y no en `tests/` por dos motivos:

- **a suite inteira roda offline.** Um estagio que so funciona com chave de API
  no tendría pruebas, y el investigador y el juez son precisamente donde un error cuesta
  caro: dossie com fato sem fonte passa despercebido.
- **o eval do M5 vai reexecutar respostas gravadas.** Comparar provedores na
  mesma rubrica exige poder rejulgar o mesmo material sem gastar cota outra vez,
  este es el punto donde entra la respuesta grabada. No está expuesto en la CLI hoy
  porque todavía no hay nada que reejecutar.

No finge ser un modelo: devuelve las respuestas en el orden recibido. La inteligencia
que aporta consiste en registrar las llamadas, lo que permite al test afirmar que el
investigador envió **una llamada por fuente**, lo que garantiza que la URL de
cada hecho venga de nosotros y no del modelo.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agent.ports.llm import Completion, LLMError, Usage


@dataclass
class Call:
    prompt: str
    system: str
    schema: dict | None


@dataclass
class ScriptedLLM:
    """Devolve `responses` em ordem, ou o resultado de `responder(prompt)`."""

    responses: list[str] = field(default_factory=list)
    responder: Callable[[str], str] | None = None
    model: str = "scripted-1"
    provider: str = "scripted"
    calls: list[Call] = field(default_factory=list)

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        schema: dict | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> Completion:
        self.calls.append(Call(prompt=prompt, system=system, schema=schema))
        inicio = time.monotonic()

        if self.responder is not None:
            texto = self.responder(prompt)
        elif self.responses:
            texto = self.responses.pop(0)
        else:
            # Error de prueba, no de dominio: significa que la etapa llamó al
            # modelo mais vezes do que o teste previu, e silenciar isso esconderia
            # justamente a chamada extra que se queria contar.
            raise LLMError("ScriptedLLM sem resposta restante para esta chamada")

        return Completion(
            text=texto,
            model=self.model,
            provider=self.provider,
            # Estimación aproximada por palabras. No sirve como medida de coste: es
            # so o suficiente para que o encanamento de uso do M5 seja exercitado
            # pelos testes em vez de existir sem nunca ser somado.
            usage=Usage(
                input_tokens=len((system + " " + prompt).split()),
                output_tokens=len(texto.split()),
            ),
            latency_s=round(time.monotonic() - inicio, 4),
            finish_reason="stop",
        )
