"""Adaptador da porta LLM que nao chama modelo nenhum.

Vive em `src/` e nao em `tests/` por dois motivos:

- **a suite inteira roda offline.** Um estagio que so funciona com chave de API
  nao teria teste, e o pesquisador e o juiz sao exatamente onde um bug custa
  caro: dossie com fato sem fonte passa despercebido.
- **o eval do M5 vai reexecutar respostas gravadas.** Comparar provedores na
  mesma rubrica exige poder rejulgar o mesmo material sem gastar cota outra vez,
  e este e o encaixe onde a resposta gravada entra. Nao esta exposto na CLI hoje
  porque ainda nao ha o que reexecutar.

Nao finge ser modelo: devolve respostas na ordem em que foram dadas. A esperteza
que ele tem e registrar as chamadas, o que permite ao teste afirmar que o
pesquisador mandou **uma chamada por fonte** -- que e o que garante que a URL de
cada fato vem de nos e nao do modelo.
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
            # Erro de teste, nao de dominio: significa que o estagio chamou o
            # modelo mais vezes do que o teste previu, e silenciar isso esconderia
            # justamente a chamada extra que se queria contar.
            raise LLMError("ScriptedLLM sem resposta restante para esta chamada")

        return Completion(
            text=texto,
            model=self.model,
            provider=self.provider,
            # Contagem grosseira por palavra. Nao vale como medida de custo -- e
            # so o suficiente para que o encanamento de uso do M5 seja exercitado
            # pelos testes em vez de existir sem nunca ser somado.
            usage=Usage(
                input_tokens=len((system + " " + prompt).split()),
                output_tokens=len(texto.split()),
            ),
            latency_s=round(time.monotonic() - inicio, 4),
            finish_reason="stop",
        )
