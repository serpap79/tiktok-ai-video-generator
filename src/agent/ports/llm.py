"""Puerto LLM: el primer modelo del proyecto entra por aquí.

E porta por dois motivos distintos.

O primeiro e a restricao de $0: o caminho padrao precisa ser free tier (Gemini
Flash, Groq), e free tier cai, muda id de modelo e devolve 429. Trocar de
cambiar de proveedor en mitad de una ejecución no puede exigir tocar ninguna etapa.

O segundo e o eval do M5, que e o artefato de portfolio: comparar free tier
contra modelo pago na mesma rubrica so tem valor se a troca for de uma linha e
si el **coste se mide**, no se estima. Por eso `Completion` incluye tokens y
latencia: sem isso a comparacao viraria "achei o roteiro do Claude melhor", que
no es prueba de nada.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class LLMError(RuntimeError):
    """Falha ao falar com o provedor, ou resposta inutilizavel."""


class LLMUnavailable(LLMError):
    """Provedor fora do ar, sem cota ou com rate limit estourado.

    Esperado, no excepcional: el nivel gratuito tiene límites por minuto y por día. Quien
    llama registra el error y continúa con lo disponible, como hace el radar con una fuente caída.
    """


class LLMQuotaExhausted(LLMUnavailable):
    """Cota negada, com o alcance dela: minuto, dia ou pedido grande demais.

    Subclasse de LLMUnavailable para quem ja trata "provedor fora" continuar
    funcionando. O alcance existe porque pede tres acoes diferentes, e foi o
    que faltou no 19/09 (gemini-2.5-flash com 20 pedidos/dia esgotou e cada
    estagio so via "429"):

    - ``minute``: esperar ``retry_after_s`` e tentar o MESMO modelo;
    - ``day``: esse modelo acabou ate o reset -- tentar o proximo da rota;
    - ``request``: la petición no cabe en el límite por minuto de ese modelo (413 de
      Groq); esperar no resuelve, solo otro modelo.
    """

    def __init__(self, message: str, *, scope: str = "unknown",
                 retry_after_s: float | None = None, limit: str = ""):
        super().__init__(message)
        self.scope = scope
        self.retry_after_s = retry_after_s
        self.limit = limit


class LLMBlocked(LLMError):
    """El proveedor rechazó la respuesta por un filtro de contenido.

    Tem tipo proprio porque pede acao oposta a de LLMUnavailable: repetir a
    la misma llamada no resuelve y puede que el tema no sea publicable.
    El curador ya bloquea la mayor parte, pero el filtro del proveedor es más
    amplo que o nosso e dispara em noticia de tech com vitima ou arma.
    """


class Usage(BaseModel):
    """Tokens cobrados pela chamada, como o provedor os reportou.

    Cero significa «el proveedor no informó», distinto de «no gastó», aunque
    aquí ambos conducen a la misma acción (no se puede comparar el coste) e inflar
    com estimativa propria seria pior que admitir a lacuna.
    """

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, outro: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + outro.input_tokens,
            output_tokens=self.output_tokens + outro.output_tokens,
        )


class Completion(BaseModel):
    """Lo que devolvió el modelo y cuánto costó medirlo."""

    text: str
    model: str
    provider: str
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = Field(default=0.0, ge=0)
    # "stop", "length", "safety"... Vem cru do provedor de proposito: truncar por
    # limite de tokens produz JSON invalido, e a mensagem de erro precisa poder
    # dizer que foi isso em vez de "resposta malformada".
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        return self.finish_reason.lower() in {"length", "max_tokens"}


@runtime_checkable
class LLM(Protocol):
    provider: str
    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        schema: dict | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> Completion:
        """Uma pergunta, uma resposta. Sem historico e sem estado.

        `schema` e um JSON Schema do formato esperado. Cada adaptador o traduz
        para o recurso que o provedor tem (saida estruturada nativa, modo JSON,
        ou instrucao no prompt) -- e nenhum deles garante aderencia, entao quem
        chama valida com Pydantic de todo jeito.

        Levanta LLMUnavailable cuando el proveedor no responde o deniega la cuota,
        LLMBlocked cuando rechaza por contenido y LLMError en el resto de casos.
        """
        ...


_CERCA = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_object(texto: str) -> dict:
    """Le o objeto JSON de uma resposta de modelo, tolerando o embrulho comum.

    Vive en el puerto y no en cada adaptador porque el defecto es genérico y no
    do provedor: mesmo em modo JSON, modelo de free tier as vezes devolve o
    objeto dentro de cerca de codigo ou com um "Aqui esta:" na frente. Tratar
    isso em tres lugares garantiria tres comportamentos diferentes.

    No intenta reparar JSON truncado: una respuesta cortada a mitad es un error de
    orcamento de tokens, e mascarar isso com regex produziria um dossie com
    metade dos fatos e nenhuma pista do motivo.
    """
    limpo = _CERCA.sub("", texto.strip())
    if limpo.startswith("["):
        # Lista no topo significa que o modelo ignorou o schema. Aproveitar o
        # primeiro elemento devolveria um dossie com um fato e nenhum aviso de
        # que os outros foram jogados fora.
        raise LLMError("esperava objeto JSON com a chave pedida, veio uma lista")

    inicio, fim = limpo.find("{"), limpo.rfind("}")
    if inicio == -1:
        raise LLMError(f"resposta sem objeto JSON: {texto[:200]!r}")
    if fim <= inicio:
        raise LLMError(
            "respuesta truncada: abrió un objeto y no lo cerró "
            "(presupuesto de tokens insuficiente para lo solicitado)"
        )
    try:
        valor = json.loads(limpo[inicio : fim + 1])
    except ValueError as exc:
        raise LLMError(f"JSON invalido na resposta do modelo: {exc}") from exc
    if not isinstance(valor, dict):
        raise LLMError(f"esperava objeto JSON, veio {type(valor).__name__}")
    return valor
