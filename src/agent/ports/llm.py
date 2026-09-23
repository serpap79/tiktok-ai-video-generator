"""Porta LLM: o primeiro modelo do projeto entra por aqui.

E porta por dois motivos distintos.

O primeiro e a restricao de $0: o caminho padrao precisa ser free tier (Gemini
Flash, Groq), e free tier cai, muda id de modelo e devolve 429. Trocar de
provedor no meio de uma execucao nao pode exigir tocar em nenhum estagio.

O segundo e o eval do M5, que e o artefato de portfolio: comparar free tier
contra modelo pago na mesma rubrica so tem valor se a troca for de uma linha e
se o **custo for medido**, nao estimado. Por isso `Completion` carrega tokens e
latencia: sem isso a comparacao viraria "achei o roteiro do Claude melhor", que
nao e evidencia de nada.
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

    Esperado, nao excepcional: free tier tem limite por minuto e por dia. Quem
    chama registra e segue com o que tem, como o radar faz com uma fonte fora.
    """


class LLMQuotaExhausted(LLMUnavailable):
    """Cota negada, com o alcance dela: minuto, dia ou pedido grande demais.

    Subclasse de LLMUnavailable para quem ja trata "provedor fora" continuar
    funcionando. O alcance existe porque pede tres acoes diferentes, e foi o
    que faltou no 19/09 (gemini-2.5-flash com 20 pedidos/dia esgotou e cada
    estagio so via "429"):

    - ``minute``: esperar ``retry_after_s`` e tentar o MESMO modelo;
    - ``day``: esse modelo acabou ate o reset -- tentar o proximo da rota;
    - ``request``: o pedido nao cabe no teto por minuto desse modelo (413 do
      Groq); esperar nao resolve, so outro modelo.
    """

    def __init__(self, message: str, *, scope: str = "unknown",
                 retry_after_s: float | None = None, limit: str = ""):
        super().__init__(message)
        self.scope = scope
        self.retry_after_s = retry_after_s
        self.limit = limit


class LLMBlocked(LLMError):
    """O provedor recusou por filtro de conteudo.

    Tem tipo proprio porque pede acao oposta a de LLMUnavailable: repetir a
    mesma chamada nao resolve, e o tema pode simplesmente nao ser publicavel.
    O curador ja barra a maior parte disso, mas o filtro do provedor e mais
    amplo que o nosso e dispara em noticia de tech com vitima ou arma.
    """


class Usage(BaseModel):
    """Tokens cobrados pela chamada, como o provedor os reportou.

    Zero significa "o provedor nao informou", que e diferente de "nao gastou" --
    mas aqui os dois levam a mesma acao (nao da para comparar custo), e inflar
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
    """O que o modelo devolveu, e quanto custou medir isso."""

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

        Levanta LLMUnavailable quando o provedor nao responde ou nega cota,
        LLMBlocked quando recusa por conteudo, LLMError no resto.
        """
        ...


_CERCA = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_object(texto: str) -> dict:
    """Le o objeto JSON de uma resposta de modelo, tolerando o embrulho comum.

    Vive na porta, e nao em cada adaptador, porque o defeito e do genero e nao
    do provedor: mesmo em modo JSON, modelo de free tier as vezes devolve o
    objeto dentro de cerca de codigo ou com um "Aqui esta:" na frente. Tratar
    isso em tres lugares garantiria tres comportamentos diferentes.

    Nao tenta consertar JSON truncado: resposta cortada no meio e erro de
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
            "resposta truncada: abriu objeto e nao fechou "
            "(orcamento de tokens curto para o que foi pedido)"
        )
    try:
        valor = json.loads(limpo[inicio : fim + 1])
    except ValueError as exc:
        raise LLMError(f"JSON invalido na resposta do modelo: {exc}") from exc
    if not isinstance(valor, dict):
        raise LLMError(f"esperava objeto JSON, veio {type(valor).__name__}")
    return valor
