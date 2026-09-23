"""Adaptador da porta LLM para o OpenRouter (endpoint compativel com OpenAI).

Principal desde 20/09/2026, por decisao do autor: a cota do Gemini no AI
Studio (20 pedidos/dia no free tier) disputa requisicao com outras automacoes
dele, e o endpoint pago do OpenRouter nao consome aquela cota. O Gemini continua
na rota, mas POR ULTIMO -- reserva que nao disputa nada enquanto houver
credito.

Uso consciente do credito ($100 ate 03/2027): os modelos baratos vao primeiro
na rota (`deepseek-v4-flash` ~$0.04/M tokens de entrada, `gemini-2.5-flash-lite`
~$0.10/M). Um video consome ~10 chamadas de ~2-6K tokens: sai por centavos de
dolar, e cada chamada grava tokens no livro (`llm_calls`), entao o gasto e
auditavel por slot.

Diferencas que importam contra o Groq:

- `response_format=json_object` existe e se comporta igual (JSON valido, sem
  garantia de formato): schema vai tambem no prompt e a validacao continua
  Pydantic. O modo estrito (`json_schema`) fica atras de allowlist, preenchida
  so com modelo MEDIDO aqui -- assumir que funciona e como o
  `json_validate_failed` voltou da ultima vez.
- erros proprios: 402 (credito esgotado) vira cota do dia com espera longa --
  nao adianta repetir hoje, segue para groq/gemini; 429 vira minuto com
  `retry-after`, igual ao Groq.
- `reasoning.effort` (low/medium/high) e o dialeto do OpenRouter para esforco
  de raciocinio; so vai no payload quando a rota pede (`@low`), porque modelo
  que nao raciocina ignora -- ou cobra -- o parametro.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from agent.ports.llm import (
    Completion,
    LLMBlocked,
    LLMError,
    LLMQuotaExhausted,
    LLMUnavailable,
    Usage,
)

BASE_URL = "https://openrouter.ai/api/v1"

# Modelos com `json_schema` estrito MEDIDO por aqui (decodificacao restrita ao
# schema). Medido em 20/09/2026 com a chave do projeto: os quatro responderam
# 200 com JSON valido e aderente ao schema pedido (flash e lite no primeiro
# teste; 2.5-flash e v4-pro no segundo). Neles o schema nao vai como texto no
# prompt -- vai no `response_format`, o que tambem economiza tokens de entrada
# (credito) em toda chamada.
STRICT_SCHEMA_MODELS = frozenset({
    "deepseek/deepseek-v4-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-v4-pro",
})
# Effort so e enviado quando a rota pede; o OpenRouter repassa ao modelo e os
# que nao raciocinam o ignoram.
REASONING_PARAM = "reasoning"


class OpenRouter:
    provider = "openrouter"

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek/deepseek-v4-flash",
        client: httpx.Client | None = None,
        timeout_s: float = 120.0,
        reasoning_effort: str = "",
        app_url: str = "",
        app_title: str = "tiktok-viral-generator",
    ):
        if not api_key:
            raise LLMError("AGENT_OPENROUTER_API_KEY vazia; a chave esta em openrouter.ai/keys")
        self.model = model
        self._reasoning_effort = reasoning_effort
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_title:
            headers["X-Title"] = app_title
        self._client = client or httpx.Client(
            base_url=BASE_URL,
            timeout=httpx.Timeout(timeout_s),
            headers=headers,
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        schema: dict | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> Completion:
        payload = self.build_payload(
            prompt, system=system, schema=schema, model=self.model,
            temperature=temperature, max_output_tokens=max_output_tokens,
            reasoning_effort=self._reasoning_effort,
        )
        inicio = time.monotonic()
        try:
            r = self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"openrouter inacessivel: {exc}") from exc
        latencia = round(time.monotonic() - inicio, 3)

        if r.status_code == 402:
            raise quota_error(r, self.model)
        if r.status_code == 429:
            raise quota_error(r, self.model)
        if r.status_code in (500, 502, 503, 504):
            raise LLMUnavailable(f"openrouter instavel ({r.status_code})")
        if r.status_code != 200:
            raise LLMError(f"openrouter devolveu {r.status_code}: {r.text[:300]}")

        try:
            corpo = r.json()
        except ValueError as exc:
            raise LLMError("openrouter devolveu resposta nao-JSON") from exc

        return self.parse(corpo, model=self.model, latency_s=latencia)

    @staticmethod
    def build_payload(
        prompt: str, *, system: str, schema: dict | None, model: str,
        temperature: float, max_output_tokens: int, reasoning_effort: str = "",
    ) -> dict[str, Any]:
        """Monta o corpo do chat/completions. Separado para ser testavel sem rede."""
        mensagens: list[dict[str, str]] = []
        instrucao = system
        estrito = schema is not None and model in STRICT_SCHEMA_MODELS
        if schema is not None and not estrito:
            instrucao = (
                f"{system}\n\n" if system else ""
            ) + (
                "Responda apenas com um objeto json valido, sem texto em volta, "
                "obedecendo exatamente a este schema:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            )
        if instrucao:
            mensagens.append({"role": "system", "content": instrucao})
        mensagens.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": mensagens,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if estrito:
            from agent.adapters.groq import strict_schema

            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "resposta", "strict": True,
                                "schema": strict_schema(schema or {})},
            }
        elif schema is not None:
            payload["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            payload[REASONING_PARAM] = {"effort": reasoning_effort}
        return payload

    @staticmethod
    def parse(corpo: dict, *, model: str, latency_s: float) -> Completion:
        escolhas = corpo.get("choices") or []
        if not escolhas:
            # O OpenRouter devolve o erro do provedor final em `error` quando
            # nao ha choice -- repassar a mensagem economiza uma adivinhacao.
            erro = corpo.get("error") or {}
            detalhe = str(erro.get("message") or "")[:200]
            sufixo = f": {detalhe}" if detalhe else ""
            raise LLMError(f"openrouter devolveu resposta sem choice{sufixo}")

        escolha = escolhas[0]
        motivo = str(escolha.get("finish_reason") or "")
        if motivo == "content_filter":
            raise LLMBlocked("openrouter bloqueou a resposta por filtro de conteudo")

        texto = ((escolha.get("message") or {}).get("content") or "")
        if not texto.strip():
            raise LLMError(f"openrouter devolveu choice sem conteudo (finish_reason={motivo})")

        uso = corpo.get("usage") or {}
        return Completion(
            text=texto,
            model=str(corpo.get("model") or model),
            provider=OpenRouter.provider,
            usage=Usage(
                input_tokens=int(uso.get("prompt_tokens") or 0),
                output_tokens=int(uso.get("completion_tokens") or 0),
            ),
            latency_s=latency_s,
        )


def quota_error(r: httpx.Response, model: str) -> LLMQuotaExhausted:
    """402/429 do OpenRouter traduzido em alcance.

    402 e credito esgotado: repetir hoje nao adianta, entao escopo do dia com
    espera de 24h -- a rota segue para groq/gemini e o livro registra. 429 e
    limite de requisicoes por minuto: espera o `retry-after` e repete.
    """
    if r.status_code == 402:
        return LLMQuotaExhausted(
            f"openrouter {model} sem credito (402)",
            scope="day", retry_after_s=86400.0, limit="")
    espera: float | None = None
    valor = r.headers.get("retry-after")
    if valor:
        try:
            espera = float(valor)
        except ValueError:
            espera = None
    detalhe = f"; tente em {espera:g}s" if espera is not None else ""
    return LLMQuotaExhausted(
        f"openrouter {model} negou cota (429{detalhe})",
        scope="minute", retry_after_s=espera, limit="")


__all__ = ["OpenRouter", "STRICT_SCHEMA_MODELS", "quota_error"]
