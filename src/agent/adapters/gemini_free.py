"""Adaptador da porta LLM para a API Gemini (free tier do AI Studio).

Caminho padrao do projeto por tres razoes praticas, nao por preferencia:

1. chave gratuita sem cartao, com cota diaria que cobre folgado um video por dia;
2. saida estruturada nativa (`responseSchema`), que e o que impede o pesquisador
   de receber prosa onde esperava uma lista de fatos;
3. pt-BR decente -- o roteirista (M3, fatia 2) escreve narracao falada, e modelo
   que tropeca em concordancia gera audio que soa errado mesmo estando certo.

A cota do free tier e por minuto **e** por dia. Estourar devolve 429, que sobe
como LLMUnavailable: repetir na hora nao resolve e segurar a execucao esperando
a janela abrir custaria mais que perder a fonte.

O raciocinio interno do 2.5 Flash vem **desligado** por padrao aqui, e isso foi
decidido medindo: com ele ligado, uma chamada do roteirista truncou o JSON no
meio e outra estourou o timeout de leitura. Ele sai do mesmo orcamento de saida e
do mesmo relogio da resposta, entao ligado ele troca previsibilidade por
qualidade que ainda nao foi medida -- e medir isso e experimento do M5.
"""

from __future__ import annotations

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

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Tipos que o Gemini aceita em responseSchema sao um subconjunto do OpenAPI 3.0,
# com os nomes em maiuscula. JSON Schema puro passa em alguns casos e e recusado
# em outros, entao a traducao e explicita.
_TIPOS = {
    "object": "OBJECT", "array": "ARRAY", "string": "STRING",
    "number": "NUMBER", "integer": "INTEGER", "boolean": "BOOLEAN",
}
# Chaves de JSON Schema que o endpoint recusa com 400 em vez de ignorar.
_CHAVES_RECUSADAS = frozenset({"additionalProperties", "$schema", "title", "default"})


class GeminiFree:
    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        client: httpx.Client | None = None,
        timeout_s: float = 120.0,
        thinking_budget: int = 0,
    ):
        if not api_key:
            raise LLMError(
                "AGENT_GEMINI_API_KEY vazia; a chave e gratuita em aistudio.google.com/apikey"
            )
        self.model = model
        self._thinking_budget = thinking_budget
        self._client = client or httpx.Client(
            base_url=BASE_URL,
            timeout=httpx.Timeout(timeout_s),
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
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
            prompt, system=system, schema=schema,
            temperature=temperature, max_output_tokens=max_output_tokens,
            thinking_budget=self._thinking_budget,
        )
        inicio = time.monotonic()
        try:
            r = self._client.post(f"/models/{self.model}:generateContent", json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"gemini inacessivel: {exc}") from exc
        latencia = round(time.monotonic() - inicio, 3)

        if r.status_code == 429:
            raise quota_error(r, self.model)
        if r.status_code in (500, 502, 503, 504):
            # 503 "high demand" e comum nos 3.x no free tier (medido em
            # 19/09/2026): e transitorio, e o roteador tenta de novo ou segue.
            raise LLMUnavailable(f"gemini instavel ({r.status_code})")
        if r.status_code != 200:
            raise LLMError(f"gemini devolveu {r.status_code}: {r.text[:300]}")

        try:
            corpo = r.json()
        except ValueError as exc:
            raise LLMError("gemini devolveu resposta nao-JSON") from exc

        return self.parse(corpo, model=self.model, latency_s=latencia)

    @staticmethod
    def build_payload(
        prompt: str, *, system: str, schema: dict | None,
        temperature: float, max_output_tokens: int, thinking_budget: int = 0,
    ) -> dict[str, Any]:
        """Monta o corpo do generateContent. Separado para ser testavel sem rede."""
        config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if thinking_budget >= 0:
            # Sem isso o raciocinio consome o maxOutputTokens e a resposta chega
            # truncada, com o objeto JSON aberto e nao fechado. Orcamento
            # negativo omite o campo e deixa o provedor decidir.
            config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        if schema is not None:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = to_openapi_schema(schema)

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    @staticmethod
    def parse(corpo: dict, *, model: str, latency_s: float) -> Completion:
        bloqueio = (corpo.get("promptFeedback") or {}).get("blockReason")
        if bloqueio:
            raise LLMBlocked(f"gemini bloqueou o prompt: {bloqueio}")

        candidatos = corpo.get("candidates") or []
        if not candidatos:
            raise LLMError("gemini devolveu resposta sem candidato")

        cand = candidatos[0]
        motivo = str(cand.get("finishReason") or "")
        if motivo.upper() == "SAFETY":
            raise LLMBlocked("gemini bloqueou a resposta por filtro de conteudo")

        partes = (cand.get("content") or {}).get("parts") or []
        texto = "".join(p.get("text", "") for p in partes)
        if not texto.strip():
            # Acontece com MAX_TOKENS logo apos o "thinking": o candidato volta
            # sem parte de texto. Dizer isso e melhor que devolver string vazia
            # e deixar o parser de JSON reclamar de outra coisa.
            raise LLMError(f"gemini devolveu candidato sem texto (finishReason={motivo})")

        uso = corpo.get("usageMetadata") or {}
        return Completion(
            text=texto,
            model=str(corpo.get("modelVersion") or model),
            provider=GeminiFree.provider,
            usage=Usage(
                input_tokens=int(uso.get("promptTokenCount") or 0),
                # O Gemini Flash cobra o raciocinio interno separado do texto
                # devolvido. Os dois saem do mesmo orcamento, entao entram juntos
                # -- ignorar o thinking subestimaria o custo no eval do M5.
                output_tokens=int(uso.get("candidatesTokenCount") or 0)
                + int(uso.get("thoughtsTokenCount") or 0),
            ),
            latency_s=latency_s,
            finish_reason=motivo,
        )


def to_openapi_schema(schema: dict) -> dict:
    """Converte JSON Schema para o dialeto OpenAPI que o Gemini aceita."""
    saida: dict[str, Any] = {}
    for chave, valor in schema.items():
        if chave in _CHAVES_RECUSADAS:
            continue
        if chave == "type" and isinstance(valor, str):
            saida["type"] = _TIPOS.get(valor.lower(), valor.upper())
        elif chave == "properties" and isinstance(valor, dict):
            saida["properties"] = {k: to_openapi_schema(v) for k, v in valor.items()}
            # Sem isso o modelo escolhe a ordem das chaves, e a ordem muda o que
            # ele escreve: pedir a afirmacao antes do numero produz fato solto.
            saida.setdefault("propertyOrdering", list(valor))
        elif chave == "items" and isinstance(valor, dict):
            saida["items"] = to_openapi_schema(valor)
        else:
            saida[chave] = valor
    return saida


def quota_error(r: httpx.Response, model: str) -> LLMQuotaExhausted:
    """429 do Gemini traduzido em alcance: minuto ou dia.

    O corpo diz qual cota estourou (`QuotaFailure.violations[].quotaId`) e o
    teto dela (`quotaValue`). Medido em 19/09/2026: o gemini-2.5-flash no free
    tier tem `GenerateRequestsPerDayPerProjectPerModel-FreeTier` = 20 -- vinte
    pedidos por dia, que um unico video (pesquisa + roteiro + juiz) consome
    quase inteiro. Sem ler isso, "429" parecia instabilidade e o pipeline
    insistia no mesmo modelo.
    """
    escopo, teto, espera = "unknown", "", None
    try:
        erro = (r.json() or {}).get("error") or {}
    except ValueError:
        erro = {}
    for det in erro.get("details") or []:
        tipo = str(det.get("@type", ""))
        if tipo.endswith("QuotaFailure"):
            for v in det.get("violations") or []:
                qid = str(v.get("quotaId", ""))
                teto = str(v.get("quotaValue", teto))
                if "PerDay" in qid:
                    escopo = "day"
                elif "PerMinute" in qid and escopo != "day":
                    escopo = "minute"
        elif tipo.endswith("RetryInfo"):
            espera = _segundos(str(det.get("retryDelay", "")))
    if espera is None:
        valor = r.headers.get("retry-after")
        espera = _segundos(valor) if valor else None
    if teto == "0":
        # Modelo fora do free tier (ex. lyria): nao volta no reset.
        escopo = "day"
    detalhe = f"; teto {teto}" if teto else ""
    if espera is not None:
        detalhe += f"; tente em {espera:g}s"
    return LLMQuotaExhausted(
        f"gemini {model} negou cota (429, {escopo}{detalhe})",
        scope=escopo, retry_after_s=espera, limit=teto)


def _segundos(valor: str) -> float | None:
    """'11s', '11.5s', '11' -> 11.0. Vazio ou lixo -> None."""
    limpo = valor.strip().removesuffix("s")
    try:
        return float(limpo)
    except ValueError:
        return None
