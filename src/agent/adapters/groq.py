"""Adaptador da porta LLM para o Groq (free tier, endpoint compativel com OpenAI).

Existe junto a Gemini desde M3, y no después, porque una puerta con un único
adaptador no es una puerta: es indirección. Mantener dos desde el principio demuestra
que cambiar de proveedor no se filtra a las etapas y ya aporta la mitad del brazo
livre do eval do M5.

Diferencia importante frente a Gemini: aquí no existe salida estructurada mediante
esquema en todos los modelos. Lo que siempre existe es `response_format=json_object`,
lo que garantiza JSON válido, pero **no** el formato solicitado. Por eso el esquema se incluye
tambem no prompt, como texto, e quem chama valida com Pydantic do mesmo jeito.

Os modelos sao open weights (GPT-OSS, Qwen). A latencia medida aqui e muito menor
que a do Gemini Flash -- 1,0s contra 6,6s na mesma chamada de teste, em
18/09/2026: el coste en el nivel gratuito también es cero. Si la escritura en castellano de España
compensa la diferencia es lo que mide la evaluación M5; por ahora es una impresión, no
resultado.

Id de modelo aqui e volatil: o padrao anterior (`llama-3.3-70b-versatile`) foi
descontinuado y devuelve 404. Por eso procede de la configuración y existe el
`agent llm-health`, que confere contra o provedor em vez de confiar no padrao.
"""

from __future__ import annotations

import copy
import json
import re
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

BASE_URL = "https://api.groq.com/openai/v1"

# Modelos com saida estruturada estrita (decodificacao restrita ao schema),
# conforme a doc de Structured Outputs do Groq em 19/09/2026. Neles o schema
# nao vai como texto no prompt -- vai no `response_format`, e a aderencia e
# garantida pelo decodificador. Isso resolve o `json_validate_failed` que o
# roteirista gpt-oss-120b batia no modo json_object, e ainda tira o schema do
# prompt (tokens de entrada a menos em toda chamada).
STRICT_SCHEMA_MODELS = frozenset({
    "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b",
})
# So a familia gpt-oss aceita `reasoning_effort`; mandar para outro modelo
# devolve 400.
REASONING_EFFORT_MODELS = frozenset({"openai/gpt-oss-120b", "openai/gpt-oss-20b"})


class Groq:
    provider = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        client: httpx.Client | None = None,
        timeout_s: float = 120.0,
        reasoning_effort: str = "",
    ):
        if not api_key:
            raise LLMError("AGENT_GROQ_API_KEY vazia; a chave e gratuita em console.groq.com/keys")
        self.model = model
        self._reasoning_effort = reasoning_effort
        self._client = client or httpx.Client(
            base_url=BASE_URL,
            timeout=httpx.Timeout(timeout_s),
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
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
            reasoning_effort=(self._reasoning_effort
                              if self.model in REASONING_EFFORT_MODELS else ""),
        )
        inicio = time.monotonic()
        try:
            r = self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"groq inacessivel: {exc}") from exc
        latencia = round(time.monotonic() - inicio, 3)

        if r.status_code in (429, 413):
            raise quota_error(r, self.model)
        if r.status_code in (500, 502, 503, 504):
            raise LLMUnavailable(f"groq instavel ({r.status_code})")
        if r.status_code != 200:
            raise LLMError(f"groq devolveu {r.status_code}: {r.text[:300]}")

        try:
            corpo = r.json()
        except ValueError as exc:
            raise LLMError("groq devolvió una respuesta que no es JSON") from exc

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
            # El modo JSON del endpoint exige la palabra «json» en el prompt y no
            # acepta un esquema; adjuntarlo como texto es lo que queda para solicitar
            # un formato. Va en el sistema para no competir con el contenido.
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
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "resposta", "strict": True,
                                "schema": strict_schema(schema)},
            }
        elif schema is not None:
            payload["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            # Só os modelos de raciocinio aceitam; mandar para os outros devolve
            # 400, por lo que la opción predeterminada es no enviarlo.
            payload["reasoning_effort"] = reasoning_effort
        return payload

    @staticmethod
    def parse(corpo: dict, *, model: str, latency_s: float) -> Completion:
        escolhas = corpo.get("choices") or []
        if not escolhas:
            raise LLMError("groq devolveu resposta sem choice")

        escolha = escolhas[0]
        motivo = str(escolha.get("finish_reason") or "")
        if motivo == "content_filter":
            raise LLMBlocked("groq bloqueó la respuesta por un filtro de contenido")

        texto = ((escolha.get("message") or {}).get("content") or "")
        if not texto.strip():
            raise LLMError(f"groq devolvió una opción sin contenido (finish_reason={motivo})")

        uso = corpo.get("usage") or {}
        return Completion(
            text=texto,
            model=str(corpo.get("model") or model),
            provider=Groq.provider,
            usage=Usage(
                input_tokens=int(uso.get("prompt_tokens") or 0),
                output_tokens=int(uso.get("completion_tokens") or 0),
            ),
            latency_s=latency_s,
            finish_reason=motivo,
        )


def strict_schema(schema: dict) -> dict:
    """O schema no dialeto do modo estrito: objeto fechado, tudo obrigatorio.

    O modo estrito do Groq recusa objeto sem `additionalProperties: false` e
    propiedad fuera de `required`. Los esquemas del proyecto ya piden todo; lo que
    falta es cerrar los objetos, algo que se hace aquí para que las etapas no conozcan el
    dialeto de provedor nenhum.
    """
    saida = copy.deepcopy(schema)

    def fechar(no: Any) -> None:
        if isinstance(no, dict):
            if no.get("type") == "object" and isinstance(no.get("properties"), dict):
                no["additionalProperties"] = False
                no["required"] = list(no["properties"])
            for valor in no.values():
                fechar(valor)
        elif isinstance(no, list):
            for item in no:
                fechar(item)

    fechar(saida)
    return saida


_ESPERA = re.compile(r"try again in ((?:[\d.]+(?:ms|h|m|s))+)", re.IGNORECASE)
_DURACAO = re.compile(r"([\d.]+)(ms|h|m|s)", re.IGNORECASE)
_UNIDADE = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def duration_s(texto: str) -> float | None:
    """Duracao no formato do Groq ('7.07s', '510ms', '1h2m3.5s') em segundos."""
    partes = _DURACAO.findall(texto)
    if not partes:
        return None
    return sum(float(n) * _UNIDADE[u.lower()] for n, u in partes)


def quota_error(r: httpx.Response, model: str) -> LLMQuotaExhausted:
    """429/413 do Groq traduzido em alcance.

    O corpo diz qual teto estourou ("tokens per minute (TPM)", "requests per
    day (RPD)"...). No free tier o gpt-oss-120b tem 8K tokens por minuto e
    200K por dia (doc de rate limits, 19/09/2026): um roteiro com raciocinio
    supera 5K, el de minutos es el aplicable y se resuelve esperando
    unos segundos, no cambiando de proveedor. El 413 es una petición que ni siquiera
    cabe en el límite por minuto: esperar no ayuda.
    """
    try:
        erro = (r.json() or {}).get("error") or {}
    except ValueError:
        erro = {}
    mensagem = str(erro.get("message", ""))
    baixa = mensagem.lower()
    if r.status_code == 413:
        escopo = "request"
    elif "per day" in baixa:
        escopo = "day"
    elif "per minute" in baixa:
        escopo = "minute"
    else:
        escopo = "unknown"
    espera: float | None = None
    valor = r.headers.get("retry-after")
    if valor:
        try:
            espera = float(valor)
        except ValueError:
            espera = None
    if espera is None:
        m = _ESPERA.search(mensagem)
        if m:
            espera = duration_s(m.group(1))
    teto = next((t for t in ("TPM", "RPM", "TPD", "RPD") if f"({t.lower()})" in baixa), "")
    detalhe = f", {teto}" if teto else ""
    if espera is not None:
        detalhe += f"; tente em {espera:g}s"
    return LLMQuotaExhausted(
        f"groq {model} negou cota ({r.status_code}, {escopo}{detalhe})",
        scope=escopo, retry_after_s=espera, limit=teto)
