"""Roteador de LLM: uma porta, varios modelos gratuitos, cota como criterio.

Implementa a porta `LLM` sobre uma lista ORDENADA de modelos. Nenhum estagio
sabe que existe mais de um: o pesquisador continua chamando `complete()`.

Por que existe (medido em 19/09/2026, nao suposto):

- o gemini-2.5-flash, padrao ate aqui, tem **20 pedidos por dia** no free
  tier. Um video consome ~10 (pesquisa por fonte + roteiro + juiz), entao o
  padrao antigo nao sustentava nem dois posts por dia;
- cada modelo tem cota PROPRIA: 3.8-flash, 3.5-flash, os flash-lite e os
  modelos do Groq sao baldes separados. Trocar de modelo quando um esgota
  multiplica a capacidade diaria sem custar nada;
- o Groq bate no teto de TOKENS POR MINUTO (8K no gpt-oss-120b) muito antes
  do diario, e esse se resolve esperando segundos -- trocar de provedor ali
  jogaria fora o modelo que ainda tem o dia inteiro de cota.

Entao cada negativa vira uma acao diferente:

| negativa                     | acao                                         |
|------------------------------|----------------------------------------------|
| cota do minuto, espera curta | dorme o `retry_after` e repete o mesmo modelo |
| cota do dia                  | grava no livro ate o reset e segue a rota    |
| pedido maior que o teto (413)| segue a rota, sem marcar esgotado            |
| 503 (sobrecarga)             | fora da rota por 10 min, segue a rota         |
| timeout / rede               | uma nova tentativa, depois segue a rota      |
| modelo inexistente (404)     | grava por 24h e segue a rota                 |
| filtro de conteudo           | sobe: outro modelo nao torna o tema publicavel|

Toda decisao vira linha no livro (`llm_calls`) e no `trail`, que a execucao
do slot grava junto do roteiro -- "por que este texto saiu do modelo X" tem
resposta.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from agent.memory.llm_ledger import LLMLedger, next_pacific_midnight
from agent.ports.llm import (
    LLM,
    Completion,
    LLMBlocked,
    LLMError,
    LLMQuotaExhausted,
    LLMUnavailable,
)

# Espera maxima que vale a pena por cota do minuto. Acima disso, um slot de
# 40 minutos de producao prefere o proximo modelo a ficar parado.
MAX_WAIT_S = 45.0
# Quantas vezes esperar pelo mesmo modelo numa mesma chamada.
MAX_WAITS = 2
# Falha de rede/timeout: uma segunda chance depois de uma pausa curta.
UNAVAILABLE_RETRIES = 1
UNAVAILABLE_BACKOFF_S = 4.0
# 503 "high demand" dos Gemini 3.x no free tier: medido no primeiro slot real
# (20/09/2026), cada 503 levou 10-20s para voltar e o mesmo modelo repetiu o
# 503 na segunda chance. Um 503 tira o modelo da rota por alguns minutos --
# no mesmo processo e nos proximos -- em vez de pagar a espera a cada chamada.
SOBRECARGA_S = 600


@dataclass(frozen=True)
class Route:
    """Um modelo numa rota: `provedor:modelo`, com opcoes do adaptador."""

    provider: str
    model: str
    reasoning_effort: str = ""

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    @classmethod
    def parse(cls, texto: str) -> Route:
        """'groq:openai/gpt-oss-120b@low' -> Route(groq, openai/gpt-oss-120b, low)."""
        base, _, esforco = texto.strip().partition("@")
        provedor, sep, modelo = base.partition(":")
        if not sep or not provedor or not modelo:
            raise ValueError(f"rota invalida {texto!r}; use provedor:modelo[@esforco]")
        return cls(provider=provedor.strip().lower(), model=modelo.strip(),
                   reasoning_effort=esforco.strip())


Builder = Callable[[Route], LLM]


@dataclass
class RoutedLLM:
    """Porta LLM sobre uma rota. `provider`/`model` dizem quem respondeu por ultimo."""

    stage: str
    routes: list[Route]
    builder: Builder
    ledger: LLMLedger | None = None
    sleeper: Callable[[float], None] = time.sleep
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    trail: list[str] = field(default_factory=list)
    _cache: dict[str, LLM] = field(default_factory=dict, repr=False)
    _last: Route | None = field(default=None, repr=False)
    # Rotas que falharam nesta instancia sem marcar o livro (413, chave
    # ausente): nao adianta tenta-las de novo no mesmo processo.
    _skip: set[str] = field(default_factory=set, repr=False)

    @property
    def provider(self) -> str:
        rota = self._last or (self.routes[0] if self.routes else None)
        return rota.provider if rota else ""

    @property
    def model(self) -> str:
        rota = self._last or (self.routes[0] if self.routes else None)
        return rota.model if rota else ""

    @property
    def last_route(self) -> str:
        return self._last.key if self._last else ""

    def prefer_other_than(self, provider: str) -> RoutedLLM:
        """A mesma rota com outros provedores na frente (ordem estavel).

        Usado pelo juiz: modelo julgando o proprio texto tende a se dar nota
        alta (o eval de 19/09 mediu o juiz groq 2 pontos mais generoso que o
        gemini no mesmo roteiro). Familia diferente primeiro, quando houver.
        """
        outros = [r for r in self.routes if r.provider != provider]
        mesmos = [r for r in self.routes if r.provider == provider]
        return RoutedLLM(stage=self.stage, routes=outros + mesmos, builder=self.builder,
                         ledger=self.ledger, sleeper=self.sleeper, clock=self.clock)

    def complete(self, prompt: str, *, system: str = "", schema: dict | None = None,
                 temperature: float = 0.2, max_output_tokens: int = 2048) -> Completion:
        motivos: list[str] = []
        for rota in self.routes:
            if rota.key in self._skip:
                continue
            ate = self.ledger.exhausted_until(rota.key, self.clock()) if self.ledger else None
            if ate is not None:
                motivos.append(f"{rota.key} esgotado ate {ate:%H:%M} UTC")
                continue
            try:
                llm = self._build(rota)
            except LLMError as exc:
                self._skip.add(rota.key)
                motivos.append(f"{rota.key}: {exc}")
                continue

            resultado = self._tentar(rota, llm, prompt, system, schema,
                                     temperature, max_output_tokens, motivos)
            if resultado is not None:
                self._last = rota
                if motivos:
                    self.trail.append(f"{self.stage}: " + "; ".join(motivos)
                                      + f" -> {rota.key}")
                return resultado

        resumo = "; ".join(motivos) or "rota vazia"
        self.trail.append(f"{self.stage}: todos indisponiveis ({resumo})")
        raise LLMUnavailable(f"nenhum modelo da rota '{self.stage}' respondeu: {resumo}")

    # ------------------------------------------------------------------ internos

    def _build(self, rota: Route) -> LLM:
        if rota.key not in self._cache:
            self._cache[rota.key] = self.builder(rota)
        return self._cache[rota.key]

    def _tentar(self, rota: Route, llm: LLM, prompt: str, system: str,
                schema: dict | None, temperature: float, max_output_tokens: int,
                motivos: list[str]) -> Completion | None:
        esperas = 0
        instabilidades = 0
        while True:
            try:
                resposta = llm.complete(prompt, system=system, schema=schema,
                                        temperature=temperature,
                                        max_output_tokens=max_output_tokens)
            except LLMBlocked:
                self._registrar(rota, ok=False, erro="bloqueado por filtro")
                raise
            except LLMQuotaExhausted as exc:
                self._registrar(rota, ok=False, erro=str(exc))
                if (exc.scope in ("minute", "unknown") and esperas < MAX_WAITS
                        and exc.retry_after_s is not None
                        and exc.retry_after_s <= MAX_WAIT_S):
                    esperas += 1
                    self.sleeper(exc.retry_after_s + 0.5)
                    continue
                if exc.scope == "request":
                    self._skip.add(rota.key)
                    motivos.append(f"{rota.key}: pedido maior que o teto ({exc.limit})")
                    return None
                self._marcar(rota, exc)
                motivos.append(f"{rota.key}: {exc}")
                return None
            except LLMUnavailable as exc:
                self._registrar(rota, ok=False, erro=str(exc))
                if "503" in str(exc):
                    if self.ledger is not None:
                        self.ledger.mark_exhausted(
                            rota.key, self.clock() + timedelta(seconds=SOBRECARGA_S),
                            f"sobrecarga do provedor: {exc}")
                    motivos.append(f"{rota.key}: sobrecarregado (503)")
                    return None
                if instabilidades < UNAVAILABLE_RETRIES:
                    instabilidades += 1
                    self.sleeper(UNAVAILABLE_BACKOFF_S)
                    continue
                motivos.append(f"{rota.key}: {exc}")
                return None
            except LLMError as exc:
                # 400/404 e resposta inutilizavel (candidato sem texto): nao e
                # cota, mas outro modelo pode responder. 404 e id descontinuado
                # -- fica fora por um dia para nao pagar o erro a cada chamada.
                self._registrar(rota, ok=False, erro=str(exc))
                if " 404" in str(exc) or "not found" in str(exc).lower():
                    if self.ledger is not None:
                        self.ledger.mark_exhausted(
                            rota.key, self.clock() + timedelta(hours=24),
                            f"modelo inexistente: {exc}")
                motivos.append(f"{rota.key}: {exc}")
                return None
            self._registrar(rota, ok=True, resposta=resposta)
            return resposta

    def _marcar(self, rota: Route, exc: LLMQuotaExhausted) -> None:
        if self.ledger is None:
            return
        agora = self.clock()
        if exc.scope == "day":
            if rota.provider == "gemini":
                ate = next_pacific_midnight(agora)
            else:
                ate = agora + timedelta(seconds=exc.retry_after_s or 3600)
        else:
            # Minuto com espera longa demais, ou alcance desconhecido: fora
            # pelo tempo pedido (ou 2 min), sem condenar o dia inteiro.
            ate = agora + timedelta(seconds=max(exc.retry_after_s or 120, 60))
        self.ledger.mark_exhausted(rota.key, ate, str(exc))

    def _registrar(self, rota: Route, *, ok: bool, erro: str = "",
                   resposta: Completion | None = None) -> None:
        if self.ledger is None:
            return
        self.ledger.record_call(
            self.stage, rota.key, ok=ok, error=erro,
            input_tokens=resposta.usage.input_tokens if resposta else 0,
            output_tokens=resposta.usage.output_tokens if resposta else 0,
            latency_s=resposta.latency_s if resposta else 0.0,
            at=self.clock(),
        )


__all__ = ["MAX_WAIT_S", "Route", "RoutedLLM"]
