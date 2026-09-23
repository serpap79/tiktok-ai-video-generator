"""Escolhe o adaptador de LLM a partir da configuracao.

Existe para que nenhum estagio precise saber qual provedor esta em uso: o
pesquisador recebe uma porta `LLM` pronta e nao importa nada de `adapters/`. E
tambem o unico lugar que sabe qual variavel de ambiente carrega qual chave, o
que mantem a mensagem de erro de chave faltando em um lugar so.

Dois caminhos:

- `build_llm(provider)`: um modelo so, o do `.env` -- o que os comandos
  manuais (`research`, `write`, `judge`) sempre usaram;
- `build_routed(stage)`: a rota do estagio, com troca de modelo por cota
  (`adapters/router.py`) -- o que o piloto automatico usa.
"""

from __future__ import annotations

from agent.adapters.router import Route, RoutedLLM
from agent.config import Settings
from agent.config import settings as default_settings
from agent.memory.llm_ledger import LLMLedger
from agent.ports.llm import LLM, LLMError

PROVEDORES = ("openrouter", "groq", "gemini")

# Rotas por estagio, da preferencia para a reserva. Cada modelo e um balde de
# cota separado; a ordem diz onde gastar o que e escasso.
#
# Desde 20/09/2026 o OpenRouter abre toda rota: a cota do Gemini no AI Studio
# disputa requisicao com outras automacoes do autor, e o endpoint pago do
# OpenRouter nao consome aquela cota -- o Gemini direto fica POR ULTIMO, como
# reserva que nao disputa nada. Dentro do OpenRouter, o preco segue a
# exigencia da tarefa (medido o teto do autor em ~$0.10-0.15/video):
# extracao e rankeamento ficam nos centavos (deepseek-v4-flash); o texto que
# vira audio e o julgamento vao no degrau acima (gemini-2.5-flash no escritor,
# deepseek-v4-pro no juiz, que ainda cruza familia com o escritor). Mesmo
# assim o video fecha em ~$0.03-0.05 -- folga de 3-5x no teto, de proposito:
# pagar mais sem medida de qualidade seria queimar credito a toa.
#
# - research: extracao literal, 1 chamada por fonte (o estagio de maior
#   volume). deepseek-v4-flash primeiro: ~$0.04/M tokens de entrada.
# - writer/humanize: o texto que vira audio. gemini-2.5-flash-lite primeiro:
#   mesma familia do escritor medido (flash), sem tocar a cota disputada.
# - judge: o `prefer_other_than` poe a familia diferente do escritor na frente.
DEFAULT_ROUTES: dict[str, tuple[str, ...]] = {
    "research": (
        "openrouter:deepseek/deepseek-v4-flash",
        "groq:openai/gpt-oss-120b@low",
        "groq:openai/gpt-oss-20b@low",
        "groq:qwen/qwen3.8-27b",
        "gemini:gemini-3.1-flash-lite",
        "gemini:gemini-2.5-flash-lite",
    ),
    "writer": (
        "openrouter:google/gemini-2.5-flash",
        "openrouter:google/gemini-2.5-flash-lite",
        "openrouter:deepseek/deepseek-v4-flash",
        "groq:openai/gpt-oss-120b@low",
        "gemini:gemini-3.8-flash",
        "gemini:gemini-3.5-flash",
        "gemini:gemini-2.5-flash",
        "groq:qwen/qwen3.8-27b",
    ),
    "humanize": (
        "openrouter:google/gemini-2.5-flash",
        "openrouter:google/gemini-2.5-flash-lite",
        "openrouter:deepseek/deepseek-v4-flash",
        "groq:openai/gpt-oss-120b@low",
        "gemini:gemini-3.8-flash",
        "gemini:gemini-3.5-flash",
        "groq:qwen/qwen3.8-27b",
    ),
    # writer_short: o curto tem teto mecanico de 30-50 palavras, e a familia
    # Gemini estoura (60-80, medido em 19/09 e repetido pelo 2.5-flash em
    # 20/09: 65 e 53 palavras). O gpt-oss com raciocinio baixo escreveu curto
    # de 1a (42) -- entao o curto abre no Groq. OpenRouter segue antes do
    # Gemini direto (sem disputar a cota das outras automacoes).
    "writer_short": (
        "groq:openai/gpt-oss-120b@low",
        "groq:openai/gpt-oss-20b@low",
        "openrouter:deepseek/deepseek-v4-flash",
        "openrouter:google/gemini-2.5-flash-lite",
        "gemini:gemini-3.1-flash-lite",
        "gemini:gemini-2.5-flash-lite",
        "groq:qwen/qwen3.8-27b",
    ),
    # ranker: 1 chamada por slot com os ~10 melhores temas. Modelo rapido;
    # a nota e combinada com o radar, nunca decide sozinha.
    "ranker": (
        "openrouter:deepseek/deepseek-v4-flash",
        "groq:openai/gpt-oss-120b@low",
        "groq:openai/gpt-oss-20b@low",
        "gemini:gemini-3.1-flash-lite",
        "gemini:gemini-2.5-flash-lite",
    ),
    "judge": (
        "openrouter:deepseek/deepseek-v4-pro",
        "openrouter:deepseek/deepseek-v4-flash",
        "groq:openai/gpt-oss-120b@medium",
        "gemini:gemini-3.5-flash",
        "gemini:gemini-3.8-flash",
        "groq:qwen/qwen3.8-27b",
    ),
}


def build_llm(provider: str | None = None, settings: Settings | None = None) -> LLM:
    cfg = settings or default_settings
    nome = (provider or cfg.llm_provider).strip().lower()

    if nome == "openrouter":
        return build_route(Route("openrouter", cfg.openrouter_model), cfg)
    if nome == "gemini":
        return build_route(Route("gemini", cfg.gemini_model), cfg)
    if nome == "groq":
        return build_route(Route("groq", cfg.groq_model, cfg.groq_reasoning_effort), cfg)
    raise LLMError(f"provedor de LLM desconhecido: {nome!r}; use um de {PROVEDORES}")


def build_route(route: Route, settings: Settings | None = None) -> LLM:
    """Adaptador de um modelo especifico. Levanta LLMError sem chave."""
    cfg = settings or default_settings
    if route.provider == "openrouter":
        from agent.adapters.openrouter import OpenRouter

        return OpenRouter(
            api_key=cfg.openrouter_api_key, model=route.model,
            timeout_s=cfg.llm_timeout_s, reasoning_effort=route.reasoning_effort,
            app_url=cfg.openrouter_app_url, app_title=cfg.openrouter_app_title,
        )
    if route.provider == "gemini":
        from agent.adapters.gemini_free import GeminiFree

        return GeminiFree(
            api_key=cfg.gemini_api_key, model=route.model,
            timeout_s=cfg.llm_timeout_s, thinking_budget=cfg.gemini_thinking_budget,
        )
    if route.provider == "groq":
        from agent.adapters.groq import Groq

        return Groq(
            api_key=cfg.groq_api_key, model=route.model,
            timeout_s=cfg.llm_timeout_s, reasoning_effort=route.reasoning_effort,
        )
    raise LLMError(f"provedor de LLM desconhecido: {route.provider!r}")


def routes_for(stage: str, settings: Settings | None = None) -> list[Route]:
    """A rota do estagio: a do `.env` (AGENT_LLM_ROUTES) ou a padrao."""
    cfg = settings or default_settings
    bruto = (cfg.llm_routes or {}).get(stage)
    textos = ([t for t in bruto.split(",") if t.strip()] if bruto
              else list(DEFAULT_ROUTES.get(stage, DEFAULT_ROUTES["writer"])))
    return [Route.parse(t) for t in textos]


def build_routed(stage: str, settings: Settings | None = None,
                 ledger: LLMLedger | None = None) -> RoutedLLM:
    """Porta LLM com troca de modelo por cota para o estagio."""
    cfg = settings or default_settings
    livro = ledger if ledger is not None else LLMLedger(cfg.db_path)
    return RoutedLLM(stage=stage, routes=routes_for(stage, cfg),
                     builder=lambda rota: build_route(rota, cfg), ledger=livro)


def configured(settings: Settings | None = None) -> list[str]:
    """Provedores que tem chave preenchida, na ordem de preferencia.

    Usado por `agent llm-health` para checar o que da para checar, em vez de
    falhar em quem o usuario nunca configurou.
    """
    cfg = settings or default_settings
    chaves = {"openrouter": cfg.openrouter_api_key, "gemini": cfg.gemini_api_key,
              "groq": cfg.groq_api_key}
    return [p for p in PROVEDORES if chaves[p]]


__all__ = ["DEFAULT_ROUTES", "PROVEDORES", "build_llm", "build_route",
           "build_routed", "configured", "routes_for"]
