"""Configuracao do agente, lida de variaveis de ambiente ou .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fonte do MoneyPrinterTurbo que cobre acentos pt-BR. As outras que vem no repo
# dele sao chinesas (STHeiti, MicrosoftYaHei) ou vietnamitas, e renderizam
# tofu no lugar de "ç" e "ã". Verificado na cmap: 459 glifos, zero faltando.
FONT_PTBR = "BeVietnamPro-Bold.ttf"

# Vozes pt-BR gratuitas do edge-tts (sem chave, sem conta).
VOICES_PTBR = (
    "pt-BR-ThalitaMultilingualNeural",
    "pt-BR-FranciscaNeural",
    "pt-BR-AntonioNeural",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AGENT_", extra="ignore"
    )

    # --- renderizador (MoneyPrinterTurbo rodando local) ---
    renderer_url: str = "http://127.0.0.1:8080"
    renderer_api_key: str = ""
    renderer_dir: Path = PROJECT_ROOT / ".renderer"
    # Render de 60-90s em CPU leva minutos; o timeout cobre o pior caso.
    renderer_timeout_s: float = 900.0
    renderer_poll_interval_s: float = 5.0

    # --- material visual ---
    # "pexels" em producao; "local" para provar a cadeia de producao (TTS,
    # legenda, montagem) sem depender de chave de API nem de rede.
    video_source: str = "pexels"
    local_materials: list[Path] = []
    # Chave do Pexels com prefixo do projeto (o .env herdado do MPT usa
    # PEXELS_API_KEY sem prefixo; os dois valem, o com prefixo primeiro).
    pexels_api_key: str = ""

    # --- producao ---
    # "ffmpeg": renderizador proprio (narracao com pronuncia corrigida, legenda
    # da marca, ~20s por video). "mpt": o MoneyPrinterTurbo, que fica de reserva.
    renderer: str = "ffmpeg"
    # Vozes do renderizador proprio: monolingues pt-BR. A Multilingual (padrao
    # do MPT, abaixo) troca o sotaque no meio da frase -- reportado no
    # primeiro video do piloto em 20/09/2026 ("Discorda?").
    narrator_voice: str = "pt-BR-FranciscaNeural"
    narrator_voice_male: str = "pt-BR-AntonioNeural"
    voice_name: str = VOICES_PTBR[0]
    font_name: str = FONT_PTBR
    # Legenda palavra-a-palavra: uma palavra por vez precisa ser grande e ter
    # contorno grosso para ler sobre clipe claro. Posicao "custom" a 60% da
    # altura: abaixo do cartao do gancho (terco de cima) e acima da faixa que
    # a interface do TikTok cobre (legenda, perfil, musica).
    font_size: int = 84
    subtitle_position: str = "custom"
    subtitle_custom_position: float = 60.0
    subtitle_stroke_width: float = 3.0

    # --- LLM (primeiro modelo do projeto, M3) ---
    # Sob a restricao de $0 o padrao nao e Claude: e free tier. O adaptador do
    # Claude existe para o braco pago do eval do M5 -- a comparacao medida e o
    # artefato, nao o modelo escolhido.
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    # Ids de modelo de free tier mudam e sao descontinuados sem aviso. Ficam
    # configuraveis, e `agent llm-health` confere contra o provedor em vez de
    # confiar que o padrao ainda existe.
    gemini_model: str = "gemini-2.5-flash"
    # O 2.5 Flash raciocina por padrao, e o raciocinio sai do MESMO orcamento de
    # saida e do mesmo relogio. Medido em 18/09/2026, com thinking ligado: uma
    # chamada do roteirista truncou o JSON no meio (o objeto abriu e nao fechou) e
    # outra estourou 60s de leitura. Zero desliga. Nao e economia de token: e o que
    # torna a resposta previsivel o bastante para ser validada por contrato.
    # Negativo deixa o provedor decidir (dinamico); se um dia houver evidencia de
    # que raciocinio melhora a rubrica, isso vira experimento do M5 e nao palpite.
    gemini_thinking_budget: int = 0
    groq_api_key: str = ""
    # Medido em 18/09/2026: o `llama-3.3-70b-versatile`, que era o padrao obvio,
    # saiu do catalogo do Groq (404 model_not_found) e nao ha mais nenhum Llama de
    # chat lá. Entre os que existem, este e o mais forte de uso geral -- o que
    # importa porque o Groq e o braco que o eval do M5 compara contra o pago.
    # Alternativa mais rapida e barata: "qwen/qwen3.8-27b" (0,5s contra 1,0s, e
    # menos da metade dos tokens). Os `groq/compound*` ficam FORA de proposito:
    # sao sistemas agenticos com busca web embutida, e aqui o texto da fonte quem
    # entrega e o pesquisador -- modelo que sai buscando sozinho quebra a
    # ancoragem.
    groq_model: str = "openai/gpt-oss-120b"
    # Vazio nao envia o parametro. Os `openai/gpt-oss-*` aceitam low/medium/high;
    # mandar isso para modelo que nao suporta devolve 400, e o caminho verificado
    # em 18/09/2026 foi sem o parametro.
    groq_reasoning_effort: str = ""
    # --- OpenRouter (principal desde 20/09/2026) ---
    # Chave com credito ($100 ate 03/2027). Virou a rota principal por dois
    # motivos do autor: a cota do Gemini no AI Studio disputa requisicao
    # com outras automacoes dele, e o endpoint pago do OpenRouter nao consome
    # aquela cota. Uso consciente: os modelos baratos primeiro na rota
    # (deepseek-v4-flash ~$0.04/M tokens, gemini-2.5-flash-lite ~$0.10/M), os
    # caros so como reserva -- um video sai por centavos de dolar.
    openrouter_api_key: str = ""
    # Padrao dos comandos manuais (`llm-health`, `write --provider openrouter`).
    openrouter_model: str = "deepseek/deepseek-v4-flash"
    # OpenRouter pede identificacao do app (ranking de apps, sem custo).
    openrouter_app_url: str = ""
    openrouter_app_title: str = "tiktok-viral-generator"
    # 60s nao bastavam: geracao de roteiro no free tier passa disso mesmo com o
    # raciocinio desligado, e o timeout caia no meio da chamada -- gastando a cota
    # sem receber a resposta.
    llm_timeout_s: float = 120.0
    # Rota por estagio do piloto automatico, sobrescrevendo a padrao de
    # `adapters/llm_factory.py`. JSON no .env, ex.:
    # AGENT_LLM_ROUTES='{"writer": "gemini:gemini-3.8-flash,groq:openai/gpt-oss-120b@low"}'
    llm_routes: dict[str, str] = {}

    # --- pesquisador ---
    # Cinco fontes cobrem um tema sem estourar a cota por minuto do free tier
    # (uma chamada de modelo por fonte).
    research_max_sources: int = 5
    research_max_facts_per_source: int = 4
    # Caracteres de cada pagina que vao no prompt. Pagina de noticia inteira e
    # cota gasta em menu e rodape.
    research_page_chars: int = 8000

    # --- publicador (TikTok Content Posting API, inbox, M4) ---
    # Inbox nao exige auditoria do app; o preco e que titulo, descricao e o
    # rotulo AIGC sao aplicados no app, nao pela API. Segredos so no .env.
    tiktok_client_key: str = ""
    tiktok_client_secret: str = ""
    tiktok_redirect_uri: str = ""
    tiktok_access_token: str = ""
    tiktok_refresh_token: str = ""
    # Chunk do PUT de bytes. < 5 MB sobe em 1 chunk; > 64 MB exige multiplos.
    tiktok_chunk_size: int = 10_000_000
    tiktok_timeout_s: float = 60.0

    # --- carrossel por API (opcional) ---
    # Clone de um repo com GitHub Pages + a URL publica dele, verificada como
    # prefixo no portal do TikTok. Vazio = carrossel sai como pacote manual.
    media_repo_dir: Path | None = None
    media_base_url: str = ""

    # --- piloto automatico ---
    # Push no celular via ntfy.sh (gratis, sem conta). Vazio = so aviso local.
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"
    # Minutos de antecedencia com que o timer dispara antes do horario do post.
    autopilot_lead_min: int = 35

    # --- armazenamento ---
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "output"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agent.db"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
