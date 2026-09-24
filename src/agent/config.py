"""Configuracion del agente, leida de variables de entorno o .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fuente de MoneyPrinterTurbo que cubre los acentos del castellano. Las otras
# que trae su repo son chinas (STHeiti, MicrosoftYaHei) o vietnamitas, y
# renderizan tofu en lugar de la "ñ" y los acentos. Verificado en la cmap.
FONT_ES = "BeVietnamPro-Bold.ttf"

# Voces gratuitas del edge-tts para Espana (sin clave, sin cuenta).
VOICES_ES = (
    "es-ES-XimenaMultilingualNeural",
    "es-ES-ElviraNeural",
    "es-ES-AlvaroNeural",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AGENT_", extra="ignore"
    )

    # --- renderizador (MoneyPrinterTurbo en local) ---
    renderer_url: str = "http://127.0.0.1:8080"
    renderer_api_key: str = ""
    renderer_dir: Path = PROJECT_ROOT / ".renderer"
    # Un render de 60-90s en CPU tarda minutos; el timeout cubre el peor caso.
    renderer_timeout_s: float = 900.0
    renderer_poll_interval_s: float = 5.0

    # --- material visual ---
    # "pexels" en produccion; "local" para probar la cadena de produccion (TTS,
    # subtitulo, montaje) sin depender de clave de API ni de red.
    video_source: str = "pexels"
    local_materials: list[Path] = []
    # Clave del Pexels con prefijo del proyecto (el .env heredado del MPT usa
    # PEXELS_API_KEY sin prefijo; ambos valen, el que tiene prefijo primero).
    pexels_api_key: str = ""

    # --- produccion ---
    # "ffmpeg": renderizador propio (narracion con pronuncia corregida, subtitulo
    # de la marca, ~20s por video). "mpt": el MoneyPrinterTurbo, de reserva.
    renderer: str = "ffmpeg"
    # Voces del renderizador propio: monolingues es-ES. La Multilingual (por
    # defecto en el MPT) cambia el acento en medio de la frase.
    narrator_voice: str = "es-ES-ElviraNeural"
    narrator_voice_male: str = "es-ES-AlvaroNeural"
    voice_name: str = VOICES_ES[0]
    font_name: str = FONT_ES
    # Subtitulo palabra a palabra: una palabra cada vez necesita ser grande y
    # tener contorno grueso para leerse sobre un clip claro. Posicion "custom"
    # al 60% de la altura: por debajo de la tarjeta del gancho (tercio superior)
    # y por encima de la franja que la interfaz de TikTok cubre (leyenda,
    # perfil, musica).
    font_size: int = 84
    subtitle_position: str = "custom"
    subtitle_custom_position: float = 60.0
    subtitle_stroke_width: float = 3.0

    # --- LLM (primer modelo del proyecto, M3) ---
    # Bajo la restriccion de $0 el estandar no es Claude: es free tier. El
    # adaptador de Claude existe para el brazo de pago del eval del M5 -- la
    # comparacion medida es el artefacto, no el modelo elegido.
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    # Los ids de modelo de free tier cambian y se descontinuan sin aviso. Son
    # configurables, y `agent llm-health` comprueba contra el proveedor en vez
    # de confiar en que el estandar todavia exista.
    gemini_model: str = "gemini-2.5-flash"
    # El 2.5 Flash razona por defecto, y el razonamiento sale del MISMO
    # presupuesto de salida y del mismo reloj. Medido en 18/09/2026, con
    # thinking activado: una llamada del guionista trunco el JSON a mitad
    # (el objeto abrio y no cerro) y otra rebaso 60s de lectura. Cero lo
    # desactiva. No es ahorro de token: es lo que hace la respuesta lo
    # bastante previsible para validarse por contrato.
    # Negativo deja decidir al proveedor (dinamico); si un dia hubiera
    # evidencia de que el razonamiento mejora la rubrica, eso se convierte en
    # experimento del M5 y no en corazonada.
    gemini_thinking_budget: int = 0
    groq_api_key: str = ""
    # Medido en 18/09/2026: el `llama-3.3-70b-versatile`, que era el estandar
    # obvio, salio del catalogo de Groq (404 model_not_found) y ya no queda
    # ningun Llama de chat alli. Entre los que existen, este es el mas fuerte
    # de uso general -- lo que importa porque Groq es el brazo que el eval del
    # M5 compara contra el de pago.
    # Alternativa mas rapida y barata: "qwen/qwen3.8-27b" (0,5s frente a 1,0s,
    # y menos de la mitad de los tokens). Los `groq/compound*` quedan FUERA a
    # proposito: son sistemas agenticos con busqueda web incluida, y aqui el
    # texto de la fuente lo entrega el investigador -- un modelo que sale a
    # buscar solo rompe el anclaje.
    groq_model: str = "openai/gpt-oss-120b"
    # Vacio no envia el parametro. Los `openai/gpt-oss-*` aceptan low/medium/high;
    # mandarlo a un modelo que no lo soporta devuelve 400, y el camino verificado
    # en 18/09/2026 fue sin el parametro.
    groq_reasoning_effort: str = ""
    # --- OpenRouter (principal desde 20/09/2026) ---
    # Clave con credito ($100 hasta 03/2027). Se convirtio en la ruta principal
    # por dos motivos del autor: la cuota de Gemini en AI Studio disputa cada
    # peticion con otras automatizaciones, y el endpoint de pago de OpenRouter
    # no consume esa cuota. Uso consciente: los modelos baratos primero en la
    # ruta (deepseek-v4-flash ~$0.04/M tokens, gemini-2.5-flash-lite ~$0.10/M),
    # los caros solo de reserva -- un video sale por centimos de dolar.
    openrouter_api_key: str = ""
    # Estandar de los comandos manuales (`llm-health`, `write --provider openrouter`).
    openrouter_model: str = "deepseek/deepseek-v4-flash"
    # OpenRouter pide identificacion del app (ranking de apps, sin coste).
    openrouter_app_url: str = ""
    openrouter_app_title: str = "tiktok-viral-generator"
    # 60s no bastaban: la generacion de guion en free tier pasa de eso incluso
    # con el razonamiento desactivado, y el timeout caia en medio de la llamada
    # -- gastando la cuota sin recibir la respuesta.
    llm_timeout_s: float = 120.0
    # Ruta por etapa del piloto automatico, sobrescribiendo la estandar de
    # `adapters/llm_factory.py`. JSON en el .env, p. ej.:
    # AGENT_LLM_ROUTES='{"writer": "gemini:gemini-3.8-flash,groq:openai/gpt-oss-120b@low"}'
    llm_routes: dict[str, str] = {}

    # --- investigador ---
    # Cinco fuentes cubren un tema sin rebasar la cuota por minuto del free tier
    # (una llamada de modelo por fuente).
    research_max_sources: int = 5
    research_max_facts_per_source: int = 4
    # Caracteres de cada pagina que entran en el prompt. Pagina de noticia entera
    # es cuota gastada en menus y pie de pagina.
    research_page_chars: int = 8000

    # --- publicador (TikTok Content Posting API, inbox, M4) ---
    # La inbox no exige auditoria del app; el precio es que titulo, descripcion
    # y la etiqueta AIGC se aplican en la app, no por la API. Secretos solo en
    # el .env.
    tiktok_client_key: str = ""
    tiktok_client_secret: str = ""
    tiktok_redirect_uri: str = ""
    tiktok_access_token: str = ""
    tiktok_refresh_token: str = ""
    # Chunk del PUT de bytes. < 5 MB sube en 1 chunk; > 64 MB exige varios.
    tiktok_chunk_size: int = 10_000_000
    tiktok_timeout_s: float = 60.0

    # --- carrusel por API (opcional) ---
    # Clon de un repo con GitHub Pages + su URL publica, verificada como prefijo
    # en el portal de TikTok. Vacio = el carrusel sale como paquete manual.
    media_repo_dir: Path | None = None
    media_base_url: str = ""

    # --- piloto automatico ---
    # Push en el movil via ntfy.sh (gratis, sin cuenta). Vacio = solo aviso local.
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"
    # Minutos de antelacion con que el timer dispara antes de la hora del post.
    autopilot_lead_min: int = 35

    # --- almacenamiento ---
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "output"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agent.db"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
