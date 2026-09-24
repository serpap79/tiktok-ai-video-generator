"""Tests de las tres puertas del curador: política, nicho y duplicado."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.curator import niche, policy
from agent.curator.dedup import LexicalDeduplicator, jaccard
from agent.ports.dedup import Deduplicator
from agent.radar.sources.hacker_news import HackerNews
from agent.text import content_tokens, normalize

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "radar"


class TestPolitica:
    def test_pluma_de_adelgazamiento_bloqueada(self):
        """Caso obligatorio del plan. Fue el tema con más tráfico en Google
        Trends ES en 17/09/2026: salud, medicamento y política a la vez."""
        v = policy.check("Sanchez defiende la pluma de adelgazamiento gratis en la Sanidad")
        assert not v.allowed

    @pytest.mark.parametrize("termino", [
        "Tribunal Constitucional",
        "Pedro Sanchez",
        "Feijoo",
        "Encuestas de opinion para las elecciones generales",
        "Lista de magistrados del Tribunal Constitucional",
    ])
    def test_politica_partidista_real_del_radar(self, termino):
        """Todos aparecieron en lo alto de la Wikipedia es en una colecta real."""
        assert not policy.check(termino).allowed

    @pytest.mark.parametrize("termino", [
        "Accidente aereo deja victimas",
        "Científico muere a los 90 años",
        "Shooting at tech conference",
    ])
    def test_tragedia_con_victima_bloqueada(self, termino):
        assert not policy.check(termino).allowed

    def test_acento_no_escapa_de_la_regla(self):
        """Las reglas se escriben sin acento y el texto se normaliza antes; sin
        eso, "elección" pasaría por no casar con "eleccion"."""
        assert normalize("elección presidencial") == "eleccion presidencial"
        assert not policy.check("elección presidencial").allowed

    @pytest.mark.parametrize("termino", [
        "Bonsai 2 27B: Near-Lossless Compression in a 9x Smaller Footprint",
        "NASA lanza telescopio para observar exoplanetas",
        "Qwen 3.8 Omni Flash",
    ])
    def test_tema_del_nicho_pasa_libre(self, termino):
        assert policy.check(termino).allowed

    def test_frontera_de_palabra_evita_casamiento_dentro_de_otra(self):
        """"muerte" no puede casar dentro de "mortero" ni "ia" dentro de "media"."""
        assert policy.check("Nuevo compilador para arquitectura importante").allowed

    def test_limitacion_conocida_del_filtro_lexico(self):
        """El filtro es léxico: no hay lista de apodos personales.

        Un tema legítimo de biología marina con nombre de persona o apodo
        ambiguo puede colarse. El error es asimétrico a propósito: dejar pasar
        política partidista cuesta mucho más que un falso negativo puntual.
        """
        # "Elon Musk lanza robotaxi" pasa porque no casa ninguna regla de
        # política (no es cargo ni partido): el nicho y el juez filtran el
        # resto. Documentado, no ignorado.
        assert policy.check("Calamar gigante grabado a 900 metros").allowed


class TestNicho:
    @pytest.fixture
    def titulos_hn(self) -> list:
        payload = json.loads((FIXTURES / "hacker_news.json").read_text(encoding="utf-8"))
        return HackerNews.parse(payload, now=datetime.now(UTC))

    def test_ai_e_ia_sobreviven_a_la_tokenizacion(self):
        """Regresión: `content_tokens` cortaba tokens con menos de 3 caracteres,
        lo que mataba "ai" e "ia" -- los dos términos más centrales del léxico
        -- antes de llegar a la comparación. El nicho tokeniza con suelo 2."""
        assert "ai" in content_tokens("Microsoft exec called AI scraping", min_len=2)
        assert "ai" not in content_tokens("Microsoft exec called AI scraping", min_len=3)
        fit = niche.fit("Microsoft exec called AI scraping")
        assert fit >= niche.UMBRAL_DEFECTO

    def test_termino_de_nucleo_solo_aprueba(self):
        """"NASA" es asunto del canal aunque no haya más pista en el título."""
        assert niche.fit("NASA") >= niche.UMBRAL_DEFECTO

    def test_solo_termino_de_apoyo_no_aprueba(self):
        """"lanzamiento récord" puede ser de fútbol."""
        assert niche.fit("Lanzamiento bate record de publico") < niche.UMBRAL_DEFECTO

    def test_prior_de_la_fuente_es_suelo_y_no_pase_libre(self):
        """Hacker News está curado por asunto, así que sus títulos ganan
        ventaja inicial -- pero por debajo del umbral, para que un título sin
        ninguna señal técnica siga cortándose."""
        termino = "Warren Buffett Steps Down as Berkshire Chairman"
        assert niche.fit(termino, "hacker_news") == niche.PRIOR_POR_FUENTE["hacker_news"]
        assert niche.fit(termino, "hacker_news") < niche.UMBRAL_DEFECTO
        assert niche.fit(termino, "wikipedia") == 0.0

    def test_prior_no_rebaja_un_encaje_lexico_alto(self):
        sin_fuente = niche.fit("GPU quantization benchmark")
        assert niche.fit("GPU quantization benchmark", "hacker_news") == sin_fuente

    def test_calibracion_contra_titulos_reales(self, titulos_hn):
        """Traba la calibración de la puerta contra los 20 títulos reales
        capturados.

        Antes de la corrección del suelo de token y del prior por fuente, solo
        2 de 20 pasaban -- incluido el "Bonsai 2 27B", que es el tema del
        fixture de guion del M0. Este test rompe si un cambio del léxico hace
        retroceder eso.
        """
        aprobados = {
            s.term for s in titulos_hn if niche.fit(s.term, s.source) >= niche.UMBRAL_DEFECTO
        }
        assert len(aprobados) >= 14, f"recall cayó a {len(aprobados)}/20"

        deben_pasar = [
            "Bonsai 2 27B", "Qwen 3.8", "Coding Agents", "passkeys", "x86 emulation",
        ]
        for trozo in deben_pasar:
            assert any(trozo in t for t in aprobados), f"{trozo!r} debería pasar"

        deben_cortar = ["Warren Buffett", "product decision"]
        for trozo in deben_cortar:
            assert not any(trozo in t for t in aprobados), f"{trozo!r} debería cortarse"

    def test_matched_terms_justifica_la_decision(self):
        nucleo, _ = niche.matched_terms("Bonsai 2 27B: Near-Lossless Compression")
        assert "compression" in nucleo


class TestDeduplicacion:
    def test_implementacion_satisface_el_puerto(self):
        assert isinstance(LexicalDeduplicator(), Deduplicator)

    def test_misma_historia_reformulada_se_caza(self):
        dedup = LexicalDeduplicator()
        anterior = "Bonsai 2 27B: Near-Lossless Compression in a 9x Smaller Footprint"
        hallado = dedup.find_duplicate(
            "Bonsai 2 27B Near Lossless Compression Footprint", [anterior]
        )
        assert hallado is not None
        original, sim = hallado
        assert original == anterior
        assert sim >= 0.45

    def test_asunto_nuevo_no_es_duplicado(self):
        dedup = LexicalDeduplicator()
        assert dedup.find_duplicate(
            "NASA lanza telescopio para observar exoplanetas",
            ["Bonsai 2 27B: Near-Lossless Compression"],
        ) is None

    def test_vocabulario_generico_compartido_no_basta(self):
        """Dos temas de IA distintos comparten "modelo" e "IA" y aun así son
        asuntos distintos."""
        dedup = LexicalDeduplicator()
        assert dedup.find_duplicate(
            "Nuevo modelo de IA de Google supera benchmark de codigo",
            ["Nuevo modelo de IA de Anthropic reduce coste de inferencia"],
        ) is None

    def test_elige_el_duplicado_mas_parecido(self):
        dedup = LexicalDeduplicator()
        anteriores = [
            "Bonsai 2 comprime modelo",
            "Bonsai 2 27B Near Lossless Compression Smaller Footprint",
        ]
        original, _ = dedup.find_duplicate(
            "Bonsai 2 27B Near-Lossless Compression in a Smaller Footprint", anteriores
        )
        assert original == anteriores[1]

    def test_limitacion_conocida_parafrasis_sin_palabra_en_comun(self):
        """Lo que la deduplicación léxica NO caza, documentado a propósito.

        Esta es la laguna que justificaría embeddings. Por estar detrás del
        puerto Deduplicator, cambiar la técnica y medir contra esta misma base
        es barato -- y es el tipo de evidencia que produce el M5.
        """
        dedup = LexicalDeduplicator()
        assert dedup.find_duplicate(
            "PrismML reduce la huella nueve veces",
            ["Bonsai 2 27B: compresión casi sin pérdida"],
        ) is None

    def test_jaccard_lida_con_conjunto_vacio(self):
        assert jaccard(set(), {"a"}) == 0.0
        assert jaccard({"a"}, set()) == 0.0


class TestContenidoComercial:
    """Guía de compra y producto financiero no son pauta del canal (radar del 19/09)."""

    def test_seguro_y_promocion_bloquean_con_motivo(self):
        from agent.curator import policy
        v = policy.check("Seguro para movil en 2026: que planes cubren el hurto de datos y Bizum?")
        assert not v.allowed and v.rule == "comercial"
        assert not policy.check("Black Friday: mejores descuentos en portatiles").allowed

    def test_seguridad_digital_sigue_siendo_pauta(self):
        from agent.curator import policy
        assert policy.check("Estafa del SIM swap: como la IA detecta el fraude en segundos").allowed
