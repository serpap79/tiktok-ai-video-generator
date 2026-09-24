"""Fuentes nuevas del radar: RSS agrupado por historia, Hugging Face, historia/archivo."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent.curator import niche
from agent.radar.sources.arquivo import ARCHIVO, Archivo, WikipediaOnThisDay
from agent.radar.sources.huggingface import HuggingFaceTrending, display_name
from agent.radar.sources.rss import Feed, cluster, parse_feed

AHORA = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def rss(*items: tuple[str, str, datetime]) -> bytes:
    cuerpo = "".join(
        f"<item><title>{t}</title><link>{u}</link>"
        f"<pubDate>{d.strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate></item>"
        for t, u, d in items)
    return f"<rss><channel>{cuerpo}</channel></rss>".encode()


class TestRss:
    def test_ventana_y_fecha_obligatoria(self):
        feed = Feed("x", "https://x", "rss_tech")
        items = parse_feed(rss(
            ("OpenAI lanza GPT-6", "https://x.com/1", AHORA - timedelta(hours=2)),
            ("Noticia vieja", "https://x.com/2", AHORA - timedelta(days=5)),
        ), feed, now=AHORA)
        assert [i.title for i in items] == ["OpenAI lanza GPT-6"]

    def test_misma_historia_en_varios_medios_se_convierte_en_una_senal(self):
        a, b, c = (Feed(n, f"https://{n}", "rss_tech_br") for n in ("tecnoblog", "canaltech",
                                                                    "olhar"))
        items = (parse_feed(rss(("OpenAI lanza GPT-6 con razonamiento", "https://t.com/1",
                                  AHORA - timedelta(hours=3))), a, now=AHORA)
                 + parse_feed(rss(("El GPT-6 de OpenAI llega hoy", "https://c.com/1",
                                   AHORA - timedelta(hours=2))), b, now=AHORA)
                 + parse_feed(rss(("Una consola nueva llega a las tiendas", "https://o.com/1",
                                   AHORA - timedelta(hours=1))), c, now=AHORA))
        senales = cluster(items, now=AHORA)
        gpt = next(s for s in senales if "GPT-6" in s.term)
        assert gpt.volume == 2 and gpt.unit == "medios"
        assert gpt.velocity == round(2 / 3, 3)
        assert [str(n.url) for n in gpt.news_items] == ["https://c.com/1"]
        assert len(senales) == 2

    def test_atom_tambien_se_lee(self):
        atom = (b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Chip nuevo</title>'
                b'<link href="https://v.com/a"/><published>2026-09-20T10:00:00Z</published>'
                b'</entry></feed>')
        (item,) = parse_feed(atom, Feed("verge", "https://v", "rss_tech"), now=AHORA)
        assert item.url == "https://v.com/a"


class TestHuggingFace:
    def test_variante_cuantizada_es_el_mismo_modelo(self):
        assert display_name("prism-ml/Ternary-Bonsai-2-27B-gguf") == "Ternary Bonsai 2 27B"
        assert display_name("Qwen/Qwen3.8-27B") == "Qwen3.8 27B"

    def test_filtra_pocos_me_gustos_viejo_y_duplicado(self):
        modelos = [
            {"id": "a/Modelo-X", "likes": 900, "trendingScore": 500,
             "createdAt": "2026-09-10T00:00:00Z"},
            {"id": "b/Modelo-X-GGUF", "likes": 300, "trendingScore": 400,
             "createdAt": "2026-09-11T00:00:00Z"},
            {"id": "c/Pequeño", "likes": 3, "trendingScore": 300,
             "createdAt": "2026-09-11T00:00:00Z"},
            {"id": "d/Antiguo", "likes": 5000, "trendingScore": 200,
             "createdAt": "2025-01-01T00:00:00Z"},
        ]
        senales = HuggingFaceTrending.parse(modelos, now=AHORA)
        assert [s.term for s in senales] == ["Modelo X (a)"]
        assert senales[0].velocity == 500 and str(senales[0].url).endswith("a/Modelo-X")

    def test_hugging_face_es_nicho_por_construccion(self):
        assert niche.fit("Modelo X (a)", "huggingface") >= niche.UMBRAL_DEFECTO


class TestHistoria:
    def test_un_dia_como_hoy_se_convierte_en_aniversario_con_pagina(self):
        eventos = [{"text": "Lanzado el primer microprocesador comercial.", "year": 1971,
                    "pages": [{"content_urls": {"desktop": {
                        "page": "https://es.wikipedia.org/wiki/Intel_4004"}}}]}]
        (s,) = WikipediaOnThisDay.parse(eventos, now=AHORA)
        assert s.term.startswith("Hace 55 anos:") and s.unit == "anos"

    def test_archivo_rota_por_dia_y_es_reproducible(self):
        a = Archivo().pick(AHORA)
        assert [s.term for s in a] == [s.term for s in Archivo().pick(AHORA)]
        otro = Archivo().pick(AHORA + timedelta(days=1))
        assert [s.term for s in a] != [s.term for s in otro]
        assert all(str(s.url).startswith("https://es.wikipedia.org/") for s in a)

    def test_archivo_pasa_la_puerta_de_nicho(self):
        for tema, _ in ARCHIVO:
            assert niche.fit(tema, "archivo") >= niche.UMBRAL_DEFECTO, tema


class TestFuentesRelacionadas:
    def test_mismo_nombre_en_otro_idioma_se_convierte_en_fuente(self):
        from agent.models import Decision, Signal, Verdict
        from agent.research.related import names, related_items

        assert {"gemini", "google"} <= names("Google's Gemini went rogue and hacked companies")
        decision = Decision(term="Gemini went rogue, hacked three companies, and Google hid it",
                            source="rss_tech", verdict=Verdict.selected, reason="prueba",
                            score=0.9,
                            niche_fit=0.8, decided_at=AHORA, url="https://theverge.com/a")
        senales = [
            Signal(term="Gemini, de Google, irrumpió en tres empresas en prueba",
                   source="rss_tech_br",
                   volume=1, unit="medios", seen_at=AHORA, url="https://tecnoblog.net/b"),
            Signal(term="Google lanza un nuevo Pixel", source="rss_tech_br", volume=1,
                   unit="medios", seen_at=AHORA, url="https://tecnoblog.net/c"),
            Signal(term="Gemini hack discussion", source="hacker_news", volume=300,
                   unit="points", seen_at=AHORA, url="https://news.ycombinator.com/item?id=1"),
        ]
        (item,) = related_items(decision, senales)
        assert str(item.url) == "https://tecnoblog.net/b"

    def test_nombre_en_clave_con_digito_basta_solo(self):
        from agent.models import Decision, Signal, Verdict
        from agent.research.related import related_items

        decision = Decision(term="Qwen3.8 27B (Qwen)", source="huggingface",
                            verdict=Verdict.selected, reason="prueba", score=0.9,
                            niche_fit=0.8,
                            decided_at=AHORA, url="https://huggingface.co/Qwen/Qwen3.8-27B")
        senales = [Signal(term="Alibaba libera el Qwen3.8 para funcionar en casa",
                          source="rss_tech_br",
                          volume=1, unit="medios", seen_at=AHORA, url="https://x.com/q")]
        assert [str(i.url) for i in related_items(decision, senales)] == ["https://x.com/q"]
