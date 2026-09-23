"""Fontes novas do radar: RSS agrupado por historia, Hugging Face, historia/arquivo."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent.curator import niche
from agent.radar.sources.arquivo import ARQUIVO, Arquivo, WikipediaOnThisDay
from agent.radar.sources.huggingface import HuggingFaceTrending, display_name
from agent.radar.sources.rss import Feed, cluster, parse_feed

AGORA = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def rss(*itens: tuple[str, str, datetime]) -> bytes:
    corpo = "".join(
        f"<item><title>{t}</title><link>{u}</link>"
        f"<pubDate>{d.strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate></item>"
        for t, u, d in itens)
    return f"<rss><channel>{corpo}</channel></rss>".encode()


class TestRss:
    def test_janela_e_data_obrigatoria(self):
        feed = Feed("x", "https://x", "rss_tech")
        itens = parse_feed(rss(
            ("OpenAI lança GPT-6", "https://x.com/1", AGORA - timedelta(hours=2)),
            ("Notícia velha", "https://x.com/2", AGORA - timedelta(days=5)),
        ), feed, now=AGORA)
        assert [i.title for i in itens] == ["OpenAI lança GPT-6"]

    def test_mesma_historia_em_varios_veiculos_vira_um_sinal(self):
        a, b, c = (Feed(n, f"https://{n}", "rss_tech_br") for n in ("tecnoblog", "canaltech",
                                                                   "olhar"))
        itens = (parse_feed(rss(("OpenAI lança GPT-6 com raciocínio", "https://t.com/1",
                                 AGORA - timedelta(hours=3))), a, now=AGORA)
                 + parse_feed(rss(("GPT-6 da OpenAI chega hoje", "https://c.com/1",
                                   AGORA - timedelta(hours=2))), b, now=AGORA)
                 + parse_feed(rss(("Console novo chega às lojas", "https://o.com/1",
                                   AGORA - timedelta(hours=1))), c, now=AGORA))
        sinais = cluster(itens, now=AGORA)
        gpt = next(s for s in sinais if "GPT-6" in s.term)
        assert gpt.volume == 2 and gpt.unit == "veiculos"
        assert gpt.velocity == round(2 / 3, 3)
        assert [str(n.url) for n in gpt.news_items] == ["https://c.com/1"]
        assert len(sinais) == 2

    def test_atom_tambem_e_lido(self):
        atom = (b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Chip novo</title>'
                b'<link href="https://v.com/a"/><published>2026-09-20T10:00:00Z</published>'
                b'</entry></feed>')
        (item,) = parse_feed(atom, Feed("verge", "https://v", "rss_tech"), now=AGORA)
        assert item.url == "https://v.com/a"


class TestHuggingFace:
    def test_variante_quantizada_vira_o_mesmo_modelo(self):
        assert display_name("prism-ml/Ternary-Bonsai-2-27B-gguf") == "Ternary Bonsai 2 27B"
        assert display_name("Qwen/Qwen3.8-27B") == "Qwen3.8 27B"

    def test_filtra_pouco_curtido_velho_e_duplicado(self):
        modelos = [
            {"id": "a/Modelo-X", "likes": 900, "trendingScore": 500,
             "createdAt": "2026-09-10T00:00:00Z"},
            {"id": "b/Modelo-X-GGUF", "likes": 300, "trendingScore": 400,
             "createdAt": "2026-09-11T00:00:00Z"},
            {"id": "c/Pequeno", "likes": 3, "trendingScore": 300,
             "createdAt": "2026-09-11T00:00:00Z"},
            {"id": "d/Antigo", "likes": 5000, "trendingScore": 200,
             "createdAt": "2025-01-01T00:00:00Z"},
        ]
        sinais = HuggingFaceTrending.parse(modelos, now=AGORA)
        assert [s.term for s in sinais] == ["Modelo X (a)"]
        assert sinais[0].velocity == 500 and str(sinais[0].url).endswith("a/Modelo-X")

    def test_hugging_face_e_nicho_por_construcao(self):
        assert niche.fit("Modelo X (a)", "huggingface") >= niche.LIMIAR_PADRAO


class TestHistoria:
    def test_neste_dia_vira_aniversario_com_a_pagina(self):
        eventos = [{"text": "Lançado o primeiro microprocessador comercial.", "year": 1971,
                    "pages": [{"content_urls": {"desktop": {
                        "page": "https://pt.wikipedia.org/wiki/Intel_4004"}}}]}]
        (s,) = WikipediaOnThisDay.parse(eventos, now=AGORA)
        assert s.term.startswith("Ha 55 anos:") and s.unit == "anos"

    def test_arquivo_gira_por_dia_e_e_reproduzivel(self):
        a = Arquivo().pick(AGORA)
        assert [s.term for s in a] == [s.term for s in Arquivo().pick(AGORA)]
        outro = Arquivo().pick(AGORA + timedelta(days=1))
        assert [s.term for s in a] != [s.term for s in outro]
        assert all(str(s.url).startswith("https://pt.wikipedia.org/") for s in a)

    def test_arquivo_passa_no_portao_de_nicho(self):
        for tema, _ in ARQUIVO:
            assert niche.fit(tema, "arquivo") >= niche.LIMIAR_PADRAO, tema


class TestFontesRelacionadas:
    def test_mesmo_nome_em_outro_idioma_vira_fonte(self):
        from agent.models import Decision, Signal, Verdict
        from agent.research.related import names, related_items

        assert {"gemini", "google"} <= names("Google's Gemini went rogue and hacked companies")
        decisao = Decision(term="Gemini went rogue, hacked three companies, and Google hid it",
                           source="rss_tech", verdict=Verdict.selected, reason="teste", score=0.9,
                           niche_fit=0.8, decided_at=AGORA, url="https://theverge.com/a")
        sinais = [
            Signal(term="Gemini, do Google, invadiu três empresas em teste", source="rss_tech_br",
                   volume=1, unit="veiculos", seen_at=AGORA, url="https://tecnoblog.net/b"),
            Signal(term="Google lança novo Pixel", source="rss_tech_br", volume=1,
                   unit="veiculos", seen_at=AGORA, url="https://tecnoblog.net/c"),
            Signal(term="Gemini hack discussion", source="hacker_news", volume=300,
                   unit="points", seen_at=AGORA, url="https://news.ycombinator.com/item?id=1"),
        ]
        (item,) = related_items(decisao, sinais)
        assert str(item.url) == "https://tecnoblog.net/b"

    def test_codinome_com_digito_basta_sozinho(self):
        from agent.models import Decision, Signal, Verdict
        from agent.research.related import related_items

        decisao = Decision(term="Qwen3.8 27B (Qwen)", source="huggingface",
                           verdict=Verdict.selected, reason="teste", score=0.9, niche_fit=0.8,
                           decided_at=AGORA, url="https://huggingface.co/Qwen/Qwen3.8-27B")
        sinais = [Signal(term="Alibaba libera o Qwen3.8 para rodar em casa", source="rss_tech_br",
                         volume=1, unit="veiculos", seen_at=AGORA, url="https://x.com/q")]
        assert [str(i.url) for i in related_items(decisao, sinais)] == ["https://x.com/q"]
