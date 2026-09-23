"""Testes dos tres portoes do curador: politica, nicho e duplicata."""

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
    def test_caneta_emagrecedora_e_bloqueada(self):
        """Caso obrigatorio do plano. Foi o tema com mais trafego no Google
        Trends BR em 17/09/2026: saude, medicamento e politica de uma vez."""
        v = policy.check("Lula defende caneta emagrecedora de graca no SUS")
        assert not v.allowed

    @pytest.mark.parametrize("termo", [
        "Supremo Tribunal Federal",
        "Luiz Inácio Lula da Silva",
        "Flávio Bolsonaro",
        "Pesquisas de opinião para a eleição presidencial",
        "Lista de ministros do Supremo Tribunal Federal",
    ])
    def test_politica_partidaria_real_do_radar(self, termo):
        """Todos apareceram no topo da Wikipedia pt numa coleta real."""
        assert not policy.check(termo).allowed

    @pytest.mark.parametrize("termo", [
        "Acidente aereo deixa vitimas",
        "Cientista morre aos 90 anos",
        "Shooting at tech conference",
    ])
    def test_tragedia_com_vitima_e_bloqueada(self, termo):
        assert not policy.check(termo).allowed

    def test_acento_nao_escapa_da_regra(self):
        """As regras sao escritas sem acento e o texto e normalizado antes; sem
        isso, "eleição" passaria por nao casar com "eleicao"."""
        assert normalize("eleição presidencial") == "eleicao presidencial"
        assert not policy.check("eleição presidencial").allowed

    @pytest.mark.parametrize("termo", [
        "Bonsai 2 27B: Near-Lossless Compression in a 9x Smaller Footprint",
        "NASA lanca telescopio para observar exoplanetas",
        "Qwen 3.8 Omni Flash",
    ])
    def test_tema_do_nicho_passa_livre(self, termo):
        assert policy.check(termo).allowed

    def test_fronteira_de_palavra_evita_casamento_dentro_de_outra(self):
        """"morte" nao pode casar dentro de "mortero" nem "cpi" dentro de "cpim"."""
        assert policy.check("Novo compilador para arquitetura importante").allowed

    def test_limitacao_conhecida_lula_tambem_e_molusco(self):
        """Em portugues "lula" e presidente e tambem e o animal.

        Um tema legitimo de biologia marinha ("lula gigante") e bloqueado por
        engano. O erro e assimetrico de proposito: deixar passar politica
        partidaria num canal ligado ao nome do autor custa muito mais caro que
        perder um video sobre cefalopodes.
        """
        assert not policy.check("Lula gigante e filmada a 900 metros").allowed


class TestNicho:
    @pytest.fixture
    def titulos_hn(self) -> list:
        payload = json.loads((FIXTURES / "hacker_news.json").read_text(encoding="utf-8"))
        return HackerNews.parse(payload, now=datetime.now(UTC))

    def test_ai_e_ia_sobrevivem_a_tokenizacao(self):
        """Regressao: `content_tokens` cortava tokens com menos de 3 caracteres,
        o que matava "ai" e "ia" -- os dois termos mais centrais do lexico --
        antes de chegarem na comparacao. O nicho tokeniza com piso 2."""
        assert "ai" in content_tokens("Microsoft exec called AI scraping", min_len=2)
        assert "ai" not in content_tokens("Microsoft exec called AI scraping", min_len=3)
        fit = niche.fit("Microsoft exec called AI scraping")
        assert fit >= niche.LIMIAR_PADRAO

    def test_termo_de_nucleo_sozinho_aprova(self):
        """"NASA" e assunto do canal mesmo sem mais nenhuma pista no titulo."""
        assert niche.fit("NASA") >= niche.LIMIAR_PADRAO

    def test_so_termo_de_apoio_nao_aprova(self):
        """"lancamento recorde" pode ser de futebol."""
        assert niche.fit("Lancamento bate recorde de publico") < niche.LIMIAR_PADRAO

    def test_prior_da_fonte_e_piso_e_nao_passe_livre(self):
        """O Hacker News e curado por assunto, entao seus titulos ganham
        vantagem inicial -- mas abaixo do limiar, para que um titulo sem nenhum
        sinal tecnico continue sendo cortado."""
        termo = "Warren Buffett Steps Down as Berkshire Chairman"
        assert niche.fit(termo, "hacker_news") == niche.PRIOR_POR_FONTE["hacker_news"]
        assert niche.fit(termo, "hacker_news") < niche.LIMIAR_PADRAO
        assert niche.fit(termo, "wikipedia") == 0.0

    def test_prior_nao_rebaixa_um_encaixe_lexico_alto(self):
        sem_fonte = niche.fit("GPU quantization benchmark")
        assert niche.fit("GPU quantization benchmark", "hacker_news") == sem_fonte

    def test_calibracao_contra_titulos_reais(self, titulos_hn):
        """Trava a calibracao do portao contra os 20 titulos reais capturados.

        Antes da correcao do piso de token e do prior por fonte, apenas 2 dos 20
        passavam -- inclusive o "Bonsai 2 27B", que e o tema do fixture de
        roteiro do M0. Este teste quebra se uma mudanca no lexico regredir isso.
        """
        aprovados = {
            s.term for s in titulos_hn if niche.fit(s.term, s.source) >= niche.LIMIAR_PADRAO
        }
        assert len(aprovados) >= 14, f"recall caiu para {len(aprovados)}/20"

        devem_passar = [
            "Bonsai 2 27B", "Qwen 3.8", "Coding Agents", "passkeys", "x86 emulation",
        ]
        for trecho in devem_passar:
            assert any(trecho in t for t in aprovados), f"{trecho!r} deveria passar"

        devem_cortar = ["Warren Buffett", "product decision"]
        for trecho in devem_cortar:
            assert not any(trecho in t for t in aprovados), f"{trecho!r} deveria ser cortado"

    def test_matched_terms_justifica_a_decisao(self):
        nucleo, _ = niche.matched_terms("Bonsai 2 27B: Near-Lossless Compression")
        assert "compression" in nucleo


class TestDeduplicacao:
    def test_implementacao_satisfaz_a_porta(self):
        assert isinstance(LexicalDeduplicator(), Deduplicator)

    def test_mesma_historia_reformulada_e_pega(self):
        dedup = LexicalDeduplicator()
        anterior = "Bonsai 2 27B: Near-Lossless Compression in a 9x Smaller Footprint"
        achado = dedup.find_duplicate(
            "Bonsai 2 27B Near Lossless Compression Footprint", [anterior]
        )
        assert achado is not None
        original, sim = achado
        assert original == anterior
        assert sim >= 0.45

    def test_assunto_novo_nao_e_duplicata(self):
        dedup = LexicalDeduplicator()
        assert dedup.find_duplicate(
            "NASA lanca telescopio para observar exoplanetas",
            ["Bonsai 2 27B: Near-Lossless Compression"],
        ) is None

    def test_vocabulario_generico_compartilhado_nao_basta(self):
        """Dois temas de IA diferentes compartilham "modelo" e "IA" e ainda assim
        sao assuntos distintos."""
        dedup = LexicalDeduplicator()
        assert dedup.find_duplicate(
            "Novo modelo de IA da Google supera benchmark de codigo",
            ["Novo modelo de IA da Anthropic reduz custo de inferencia"],
        ) is None

    def test_escolhe_a_duplicata_mais_parecida(self):
        dedup = LexicalDeduplicator()
        anteriores = [
            "Bonsai 2 comprime modelo",
            "Bonsai 2 27B Near Lossless Compression Smaller Footprint",
        ]
        original, _ = dedup.find_duplicate(
            "Bonsai 2 27B Near-Lossless Compression in a Smaller Footprint", anteriores
        )
        assert original == anteriores[1]

    def test_limitacao_conhecida_parafrase_sem_palavra_em_comum(self):
        """O que a deduplicacao lexica NAO pega, documentado de proposito.

        Esta e a lacuna que justificaria embeddings. Por estar atras da porta
        Deduplicator, trocar a tecnica e medir contra esta mesma base e barato --
        e e o tipo de evidencia que o M5 produz.
        """
        dedup = LexicalDeduplicator()
        assert dedup.find_duplicate(
            "PrismML reduz footprint em nove vezes",
            ["Bonsai 2 27B: compressao quase sem perda"],
        ) is None

    def test_jaccard_lida_com_conjunto_vazio(self):
        assert jaccard(set(), {"a"}) == 0.0
        assert jaccard({"a"}, set()) == 0.0


class TestConteudoComercial:
    """Guia de compra e produto financeiro nao sao pauta do canal (radar de 19/09)."""

    def test_seguro_e_promocao_bloqueiam_com_motivo(self):
        from agent.curator import policy
        v = policy.check("Seguro para celular em 2026: quais planos cobrem furto de dados e Pix?")
        assert not v.allowed and v.rule == "comercial"
        assert not policy.check("Black Friday: melhores descontos em notebooks").allowed

    def test_seguranca_digital_continua_pauta(self):
        from agent.curator import policy
        assert policy.check("Golpe do Pix: como a IA detecta fraude em segundos").allowed
