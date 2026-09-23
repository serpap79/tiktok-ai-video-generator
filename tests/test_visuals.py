"""Vocabulario visual do canal: 4 pilares, tags fixas, sem termo generico.

O que este arquivo trava: toda tag e ASCII (vai direto ao Pexels), termo fora
do pool reprova com a lista do que vale, pilares misturados reprovam (um
roteiro, um pilar), e a sugestao pelo assunto e orientacao -- quem decide e o
modelo, o portao so confere pertencimento.
"""

from __future__ import annotations

from agent.writer.visuals import (
    PILLARS,
    brief,
    pillar_of,
    suggest_pillar,
    validate_terms,
)
from agent.writer.writer import build_prompt
from tests.test_writer import dossie


class TestPool:
    def test_toda_tag_e_ascii_e_cena_concreta(self):
        tags = [t for p in PILLARS.values() for t in p.tags]
        assert len(tags) == len({t.lower() for t in tags})  # sem duplicada
        assert all(t.isascii() for t in tags)
        assert all(len(t.split()) >= 2 for t in tags)  # conceito vira cena

    def test_cinco_pilares(self):
        """Os 4 de tech + E (ciencia e espaco, 19/09/2026)."""
        assert sorted(PILLARS) == ["A", "B", "C", "D", "E"]


class TestValidacao:
    def test_tag_do_pool_com_caixa_alta_passa(self):
        assert validate_terms(["Neural Network Nodes", "  ai deep learning loop "]) == []

    def test_parafrase_reprova_trazendo_o_que_vale(self):
        (problema,) = validate_terms(["neural network nodes", "server rack blue lights"])
        assert "ESTETICA" in problema or "pilar" in problema
        assert "server rack blue lights" in problema

    def test_pilares_misturados_reprovam(self):
        problemas = validate_terms(["sentient ai", "matrix code rain"])
        assert any("misturam" in p and "A" in p and "B" in p for p in problemas)

    def test_um_pilar_so_passa(self):
        assert validate_terms(["sentient ai", "bionic eye neon"]) == []


class TestPilar:
    def test_maioria_vence(self):
        assert pillar_of(["sentient ai", "matrix code rain", "bionic eye neon"]) == "A"

    def test_nenhum_casamento_e_none(self):
        assert pillar_of(["innovation", "future"]) is None

    def test_sugestao_por_assunto(self):
        assert suggest_pillar("Robo humanoide ganha consciencia") == "A"
        assert suggest_pillar("Vazamento expoe falha de seguranca em servidor") == "B"
        assert suggest_pillar("Bonsai 2 27B: modelo de 27B em 5,9 GB") == "C"
        assert suggest_pillar("Prefeitura lanca aplicativo de onibus") == "D"
        assert suggest_pillar("O lado roxo de Marte: gelo, poeira e luz") == "E"
        assert suggest_pillar("Robo humanoide explora Marte") == "A"


class TestPrompt:
    def test_brief_vai_ao_prompt_com_sugestao_marcada(self):
        texto = build_prompt(dossie(), None)
        assert "ESTETICA" in texto
        assert "neural network nodes" in texto
        assert brief("C").splitlines()[0].startswith("ESTETICA")
