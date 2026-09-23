"""Vetor de marca: uma fonte, portoes medidos, conformidade testada.

Se o guia mudar, muda `brand/brand.json` e estes testes dizem onde o codigo
divergiu -- em vez de a identidade apodrecer em silencio.
"""

from __future__ import annotations

import re

from agent.brand.brand import (
    avatar_prompt,
    load,
    suggest_content_pillar,
    voice_brief,
)
from agent.brand.checks import (
    check_emoji_bordao,
    check_hook,
    check_numbers,
)


class TestVetor:
    def test_sete_pilares_um_acento_por_peca(self):
        """Os seis do guia + historia (20/09/2026, pedido do autor)."""
        brand = load()
        assert sorted(brand.pillars) == ["analise", "fato", "futuro", "historia",
                                         "news", "tutorial", "vs"]
        for p in brand.pillars.values():
            assert re.fullmatch(r"#[0-9A-F]{6}", p.accent)
        assert brand.accent_for("fato") == "#00E0FF"
        assert brand.accent_for("news") == "#39FF88"
        assert brand.accent_for("inexistente") == "#39FF88"

    def test_hashtags_e_bio(self):
        brand = load()
        assert brand.hashtags == ("#ia", "#inteligenciaartificial", "#tecnologia",
                                  "#ai", "#seucanal")
        assert "@seucanal" in brand.handle and "Sinais do futuro" in brand.tagline

    def test_sugestao_por_assunto(self):
        assert suggest_content_pillar("Como usar prompts no dia a dia") == "tutorial"
        assert suggest_content_pillar("Modelo contra modelo: qual vence") == "vs"
        assert suggest_content_pillar("Anuncio de modelo novo") == "news"

    def test_voz_tem_as_regras(self):
        texto = voice_brief()
        assert "12 palavras" in texto and "emoji" in texto


class TestApresentadores:
    def test_elenco_com_seeds_e_formatos(self):
        brand = load()
        assert sorted(brand.presenters) == ["iris", "theo"]
        assert brand.presenters["iris"].seed == 481502
        assert brand.presenters["theo"].seed == 907314
        # Desde a noite de 20/09/2026 o THEO apresenta TODOS os pilares: so
        # ele tem clipe base filmado, e misturar um apresentador filmado com um
        # sintetizado no mesmo canal seria uma diferenca de qualidade visivel.
        for pilar in ("analise", "tutorial", "news", "futuro", "historia",
                      "fato", "vs"):
            assert brand.presenter_for(pilar).id == "theo", pilar
        # A Iris nao foi removida, so ficou sem formato -- ela volta a disputar
        # pilar sozinha quando `iris_base.json` existir.
        assert brand.presenters["iris"].formats == ()

    def test_vozes_mapeadas_sem_invencao(self):
        brand = load()
        assert brand.presenters["theo"].library_voice == "jeff"
        assert brand.presenters["iris"].library_voice is None

    def test_prompt_travado_com_variaveis(self):
        texto = avatar_prompt("iris", expressao="concentrada")
        assert "seed: 481502" in texto and "concentrada" in texto
        assert "fundo transparente" in texto

    def test_variavel_fora_da_lista_falha(self):
        import pytest
        with pytest.raises(ValueError):
            avatar_prompt("theo", gesto="dancando")


class TestPortoes:
    def test_gancho_ate_12(self):
        assert check_hook(" ".join(["palavra"] * 12)) is None
        falha = check_hook(" ".join(["palavra"] * 13))
        assert falha is not None and "12" in falha

    def test_nome_proprio_nao_conta_como_numero(self):
        assert check_numbers("O Bonsai 27B roda na RTX 5090.") == []
        assert check_numbers("Escala FP16 com 1.76 bits.") == []

    def test_intervalo_conta_como_um(self):
        assert check_numbers("Suporta de 25 a 300 contas por chave.") == []

    def test_dois_numeros_pedem_duas_frases(self):
        texto = "Ocupa 5,9 GB e retem 98,2% do desempenho."
        (falha,) = check_numbers(texto)
        assert "um por frase" in falha
        assert check_numbers("Ocupa 5,9 GB no disco. Retem quase tudo.") == []

    def test_emoji_e_bordao(self):
        assert any("emoji" in p for p in check_emoji_bordao("Olha isso \U0001F600"))
        assert any("fala galera" in p for p in check_emoji_bordao("Fala galera!"))
        assert check_emoji_bordao("Texto limpo.") == []


class TestSlidesNaMarca:
    def test_legenda_leva_hashtags(self, tmp_path):
        from agent.models import Carousel
        from agent.render.carousel import render_carousel
        from tests.test_carousel import slides
        carrossel = Carousel.model_validate_json(slides())
        render_carousel(carrossel, tmp_path / "car", "fato")
        caption = (tmp_path / "car" / "caption.txt").read_text(encoding="utf-8")
        assert "#seucanal" in caption
