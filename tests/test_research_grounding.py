"""Testes do portao de ancoragem numerica.

Ele e a unica verificacao de fidelidade que nao depende do modelo, e por isso
tem teste nas duas direcoes: deixar passar numero inventado e tao ruim quanto
derrubar fato bom por diferenca de notacao.
"""

from __future__ import annotations

import pytest

from agent.research.grounding import canonical_numbers, missing_numbers


class TestNotacao:
    """pt-BR escreve 5,9 e ingles escreve 5.9 para o mesmo valor.

    As fontes de tech sao quase todas em ingles e a afirmacao sai em pt-BR: se o
    portao nao casasse as duas notacoes, ele reprovaria justamente os fatos bem
    traduzidos.
    """

    def test_decimal_com_virgula_casa_com_decimal_com_ponto(self):
        assert missing_numbers("ocupa 5,9 GB", "it occupies 5.9 GB on disk") == []

    def test_decimal_com_ponto_casa_com_decimal_com_virgula(self):
        assert missing_numbers("retains 98.2%", "retem 98,2% do desempenho") == []

    def test_milhar_nas_duas_convencoes(self):
        assert missing_numbers("1.500 GPUs", "a cluster of 1,500 GPUs") == []

    def test_digitos_sao_o_que_conta(self):
        assert canonical_numbers("5,9 GB e 1.500 GPUs em 2026") == ["59", "1500", "2026"]


class TestDeteccaoDeInvencao:
    def test_numero_ausente_e_apontado(self):
        ausentes = missing_numbers(
            "o modelo ocupa 5,9 GB e custa 200 dolares", "it occupies 5.9 GB on disk"
        )
        assert ausentes == ["200"]

    def test_varios_ausentes_saem_na_ordem_da_afirmacao(self):
        assert missing_numbers("subiu de 155 para 453 pontos", "texto sem numero") == [
            "155", "453",
        ]

    def test_repetido_aparece_uma_vez(self):
        assert missing_numbers("42 e 42 de novo", "nada") == ["42"]

    def test_unidade_convertida_pelo_modelo_e_reprovada(self):
        """"5900 MB" pode ate estar certo, mas nao esta na pagina.

        O portao nao converte unidade de proposito: quem converte e o modelo, e
        e exatamente ai que ele erra sem avisar.
        """
        assert missing_numbers("ocupa 5900 MB", "it occupies 5.9 GB") == ["5900"]


class TestLimitesConhecidos:
    def test_afirmacao_sem_numero_passa(self):
        """Ela ainda tem URL e trecho conferido; julgar o resto e do juiz."""
        assert missing_numbers("o modelo usa pesos ternarios", "ternary weights") == []

    @pytest.mark.parametrize("fonte", [
        "5,9 milhoes de downloads",
        "lancado em 5.9 de alguma outra escala",
    ])
    def test_nao_confere_contexto_nem_unidade(self, fonte):
        """Limite aceito e documentado: casar contexto exigiria pedir ao proprio
        modelo para se auditar, o que nao e verificacao."""
        assert missing_numbers("ocupa 5,9 GB", fonte) == []
