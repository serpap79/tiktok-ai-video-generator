"""Sujeito obrigatorio: video anonimo nao tem busca nem credibilidade."""

from __future__ import annotations

from agent.research.subject import missing_subject, subject_terms


class TestTermos:
    def test_nome_e_identificador_saem_do_topico(self):
        assert subject_terms("Bonsai 2 27B: modelo de 27B em 5,9 GB") == [
            "bonsai", "27b"]

    def test_categoria_e_numero_puro_nao_sao_identidade(self):
        assert "modelo" not in subject_terms("Modelo novo da Apple")
        assert "5,9" not in subject_terms("Ocupa 5,9 GB")


class TestPortao:
    def test_video_anonimo_reprova_nomeando_o_que_falta(self):
        faltam = missing_subject("Um modelo gigante cabe no bolso.",
                                 "Bonsai 2 27B: modelo de 27B em 5,9 GB")
        assert faltam == ["bonsai", "27b"]

    def test_video_nomeado_passa(self):
        texto = "O Bonsai 27B ocupa pouco espaco."
        assert missing_subject(texto, "Bonsai 2 27B: modelo de 27B em 5,9 GB") == []

    def test_carrossel_pede_um_so(self):
        texto = "5 dados do Bonsai em 5,9 GB"
        assert missing_subject(texto, "Bonsai 2 27B", minimum=1) == []
        assert missing_subject("Resumo da semana", "Bonsai 2 27B", minimum=1) != []
