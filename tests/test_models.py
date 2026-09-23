"""Testes dos contratos entre estagios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.models import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    Dossier,
    RenderResult,
    RenderState,
    Script,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "roteiro_manual.json"


def load_fixture() -> Script:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    return Script.model_validate(raw)


class TestScript:
    def test_fixture_e_valido(self):
        script = load_fixture()
        assert script.topic
        assert script.facts, "o fixture precisa carregar fatos com fonte"

    def test_fixture_cabe_na_faixa_monetizavel(self):
        """A estimativa pre-TTS precisa cair na faixa do Creator Rewards.

        Nao substitui a medicao do MP4 - so evita mandar para render um roteiro
        que ja nasce fora da faixa.
        """
        script = load_fixture()
        assert MIN_DURATION_S <= script.estimated_duration_s <= MAX_DURATION_S, (
            f"{script.word_count} palavras => ~{script.estimated_duration_s:.0f}s, "
            f"fora de {MIN_DURATION_S}-{MAX_DURATION_S}s"
        )

    def test_narracao_usa_portugues_acentuado(self):
        """O fixture e o modelo do que o roteirista deve gerar no M3.

        Portugues sem acento aparece na legenda palavra-por-palavra e parece
        desleixo num canal. A fonte (BeVietnamPro-Bold) cobre todos os acentos,
        entao nao ha desculpa tecnica para escrever sem eles.
        """
        import unicodedata

        narracao = load_fixture().narration
        acentos = sum(
            1 for c in unicodedata.normalize("NFD", narracao) if unicodedata.combining(c)
        )
        assert acentos >= 20, f"apenas {acentos} acentos em {len(narracao)} caracteres"

    def test_narracao_preserva_a_ordem(self):
        script = load_fixture()
        narration = script.narration
        assert narration.index(script.hook) < narration.index(script.closing)

    def test_termo_de_busca_com_acento_e_rejeitado(self):
        """Termo em pt-BR vai direto para o Pexels e devolve resultado vazio.

        Falhar na validacao custa milissegundos; descobrir depois custa um render
        inteiro com material errado.
        """
        with pytest.raises(ValidationError, match="ASCII"):
            Script(
                topic="teste",
                hook="um hook qualquer com tamanho suficiente",
                body="um corpo de roteiro com pelo menos cinquenta caracteres para passar",
                closing="um fechamento qualquer",
                search_terms=["placa de vídeo", "data center", "código"],
            )

    def test_poucos_termos_sao_rejeitados(self):
        with pytest.raises(ValidationError):
            Script(
                topic="teste",
                hook="um hook qualquer com tamanho suficiente",
                body="um corpo de roteiro com pelo menos cinquenta caracteres para passar",
                closing="um fechamento qualquer",
                search_terms=["only one"],
            )

    def test_verificacao_de_fonte_nao_e_do_modelo(self):
        """Casar afirmacao com fonte exige julgamento semantico, nao string match."""
        with pytest.raises(NotImplementedError):
            _ = load_fixture().unsourced


class TestDossier:
    def test_dossie_sem_fato_e_rejeitado(self):
        from datetime import datetime

        with pytest.raises(ValidationError, match="sem fato"):
            Dossier(topic="t", facts=[], collected_at=datetime.now())

    def test_fato_sem_url_e_rejeitado(self):
        from datetime import datetime

        with pytest.raises(ValidationError):
            Dossier.model_validate(
                {
                    "topic": "t",
                    "collected_at": datetime.now(),
                    "facts": [{"claim": "uma afirmacao qualquer", "source_name": "Fonte"}],
                }
            )


class TestRenderResult:
    def test_completo_sem_caminho_e_incoerente(self):
        with pytest.raises(ValidationError, match="sem video_path"):
            RenderResult(state=RenderState.complete)

    def test_falha_sem_motivo_e_incoerente(self):
        with pytest.raises(ValidationError, match="sem mensagem de erro"):
            RenderResult(state=RenderState.failed)

    def test_aceite_do_m0(self):
        ok = RenderResult(
            state=RenderState.complete,
            video_path="/tmp/x.mp4",
            width=1080,
            height=1920,
            duration_s=78.4,
            has_audio=True,
        )
        assert ok.is_portrait_1080x1920
        assert ok.duration_in_monetizable_range
        assert ok.has_audio

    def test_video_mudo_nao_e_aceite(self):
        """Regressao de um caso real: o primeiro render saiu sem trilha de audio
        e passou nas checagens de dimensao e duracao. `has_audio` e falso por
        padrao justamente para que o silencio nunca seja o default aprovado."""
        mudo = RenderResult(
            state=RenderState.complete,
            video_path="/tmp/x.mp4",
            width=1080,
            height=1920,
            duration_s=78.4,
        )
        assert mudo.is_portrait_1080x1920
        assert mudo.duration_in_monetizable_range
        assert not mudo.has_audio

    def test_video_curto_demais_reprova(self):
        """59s nao e elegivel ao Creator Rewards, por mais bonito que esteja."""
        curto = RenderResult(
            state=RenderState.complete,
            video_path="/tmp/x.mp4",
            width=1080,
            height=1920,
            duration_s=59.0,
        )
        assert curto.is_portrait_1080x1920
        assert not curto.duration_in_monetizable_range

    def test_duracao_ausente_nao_conta_como_aprovada(self):
        sem_medida = RenderResult(state=RenderState.complete, video_path="/tmp/x.mp4")
        assert not sem_medida.duration_in_monetizable_range
