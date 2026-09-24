"""Eval do M5, sem rede e sem chave.

O harness agrega o que ja esta gravado e nunca chama modelo: estes testes
semeiam linhas sinteticas e conferem as tres regras que sustentam a tabela --
media de juiz exclui parecer interrompido, matriz pareada exige roteiro
conhecido, e numero ausente (braco pago) nao aparece como zero.
"""

from __future__ import annotations

from datetime import UTC, datetime

from agent.eval.eval import (
    ReviewRow,
    ScriptRow,
    build_report,
    chave,
    format_text,
    paired_matrix,
    summarize_judges,
    summarize_writers,
)
from agent.models import Criterion, CriterionScore, Review

TEMA = "Bonsai 2 27B: modelo de 27B em 5,9 GB"


def _nota(criterion: Criterion, score: int, **kw) -> CriterionScore:
    base = {"criterion": criterion, "score": score,
            "reason": f"motivo de {criterion.value}"}
    base.update(kw)
    return CriterionScore(**base)


def _parecer(notas: dict[Criterion, int], **kw) -> Review:
    return Review(
        topic=TEMA,
        scores=[_nota(c, s) for c, s in notas.items()],
        reviewed_at=datetime.now(UTC),
        **kw,
    )


NOTAS_12 = {
    Criterion.hook: 1, Criterion.fuente: 2, Criterion.duracion: 2,
    Criterion.punto_de_vista: 1, Criterion.politica: 2,
    Criterion.idioma: 2, Criterion.cta: 2,
}
NOTAS_14 = {
    Criterion.hook: 2, Criterion.fuente: 2, Criterion.duracion: 2,
    Criterion.punto_de_vista: 2, Criterion.politica: 2,
    Criterion.idioma: 2, Criterion.cta: 2,
}


def _roteiro(id: int, provider: str, model: str, **kw) -> ScriptRow:
    base = {"id": id, "topic": TEMA, "model": model, "provider": provider,
            "word_count": 180, "attempts": 1,
            "input_tokens": 1000, "output_tokens": 300, "latency_s": 5.0}
    base.update(kw)
    return ScriptRow(**base)


def _parecer_linha(id: int, review: Review, provider: str, model: str,
                   **kw) -> ReviewRow:
    base = {"id": id, "topic": TEMA, "script_id": None,
            "model": model, "provider": provider, "review": review,
            "input_tokens": 1200, "output_tokens": 200, "latency_s": 2.0}
    base.update(kw)
    return ReviewRow(**base)


def _interrompido() -> Review:
    """Reprovado na medida: duracao e politica medidos, resto nao avaliado."""
    return Review(
        topic=TEMA,
        scores=[
            _nota(Criterion.duracion, 0, measured=True),
            _nota(Criterion.politica, 2, measured=True),
            _nota(Criterion.hook, 0, evaluated=False),
            _nota(Criterion.fuente, 0, evaluated=False),
            _nota(Criterion.punto_de_vista, 0, evaluated=False),
            _nota(Criterion.idioma, 0, evaluated=False),
            _nota(Criterion.cta, 0, evaluated=False),
        ],
        reviewed_at=datetime.now(UTC),
    )


class TestChave:
    def test_modelo_vazio_nao_some(self):
        assert chave("gemini", "") == "gemini/?"


class TestEscritores:
    def test_agrega_custo_e_palavras(self):
        saidas = summarize_writers([
            _roteiro(1, "gemini", "gemini-2.5-flash"),
            _roteiro(2, "gemini", "gemini-2.5-flash", word_count=186,
                     input_tokens=2038, output_tokens=648),
        ])
        assert len(saidas) == 1
        w = saidas[0]
        assert w.n == 2
        assert w.avg_words == 183.0
        assert w.input_tokens == 3038
        assert w.output_tokens == 948


class TestJuizes:
    def test_interrompido_fora_da_media(self):
        (j,) = summarize_judges([
            _parecer_linha(1, _parecer(NOTAS_12), "gemini", "gemini-2.5-flash"),
            _parecer_linha(2, _interrompido(), "gemini", "gemini-2.5-flash"),
        ])
        assert j.n_total == 2
        assert j.n_full == 1
        assert j.avg_total == 12.0
        assert j.approval_rate == 1.0
        assert j.avg_by_criterion["hook"] == 1.0

    def test_generosidade_aparece_no_numero(self):
        gemini, groq = summarize_judges([
            _parecer_linha(1, _parecer(NOTAS_12), "gemini", "gemini-2.5-flash"),
            _parecer_linha(2, _parecer(NOTAS_14), "groq", "openai/gpt-oss-120b"),
        ])
        assert gemini.avg_total == 12.0
        assert groq.avg_total == 14.0


class TestPareado:
    def test_so_parecer_com_roteiro_conhecido(self):
        scripts = [_roteiro(7, "gemini", "gemini-2.5-flash")]
        cells = paired_matrix(scripts, [
            _parecer_linha(1, _parecer(NOTAS_12), "gemini",
                           "gemini-2.5-flash", script_id=7),
            _parecer_linha(2, _parecer(NOTAS_14), "groq",
                           "openai/gpt-oss-120b", script_id=7),
            _parecer_linha(3, _parecer(NOTAS_14), "groq",
                           "openai/gpt-oss-120b", script_id=None),
            _parecer_linha(4, _parecer(NOTAS_14), "groq",
                           "openai/gpt-oss-120b", script_id=999),
            _parecer_linha(5, _interrompido(), "gemini",
                           "gemini-2.5-flash", script_id=7),
        ])
        assert [(c.writer, c.judge, c.n) for c in cells] == [
            ("gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash", 1),
            ("gemini/gemini-2.5-flash", "groq/openai/gpt-oss-120b", 1),
        ]
        assert [c.avg_total for c in cells] == [12.0, 14.0]


class TestRelatorio:
    def test_vazio_nao_quebra(self):
        report = build_report([], [])
        assert report.n_scripts == 0
        assert "nenhum" in format_text(report)

    def test_custo_soma_escritor_e_juiz(self):
        report = build_report(
            [_roteiro(1, "gemini", "gemini-2.5-flash")],
            [_parecer_linha(1, _parecer(NOTAS_12), "gemini",
                            "gemini-2.5-flash", script_id=1)],
        )
        assert report.input_tokens == 2200
        assert report.topics == [TEMA]
        texto = format_text(report)
        assert "gemini/gemini-2.5-flash x gemini/gemini-2.5-flash" in texto
