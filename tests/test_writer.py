"""Testes do roteirista -- fatia 2 do M3.

O criterio aqui e que **nada mecanico chegue ao juiz**: contagem de palavra fora
da faixa de monetizacao, termo de busca em portugues, indice de fato inexistente
e numero que nao esta no dossie sao defeitos verificaveis sem julgamento. Gastar
uma rodada de revisao do juiz com esses erros seria queimar cota do free tier.

O que este arquivo NAO testa, de proposito: se o hook e bom, se o fechamento tem
ponto de vista proprio, se o portugues soa falado. Isso e a rubrica do juiz
(fatia 3) e nao da para afirmar por regra.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agent.adapters.scripted_llm import ScriptedLLM
from agent.models import Dossier, Fact, Script
from agent.ports.llm import LLMUnavailable
from agent.writer.writer import (
    MAX_PALAVRAS,
    MIN_PALAVRAS,
    Screenwriter,
    build_prompt,
)

AGORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

HOOK = "Um modelo gigante agora cabe no seu bolso."
FECHAMENTO = ("Numero sem base de comparacao nao e medicao, e marketing. Leve essa "
              "pergunta para o proximo anuncio que voce ler.")
TERMOS = [
    "neural network nodes",
    "ai deep learning loop",
    "data stream tunnel",
    "abstract digital plexus",
]


def dossie() -> Dossier:
    return Dossier(
        topic="Bonsai 2 27B: modelo de 27B em 5,9 GB",
        facts=[
            Fact(claim="O Bonsai 2 27B ocupa 5,9 GB, mais de 9x menor que o original",
                 source_url="https://prismml.com/news/bonsai-2-27b", source_name="PrismML",
                 quote="that occupies 5.9 GB on disk, more than 9x smaller"),
            Fact(claim="Retem 98,2% do desempenho agregado nos benchmarks",
                 source_url="https://prismml.com/news/bonsai-2-27b", source_name="PrismML",
                 quote="the model retains 98.2% of the original score"),
            Fact(claim="Roda a 143 tokens por segundo numa RTX 5090",
                 source_url="https://cienciahoje.com.br/ia/bonsai",
                 source_name="Ciencia Hoje",
                 quote="Throughput reaches 143 tokens per second on a single RTX 5090"),
        ],
        collected_at=AGORA,
    )


def resposta(
    total: int = 187,
    used: tuple[int, ...] = (0, 1),
    termos: list[str] | None = None,
    extra: str = "",
    hook: str = HOOK,
    closing: str = FECHAMENTO,
) -> str:
    """Resposta do modelo no formato do schema, com contagem de palavras exata."""
    sujeito = "O Bonsai 27B e um modelo. "
    fixos = len((hook + " " + sujeito + closing + " " + extra).split())
    corpo = sujeito + " ".join(["detalhe"] * max(total - fixos, 1))
    return json.dumps({
        "hook": hook,
        "body": (extra + " " + corpo).strip(),
        "closing": closing,
        "search_terms": TERMOS if termos is None else termos,
        "caption": "Modelo gigante, disco pequeno.\nO Bonsai 27B mostra o que a compressao ja faz.",
        "used_facts": list(used),
    }, ensure_ascii=False)


def escrever(llm, **kwargs):
    return Screenwriter(llm, **kwargs).write(dossie())


class TestAceiteDaFatia:
    def test_roteiro_valido_na_primeira_tentativa(self):
        report = escrever(ScriptedLLM(responses=[resposta()]))
        assert report.ok
        assert len(report.attempts) == 1
        assert MIN_PALAVRAS <= report.script.word_count <= MAX_PALAVRAS
        assert 60 <= report.script.estimated_duration_s <= 90

    def test_o_modelo_aponta_o_fato_e_nao_reescreve(self):
        """Se o modelo pudesse redigir o fato, a afirmacao do roteiro deixaria de
        ser rastreavel ao que a fonte diz -- todo o ponto de haver dossie."""
        report = escrever(ScriptedLLM(responses=[resposta(used=(0, 2))]))
        d = dossie()
        assert [f.claim for f in report.script.facts] == [
            d.facts[0].claim, d.facts[2].claim,
        ]
        assert all(str(f.source_url).startswith("http") for f in report.script.facts)

    def test_roteiro_serve_de_entrada_para_o_renderizador(self):
        """O contrato Script e o mesmo que o M0 ja renderiza: o que sai daqui
        entra no `render` sem adaptacao."""
        report = escrever(ScriptedLLM(responses=[resposta()]))
        assert report.script.narration.startswith(HOOK.split(".")[0])
        assert len(report.script.search_terms) >= 3
        assert all(t.isascii() for t in report.script.search_terms)


class TestFaixaDeDuracao:
    def test_curto_demais_volta_com_a_contagem_medida(self):
        """Abaixo de 60s o video nao e elegivel ao Creator Rewards. O defeito e
        contavel, entao o roteirista corrige sozinho em vez de ocupar o juiz."""
        llm = ScriptedLLM(responses=[resposta(total=90), resposta(total=180)])
        report = escrever(llm)
        assert report.ok
        assert len(report.attempts) == 2
        (violacao,) = report.attempts[0].violations
        assert "90 palavras" in violacao
        assert f"{MIN_PALAVRAS}" in violacao and f"{MAX_PALAVRAS}" in violacao
        # A correcao medida volta ao modelo em texto, na segunda chamada.
        assert "90 palavras" in llm.calls[1].prompt
        assert "CORRIJA A TENTATIVA ANTERIOR" in llm.calls[1].prompt

    def test_longo_demais_tambem_e_corrigido(self):
        report = escrever(ScriptedLLM(responses=[resposta(total=400), resposta(total=200)]))
        assert report.ok
        assert "400 palavras" in report.attempts[0].violations[0]

    def test_teto_de_tentativas_nao_entrega_roteiro_ruim(self):
        """Tres tentativas erradas e problema de instrucao, nao de sorte.
        Insistir queima cota; entregar fora da faixa quebraria a monetizacao."""
        llm = ScriptedLLM(responses=[resposta(total=90)] * 3)
        report = escrever(llm)
        assert not report.ok
        assert len(report.attempts) == 3
        assert len(llm.calls) == 3
        assert report.violations  # o motivo da ultima reprovacao fica no relatorio


class TestAncoragemNoDossie:
    def test_numero_em_digito_fora_do_dossie_reprova(self):
        llm = ScriptedLLM(responses=[
            resposta(extra="O treinamento usou 12000 GPUs."),
            resposta(),
        ])
        report = escrever(llm)
        assert report.ok
        assert "12000" in report.attempts[0].violations[0]

    def test_numero_do_dossie_passa_mesmo_em_notacao_pt_br(self):
        report = escrever(ScriptedLLM(responses=[
            resposta(extra="Ele ocupa 5,9 GB no disco. Retem quase tudo."),
        ]))
        assert report.ok

    def test_indice_inexistente_reprova_dizendo_a_faixa_valida(self):
        llm = ScriptedLLM(responses=[resposta(used=(0, 9)), resposta()])
        report = escrever(llm)
        (violacao,) = llm_violacoes(report, 0)
        assert "[9]" in violacao
        assert "0 a 2" in violacao

    def test_roteiro_sem_fato_reprova(self):
        """Roteiro sem fonte e exatamente o que o projeto existe para nao fazer."""
        llm = ScriptedLLM(responses=[resposta(used=()), resposta()])
        report = escrever(llm)
        assert any("used_facts esta vazio" in v for v in llm_violacoes(report, 0))

    def test_indice_repetido_nao_duplica_o_fato(self):
        report = escrever(ScriptedLLM(responses=[resposta(used=(1, 1, 1))]))
        assert len(report.script.facts) == 1


class TestContratoDoScript:
    def test_termo_de_busca_em_portugues_reprova(self):
        """Os termos vao direto para o Pexels, sem traducao: acento quase sempre
        significa que o modelo respondeu em pt-BR, e o material volta errado."""
        llm = ScriptedLLM(responses=[
            resposta(termos=["placa de vídeo", "chip de memória", "sala de servidores"]),
            resposta(),
        ])
        report = escrever(llm)
        assert report.ok
        assert any("search_terms" in v for v in llm_violacoes(report, 0))

    def test_termos_de_menos_reprova(self):
        llm = ScriptedLLM(responses=[resposta(termos=["one term"]), resposta()])
        report = escrever(llm)
        assert any("search_terms" in v for v in llm_violacoes(report, 0))

    def test_campo_faltando_reprova_com_o_nome_do_campo(self):
        llm = ScriptedLLM(responses=[json.dumps({"hook": "curto"}), resposta()])
        report = escrever(llm)
        violacoes = " ".join(llm_violacoes(report, 0))
        assert "body" in violacoes and "search_terms" in violacoes


class TestRespostaDefeituosa:
    def test_json_quebrado_e_violacao_corrigivel_e_nao_falha_do_estagio(self):
        report = escrever(ScriptedLLM(responses=["nao consigo ajudar", resposta()]))
        assert report.ok
        assert "nao veio como objeto JSON" in report.attempts[0].violations[0]

    def test_resposta_truncada_pede_texto_mais_curto(self):
        """Truncamento e orcamento de token, nao erro de escrita: mandar o
        modelo 'corrigir o JSON' nao resolveria nada."""
        class Truncado(ScriptedLLM):
            def complete(self, prompt, **kwargs):
                c = super().complete(prompt, **kwargs)
                return c.model_copy(update={"finish_reason": "length"})

        report = escrever(Truncado(responses=[resposta()] * 3))
        assert not report.ok
        assert "cortada por limite de tokens" in report.attempts[0].violations[0]

    def test_cota_estourada_sobe_para_quem_chamou(self):
        """Nao ha o que corrigir no prompt: repetir so gasta a cota que falta."""
        def responder(prompt: str) -> str:
            raise LLMUnavailable("cota diaria estourada (429)")

        with pytest.raises(LLMUnavailable):
            escrever(ScriptedLLM(responder=responder))


class TestCustoEHistorico:
    def test_custo_soma_as_tentativas_que_falharam(self):
        """Tentativa reprovada tambem foi cobrada. Contar so a que passou
        subestimaria o custo do roteiro no eval do M5."""
        report = escrever(ScriptedLLM(responses=[resposta(total=90), resposta()]))
        primeira, segunda = report.attempts
        assert report.usage.input_tokens == (
            primeira.usage.input_tokens + segunda.usage.input_tokens
        )
        assert report.usage.output_tokens > 0

    def test_tentativas_ficam_no_relatorio_mesmo_no_sucesso(self):
        """E o que revela prompt fraco: se toda execucao gasta duas rodadas no
        mesmo defeito, o problema e a instrucao, nao o modelo."""
        report = escrever(ScriptedLLM(responses=[resposta(total=90), resposta()]))
        assert len(report.attempts) == 2
        assert report.attempts[0].violations and not report.attempts[1].violations
        assert report.attempts[0].word_count == 90

    def test_modelo_e_provedor_ficam_registrados(self):
        report = escrever(ScriptedLLM(responses=[resposta()]))
        assert report.model == "scripted-1"
        assert report.provider == "scripted"


class TestPrompt:
    def test_fatos_vao_indexados_com_fonte_e_trecho(self):
        prompt = build_prompt(dossie())
        assert "[0] O Bonsai 2 27B ocupa 5,9 GB" in prompt
        assert "fonte: PrismML" in prompt
        assert "that occupies 5.9 GB on disk" in prompt

    def test_faixa_de_palavras_e_a_regra_de_monetizacao_no_texto(self):
        prompt = build_prompt(dossie())
        assert str(MIN_PALAVRAS) in prompt and str(MAX_PALAVRAS) in prompt
        assert "monetizacao" in prompt

    def test_termos_em_ingles_e_ordem_cronologica_sao_pedidos(self):
        prompt = build_prompt(dossie())
        assert "EM INGLES" in prompt
        assert "cronologica" in prompt

    def test_primeira_tentativa_nao_tem_secao_de_correcao(self):
        assert "CORRIJA" not in build_prompt(dossie())


def llm_violacoes(report, indice: int) -> list[str]:
    return report.attempts[indice].violations


class TestMarcadorDeCitacao:
    """Defeito medido em execucao real, nao imaginado.

    O modelo escreveu "...no seu projeto [0]." e "...em um projeto [0, 3]." --
    echoando no texto FALADO o indice que devia ir so em used_facts. O
    sintetizador leria "zero" e "tres" em voz alta no video.
    """

    def test_marcador_no_texto_reprova_com_o_motivo_certo(self):
        llm = ScriptedLLM(responses=[
            resposta(extra="A leitura muda sem CLAUDE.md [0] e isso vale para todos [1, 3]."),
            resposta(),
        ])
        report = escrever(llm)
        assert report.ok
        (violacao,) = [v for v in report.attempts[0].violations if "marcador" in v]
        assert "[0]" in violacao and "[1, 3]" in violacao
        assert "voz alta" in violacao

    def test_marcador_nao_vira_acusacao_de_numero_inventado(self):
        """A mensagem errada era "numero 0, 1, 2 sem respaldo": verdadeira e
        inutil para saber o que fazer."""
        llm = ScriptedLLM(responses=[resposta(extra="Vale para o projeto [0]."), resposta()])
        report = escrever(llm)
        assert not any("numero que nao esta no dossie" in v
                       for v in report.attempts[0].violations)

    def test_numero_inventado_de_verdade_continua_sendo_pego(self):
        llm = ScriptedLLM(responses=[
            resposta(extra="Segundo o fato [0], foram 12000 GPUs."), resposta(),
        ])
        report = escrever(llm)
        violacoes = " ".join(report.attempts[0].violations)
        assert "marcador" in violacoes
        assert "12000" in violacoes

    def test_ano_no_texto_nao_e_confundido_com_marcador(self):
        llm = ScriptedLLM(responses=[
            resposta(extra="Ele ocupa 5,9 GB desde 2026."),
            resposta(),
        ])
        report = escrever(llm)
        assert report.ok
        violacoes = " ".join(report.attempts[0].violations)
        assert "marcador" not in violacoes
        assert "2026" in violacoes

    def test_o_prompt_proibe_o_marcador(self):
        from agent.writer.writer import build_prompt
        prompt = build_prompt(dossie())
        assert "'[0]'" in prompt and "used_facts" in prompt


class TestDossieFino:
    """Medir antes de pagar, como o curador e o juiz fazem.

    Um dossie de 4 fatos tirados de UMA frase levou o roteirista a tres
    tentativas, todas entre 104 e 157 palavras: faltava assunto, nao instrucao.
    """

    def test_dossie_com_menos_de_tres_fatos_nao_gasta_chamada(self):
        from agent.models import Dossier
        from agent.writer.writer import Screenwriter

        magro = Dossier(topic="t", facts=dossie().facts[:2], collected_at=AGORA)
        llm = ScriptedLLM(responses=[])
        report = Screenwriter(llm).write(magro)

        assert not report.ok
        assert llm.calls == []
        assert report.attempts == []
        assert "dossie fino: 2 fato(s)" in report.refusal
        assert report.usage.total_tokens == 0

    def test_recusa_diz_o_que_fazer(self):
        from agent.models import Dossier
        from agent.writer.writer import Screenwriter

        magro = Dossier(topic="t", facts=dossie().facts[:1], collected_at=AGORA)
        report = Screenwriter(ScriptedLLM()).write(magro)
        assert "Pesquise outras fontes" in report.refusal

    def test_tres_fatos_ja_autorizam_a_tentativa(self):
        report = escrever(ScriptedLLM(responses=[resposta()]))
        assert report.ok
        assert report.refusal == ""

    def test_o_roteiro_de_referencia_do_m0_nao_seria_recusado(self):
        """Cinco fatos de uma unica fonte rendem roteiro: o numero de FONTES nao
        entra na regra, o de fatos distintos entra."""
        import json
        from pathlib import Path

        from agent.models import Dossier, Script
        from agent.writer.writer import thin_dossier_reason

        bruto = json.loads(
            (Path(__file__).resolve().parent.parent / "fixtures" / "roteiro_manual.json")
            .read_text(encoding="utf-8")
        )
        bruto.pop("_comment", None)
        referencia = Script.model_validate(bruto)
        assert thin_dossier_reason(
            Dossier(topic=referencia.topic, facts=referencia.facts, collected_at=AGORA)
        ) == ""


class TestLegendaDoPost:
    """Guia da marca: gancho escrito SEM repetir o audio + contexto; sem link/hashtag."""

    def _script(self, caption: str) -> Script:
        return Script(topic="Tema", hook="Um modelo gigante agora cabe num pendrive.",
                      body=" ".join(["palavra"] * 60), closing="E voce, confia?",
                      search_terms=["neural network nodes", "data stream tunnel"],
                      caption=caption)

    def test_legenda_boa_passa(self):
        from agent.writer.writer import caption_problems
        assert caption_problems(self._script(
            "Disco pequeno, modelo enorme.\nO Bonsai 27B reduz 9x o tamanho.")) == []

    def test_repetir_o_hook_reprova(self):
        from agent.writer.writer import caption_problems
        (p,) = caption_problems(self._script("Um modelo gigante agora cabe num pendrive."))
        assert "repete o hook" in p

    def test_hashtag_link_e_tres_linhas_reprovam(self):
        from agent.writer.writer import caption_problems
        problemas = " ".join(caption_problems(self._script(
            "Linha um #ia\nveja em prismml.com\nlinha tres")))
        assert "hashtag" in problemas and "link" in problemas and "linha" in problemas

    def test_vazia_reprova_com_instrucao(self):
        from agent.writer.writer import caption_problems
        assert "2 linhas" in caption_problems(self._script(""))[0]
