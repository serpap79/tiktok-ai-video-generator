"""Testes do pesquisador -- e o aceite da primeira fatia do M3.

O criterio nao e "o dossie ficou bom": e que **nenhum fato entre sem URL
verificavel**, e que a URL venha de nos e nao do modelo. O resto dos testes
existe porque cada um deles corresponde a uma forma conhecida de o modelo
parecer certo estando errado.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from agent.adapters.scripted_llm import ScriptedLLM
from agent.models import Decision, Verdict
from agent.ports.llm import LLMBlocked, LLMUnavailable
from agent.research.fetch import PageFetcher
from agent.research.researcher import Researcher
from agent.research.sources import Candidate

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "research"
ARTIGO = (FIXTURES / "artigo.html").read_text(encoding="utf-8")
AGORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

URL_REAL = "https://prismml.com/news/bonsai-2-27b"

# Trechos copiados do texto que o extrator produz para a fixture. E o que o
# modelo teria de devolver para o portao de trecho aceitar.
TRECHO_TAMANHO = "that occupies 5.9 GB on disk, more than 9x smaller than"
TRECHO_BENCH = "the model retains 98.2% of the original score"
TRECHO_THROUGHPUT = "Throughput reaches 143 tokens per second on a single RTX 5090"


def resposta(*fatos: tuple[str, str]) -> str:
    """Resposta do modelo no formato do schema: pares (trecho, afirmacao)."""
    return json.dumps(
        {"facts": [{"quote": q, "claim": c} for q, c in fatos]}, ensure_ascii=False
    )


def decisao(term: str = "Bonsai 2 27B em 5,9 GB", source: str = "hacker_news") -> Decision:
    return Decision(
        term=term, source=source, verdict=Verdict.selected,
        reason="teste", score=0.9, niche_fit=0.8, decided_at=AGORA,
    )


def fetcher(handler=None) -> PageFetcher:
    handler = handler or (
        lambda r: httpx.Response(200, text=ARTIGO, headers={"content-type": "text/html"})
    )
    return PageFetcher(
        client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    )


def pesquisar(llm, candidatos=None, **kwargs) -> tuple:
    r = Researcher(llm, fetcher=fetcher(kwargs.pop("handler", None)), **kwargs).research(
        decisao(), candidates=candidatos or [Candidate(url=URL_REAL, source_name="PrismML")]
    )
    return r


class TestAceiteDaFatia:
    def test_todo_fato_tem_url_rastreavel(self):
        llm = ScriptedLLM(responses=[resposta(
            (TRECHO_TAMANHO, "O Bonsai 2 27B ocupa 5,9 GB, mais de 9x menor que o original"),
            (TRECHO_BENCH, "O modelo retem 98,2% da nota original nos benchmarks"),
        )])
        report = pesquisar(llm)

        assert report.ok
        assert len(report.facts) == 2
        assert all(str(f.source_url) == URL_REAL for f in report.facts)
        assert all(f.source_name == "PrismML" for f in report.facts)
        assert report.dossier.source_urls == {URL_REAL}

    def test_a_url_e_estampada_por_nos_e_nao_pedida_ao_modelo(self):
        """O erro mais caro possivel aqui e fato real com fonte trocada.

        Ele parece ancorado, passa no juiz, e so aparece quando alguem clica no
        link. Por isso o modelo nem tem como opinar: o campo que ele mandar e
        ignorado, porque de onde veio o texto e informacao que nos ja temos.
        """
        inventado = json.dumps({"facts": [{
            "quote": TRECHO_TAMANHO,
            "claim": "O Bonsai 2 27B ocupa 5,9 GB, mais de 9x menor que o original",
            "source_url": "https://fonte-que-o-modelo-inventou.com/post",
            "source_name": "Veiculo Inexistente",
        }]})
        report = pesquisar(ScriptedLLM(responses=[inventado]))

        (fato,) = report.facts
        assert str(fato.source_url) == URL_REAL
        assert fato.source_name == "PrismML"

    def test_schema_pedido_ao_modelo_nao_tem_campo_de_fonte(self):
        """Nao adianta ignorar a URL do modelo e ainda pedi-la: pedir convida o
        modelo a preencher, e alguem depois usaria o campo por engano."""
        llm = ScriptedLLM(responses=[resposta((TRECHO_BENCH, "Retem 98,2% da nota"))])
        pesquisar(llm)
        propriedades = llm.calls[0].schema["properties"]["facts"]["items"]["properties"]
        assert set(propriedades) == {"quote", "claim"}


class TestUmaChamadaPorFonte:
    def test_cada_fonte_recebe_uma_chamada_so(self):
        """Uma chamada por fonte e o que permite estampar a URL certa.

        Mandar tres paginas juntas economizaria cota e devolveria fato sem dono.
        """
        llm = ScriptedLLM(responses=[
            resposta((TRECHO_TAMANHO, "Ocupa 5,9 GB, mais de 9x menor")),
            resposta((TRECHO_BENCH, "Retem 98,2% da nota original")),
        ])
        report = pesquisar(llm, candidatos=[
            Candidate(url="https://prismml.com/a", source_name="PrismML"),
            Candidate(url="https://cienciahoje.com.br/b", source_name="Ciencia Hoje"),
        ])
        assert len(llm.calls) == 2
        assert {f.source_name for f in report.facts} == {"PrismML", "Ciencia Hoje"}

    def test_o_texto_da_pagina_vai_no_prompt(self):
        llm = ScriptedLLM(responses=[resposta((TRECHO_BENCH, "Retem 98,2% da nota"))])
        pesquisar(llm)
        prompt = llm.calls[0].prompt
        assert "5.9 GB" in prompt
        assert "Bonsai 2 27B em 5,9 GB" in prompt  # o tema em apuracao

    def test_teto_de_fontes_e_respeitado(self):
        llm = ScriptedLLM(responses=[resposta((TRECHO_BENCH, "Retem 98,2% da nota"))])
        pesquisar(llm, max_sources=1, candidatos=[
            Candidate(url="https://prismml.com/a"),
            Candidate(url="https://outro.com/b"),
        ])
        assert len(llm.calls) == 1


class TestPortaoDeTrecho:
    def test_parafrase_no_lugar_de_copia_reprova(self):
        """O modelo devia copiar e reescreveu. Nao da para saber se a afirmacao
        e verdadeira -- e "nao da para saber" reprova."""
        report = pesquisar(ScriptedLLM(responses=[resposta((
            "the model keeps 98.2 percent of the original score",
            "O modelo retem 98,2% da nota original",
        ))]))
        assert not report.ok
        (descartado,) = report.discarded
        assert "nao existe na pagina" in descartado.reason
        assert descartado.source_url == URL_REAL

    def test_trecho_ausente_reprova(self):
        report = pesquisar(ScriptedLLM(responses=[resposta(
            ("", "O modelo retem 98,2% da nota original"),
        )]))
        assert not report.ok
        assert "trecho de apoio ausente" in report.discarded[0].reason

    def test_trecho_curto_demais_reprova(self):
        """"5.9 GB" aparece no menu tambem: trecho curto casa com qualquer coisa
        e nao prova nada."""
        report = pesquisar(ScriptedLLM(responses=[resposta(
            ("5.9 GB", "O modelo ocupa 5,9 GB"),
        )]))
        assert not report.ok
        assert "curto demais" in report.discarded[0].reason

    def test_espaco_estreito_dentro_do_numero_nao_reprova(self):
        """Medido em 18/09/2026: o `groq/compound-mini` devolve "5,9\u202fGB" --
        espaco estreito sem quebra dentro do numero. E formatacao tipografica, nao
        parafrase, e reprovar por isso derrubaria copia literal correta."""
        estreito = "that occupies 5.9\u202fGB on disk, more than 9x smaller than"
        report = pesquisar(ScriptedLLM(responses=[resposta(
            (estreito, "O Bonsai 2 27B ocupa 5,9\u202fGB, mais de 9x menor"),
        )]))
        assert report.ok
        assert report.facts[0].quote.startswith("that occupies 5.9")

    def test_diferenca_de_espaco_e_quebra_de_linha_nao_reprova(self):
        """O trecho atravessa quebra de linha no HTML; exigir formatacao
        identica reprovaria copia literal correta."""
        atravessa = "a ternary-weight version of Qwen3.8 27B     that occupies 5.9 GB"
        report = pesquisar(ScriptedLLM(responses=[resposta(
            (atravessa, "O Bonsai 2 27B e uma versao de pesos ternarios do Qwen3.8 27B"),
        )]))
        assert report.ok


class TestPortaoNumerico:
    def test_numero_inventado_reprova_com_o_numero_no_motivo(self):
        report = pesquisar(ScriptedLLM(responses=[resposta((
            TRECHO_TAMANHO,
            "O Bonsai 2 27B ocupa 5,9 GB e foi treinado com 12000 GPUs",
        ))]))
        assert not report.ok
        assert "12000" in report.discarded[0].reason

    def test_numero_em_notacao_pt_br_passa(self):
        report = pesquisar(ScriptedLLM(responses=[resposta((
            TRECHO_THROUGHPUT,
            "Roda a 143 tokens por segundo numa RTX 5090",
        ))]))
        assert report.ok

    def test_fato_bom_sobrevive_ao_lado_de_fato_reprovado(self):
        """Uma afirmacao ruim nao contamina a fonte inteira."""
        report = pesquisar(ScriptedLLM(responses=[resposta(
            (TRECHO_TAMANHO, "O Bonsai 2 27B ocupa 5,9 GB, mais de 9x menor"),
            (TRECHO_BENCH, "Retem 98,2% da nota e custa 500 mil dolares"),
        )]))
        assert len(report.facts) == 1
        assert len(report.discarded) == 1


class TestFalhasIsoladas:
    def test_cota_estourada_numa_fonte_nao_derruba_as_outras(self):
        """Mesma regra do radar: a janela de um tema e de horas."""
        def responder(prompt: str) -> str:
            if "cienciahoje" in prompt or "Ciencia Hoje" in prompt:
                raise LLMUnavailable("cota diaria estourada (429)")
            return resposta((TRECHO_BENCH, "Retem 98,2% da nota original"))

        report = pesquisar(ScriptedLLM(responder=responder), candidatos=[
            Candidate(url="https://prismml.com/a", source_name="PrismML"),
            Candidate(url="https://cienciahoje.com.br/b", source_name="Ciencia Hoje"),
        ])
        assert report.ok
        assert len(report.facts) == 1
        assert any("LLMUnavailable" in m for m in report.failures.values())

    def test_filtro_de_conteudo_do_provedor_e_registrado_e_nao_estoura(self):
        def responder(prompt: str) -> str:
            raise LLMBlocked("filtro de conteudo")

        report = pesquisar(ScriptedLLM(responder=responder))
        assert not report.ok
        assert any("LLMBlocked" in m for m in report.failures.values())

    def test_pagina_fora_do_ar_e_registrada_com_a_url(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403)

        report = pesquisar(ScriptedLLM(responses=[]), handler=handler)
        assert not report.ok
        assert "403" in report.failures[URL_REAL]

    def test_pagina_sem_texto_nao_gasta_chamada_de_modelo(self):
        """Cota e o recurso escasso do free tier: pagina sem conteudo nao vira
        prompt, e o motivo fica gravado."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<p>Aceite os cookies</p>",
                                  headers={"content-type": "text/html"})

        llm = ScriptedLLM(responses=[])
        report = pesquisar(llm, handler=handler)
        assert llm.calls == []
        assert "curto demais" in report.failures[URL_REAL]

    def test_resposta_do_modelo_fora_do_contrato_e_falha_da_fonte(self):
        report = pesquisar(ScriptedLLM(responses=["nao posso ajudar com isso"]))
        assert not report.ok
        assert any("LLMError" in m for m in report.failures.values())

    def test_tema_sem_fonte_candidata_nao_chama_o_modelo(self):
        llm = ScriptedLLM(responses=[])
        report = Researcher(llm, fetcher=fetcher()).research(decisao(), candidates=[])
        assert llm.calls == []
        assert not report.ok
        assert "descoberta" in report.failures


class TestCustoMedido:
    def test_tokens_e_latencia_somam_entre_as_fontes(self):
        """O M5 compara provedores por qualidade e por consumo. Numero medido na
        hora e mais confiavel que reconstruido do log depois."""
        llm = ScriptedLLM(responses=[
            resposta((TRECHO_TAMANHO, "Ocupa 5,9 GB, mais de 9x menor")),
            resposta((TRECHO_BENCH, "Retem 98,2% da nota original")),
        ])
        report = pesquisar(llm, candidatos=[
            Candidate(url="https://prismml.com/a"),
            Candidate(url="https://cienciahoje.com.br/b"),
        ])
        assert report.usage.input_tokens > 0
        assert report.usage.output_tokens > 0
        assert report.latency_s >= 0
        assert report.model == "scripted-1"

    def test_fonte_distinta_e_contada_por_dominio(self):
        llm = ScriptedLLM(responses=[
            resposta((TRECHO_TAMANHO, "Ocupa 5,9 GB, mais de 9x menor")),
            resposta((TRECHO_BENCH, "Retem 98,2% da nota original")),
        ])
        report = pesquisar(llm, candidatos=[
            Candidate(url="https://prismml.com/a"),
            Candidate(url="https://prismml.com/b"),
        ])
        assert len(report.facts) == 2
        assert report.source_count == 1


class TestHigieneDaResposta:
    def test_fato_repetido_na_mesma_fonte_entra_uma_vez(self):
        claim = "O Bonsai 2 27B ocupa 5,9 GB, mais de 9x menor"
        report = pesquisar(ScriptedLLM(responses=[resposta(
            (TRECHO_TAMANHO, claim), (TRECHO_TAMANHO, claim),
        )]))
        assert len(report.facts) == 1

    def test_teto_de_fatos_por_fonte(self):
        llm = ScriptedLLM(responses=[resposta(
            (TRECHO_TAMANHO, "Ocupa 5,9 GB, mais de 9x menor que o original"),
            (TRECHO_BENCH, "Retem 98,2% da nota original nos benchmarks"),
            (TRECHO_THROUGHPUT, "Roda a 143 tokens por segundo numa RTX 5090"),
        )])
        report = pesquisar(llm, max_facts_per_source=2)
        assert len(report.facts) == 2

    def test_afirmacao_curta_demais_para_o_contrato_e_descartada_com_motivo(self):
        report = pesquisar(ScriptedLLM(responses=[resposta((TRECHO_TAMANHO, "5,9 GB"))]))
        assert not report.ok
        assert "contrato Fact recusou" in report.discarded[0].reason

    def test_lista_vazia_do_modelo_e_resposta_valida(self):
        """Pagina que nao trata do tema deve devolver nada, nao inventar."""
        report = pesquisar(ScriptedLLM(responses=['{"facts": []}']))
        assert not report.ok
        assert report.discarded == []
        assert report.pages  # a pagina foi lida; so nao rendeu fato


def test_dossie_sem_fato_nao_autoriza_roteiro():
    """O contrato Dossier ja proibe isso; o pesquisador nao tenta burlar."""
    report = pesquisar(ScriptedLLM(responses=['{"facts": []}']))
    assert report.dossier is None
    with pytest.raises(AttributeError):
        _ = report.dossier.facts


class TestUmTrechoUmFato:
    """Regra que veio de execucao real, nao de suposicao.

    Em 18/09/2026, de UMA frase de changelog o modelo tirou quatro "fatos", tres
    deles apoiados no mesmo trecho. O dossie parecia cheio (4 fatos) e nao dava
    assunto para 60 segundos: o roteirista tentou tres vezes e nunca passou de
    157 palavras.
    """

    def test_trecho_repetido_na_mesma_fonte_vira_um_fato_so(self):
        report = pesquisar(ScriptedLLM(responses=[resposta(
            (TRECHO_TAMANHO, "O Bonsai 2 27B ocupa 5,9 GB, mais de 9x menor"),
            (TRECHO_TAMANHO, "O modelo e mais de nove vezes menor que o original"),
            (TRECHO_BENCH, "Retem 98,2% da nota original nos benchmarks"),
        )]))
        assert len(report.facts) == 2
        assert "mesmo trecho ja sustenta outro fato" in report.discarded[0].reason

    def test_diferenca_de_formatacao_nao_burla_a_regra(self):
        espacado = TRECHO_TAMANHO.replace(" ", "   ")
        report = pesquisar(ScriptedLLM(responses=[resposta(
            (TRECHO_TAMANHO, "O Bonsai 2 27B ocupa 5,9 GB, mais de 9x menor"),
            (espacado, "Afirmacao diferente sobre o mesmo trecho reformatado"),
        )]))
        assert len(report.facts) == 1

    def test_trechos_distintos_rendem_fatos_distintos(self):
        report = pesquisar(ScriptedLLM(responses=[resposta(
            (TRECHO_TAMANHO, "O Bonsai 2 27B ocupa 5,9 GB, mais de 9x menor"),
            (TRECHO_BENCH, "Retem 98,2% da nota original nos benchmarks"),
            (TRECHO_THROUGHPUT, "Roda a 143 tokens por segundo numa RTX 5090"),
        )]))
        assert len(report.facts) == 3
        assert report.discarded == []

    def test_o_prompt_pede_passagem_diferente(self):
        llm = ScriptedLLM(responses=[resposta((TRECHO_BENCH, "Retem 98,2% da nota"))])
        pesquisar(llm)
        assert "passagem DIFERENTE" in llm.calls[0].prompt
