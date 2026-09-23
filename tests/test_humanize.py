"""Humanizacao: scan sem modelo quando limpo, reescrita travada quando sujo.

Adaptacao do `blader/humanizer` (MIT) como estagio: as travas de grounding
sao nossas -- numero e faixa quebrados mantem o original. Estes testes usam
ScriptedLLM e nunca tocam rede.
"""

from __future__ import annotations

import json

from agent.adapters.scripted_llm import ScriptedLLM
from agent.writer.humanize import build_prompt, humanize, scan
from tests.test_writer import dossie


def _llm_reescrita(hook="Hook refeito.", body="", closing="Fim refeito."):
    corpo = body or " ".join(["palavra"] * 150)
    return ScriptedLLM(responses=[json.dumps({
        "hook": hook, "body": corpo, "closing": closing})])


class TestScan:
    def test_limpo_nao_acha_nada(self):
        assert scan("O modelo ocupa pouco espaco e roda rapido.") == []

    def test_marca_contraste_e_palavra_de_ia(self):
        achados = " ".join(scan("Nao e apenas rapido, e revolucionario. Mergulhe nos dados."))
        assert "contraste" in achados and "mergulh" in achados

    def test_marca_emoji_e_cta_generico(self):
        achados = " ".join(scan("Incrivel! Siga para mais."))
        assert "CTA" in achados


class TestHumanize:
    def test_sem_tell_nao_chama_modelo(self):
        llm = ScriptedLLM(responses=[])
        rel = humanize("Hook.", " ".join(["p"] * 160), "Fim.", dossie(), llm, 150, 225)
        assert not rel.changed and llm.calls == [] and rel.notes == []

    def test_com_tell_reescreve(self):
        llm = _llm_reescrita()
        rel = humanize("Hook.", "Mergulhe: " + " ".join(["p"] * 160), "Fim.",
                       dossie(), llm, 150, 225)
        assert rel.changed and rel.hook == "Hook refeito."
        assert rel.usage.total_tokens >= 0

    def test_reescrita_fora_da_faixa_mantem_original(self):
        llm = _llm_reescrita(body="curto")
        rel = humanize("Hook.", "Mergulhe: " + " ".join(["p"] * 160), "Fim.",
                       dossie(), llm, 150, 225)
        assert not rel.changed and rel.hook == "Hook."
        assert any("faixa" in n for n in rel.notes)

    def test_reescrita_que_muda_numero_mantem_original(self):
        llm = _llm_reescrita(body="Ocupa 5,9 GB e tambem 9999 GB. " + " ".join(["p"] * 150))
        rel = humanize("Hook.", "Mergulhe nos 5,9 GB. " + " ".join(["p"] * 150),
                       "Fim.", dossie(), llm, 150, 225)
        assert not rel.changed
        assert any("numero" in n for n in rel.notes)

    def test_prompt_traz_marcados_e_faixa(self):
        texto = build_prompt("h", "b", "c", ["'mergulhe' (palavra de IA)"], 150, 225)
        assert "mergulhe" in texto and "150" in texto and "225" in texto
