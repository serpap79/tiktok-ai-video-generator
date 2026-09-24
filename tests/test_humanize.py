"""Humanización: scan sin modelo cuando limpio, reescrita trabada cuando sucio.

Adaptación del `blader/humanizer` (MIT) como etapa: las trabas de grounding son
nuestras -- número y franja rotos mantienen el original. Estos tests usan
ScriptedLLM y nunca tocan red.
"""

from __future__ import annotations

import json

from agent.adapters.scripted_llm import ScriptedLLM
from agent.writer.humanize import build_prompt, humanize, scan
from tests.test_writer import dossier


def _llm_reescrita(hook="Hook rehecho.", body="", closing="Fin rehecho."):
    cuerpo = body or " ".join(["palabra"] * 150)
    return ScriptedLLM(responses=[json.dumps({
        "hook": hook, "body": cuerpo, "closing": closing})])


class TestScan:
    def test_limpio_no_encuentra_nada(self):
        assert scan("El modelo ocupa poco espacio y funciona rápido.") == []

    def test_marca_contraste_y_palabra_de_ia(self):
        hallados = " ".join(scan(
            "No es solo rápido, es revolucionario. Hay que sumergir en los datos."))
        assert "contraste" in hallados and "sumergir" in hallados

    def test_marca_emoji_y_cta_generico(self):
        hallados = " ".join(scan("Increíble! Sigue para más."))
        assert "CTA" in hallados


class TestHumanize:
    def test_sin_tell_no_llama_al_modelo(self):
        llm = ScriptedLLM(responses=[])
        rel = humanize("Hook.", " ".join(["p"] * 160), "Fin.", dossier(), llm, 150, 225)
        assert not rel.changed and llm.calls == [] and rel.notes == []

    def test_con_tell_reescribe(self):
        llm = _llm_reescrita()
        rel = humanize("Hook.", "Sumérgete: " + " ".join(["p"] * 160), "Fin.",
                       dossier(), llm, 150, 225)
        assert rel.changed and rel.hook == "Hook rehecho."
        assert rel.usage.total_tokens >= 0

    def test_reescrita_fuera_de_la_franja_mantiene_original(self):
        llm = _llm_reescrita(body="corto")
        rel = humanize("Hook.", "Sumérgete: " + " ".join(["p"] * 160), "Fin.",
                       dossier(), llm, 150, 225)
        assert not rel.changed and rel.hook == "Hook."
        assert any("franja" in n for n in rel.notes)

    def test_reescrita_que_cambia_numero_mantiene_original(self):
        llm = _llm_reescrita(body="Ocupa 5,9 GB y también 9999 GB. " + " ".join(["p"] * 150))
        rel = humanize("Hook.", "Sumérgete en los 5,9 GB. " + " ".join(["p"] * 150),
                       "Fin.", dossier(), llm, 150, 225)
        assert not rel.changed
        assert any("numero" in n for n in rel.notes)

    def test_prompt_trae_marcados_y_franja(self):
        texto = build_prompt("h", "b", "c", ["'sumérgete' (palabra de IA)"], 150, 225)
        assert "sumérgete" in texto and "150" in texto and "225" in texto
