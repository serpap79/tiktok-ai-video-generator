"""Estudio de vozes, sem baixar modelo e sem rede.

O `Narrator` fala contra a porta `TTS`, entao estes testes usam um fundo
falso deterministico (0,1s por palavra). O adaptador Piper real e exercitado
fora da suite, no `voice-fetch` + amostras de `output/vozes/`.
"""

from __future__ import annotations

from agent.voice.engine import Utterance
from agent.voice.library import STYLES, VOICES
from agent.voice.narrate import Narrator, _expandir_marcadores


class Falso:
    """0,1s por palavra a 22050 Hz, gravando o que recebeu."""

    def __init__(self):
        self.textos: list[str] = []

    def speak(self, text, *, speed=1.0, noise=0.667, seed=None):
        self.textos.append(text)
        n = max(1, int(22050 * 0.1 * len(text.split()) / speed))
        return Utterance(sample_rate=22050, pcm16=b"\x01\x02" * n)


class TestMarcacao:
    def test_pausas_viram_silencio_medido(self):
        blocos = _expandir_marcadores("Oi. [PAUSA MEDIA] Tchau.")
        assert blocos == [("fala", "Oi. "), ("pausa", 0.6), ("fala", " Tchau.")]

    def test_marcador_some_antes_da_sintese(self):
        n = Narrator(Falso())
        nar = n.narrate("[PAUSA LONGA] Uma frase.")
        assert nar.pauses_s >= 1.0
        assert all("[PAUSA" not in t for t in n.tts.textos)

    def test_enfase_nao_vai_ao_motor(self):
        falso = Falso()
        Narrator(falso).narrate("Isto e *muito* importante.")
        assert "muito" in falso.textos
        assert all("*" not in t for t in falso.textos)

    def test_frase_quebra_em_falas(self):
        falso = Falso()
        nar = Narrator(falso).narrate("Primeira. Segunda! Terceira?")
        assert nar.utterances == 3 and len(falso.textos) == 3

    def test_duracao_conta_fala_e_pausa(self, tmp_path):
        nar = Narrator(Falso(), pausa_frase_s=0.5).narrate("Uma frase.")
        assert nar.duration_s > 0.5
        caminho = str(tmp_path / "n.wav")
        assert nar.write_wav(caminho) == caminho


class TestBiblioteca:
    def test_toda_voz_tem_modelo_e_licenca(self):
        for v in VOICES.values():
            assert v.modelo_url.startswith("https://") and v.licenca

    def test_estilos_apontam_para_vozes_reais(self):
        for s in STYLES.values():
            assert s.voz in VOICES
            assert 0.5 <= s.velocidade <= 2.0
            assert 0.0 <= s.ruido <= 1.5
            assert s.pausa_frase_s >= 0
