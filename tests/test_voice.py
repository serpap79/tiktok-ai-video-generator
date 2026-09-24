"""Estudio de voces, sin descargar modelo y sin red.

El `Narrator` habla contra el puerto `TTS`, asi que estos tests usan un doble
falso determinista (0,1s por palabra). El adaptador Piper real se ejercita
fuera de la suite, en `voice-fetch` + muestras de `output/voces/`.
"""

from __future__ import annotations

from agent.voice.engine import Utterance
from agent.voice.library import STYLES, VOICES
from agent.voice.narrate import Narrator, _expandir_marcadores


class Falso:
    """0,1s por palabra a 22050 Hz, grabando lo que recibe."""

    def __init__(self):
        self.textos: list[str] = []

    def speak(self, text, *, speed=1.0, noise=0.667, seed=None):
        self.textos.append(text)
        n = max(1, int(22050 * 0.1 * len(text.split()) / speed))
        return Utterance(sample_rate=22050, pcm16=b"\x01\x02" * n)


class TestMarcas:
    def test_pausas_se_convierten_en_silencio_medido(self):
        bloques = _expandir_marcadores("Hola. [PAUSA MEDIA] Adios.")
        assert bloques == [("habla", "Hola. "), ("pausa", 0.6), ("habla", " Adios.")]

    def test_la_marca_desaparece_antes_de_la_sintesis(self):
        n = Narrator(Falso())
        nar = n.narrate("[PAUSA LARGA] Una frase.")
        assert nar.pauses_s >= 1.0
        assert all("[PAUSA" not in t for t in n.tts.textos)

    def test_el_enfasis_no_llega_al_motor(self):
        falso = Falso()
        Narrator(falso).narrate("Esto es *muy* importante.")
        assert "muy" in falso.textos
        assert all("*" not in t for t in falso.textos)

    def test_la_frase_se_rompe_en_habladas(self):
        falso = Falso()
        nar = Narrator(falso).narrate("Primera. Segunda! Tercera?")
        assert nar.utterances == 3 and len(falso.textos) == 3

    def test_la_duracion_cuenta_habla_y_pausa(self, tmp_path):
        nar = Narrator(Falso(), pausa_frase_s=0.5).narrate("Una frase.")
        assert nar.duration_s > 0.5
        camino = str(tmp_path / "n.wav")
        assert nar.write_wav(camino) == camino


class TestBiblioteca:
    def test_toda_voz_tiene_modelo_y_licencia(self):
        for v in VOICES.values():
            assert v.modelo_url.startswith("https://") and v.licencia

    def test_los_estilos_apuntan_a_voces_reales(self):
        for s in STYLES.values():
            assert s.voz in VOICES
            assert 0.5 <= s.velocidad <= 2.0
            assert 0.0 <= s.ruido <= 1.5
            assert s.pausa_frase_s >= 0
