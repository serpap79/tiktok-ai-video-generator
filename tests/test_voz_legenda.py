"""Pronunciacion, leyenda karaoke y el renderizador ffmpeg -- sin red y sin ffmpeg.

El defecto que origino todo esto vino del primer video del piloto (20/09):
"Gemini" narrado como "Zemini". La correccion tiene que valer en la VOZ y no
filtrarse a la LEYENDA, y el tiempo de la leyenda tiene que seguir correcto.
"""

from __future__ import annotations

from pathlib import Path

from agent.adapters.ffmpeg_renderer import build_command, suspicious_names
from agent.render.subtitles import align, build_ass, chunks
from agent.voice.edge import WordTiming
from agent.voice.pronounce import respell, spell


class TestPronunciacion:
    def test_gemini_vira_yemini_solo_en_la_voz(self):
        r = respell("pero el Gemini probó lo contrario.")
        assert "Yemini" in r.text and "Gemini" in " ".join(r.original)
        assert r.changes == [("Gemini", "Yemini")]

    def test_puntuacion_de_borde_se_queda_en_su_sitio(self):
        r = respell("Quién ganó, Google?")
        assert r.spoken[-1] == "Gugol?"

    def test_nombre_compuesto_mantiene_el_alineamiento(self):
        r = respell("según el Wall Street Journal ayer")
        assert r.groups == [1, 1, 3, 0, 0, 1]
        assert " ".join(r.spoken) == "según el Uól Strit Yórnal ayer"

    def test_sigla_tecnica_se_deletrea_y_version_separada(self):
        r = respell("La OpenAI lanzó el GPT-5 en una RTX5090.")
        assert "Oupen EI Ei" in r.text
        assert "YipiTi 5" in r.text
        assert "erre te equis 5090." in r.text

    def test_palabra_comun_no_cambia(self):
        r = respell("Discord? Escribe en los comentarios.")
        assert r.changes == [("Discord", "Díscord")]
        r2 = respell("Escribe en los comentarios.")
        assert r2.changes == [] and r2.text == "Escribe en los comentarios."

    def test_deletrear(self):
        assert spell("GPU") == "ge pé u"
        assert spell("APIs") == "a pé is"


class TestLeyenda:
    def test_el_tiempo_de_la_voz_vuelve_a_la_grafia_original(self):
        r = respell("pero el Gemini probó")
        tiempos = [WordTiming("pero", 0.0, 0.2), WordTiming("el", 0.2, 0.3),
                   WordTiming("Yemini", 0.3, 0.9), WordTiming("probó", 0.9, 1.3)]
        palabras = align(r, tiempos)
        assert [p.text for p in palabras] == ["pero", "el", "Gemini", "probó"]
        assert (palabras[2].start, palabras[2].end) == (0.3, 0.9)

    def test_nombre_compuesto_se_enciende_junto(self):
        r = respell("el Wall Street Journal dijo")
        tiempos = [WordTiming("el", 0.0, 0.1), WordTiming("Uól", 0.1, 0.3),
                   WordTiming("Strit", 0.3, 0.5), WordTiming("Yórnal", 0.5, 0.9),
                   WordTiming("dijo", 0.9, 1.2)]
        palabras = align(r, tiempos)
        assert palabras[1].text == "Wall" and (palabras[1].start, palabras[1].end) == (0.1, 0.9)
        assert palabras[2].start == palabras[1].start  # "Street" se enciende con "Wall"

    def test_palabra_sin_evento_se_interpola(self):
        r = respell("un dos tres")
        palabras = align(r, [WordTiming("un", 0.0, 0.2), WordTiming("tres", 0.6, 0.8)])
        assert 0.2 <= palabras[1].start <= palabras[1].end <= 0.6

    def test_bloques_de_hasta_tres_palabras_rompen_en_la_puntuacion(self):
        from agent.render.subtitles import TimedWord
        ws = [TimedWord(t, i, i + 1) for i, t in enumerate(
            "Dicen que la IA es segura, pero el Gemini probó".split())]
        bloques = [" ".join(w.text for w in b) for b in chunks(ws)]
        assert bloques[0] == "Dicen que la" and bloques[1] == "IA es segura,"
        assert all(len(b.split()) <= 3 for b in bloques)

    def test_ass_destaca_la_palabra_activa_en_el_color_de_la_marca(self, tmp_path):
        from agent.render.subtitles import TimedWord
        ws = [TimedWord("el", 0.0, 0.2), TimedWord("Gemini", 0.2, 0.8)]
        ass = build_ass(ws, tmp_path / "l.ass", accent="#39FF88",
                        font_family="CircuitoCero Display", duration=1.2).read_text()
        assert "Style: Fala,CircuitoCero Display,84" in ass
        # &HAABBGGRR: verde #39FF88 se vuelve 88FF39.
        assert "{\\c&H0088FF39}Gemini" in ass
        assert ass.count("Dialogue:") == 2


class TestConferencia:
    def test_nombre_que_el_oyente_no_reconocio_es_sospechoso(self):
        guion = "Dicen que la IA es segura. Pero el Yemini probó lo contrario."
        # Si la voz leyó el nombre de otra manera ("Geminy"), Whisper no lo
        # devuelve y el nombre entero es sospechoso.
        assert suspicious_names(guion, "pero el Geminy probo lo contrario") == ["Yemini"]
        # Si la transcripcion contiene el nombre tal cual, no hay sospecha.
        assert suspicious_names(guion, "pero el yemini probo lo contrario") == []


class TestComandoFfmpeg:
    def test_tajadas_de_cinco_segundos_cubren_la_narracion(self, tmp_path):
        clips = [tmp_path / f"c{i}.mp4" for i in range(3)]
        cmd = build_command(clips, tmp_path / "n.mp3", tmp_path / "l.ass",
                            tmp_path / "o.mp4", 12.4)
        grafo = cmd[cmd.index("-filter_complex") + 1]
        assert "concat=n=3" in grafo
        assert "crop=1080:1920" in grafo and "subtitles=" in grafo
        assert cmd[cmd.index("-t", cmd.index("-filter_complex")) + 1] == "12.40"

    def test_clip_reusado_muestra_otro_trecho(self, tmp_path):
        clips = [tmp_path / "c0.mp4", tmp_path / "c1.mp4"]
        cmd = build_command(clips, tmp_path / "n.mp3", tmp_path / "l.ass",
                            tmp_path / "o.mp4", 18.0)
        inicios = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-ss"]
        assert inicios == ["0.30", "0.30", "5.30", "5.30"]

    def test_la_narracion_es_el_audio_mapeado(self, tmp_path):
        cmd = build_command([tmp_path / "c.mp4"], tmp_path / "n.mp3", tmp_path / "l.ass",
                            tmp_path / "o.mp4", 4.0)
        assert "-map" in cmd and "1:a" in cmd
        assert str(Path(tmp_path / "n.mp3")) in cmd
