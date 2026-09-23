"""Pronuncia, legenda karaoke e o renderizador ffmpeg -- sem rede e sem ffmpeg.

O defeito que originou tudo isto veio do primeiro video do piloto (20/09):
"Gemini" narrado como "Zemini". A correcao tem de valer na VOZ e nao
vazar para a LEGENDA, e o tempo da legenda tem de continuar certo.
"""

from __future__ import annotations

from pathlib import Path

from agent.adapters.ffmpeg_renderer import build_command, suspicious_names
from agent.render.subtitles import align, build_ass, chunks
from agent.voice.edge import WordTiming
from agent.voice.pronounce import respell, spell


class TestPronuncia:
    def test_gemini_vira_djemini_so_na_voz(self):
        r = respell("mas o Gemini provou o contrário.")
        assert "Djémini" in r.text and "Gemini" in " ".join(r.original)
        assert r.changes == [("Gemini", "Djémini")]

    def test_pontuacao_de_borda_fica_no_lugar(self):
        r = respell("Quem ganhou, o Google?")
        assert r.spoken[-1] == "Gúgou?"

    def test_nome_composto_mantem_o_alinhamento(self):
        r = respell("segundo o Wall Street Journal ontem")
        assert r.groups == [1, 1, 3, 0, 0, 1]
        assert " ".join(r.spoken) == "segundo o Uól Strít Djôrnal ontem"

    def test_sigla_tecnica_e_soletrada_e_versao_separada(self):
        r = respell("A OpenAI lançou o GPT-5 numa RTX5090.")
        assert "Ôupen Ei Ái" in r.text
        assert "Gê Pê Tê 5" in r.text
        assert "érre tê xis 5090." in r.text

    def test_palavra_comum_nao_muda(self):
        r = respell("Discorda? Fala nos comentários.")
        assert r.changes == [] and r.text == "Discorda? Fala nos comentários."

    def test_soletrar(self):
        assert spell("GPU") == "gê pê ú"
        assert spell("APIs") == "á pê ís"


class TestLegenda:
    def test_tempo_da_voz_volta_para_a_grafia_original(self):
        r = respell("mas o Gemini provou")
        tempos = [WordTiming("mas", 0.0, 0.2), WordTiming("o", 0.2, 0.3),
                  WordTiming("Djémini", 0.3, 0.9), WordTiming("provou", 0.9, 1.3)]
        palavras = align(r, tempos)
        assert [p.text for p in palavras] == ["mas", "o", "Gemini", "provou"]
        assert (palavras[2].start, palavras[2].end) == (0.3, 0.9)

    def test_nome_composto_acende_junto(self):
        r = respell("o Wall Street Journal disse")
        tempos = [WordTiming("o", 0.0, 0.1), WordTiming("Uól", 0.1, 0.3),
                  WordTiming("Strít", 0.3, 0.5), WordTiming("Djôrnal", 0.5, 0.9),
                  WordTiming("disse", 0.9, 1.2)]
        palavras = align(r, tempos)
        assert palavras[1].text == "Wall" and (palavras[1].start, palavras[1].end) == (0.1, 0.9)
        assert palavras[2].start == palavras[1].start  # "Street" acende com "Wall"

    def test_palavra_sem_evento_e_interpolada(self):
        r = respell("um dois tres")
        palavras = align(r, [WordTiming("um", 0.0, 0.2), WordTiming("tres", 0.6, 0.8)])
        assert 0.2 <= palavras[1].start <= palavras[1].end <= 0.6

    def test_blocos_de_ate_tres_palavras_quebram_na_pontuacao(self):
        from agent.render.subtitles import TimedWord
        ws = [TimedWord(t, i, i + 1) for i, t in enumerate(
            "Dizem que a IA é segura, mas o Gemini provou".split())]
        blocos = [" ".join(w.text for w in b) for b in chunks(ws)]
        assert blocos[0] == "Dizem que a" and blocos[1] == "IA é segura,"
        assert all(len(b.split()) <= 3 for b in blocos)

    def test_ass_destaca_a_palavra_ativa_na_cor_da_marca(self, tmp_path):
        from agent.render.subtitles import TimedWord
        ws = [TimedWord("o", 0.0, 0.2), TimedWord("Gemini", 0.2, 0.8)]
        ass = build_ass(ws, tmp_path / "l.ass", accent="#39FF88",
                        font_family="SeuCanal Display", duration=1.2).read_text()
        assert "Style: Fala,SeuCanal Display,84" in ass
        # &HAABBGGRR: verde #39FF88 vira 88FF39.
        assert "{\\c&H0088FF39}Gemini" in ass
        assert ass.count("Dialogue:") == 2


class TestConferencia:
    def test_nome_que_o_ouvinte_nao_reconheceu_e_suspeito(self):
        roteiro = "Dizem que a IA e segura. Mas o Gemini provou o contrario."
        assert suspicious_names(roteiro, "mas o zemini provou o contrario") == ["Gemini"]
        assert suspicious_names(roteiro, "mas o Gemini provou o contrario") == []


class TestComandoFfmpeg:
    def test_fatias_de_cinco_segundos_cobrem_a_narracao(self, tmp_path):
        clips = [tmp_path / f"c{i}.mp4" for i in range(3)]
        cmd = build_command(clips, tmp_path / "n.mp3", tmp_path / "l.ass",
                            tmp_path / "o.mp4", 12.4)
        grafo = cmd[cmd.index("-filter_complex") + 1]
        assert "concat=n=3" in grafo
        assert "crop=1080:1920" in grafo and "subtitles=" in grafo
        assert cmd[cmd.index("-t", cmd.index("-filter_complex")) + 1] == "12.40"

    def test_clipe_reusado_mostra_outro_trecho(self, tmp_path):
        clips = [tmp_path / "c0.mp4", tmp_path / "c1.mp4"]
        cmd = build_command(clips, tmp_path / "n.mp3", tmp_path / "l.ass",
                            tmp_path / "o.mp4", 18.0)
        inicios = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-ss"]
        assert inicios == ["0.30", "0.30", "5.30", "5.30"]

    def test_narracao_e_o_audio_mapeado(self, tmp_path):
        cmd = build_command([tmp_path / "c.mp4"], tmp_path / "n.mp3", tmp_path / "l.ass",
                            tmp_path / "o.mp4", 4.0)
        assert "-map" in cmd and "1:a" in cmd
        assert str(Path(tmp_path / "n.mp3")) in cmd
