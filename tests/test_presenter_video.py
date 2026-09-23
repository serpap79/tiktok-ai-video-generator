"""Apresentador de video: visemas de pt-BR e a boca posta num rosto filmado.

Os testes daqui medem o **artefato** (o campo de deformacao, os pixels do
ladrilho, a trilha de abertura), nunca o codigo de saida. Foi assim que o MP4
mudo e a boca escancarada apareceram, e os dois defeitos que estes testes
travam -- o risco horizontal na bochecha e a vogal final nasalizada -- tambem
passavam por qualquer teste que so conferisse que a funcao rodou.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.render import presenter_video as pv
from agent.render.visemes import REPOUSO, VISEMAS, fonemas, trilha, visemas


class TempoFalso:
    """O que `voice/edge.WordTiming` entrega, sem TTS nenhum."""

    def __init__(self, text: str, start_s: float, end_s: float) -> None:
        self.text, self.start_s, self.end_s = text, start_s, end_s


# ===========================================================================
# grafema -> fonema
# ===========================================================================


class TestFonemas:
    def test_vogal_final_nao_nasaliza(self):
        """`"" in "mn"` e True em Python, e isso nasalizava TODA palavra.

        A regra quer dizer "vogal seguida de m ou n"; escrita como
        `prox in "mn"`, ela tambem casava com a string vazia do fim da palavra.
        Efeito medido: "bonito" saia `P o T i T o~`, ou seja, toda palavra
        terminada em vogal ganhava boca de vogal nasal. Some sozinho quando a
        comparacao e contra uma tupla de caracteres.
        """
        assert fonemas("bonito")[-1] == "u"
        assert fonemas("papai")[-1] == "i"
        assert fonemas("chuva")[-1] == "a"
        # o nasal de verdade continua nasal
        assert fonemas("bom") == ["P", "o~"]
        assert fonemas("um") == ["u~"]

    def test_c_e_g_amolecem_antes_de_vogal_acentuada(self):
        """`prox in "ei"` nao pega "e" nem "i": "voce" saia com [k] duro."""
        assert "s" in fonemas("você")
        assert fonemas("você") == ["F", "o", "s", "e"]
        assert "S" in fonemas("inteligência")
        assert fonemas("gigante")[0] == "S"     # g antes de i
        assert fonemas("gato")[0] == "K"        # g antes de a segue duro

    def test_m_antes_de_vogal_cola_o_labio_e_no_fim_nao(self):
        """A diferenca que o modulo inteiro existe para acertar."""
        assert fonemas("mamãe")[0] == "P"
        assert VISEMAS[fonemas("mamãe")[0]].fechado
        # "bom" acaba em vogal nasal: nenhum fonema de labio colado no fim
        assert not any(VISEMAS[f].fechado for f in fonemas("bom")[1:])

    def test_e_o_atonos_no_fim_reduzem_como_no_brasil(self):
        assert fonemas("modelo")[-1] == "u"
        assert fonemas("que")[-1] == "i"

    def test_l_no_fim_de_silaba_vira_semivogal(self):
        assert fonemas("sal") == ["s", "a", "u"]
        assert fonemas("lado")[0] == "T"      # l antes de vogal segue alveolar

    def test_palavra_vazia_ou_so_pontuacao_nao_quebra(self):
        assert fonemas("") == []
        assert fonemas("...") == []
        assert visemas("—") == []


# ===========================================================================
# trilha de visemas
# ===========================================================================


class TestTrilha:
    def test_labio_cola_na_consoante_bilabial(self):
        """O /m/ tem de CHEGAR a zero, nao passar perto.

        E o defeito que a envoltoria de energia sozinha nunca resolveu: no /m/
        o audio tem energia (e um som sonoro), entao a boca ficava aberta
        exatamente onde o espectador espera ver o labio colado.
        """
        n, fps = 60, 30
        abertura, _ = trilha([TempoFalso("mamãe", 0.2, 1.4)], n, fps)
        assert abertura.min() < 0.02
        assert abertura.max() > 0.4          # e a vogal continua abrindo

    def test_vogal_aberta_abre_mais_que_vogal_fechada(self):
        n, fps = 90, 30
        a_aberta, _ = trilha([TempoFalso("ah", 0.3, 1.2)], n, fps)
        a_fechada, _ = trilha([TempoFalso("iii", 0.3, 1.2)], n, fps)
        assert a_aberta.max() > a_fechada.max() + 0.3

    def test_arredondada_e_espalhada_saem_com_sinais_opostos(self):
        n, fps = 90, 30
        _, l_u = trilha([TempoFalso("uuu", 0.3, 1.2)], n, fps)
        _, l_i = trilha([TempoFalso("iii", 0.3, 1.2)], n, fps)
        assert l_u.min() < -0.4          # /u/ recolhe
        assert l_i.max() > 0.4           # /i/ espalha

    def test_boca_descansa_no_silencio_entre_palavras(self):
        n, fps = 120, 30
        t = trilha([TempoFalso("ana", 0.1, 0.6), TempoFalso("ana", 3.0, 3.5)],
                   n, fps)[0]
        # no meio (t ~ 1,8s) nao ha palavra nenhuma
        assert t[54] < 0.05

    def test_trilha_sem_palavras_nao_quebra(self):
        abertura, largura = trilha([], 30, 30)
        assert abertura.shape == (30,) and largura.shape == (30,)
        assert float(abertura.max()) == 0.0

    def test_trilha_cobre_todos_os_quadros_pedidos(self):
        abertura, largura = trilha([TempoFalso("teste", 0.0, 0.5)], 45, 30)
        assert len(abertura) == 45 and len(largura) == 45


class TestModular:
    def test_energia_modula_mas_nao_abre_o_que_o_fonema_fechou(self):
        """Audio alto num /m/ nao pode descolar o labio."""
        visema = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        energia = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        assert pv.modular(visema, energia)[0] == 0.0

    def test_piso_impede_que_silencio_apague_a_forma(self):
        visema = np.array([1.0], dtype=np.float32)
        mudo = pv.modular(visema, np.array([0.0], dtype=np.float32))
        alto = pv.modular(visema, np.array([1.0], dtype=np.float32))
        assert mudo[0] == pytest.approx(pv.ENERGIA_PISO)
        assert alto[0] > mudo[0]


# ===========================================================================
# vai-e-vem do clipe base
# ===========================================================================


class BaseFalsa:
    frames = 10
    fps = 10.0


class TestVaiVem:
    def test_vai_e_volta_sem_pular_quadro(self):
        """A emenda do loop e o quadro vizinho, nunca um corte.

        Medido no clipe do THEO: o melhor par de quadros nao vizinhos difere
        6,74/255 contra 1,13 entre vizinhos. Um corte custaria seis vezes o
        movimento normal -- visivel a cada volta.
        """
        b = BaseFalsa()
        seq = [pv.indice_vaivem(k / 10, b) for k in range(40)]
        assert max(abs(a - c) for a, c in zip(seq, seq[1:], strict=False)) <= 1
        assert max(seq) == b.frames - 1 and min(seq) == 0
        assert seq[:10] == list(range(10))          # ida
        assert seq[10] < seq[9]                     # volta

    def test_clipe_de_um_quadro_so_nao_divide_por_zero(self):
        class Um:
            frames = 1
            fps = 24.0

        assert pv.indice_vaivem(3.7, Um()) == 0


# ===========================================================================
# o campo do maxilar
# ===========================================================================


def _rosto_sintetico(h: int = 900, w: int = 700):
    """Um rosto inventado: so os pontos, que e tudo de que a boca precisa."""
    quadro = np.zeros((h, w, 4), dtype=np.uint8)
    # Xadrez: varia nos DOIS eixos, senao um deslocamento horizontal num
    # padrao constante em x nao muda pixel nenhum e o teste passa a toa.
    ys, xs = np.mgrid[0:h, 0:w]
    quadro[:, :, 0] = ((ys // 7 + xs // 7) % 2 * 180 + 40).astype(np.uint8)
    quadro[:, :, 1] = (xs % 251).astype(np.uint8)
    quadro[:, :, 3] = 255
    pts = {
        "labio_sup": (350.0, 500.0), "labio_inf": (350.0, 502.0),
        "boca_esq": (280.0, 501.0), "boca_dir": (420.0, 501.0),
        "queixo": (350.0, 620.0), "subnasal": (350.0, 450.0),
        "olho_esq_baixo": (300.0, 300.0), "olho_dir_baixo": (400.0, 300.0),
        "mand_esq": (230.0, 560.0), "mand_dir": (470.0, 560.0),
        "face_esq": (200.0, 420.0), "face_dir": (500.0, 420.0),
        "nariz_ponta": (350.0, 430.0),
        "labio_sup_out": (350.0, 478.0), "labio_inf_out": (350.0, 528.0),
    }
    return quadro, pts


def _contorno(pts, n: int = 11):
    """O labio interno de uma boca FECHADA, como vem do clipe base.

    Os dois arcos quase se tocam (a boca do clipe esta fechada o tempo todo) e
    correm em sentidos opostos, como na malha do mediapipe: o de baixo da
    esquerda para a direita, o de cima de volta.
    """
    esq, dir_ = pts["boca_esq"][0], pts["boca_dir"][0]
    y = (pts["labio_sup"][1] + pts["labio_inf"][1]) / 2
    u = np.linspace(0.0, 1.0, n, dtype=np.float32)
    arco = np.sin(np.pi * u)
    baixo = np.column_stack([esq + (dir_ - esq) * u, y + 2.0 * arco])
    cima = np.column_stack([dir_ - (dir_ - esq) * u, y - 1.0 * arco[::-1]])
    return baixo.astype(np.float32), cima.astype(np.float32)


def _campo(abertura: float = 1.0):
    """O campo de deslocamento em pixel, no quadro inteiro.

    Mede o campo direto de `campo_maxilar` em vez de tentar recuperar o
    deslocamento dos pixels depois da deformacao. A primeira versao deste
    teste punha uma rampa `valor = y` na imagem e lia o deslocamento de volta;
    parecia esperto e estava errado, porque a rampa satura em 255 e o rosto
    tem 900 linhas -- media zero em todo lugar que importava.
    """
    _, pts = _rosto_sintetico()
    caixa, campo, queda = pv.campo_maxilar(pts, 570.0, abertura, (900, 700))
    inteiro = np.zeros((900, 700), dtype=np.float32)
    x0, y0, x1, y1 = caixa
    inteiro[y0:y1, x0:x1] = campo
    return inteiro, queda


class TestMaxilar:
    def test_o_queixo_desce_e_a_altura_dos_olhos_nao(self):
        campo, queda = _campo(1.0)
        assert campo[618, 350] > 0.8 * queda    # queixo desce de verdade
        assert campo[302, 350] < 0.5            # a dobradica nao anda

    def test_fora_da_boca_o_campo_nao_tem_degrau(self):
        """O risco horizontal atravessando a bochecha, travado.

        A primeira versao descia em bloco tudo abaixo da linha dos labios. No
        centro o vao pintado escondia a emenda; do canto da boca para fora
        sobrava um degrau nu de ate 61 px de uma linha para a seguinte, que na
        tela e um risco atravessando o rosto. O osso gira em torno da orelha, e
        girando o campo cresce devagar onde nao ha boca para abrir.

        Mede o salto entre linhas vizinhas nas colunas FORA do vao.
        """
        campo, _ = _campo(1.0)
        fora = np.r_[np.arange(205, 262), np.arange(438, 495)]   # bochechas
        salto = float(np.abs(np.diff(campo[290:700, fora], axis=0)).max())
        assert salto < 4.0, f"degrau de {salto:.1f} px na bochecha"

    def test_no_centro_da_boca_o_labio_de_baixo_desce_junto(self):
        """O oposto do teste acima, e por isso os dois andam em par.

        Se a rampa central fosse suave a boca nao abriria: o labio de baixo
        desceria uma fracao do que o queixo desce e o vao viraria um risco.
        Logo abaixo da linha dos labios o campo ja tem de estar quase cheio.
        """
        campo, queda = _campo(1.0)
        assert campo[508, 350] > 0.8 * queda

    def test_o_campo_cresce_do_canto_do_maxilar_para_o_queixo(self):
        """Amplitude por distancia ate a dobradica: e o giro do osso.

        O queixo desce tudo, o canto do maxilar desce pouco, e entre os dois a
        queda cai sem volta. Translacao em bloco (a versao antiga) daria o
        mesmo valor nas tres colunas.
        """
        campo, queda = _campo(1.0)
        centro, meio, canto = campo[600, 350], campo[600, 280], campo[600, 215]
        assert centro > meio > canto
        assert centro > 0.9 * queda
        assert canto < 0.5 * queda

    def test_boca_fechada_nao_toca_um_pixel(self):
        quadro, pts = _rosto_sintetico()
        copia = quadro.copy()
        assert pv.abrir_maxilar(quadro, pts, 570.0, 0.0) == 0.0
        assert np.array_equal(quadro, copia)

    def test_o_campo_morre_antes_da_borda_da_caixa(self):
        """Costura vertical na lateral: defeito pago no apresentador parado."""
        campo, _ = _campo(1.0)
        assert float(np.abs(campo[:, :180]).max()) < 0.5
        assert float(np.abs(campo[:, 525:]).max()) < 0.5

    def test_abrir_de_verdade_mexe_nos_pixels(self):
        quadro, pts = _rosto_sintetico()
        copia = quadro.copy()
        queda = pv.abrir_maxilar(quadro, pts, 570.0, 0.9)
        assert queda > 0
        assert not np.array_equal(quadro, copia)


class TestVaoDaBoca:
    """O vao entre os labios -- recortado pelo contorno REAL, nao por elipse."""

    def _tons(self):
        return pv.Tons(interior=(42.0, 20.0, 20.0), dente=(180.0, 178.0, 172.0),
                       lingua=(96.0, 52.0, 52.0))

    def _vao(self, abertura=0.9, largura=0.0, queda=50.0):
        _, pts = _rosto_sintetico()
        return pv.vao_boca(pts, _contorno(pts), queda, abertura, largura,
                           self._tons())

    def test_a_borda_do_vao_e_suave(self):
        """O `ImageDraw` do Pillow nao tem anti-aliasing: a mascara sai em 4x.

        Sem a super-amostragem o poligono sai com ZERO pixel de borda parcial
        -- degrau duro, que na tela le como serrilhado no labio.
        """
        tile, _ = self._vao()
        alfa = np.asarray(tile)[:, :, 3]
        parciais = int(((alfa > 12) & (alfa < 243)).sum())
        assert parciais > 200, f"so {parciais} pixels de borda parcial"

    def test_o_vao_fecha_em_bico_nos_cantos_da_boca(self):
        """Canto de boca nao abre: e onde os dois labios se encontram.

        Sem o perfil de abertura o labio de baixo descia o mesmo tanto no meio
        e na ponta, e o vao saia como uma **barra retangular** de canto em
        esquadro -- defeito visto na ampliacao 2x de 20/09/2026.
        """
        tile, _ = self._vao()
        alfa = np.asarray(tile)[:, :, 3]
        alturas = (alfa > 128).sum(axis=0)
        com_vao = np.flatnonzero(alturas > 0)
        assert com_vao.size > 20
        meio = int(alturas.max())
        # nas bordas do vao a altura tem de ser uma fracao pequena do meio
        ponta = max(int(alturas[com_vao[0] + 2]), int(alturas[com_vao[-1] - 2]))
        assert ponta < 0.35 * meio, f"ponta {ponta} contra meio {meio}"

    def test_o_dente_aparece_no_alto_do_vao_e_nao_no_meio(self):
        """Dente pende do cranio, nao do osso que desce."""
        tile, _ = self._vao(abertura=1.0)
        a = np.asarray(tile)
        dentro = a[:, :, 3] > 200
        linhas = np.flatnonzero(dentro.any(axis=1))
        alto, baixo = linhas[0], linhas[-1]
        cx = a.shape[1] // 2
        coluna = a[alto:baixo + 1, cx, 0].astype(int)
        pico = int(np.argmax(coluna)) / max(len(coluna) - 1, 1)
        assert 0.02 < pico < 0.45, f"pico do dente em {pico:.2f} da altura"
        assert coluna.max() > 120

    def test_o_fundo_do_vao_nao_e_preto(self):
        """Sem lingua o vao le como buraco recortado no rosto."""
        tile, _ = self._vao(abertura=1.0)
        a = np.asarray(tile)
        dentro = a[:, :, 3] > 200
        linhas = np.flatnonzero(dentro.any(axis=1))
        fundo = a[linhas[-3], a.shape[1] // 2, :3].astype(int)
        assert fundo[0] > fundo[2] + 8, "o fundo devia puxar para o vermelho"
        assert fundo[0] > 50

    def test_boca_quase_fechada_nao_desenha_vao(self):
        assert self._vao(abertura=0.02, queda=0.4) is None

    def test_o_vao_nunca_mexe_no_alfa_da_silhueta(self):
        """Mexer no alfa ali abriria um buraco no meio do rosto."""
        quadro, pts = _rosto_sintetico()
        alfa_antes = quadro[:, :, 3].copy()
        pv.falar(quadro, pts, _contorno(pts), 570.0, 0.9, 0.2, self._tons())
        assert np.array_equal(quadro[:, :, 3], alfa_antes)

    def test_falar_de_fato_pinta_o_vao(self):
        quadro, pts = _rosto_sintetico()
        copia = quadro.copy()
        pv.falar(quadro, pts, _contorno(pts), 570.0, 0.9, 0.0, self._tons())
        assert not np.array_equal(quadro, copia)


class TestEspalhar:
    def test_arredondar_e_espalhar_mexem_e_o_neutro_nao(self):
        quadro, pts = _rosto_sintetico()
        copia = quadro.copy()
        pv.espalhar_labios(quadro, pts, 0.0)
        assert np.array_equal(quadro, copia)
        pv.espalhar_labios(quadro, pts, -0.9)
        assert not np.array_equal(quadro, copia)

    def test_o_campo_fica_na_faixa_do_labio(self):
        """Com a caixa antiga (121 px de altura) o esticar ondulava a barba.

        Lábio e o que estica; bochecha e queixo em volta, nao. Travado porque o
        defeito so aparecia na ampliacao 2x, em movimento.
        """
        quadro, pts = _rosto_sintetico()
        antes = quadro[:, :, 0].astype(int).copy()
        pv.espalhar_labios(quadro, pts, 0.9)
        mudou = np.abs(quadro[:, :, 0].astype(int) - antes) > 1
        linhas = np.flatnonzero(mudou.any(axis=1))
        assert linhas.size
        alto = pts["labio_sup_out"][1] - 14
        baixo = pts["labio_inf_out"][1] + 14
        assert linhas[0] >= alto and linhas[-1] <= baixo, (
            f"mexeu de y={linhas[0]} a {linhas[-1]}, fora da faixa "
            f"{alto:.0f}-{baixo:.0f}")


class TestVisemasBasicos:
    def test_repouso_e_boca_fechada(self):
        assert REPOUSO.abertura == 0.0

    def test_todo_visema_do_mapa_e_plausivel(self):
        for nome, v in VISEMAS.items():
            assert 0.0 <= v.abertura <= 1.0, nome
            assert -1.0 <= v.largura <= 1.0, nome
            assert v.peso > 0, nome
