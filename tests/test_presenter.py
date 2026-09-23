"""Apresentador animado: o que se mede no quadro, sem rede e sem ffmpeg.

O defeito que motivou este modulo passava em qualquer teste de codigo de saida:
o comando rodava, o MP4 saia, a duracao batia -- e o que aparecia na tela era um
retangulo de rosto com um movimento de janela por cima. Por isso aqui se mede
**o pixel**: a silhueta, a costura das camadas, a boca e a palpebra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent.render import presenter as pr

CUTOUTS = Path(__file__).resolve().parent.parent / "brand" / "assets" / "presenters"


@pytest.fixture(scope="module")
def retrato() -> pr.Retrato:
    png = CUTOUTS / "iris.png"
    if not png.exists() or not png.with_suffix(".json").exists():
        pytest.skip("recorte do apresentador ausente (scripts/make_presenter_cutouts.py)")
    return pr.carregar(png)


@pytest.fixture(scope="module")
def anim(retrato: pr.Retrato) -> pr.Animador:
    return pr.Animador(retrato, "#39FF88", "Íris")


@pytest.fixture(scope="module")
def encenacao() -> pr.Encenacao:
    return pr.plan(pr.Batidas(hook_end=3.4, closing_start=52.0, duration=68.0))


def _alfa(img: Image.Image) -> np.ndarray:
    return np.asarray(img)[:, :, 3]


class TestSilhueta:
    def test_o_quadro_nao_e_um_retangulo(self, anim, encenacao):
        """O defeito original: uma janela 340x640 recortava a pessoa em caixa.

        Prova de que isto acabou: o ombro e muito mais largo que a testa, e os
        cantos da caixa que contem o apresentador estao vazios. Retangulo nao
        tem nenhuma das duas coisas.
        """
        a = _alfa(anim.quadro(2.0, 0.4, 0.0, encenacao.marcas))
        linhas = np.flatnonzero(a.max(axis=1) > 8)
        colunas = np.flatnonzero(a.max(axis=0) > 8)
        y0, y1 = int(linhas[0]), int(linhas[-1])
        x0, x1 = int(colunas[0]), int(colunas[-1])
        alto = y1 - y0
        testa = int((a[y0 + int(0.06 * alto)] > 128).sum())
        ombro = int((a[y0 + int(0.88 * alto)] > 128).sum())
        assert 0 < testa < 0.65 * ombro
        for cy, cx in ((y0 + 2, x0 + 2), (y0 + 2, x1 - 2)):
            assert a[cy, cx] < 40

    def test_nada_encosta_na_borda_de_cima_da_camada(self, anim, encenacao):
        """Se o topo da camada tivesse tinta, o cabelo estaria cortado a faca."""
        a = _alfa(anim.quadro(2.0, 0.4, 0.0, encenacao.marcas))
        assert a[0].max() == 0

    def test_a_base_some_em_degrade(self, retrato):
        """Busto cortado reto denuncia recorte; o alfa tem de cair ate zero."""
        a = np.asarray(retrato.imagem)[:, :, 3]
        assert a[-1].max() < 12
        assert a[int(a.shape[0] * 0.62)].max() > 240


class TestCamadas:
    def test_cabeca_mais_tronco_dao_o_alfa_original(self, anim, retrato):
        """Mascara complementar: sem ela, a costura vira fenda ou fantasma."""
        soma = anim._alfa_cab.astype(np.int32) + anim._alfa_tro.astype(np.int32)
        alfa = np.asarray(pr.realce(retrato.imagem, "#39FF88"))[:, :, 3].astype(np.int32)
        assert int(np.abs(soma - alfa).max()) <= 1

    def test_a_costura_fica_no_peito_nao_no_pescoco(self, retrato):
        """Cabeca que gira sobre pescoco parado e boneco de ventriloquo."""
        peso = pr.mascara_cabeca(retrato)[:, 0]
        assert peso[int(retrato.p("queixo")[1])] > 0.99
        assert peso[retrato.neck_y] > 0.9
        meio = int(np.argmin(np.abs(peso - 0.5)))
        assert meio > retrato.neck_y + 0.3 * retrato.face_height


class TestBoca:
    def test_silencio_nao_mexe_no_rosto(self, anim):
        base = anim._rgb.copy()
        pr.aplicar_maxilar(base, anim.maxilar, 0.0)
        assert np.array_equal(base, anim._rgb)
        assert pr.tile_boca(anim.maxilar, 0.0) is None

    def test_o_queixo_desce_e_a_boca_abre_com_a_voz(self, anim, retrato):
        aberto = anim._rgb.copy()
        pr.aplicar_maxilar(aberto, anim.maxilar, 1.0)
        queixo_y = int(retrato.p("queixo")[1])
        queixo_x = int(retrato.p("queixo")[0])
        # A linha do queixo passou a mostrar o que estava acima dela.
        antes = anim._rgb[queixo_y, queixo_x - 40:queixo_x + 40].astype(int)
        depois = aberto[queixo_y, queixo_x - 40:queixo_x + 40].astype(int)
        assert np.abs(antes - depois).mean() > 3
        tile, _ = pr.tile_boca(anim.maxilar, 1.0)
        assert tile.height > 0.25 * anim.maxilar.boca_w

    def test_a_abertura_cresce_junto_com_a_envoltoria(self, anim):
        alturas = [pr.tile_boca(anim.maxilar, v)[0].height for v in (0.3, 0.6, 1.0)]
        assert alturas[0] < alturas[1] < alturas[2]

    def test_o_labio_de_cima_fica_parado(self, anim, retrato):
        """So o de baixo acompanha o maxilar -- e o que separa fala de careta."""
        x0, y0, _, _ = anim.maxilar.caixa
        acima = int(retrato.p("labio_sup")[1]) - 14 - y0
        aberto = anim._rgb.copy()
        pr.aplicar_maxilar(aberto, anim.maxilar, 1.0)
        faixa = slice(y0 + max(acima - 6, 0), y0 + acima)
        assert np.array_equal(anim._rgb[faixa], aberto[faixa])


class TestPiscada:
    def test_a_palpebra_cobre_o_olho(self, anim, retrato):
        fechado = anim._rgb.copy()
        pr.aplicar_piscada(fechado, anim.olhos, 1.0)
        x0, y0, x1, y1 = anim.olhos[0].caixa
        olho_antes = anim._rgb[y0:y1, x0:x1].astype(int)
        olho_depois = fechado[y0:y1, x0:x1].astype(int)
        assert np.abs(olho_antes - olho_depois).mean() > 4
        # Fora da caixa do olho nada mudou.
        assert np.array_equal(anim._rgb[:y0], fechado[:y0])

    def test_olho_aberto_nao_e_tocado(self, anim):
        igual = anim._rgb.copy()
        pr.aplicar_piscada(igual, anim.olhos, 0.0)
        assert np.array_equal(igual, anim._rgb)

    def test_piscada_e_reprodutivel_e_tem_ritmo_de_gente(self):
        a = pr.piscadas(60.0, "Bonsai 2")
        assert a == pr.piscadas(60.0, "Bonsai 2")
        assert a != pr.piscadas(60.0, "outro tema")
        assert 8 <= len(a) <= 30
        assert all(abs((f - i) - pr.PISCADA_S) < 1e-9 for i, f in a)

    def test_fecho_abre_e_fecha_dentro_da_janela(self):
        (ini, fim), = [pr.piscadas(6.0, "x")[0]]
        assert pr.fecho_em([(ini, fim)], ini - 0.01) == 0.0
        assert pr.fecho_em([(ini, fim)], (ini + fim) / 2) > 0.9
        assert pr.fecho_em([(ini, fim)], fim + 0.01) == 0.0


class TestEnvoltoria:
    def test_ataque_e_mais_rapido_que_o_relaxamento(self):
        """Boca abre rapido e fecha devagar; sem isso ela treme em consoante."""
        degrau = np.concatenate([np.ones(30, np.float32), np.zeros(30, np.float32)])
        y = pr._ataque_relaxamento(degrau, 30)
        subida = int(np.argmax(y >= 0.5))          # quadros ate a boca abrir
        descida = int(np.argmax(y[30:] < 0.5))     # quadros ate ela fechar
        assert subida < descida
        assert y[0] > 0.5 and y[29] > 0.95 and y[-1] < 0.02


class TestEncenacao:
    def test_grande_na_chamada_pequeno_no_corpo_volta_no_fim(self, encenacao):
        assert pr.pose(encenacao.marcas, 2.0).altura == pr.ALTURA_CHAMADA
        assert pr.pose(encenacao.marcas, 30.0).altura == pr.ALTURA_CORPO
        assert pr.pose(encenacao.marcas, 60.0).altura == pr.ALTURA_FECHAMENTO

    def test_entra_aparecendo_em_vez_de_surgir_de_uma_vez(self, encenacao):
        assert pr.pose(encenacao.marcas, 0.0).alfa == 0.0
        assert 0.0 < pr.pose(encenacao.marcas, 0.2).alfa < 1.0
        assert pr.pose(encenacao.marcas, 0.5).alfa == 1.0

    def test_a_travessia_e_suave(self, encenacao):
        alturas = [pr.pose(encenacao.marcas, 3.4 + k * 0.1).altura for k in range(8)]
        assert alturas == sorted(alturas, reverse=True)
        saltos = np.diff(alturas)
        assert abs(saltos[0]) < abs(saltos[len(saltos) // 2])

    def test_a_legenda_so_entra_quando_ele_encolhe(self, encenacao):
        """Enquanto ele esta grande quem escreve o gancho e o cartao."""
        assert encenacao.legenda_inicio == encenacao.cartao_s
        assert pr.pose(encenacao.marcas, encenacao.legenda_inicio).altura \
            == pr.ALTURA_CORPO
        assert encenacao.legenda_y == pr.Y_LEGENDA_COM_APRESENTADOR

    def test_a_legenda_passa_acima_da_cabeca_no_canto(self, encenacao):
        """Mede o vao entre o texto (ate ~+50 do centro) e o topo do avatar."""
        for t in (30.0, 60.0):
            p = pr.pose(encenacao.marcas, t)
            topo_avatar = p.base - p.altura
            assert topo_avatar > encenacao.legenda_y + 60

    def test_roteiro_curto_nao_inverte_as_batidas(self):
        enc = pr.plan(pr.Batidas(hook_end=9.0, closing_start=2.0, duration=11.0))
        tempos = [m.t for m in enc.marcas]
        assert tempos == sorted(tempos)
        assert all(0 <= t <= 11.0 for t in tempos)


class TestBatidas:
    def test_batidas_saem_do_tempo_de_palavra(self):
        class P:
            def __init__(self, texto, s, e):
                self.text, self.start, self.end = texto, s, e

        palavras = [P("Um", 0.0, 0.3), P("gancho.", 0.3, 1.1),
                    P("Corpo", 1.2, 1.8), P("aqui.", 1.8, 2.4),
                    P("Fecha", 2.5, 3.0), P("agora?", 3.0, 3.6)]
        b = pr.batidas_por_tempo("Um gancho.", "Fecha agora?", palavras, 4.0)
        assert b.hook_end == pytest.approx(1.1)
        assert b.closing_start == pytest.approx(2.5)

    def test_sem_tempo_de_palavra_estima_por_proporcao(self):
        b = pr.batidas_estimadas(n_hook=10, n_close=10, total=100, duracao=60.0)
        assert b.hook_end == pytest.approx(6.0)
        assert b.closing_start == pytest.approx(54.0)


class TestGeometria:
    def test_a_caixa_de_saida_cobre_a_camada_girada(self):
        fonte = (10, 20, 210, 320)
        alvo = (400.0, 500.0)
        pivo = (100.0, 300.0)
        caixa = pr._caixa_saida(fonte, pivo, alvo, escala=0.8, graus=3.0)
        assert caixa is not None
        ox, oy, ow, oh = caixa
        for px in (fonte[0], fonte[2]):
            for py in (fonte[1], fonte[3]):
                dx, dy = pr._giro(px - pivo[0], py - pivo[1], 3.0, 0.8)
                assert ox <= alvo[0] + dx <= ox + ow
                assert oy <= alvo[1] + dy <= oy + oh

    def test_a_base_do_recorte_pousa_onde_a_encenacao_mandou(self, anim, encenacao):
        """A altura pedida e a altura que aparece: escala errada e avatar torto."""
        p = pr.pose(encenacao.marcas, 30.0)
        a = _alfa(anim.quadro(30.0, 0.0, 0.0, encenacao.marcas))
        linhas = np.flatnonzero((a > 128).max(axis=1))
        alto = int(linhas[-1] - linhas[0])
        # O recorte tem margem transparente em volta, entao a tinta e um pouco
        # menor que a altura pedida -- mas nao muito.
        assert 0.80 * p.altura < alto <= p.altura


class TestCosturaEntreCamadas:
    def test_as_duas_camadas_giram_no_mesmo_ponto(self, anim, encenacao):
        """Pivo por camada as separaria -- e a costura e justamente ali.

        Com angulo de cabeca no maximo, a soma dos alfas do quadro composto
        continua parecida com a do quadro sem giro nenhum: nada de fenda nem de
        contorno duplicado no peito.
        """
        parado = _alfa(anim.quadro(0.0, 0.0, 0.0,
                                   [pr.Marca(t=0.0, altura=600, cx=540, base=1700),
                                    pr.Marca(t=9.9, altura=600, cx=540, base=1700)]))
        area = int((parado > 128).sum())
        for t in (1.1, 2.7, 4.3):
            a = _alfa(anim.quadro(t, 1.0, 0.0,
                                  [pr.Marca(t=0.0, altura=600, cx=540, base=1700),
                                   pr.Marca(t=9.9, altura=600, cx=540, base=1700)]))
            assert abs(int((a > 128).sum()) - area) < 0.06 * area

    def test_so_a_cabeca_responde_a_silaba_forte(self, anim):
        """Se o quadro inteiro andasse junto, seria figurinha deslizando.

        Entre abertura 0 e 1 no MESMO instante so muda `giro_c` (o acento da
        cabeca) e a boca -- as duas coisas da cabeca. A base do busto, abaixo
        da costura, tem de sair identica.
        """
        marcas = [pr.Marca(t=0.0, altura=900, cx=540, base=1700),
                  pr.Marca(t=9.9, altura=900, cx=540, base=1700)]
        calmo = np.asarray(anim.quadro(3.0, 0.0, 0.0, marcas)).astype(int)
        forte = np.asarray(anim.quadro(3.0, 1.0, 0.0, marcas)).astype(int)
        d = np.abs(calmo - forte).mean(axis=2)
        linhas = np.flatnonzero(
            (_alfa(anim.quadro(3.0, 0.0, 0.0, marcas)) > 128).any(axis=1))
        y0, y1 = int(linhas.min()), int(linhas.max())
        alto = y1 - y0
        cabeca = d[y0:y0 + int(0.25 * alto)].mean()
        peito = d[y1 - int(0.12 * alto):y1].mean()
        assert cabeca > 4 * max(peito, 0.05), f"cabeca {cabeca:.2f} x peito {peito:.2f}"


class TestPortaoNoQuadro:
    """O portao que impede o MP4 sem apresentador de passar calado.

    A camada pode sair perfeita e nao chegar ao quadro: basta um rotulo de
    fluxo trocado no `overlay`. Nada disso levanta erro -- sai um MP4 valido,
    com a duracao certa, sem avatar. Mesmo defeito do MP4 mudo.

    A cena e sintetica de proposito: o rosto e um padrao com estrutura, o
    b-roll e outro, e a pos-producao entra como eq + vinheta por cima de tudo.
    """

    ALTO, LARGO = 600, 400
    X, Y = 100, 300
    MW, MH = 200, 200

    @classmethod
    def _pecas(cls, com_avatar: bool, cartao_forte: bool = False):
        """Devolve (final, cor da camada, mascara da camada).

        `cartao_forte` liga o cartao do gancho logo acima do apresentador --
        e o caso que derrubava a medida antiga.
        """
        rng = np.random.default_rng(7)
        # Rosto: estrutura propria. B-roll: outra estrutura, sem relacao.
        rosto = rng.integers(40, 220, (cls.MH, cls.MW)).astype(np.uint8)
        broll = rng.integers(30, 200, (cls.ALTO, cls.LARGO)).astype(np.uint8)

        mascara = np.zeros((cls.MH, cls.MW), dtype=np.uint8)
        mascara[40:170, 60:150] = 255
        cor = np.where(mascara > 0, rosto, 0).astype(np.uint8)

        final = broll.astype(np.int16)
        if com_avatar:
            recorte = final[cls.Y:cls.Y + cls.MH, cls.X:cls.X + cls.MW]
            recorte[mascara > 0] = rosto[mascara > 0]
        # pos-producao: eq (ganho + offset) e vinheta (escurece as bordas)
        final = final * 1.12 + 10
        final[:80] -= 26
        final[-80:] -= 26
        if cartao_forte:
            final[cls.Y - 120:cls.Y - 10, 20:cls.LARGO - 20] = 235
        return (np.clip(final, 0, 255).astype(np.uint8), cor, mascara)

    def _rodar(self, com_avatar, tmp_path, monkeypatch, cartao_forte=False):
        final, cor, mascara = self._pecas(com_avatar, cartao_forte)
        quadros = iter([final, cor, mascara])
        monkeypatch.setattr(pr, "_um_quadro", lambda *a, **k: next(quadros))
        camada = pr.Camada(path=tmp_path / "apr.mp4", x=self.X, y=self.Y,
                           width=self.MW, height=self.MH, subtitle_y=940,
                           subtitle_start=4.0, card_s=4.0, frames=120, fps=30,
                           seconds=4.0)
        return pr.conferir_apresentador(tmp_path / "f.mp4", camada, tmp_path)

    def test_avatar_no_quadro_passa(self, tmp_path, monkeypatch):
        assert "OK" in self._rodar(True, tmp_path, monkeypatch)

    def test_avatar_que_sumiu_e_acusado(self, tmp_path, monkeypatch):
        assert "NAO CHEGOU AO QUADRO" in self._rodar(False, tmp_path, monkeypatch)

    def test_cartao_do_gancho_nao_derruba_o_veredito(self, tmp_path, monkeypatch):
        """O falso negativo que aposentou a medida antiga, travado.

        A medida antiga comparava final x bruto dentro da mascara contra um
        anel em volta. Com o apresentador fotorrealista e o cartao do gancho
        logo acima dele, o anel mudava tanto quanto a mascara e o veredito
        saia NAO CHEGOU em video que tinha o apresentador no quadro -- olhado
        a olho no artefato de 20/09/2026. Comparar com a propria camada em vez
        de com o bruto nao se importa com o que acontece do lado de fora.
        """
        assert "OK" in self._rodar(True, tmp_path, monkeypatch, cartao_forte=True)

    def test_a_medida_e_contra_a_camada_e_nao_contra_o_bruto(self, tmp_path,
                                                             monkeypatch):
        texto = self._rodar(True, tmp_path, monkeypatch)
        assert "correlacao com a camada" in texto

    def test_mascara_vazia_vira_aviso(self, tmp_path, monkeypatch):
        vazia = np.zeros((self.MH, self.MW), dtype=np.uint8)
        final, cor, _ = self._pecas(True)
        quadros = iter([final, cor, vazia])
        monkeypatch.setattr(pr, "_um_quadro", lambda *a, **k: next(quadros))
        camada = pr.Camada(path=tmp_path / "apr.mp4", x=self.X, y=self.Y,
                           width=self.MW, height=self.MH, subtitle_y=940,
                           subtitle_start=4.0, card_s=4.0, frames=120, fps=30,
                           seconds=4.0)
        assert "mascara vazia" in pr.conferir_apresentador(
            tmp_path / "f.mp4", camada, tmp_path)

    def test_quadro_indisponivel_vira_aviso_e_nao_excecao(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pr, "_um_quadro", lambda *a, **k: None)
        camada = pr.Camada(path=tmp_path / "apr.mp4", x=0, y=660, width=1080,
                           height=1160, subtitle_y=940, subtitle_start=4.0,
                           card_s=4.0, frames=120, fps=30, seconds=4.0)
        texto = pr.conferir_apresentador(tmp_path / "f.mp4", camada, tmp_path)
        assert "indisponivel" in texto


class TestBordaSuave:
    """Serrilhado e quantizacao -- o que a pessoa via como 'pixelado'.

    Os tres defeitos aqui passavam em todo teste de codigo de saida: o ladrilho
    saia, a boca abria, o MP4 rodava. So medindo o pixel da borda e a posicao
    fracionaria entre dois quadros e que eles aparecem.
    """

    def test_a_boca_tem_borda_suave_e_nao_degrau(self, anim):
        """`ImageDraw` do Pillow nao tem anti-aliasing: media era ZERO px de borda."""
        for abertura in (0.3, 0.6, 1.0):
            tile, _ = pr.tile_boca(anim.maxilar, abertura)
            alfa = _alfa(tile)
            parcial = int(((alfa > 0) & (alfa < 255)).sum())
            assert parcial > 300, f"abertura {abertura}: so {parcial} px de borda parcial"

    def test_a_boca_anda_em_fracao_de_pixel(self, anim):
        """Antes o canto era truncado: saltava 1 px inteiro e ficava 3 quadros parada."""
        centros = []
        for abertura in (0.30, 0.33, 0.36, 0.39, 0.42):
            tile, canto = pr.tile_boca(anim.maxilar, abertura)
            alfa = _alfa(tile).astype(np.float32)
            ys = np.arange(alfa.shape[0], dtype=np.float32)[:, None]
            centros.append(canto[1] + float((alfa * ys).sum() / alfa.sum()))
        passos = np.diff(centros)
        assert (passos > 0).all(), "a boca parou entre dois quadros"
        assert passos.max() < 0.99, f"saltou {passos.max():.2f} px -- voltou a quantizar"

    def test_a_boca_nasce_em_degrade_em_vez_de_aparecer(self, anim):
        """Abaixo de BOCA_MIN_PX o alfa entra por ease cubico; antes surgia opaca."""
        alturas = [pr.tile_boca(anim.maxilar, a) for a in (0.06, 0.10, 0.16)]
        picos = [int(_alfa(t).max()) for t, _ in alturas if t is not None]
        assert len(picos) >= 2
        assert picos == sorted(picos) and picos[0] < 250

    def test_a_palpebra_desce_em_fracao_de_pixel(self, anim):
        """Era `int(round(linha))`: saltava 31 px de um quadro para o outro.

        A medida e a **continuidade**: um passo minusculo de fecho tem de
        produzir uma mudanca minuscula no pixel. Com a faixa presa a uma linha
        inteira, 1% de fecho ou nao mudava nada ou mudava uma linha inteira de
        uma vez -- e e esse degrau que se via na palpebra.
        """
        x0, y0, x1, y1 = anim.olhos[0].caixa

        def com(fecho: float) -> np.ndarray:
            quadro = anim._rgb.copy()
            pr.aplicar_piscada(quadro, anim.olhos, fecho)
            return quadro[y0:y1, x0:x1].astype(int)

        fino = np.abs(com(0.50) - com(0.51)).mean()
        grosso = np.abs(com(0.50) - com(0.62)).mean()
        assert fino > 0, "1% de fecho nao mexeu um pixel: continua quantizado"
        assert fino < grosso / 4, (
            f"1% de fecho mudou {fino:.2f}/255 contra {grosso:.2f} de 12% -- "
            "a resposta nao e proporcional, ainda ha salto de linha")

    def test_nada_dobra_na_reamostragem_do_olho(self, anim):
        """Dobra = a imagem volta atras = vinco. `maximum.accumulate` impede."""
        for fecho in (0.3, 0.6, 0.9, 1.0):
            antes = anim._rgb.copy()
            pr.aplicar_piscada(antes, anim.olhos, fecho)
            assert np.isfinite(antes).all()
        # E o borrao vertical da palpebra inferior (5 px grudados na ultima
        # linha, medido em 20/09) nao pode voltar: com o olho fechado a
        # bochecha ainda tem de mostrar textura propria, nao uma linha esticada.
        fechado = anim._rgb.copy()
        pr.aplicar_piscada(fechado, anim.olhos, 1.0)
        x0, y0, x1, y1 = anim.olhos[0].caixa
        rodape = fechado[y1 - 8:y1, x0:x1].astype(int)
        variacao = np.abs(np.diff(rodape, axis=0)).mean()
        assert variacao > 0.5, "as ultimas linhas viraram a mesma linha esticada"


class TestPiscadaHumana:
    def test_fecha_rapido_e_abre_devagar(self):
        """Piscada humana e assimetrica; a anterior era `sin(pi*u)**0.6`, simetrica."""
        assert pr.PISCADA_ABRE_S > 1.8 * pr.PISCADA_FECHA_S
        jan = [(0.0, pr.PISCADA_S)]
        ts = np.arange(0, pr.PISCADA_S, 0.001)
        v = np.array([pr.fecho_em(jan, float(t)) for t in ts])
        fecha = float(ts[v >= 0.995][0])
        abre = float(pr.PISCADA_S - ts[v >= 0.995][-1])
        assert abre > 1.8 * fecha, f"fecha {fecha:.3f}s, abre {abre:.3f}s"
        # 1-2 quadros fechando, 3-4 abrindo, a 30 fps.
        assert 1.0 <= fecha * 30 <= 2.0
        assert 3.0 <= abre * 30 <= 4.5

    def test_a_curva_nao_tem_quina(self):
        """Ease cubico: chega e sai com velocidade zero, senao le como mecanismo."""
        jan = [(0.0, pr.PISCADA_S)]
        v = np.array([pr.fecho_em(jan, float(t)) for t in np.arange(0, pr.PISCADA_S, 0.001)])
        assert np.abs(np.diff(v)).max() < 0.10, "a curva tem degrau"

    def test_a_caixa_do_olho_nao_deixa_costura_vertical(self, anim):
        """A deformacao valia cheia ate a ultima coluna e zero na seguinte.

        Medido antes da pena: salto de 9,4/255 na borda contra 2,8 tipico --
        3,4x, que na pele lisa da tempora le como risco vertical. O maxilar ja
        resolvia isso com `peso_x`; o olho nao tinha pena nenhuma. Este teste
        mede o salto entre as duas colunas que se encostam na borda.
        """
        olho = anim.olhos[0]
        x0, y0, x1, y1 = olho.caixa
        quadro = anim._rgb.copy()
        pr.aplicar_piscada(quadro, anim.olhos, 0.55)

        def salto(x: int) -> float:
            col = quadro[y0:y1, x].astype(int)
            return float(np.abs(col - quadro[y0:y1, x - 1].astype(int)).mean())

        tipico = float(np.median([salto(x) for x in range(x0 + 6, x1 - 6)]))
        for borda in (x0, x1):
            assert salto(borda) < 1.6 * tipico, (
                f"borda em x={borda}: salto {salto(borda):.2f}/255 contra "
                f"{tipico:.2f} tipico -- a costura vertical voltou")

    def test_a_bochecha_e_a_palpebra_de_baixo_reagem(self, anim, retrato):
        """Piscar puxa o que esta em volta; so a palpebra de cima le como persiana."""
        olho = anim.olhos[0]
        fechado = anim._rgb.copy()
        pr.aplicar_piscada(fechado, anim.olhos, 1.0)
        x0, _, x1, y1 = olho.caixa
        # Faixa abaixo do cilio inferior: e bochecha, e tem de ter mudado.
        topo = int(olho.base_y) + 2
        antes = anim._rgb[topo:topo + 14, x0:x1].astype(int)
        depois = fechado[topo:topo + 14, x0:x1].astype(int)
        assert np.abs(antes - depois).mean() > 1.0, "a bochecha ficou parada"
        # E a caixa tem de alcancar a bochecha, nao parar no cilio.
        assert y1 > olho.base_y + 1.5 * olho.alto

    def test_a_piscada_procura_a_fronteira_de_frase(self):
        """Gente pisca na pontuacao. `batidas.sentencas` ja existe -- e de graca."""
        sentencas = tuple(round(v, 2) for v in np.arange(4.1, 65, 4.6))
        solta = pr.piscadas(65.0, "Bonsai 2")
        presa = pr.piscadas(65.0, "Bonsai 2", sentencas)
        assert len(presa) == len(solta), "a ancora mudou o ritmo"

        def perto(janelas, desloca):
            return float(np.median([min(abs(i + desloca - m) for m in sentencas)
                                    for i, _ in janelas]))

        assert perto(presa, pr.PISCADA_FECHA_S) < perto(solta, 0.0) * 0.75
        assert presa == pr.piscadas(65.0, "Bonsai 2", sentencas), "deixou de ser reprodutivel"

    def test_sem_tempo_de_palavra_o_sorteio_e_o_de_antes(self):
        """Caminho do MPT nao tem sentenca: nao pode quebrar."""
        assert pr.piscadas(30.0, "x", ()) == pr.piscadas(30.0, "x")


class TestEnvoltoriaSuave:
    def test_o_ataque_filtra_de_verdade(self):
        """Ataque menor que um quadro nao filtra nada -- e a boca abre de uma vez."""
        assert pr.ATAQUE_S * pr.FPS * pr.ENV_SUB > 3.0, "o ataque cabe em 1 amostra"
        assert pr.RELAXAMENTO_S > pr.ATAQUE_S

    def test_a_curva_desce_para_o_quadro_por_media_e_nao_por_amostra(self):
        """Filtrar na taxa alta e so entao decimar: super-amostragem no tempo."""
        assert pr.ENV_SUB >= 2
