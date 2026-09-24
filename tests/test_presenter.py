"""Presentador animado: lo que se mide en el cuadro, sin red y sin ffmpeg.

El defecto que motivo este modulo pasaba en cualquier test de codigo de salida:
el comando corria, el MP4 salia, la duracion cuadraba -- y lo que aparecia en la
pantalla era un rectangulo de rostro con un movimiento de ventana por encima.
Por eso aqui se mide **el pixel**: la silueta, la costura de las capas, la boca
y el parpado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent.render import presenter as pr

RECORTE = Path(__file__).resolve().parent.parent / "brand" / "assets" / "presenters"


@pytest.fixture(scope="module")
def retrato() -> pr.Retrato:
    png = RECORTE / "iris.png"
    if not png.exists() or not png.with_suffix(".json").exists():
        pytest.skip("recorte del presentador ausente (scripts/make_presenter_cutouts.py)")
    return pr.cargar(png)


@pytest.fixture(scope="module")
def anim(retrato: pr.Retrato) -> pr.Animador:
    return pr.Animador(retrato, "#39FF88", "Iris")


@pytest.fixture(scope="module")
def escena() -> pr.Escenificacion:
    return pr.plan(pr.Tiempos(fin_gancho=3.4, inicio_cierre=52.0, duracion=68.0))


def _alfa(img: Image.Image) -> np.ndarray:
    return np.asarray(img)[:, :, 3]


class TestSilueta:
    def test_el_cuadro_no_es_un_rectangulo(self, anim, escena):
        """El defecto original: una ventana 340x640 recortaba a la persona en caja.

        Prueba de que esto acabo: el hombro es mucho mas ancho que la frente, y
        las esquinas de la caja que contiene el presentador estan vacias. Un
        rectangulo no tiene ninguna de las dos cosas.
        """
        a = _alfa(anim.cuadro(2.0, 0.4, 0.0, escena.poses))
        lineas = np.flatnonzero(a.max(axis=1) > 8)
        columnas = np.flatnonzero(a.max(axis=0) > 8)
        y0, y1 = int(lineas[0]), int(lineas[-1])
        x0, x1 = int(columnas[0]), int(columnas[-1])
        alto = y1 - y0
        frente = int((a[y0 + int(0.06 * alto)] > 128).sum())
        hombro = int((a[y0 + int(0.88 * alto)] > 128).sum())
        assert 0 < frente < 0.65 * hombro
        for cy, cx in ((y0 + 2, x0 + 2), (y0 + 2, x1 - 2)):
            assert a[cy, cx] < 40

    def test_nada_toca_el_borde_de_arriba_de_la_capa(self, anim, escena):
        """Si el tope de la capa tuviera tinta, el pelo estaria cortado a cuchillo."""
        a = _alfa(anim.cuadro(2.0, 0.4, 0.0, escena.poses))
        assert a[0].max() == 0

    def test_la_base_se_desvanece_en_degradado(self, retrato):
        """Busto cortado a cuchillo delata recorte; el alfa tiene que caer a cero."""
        a = np.asarray(retrato.imagen)[:, :, 3]
        assert a[-1].max() < 12
        assert a[int(a.shape[0] * 0.62)].max() > 240


class TestCapas:
    def test_cabeza_mas_torso_dan_el_alfa_original(self, anim, retrato):
        """Mascara complementaria: sin ella, la costura se vuelve rendija o fantasma."""
        suma = anim._alfa_cabeza.astype(np.int32) + anim._alfa_torso.astype(np.int32)
        alfa = np.asarray(pr.realce(retrato.imagen, "#39FF88"))[:, :, 3].astype(np.int32)
        assert int(np.abs(suma - alfa).max()) <= 1

    def test_la_costura_queda_en_el_pecho_no_en_el_cuello(self, retrato):
        """Cabeza que gira sobre cuello quieto es muneco de ventrilocuo."""
        peso = pr.mascara_cabeza(retrato)[:, 0]
        assert peso[int(retrato.p("menton")[1])] > 0.99
        assert peso[retrato.neck_y] > 0.9
        medio = int(np.argmin(np.abs(peso - 0.5)))
        assert medio > retrato.neck_y + 0.3 * retrato.face_height


class TestBoca:
    def test_silencio_no_mueve_el_rostro(self, anim):
        base = anim._rgb.copy()
        pr.aplicar_maxilar(base, anim.maxilar, 0.0)
        assert np.array_equal(base, anim._rgb)
        assert pr.losa_boca(anim.maxilar, 0.0) is None

    def test_el_menton_baja_y_la_boca_abre_con_la_voz(self, anim, retrato):
        abierta = anim._rgb.copy()
        pr.aplicar_maxilar(abierta, anim.maxilar, 1.0)
        menton_y = int(retrato.p("menton")[1])
        menton_x = int(retrato.p("menton")[0])
        # La linea del menton paso a mostrar lo que estaba encima de ella.
        antes = anim._rgb[menton_y, menton_x - 40:menton_x + 40].astype(int)
        despues = abierta[menton_y, menton_x - 40:menton_x + 40].astype(int)
        assert np.abs(antes - despues).mean() > 3
        losa, _ = pr.losa_boca(anim.maxilar, 1.0)
        assert losa.height > 0.25 * anim.maxilar.boca_w

    def test_la_abertura_crece_con_la_envolvente(self, anim):
        alturas = [pr.losa_boca(anim.maxilar, v)[0].height for v in (0.3, 0.6, 1.0)]
        assert alturas[0] < alturas[1] < alturas[2]

    def test_el_labio_de_arriba_queda_quieto(self, anim, retrato):
        """Solo el de abajo acompana la mandibula -- y eso separa hablar de mueca."""
        x0, y0, _, _ = anim.maxilar.caja
        encima = int(retrato.p("labio_sup")[1]) - 14 - y0
        abierta = anim._rgb.copy()
        pr.aplicar_maxilar(abierta, anim.maxilar, 1.0)
        franja = slice(y0 + max(encima - 6, 0), y0 + encima)
        assert np.array_equal(anim._rgb[franja], abierta[franja])


class TestParpadeo:
    def test_el_parpado_cubre_el_ojo(self, anim, retrato):
        cerrado = anim._rgb.copy()
        pr.aplicar_parpadeo(cerrado, anim.ojos, 1.0)
        x0, y0, x1, y1 = anim.ojos[0].caja
        ojo_antes = anim._rgb[y0:y1, x0:x1].astype(int)
        ojo_despues = cerrado[y0:y1, x0:x1].astype(int)
        assert np.abs(ojo_antes - ojo_despues).mean() > 4
        # Fuera de la caja del ojo nada cambio.
        assert np.array_equal(anim._rgb[:y0], cerrado[:y0])

    def test_ojo_abierto_no_se_toca(self, anim):
        igual = anim._rgb.copy()
        pr.aplicar_parpadeo(igual, anim.ojos, 0.0)
        assert np.array_equal(igual, anim._rgb)

    def test_el_parpadeo_es_reproducible_y_tiene_ritmo_de_gente(self):
        a = pr.parpadeos(60.0, "Bonsai 2")
        assert a == pr.parpadeos(60.0, "Bonsai 2")
        assert a != pr.parpadeos(60.0, "otro tema")
        assert 8 <= len(a) <= 30
        assert all(abs((f - i) - pr.PARPADEO_S) < 1e-9 for i, f in a)

    def test_el_cierre_abre_y_cierra_dentro_de_la_ventana(self):
        ventana = pr.parpadeos(6.0, "x")[0]
        ini, fin = ventana
        assert pr.cierre_en([ventana], ini - 0.01) == 0.0
        assert pr.cierre_en([ventana], (ini + fin) / 2) > 0.9
        assert pr.cierre_en([ventana], fin + 0.01) == 0.0


class TestEnvolvente:
    def test_el_ataque_es_mas_rapido_que_la_relajacion(self):
        """Boca abre rapido y cierra despacio; sin eso tiembla en cada consonante."""
        escalon = np.concatenate([np.ones(30, np.float32), np.zeros(30, np.float32)])
        y = pr._ataque_relajacion(escalon, 30)
        subida = int(np.argmax(y >= 0.5))          # cuadros hasta que la boca abre
        bajada = int(np.argmax(y[30:] < 0.5))      # cuadros hasta que cierra
        assert subida < bajada
        assert y[0] > 0.5 and y[29] > 0.95 and y[-1] < 0.02


class TestEscenificacion:
    def test_grande_en_la_llamada_pequeno_en_el_cuerpo_vuelve_al_final(self, escena):
        assert pr.pose(escena.poses, 2.0).altura == pr.ALTURA_LLAMADA
        assert pr.pose(escena.poses, 30.0).altura == pr.ALTURA_CUERPO
        assert pr.pose(escena.poses, 60.0).altura == pr.ALTURA_CIERRE

    def test_entra_apareciendo_en_vez_de_surgir_de_una_vez(self, escena):
        assert pr.pose(escena.poses, 0.0).alfa == 0.0
        assert 0.0 < pr.pose(escena.poses, 0.2).alfa < 1.0
        assert pr.pose(escena.poses, 0.5).alfa == 1.0

    def test_la_travesia_es_suave(self, escena):
        alturas = [pr.pose(escena.poses, 3.4 + k * 0.1).altura for k in range(8)]
        assert alturas == sorted(alturas, reverse=True)
        saltos = np.diff(alturas)
        assert abs(saltos[0]) < abs(saltos[len(saltos) // 2])

    def test_la_leyenda_solo_entrar_cuando_el_encoge(self, escena):
        """Mientras esta grande quien escribe el gancho es la tarjeta."""
        assert escena.leyenda_inicio == escena.tarjeta_s
        assert pr.pose(escena.poses, escena.leyenda_inicio).altura == pr.ALTURA_CUERPO
        assert escena.leyenda_y == pr.Y_LEYENDA_CON_PRESENTADOR

    def test_la_leyenda_pasa_por_encima_de_la_cabeza_en_la_esquina(self, escena):
        """Mide el hueco entre el texto (hasta ~+50 del centro) y el tope del avatar."""
        for t in (30.0, 60.0):
            p = pr.pose(escena.poses, t)
            tope_avatar = p.base - p.altura
            assert tope_avatar > escena.leyenda_y + 60

    def test_guion_corto_no_invierte_los_tiempos(self):
        esc = pr.plan(pr.Tiempos(fin_gancho=9.0, inicio_cierre=2.0, duracion=11.0))
        tiempos = [m.t for m in esc.poses]
        assert tiempos == sorted(tiempos)
        assert all(0 <= t <= 11.0 for t in tiempos)


class TestTiempos:
    def test_los_tiempos_salen_del_tiempo_de_palabra(self):
        class P:
            def __init__(self, texto, s, e):
                self.text, self.start, self.end = texto, s, e

        palabras = [P("Un", 0.0, 0.3), P("gancho.", 0.3, 1.1),
                    P("Cuerpo", 1.2, 1.8), P("aqui.", 1.8, 2.4),
                    P("Cierra", 2.5, 3.0), P("ahora?", 3.0, 3.6)]
        t = pr.tiempos_por_palabras("Un gancho.", "Cierra ahora?", palabras, 4.0)
        assert t.fin_gancho == pytest.approx(1.1)
        assert t.inicio_cierre == pytest.approx(2.5)

    def test_sin_tiempo_de_palabra_estima_por_proporcion(self):
        t = pr.tiempos_estimados(n_hook=10, n_close=10, total=100, duracion=60.0)
        assert t.fin_gancho == pytest.approx(6.0)
        assert t.inicio_cierre == pytest.approx(54.0)


class TestGeometria:
    def test_la_caja_de_salida_cubre_la_capa_girada(self):
        fuente = (10, 20, 210, 320)
        destino = (400.0, 500.0)
        pivote = (100.0, 300.0)
        caja = pr._caja_salida(fuente, pivote, destino, escala=0.8, grados=3.0)
        assert caja is not None
        ox, oy, ow, oh = caja
        for px in (fuente[0], fuente[2]):
            for py in (fuente[1], fuente[3]):
                dx, dy = pr._giro(px - pivote[0], py - pivote[1], 3.0, 0.8)
                assert ox <= destino[0] + dx <= ox + ow
                assert oy <= destino[1] + dy <= oy + oh

    def test_la_base_del_recorte_posa_donde_mando_la_escenificacion(self, anim, escena):
        """La altura pedida es la altura que aparece: escala mala es avatar torcido."""
        p = pr.pose(escena.poses, 30.0)
        a = _alfa(anim.cuadro(30.0, 0.0, 0.0, escena.poses))
        lineas = np.flatnonzero((a > 128).max(axis=1))
        alto = int(lineas[-1] - lineas[0])
        # El recorte tiene margen transparente alrededor, asi que la tinta es un
        # poco menor que la altura pedida -- pero no mucho.
        assert 0.80 * p.altura < alto <= p.altura


class TestCosturaEntreCapas:
    def test_las_dos_capas_giran_en_el_mismo_punto(self, anim, escena):
        """Pivote por capa las separaria -- y la costura es justamente ahi.

        Con angulo de cabeza al maximo, la suma de los alfas del cuadro compuesto
        sigue pareciendose a la del cuadro sin giro ninguno: nada de rendija ni
        de contorno duplicado en el pecho.
        """
        quieto = _alfa(anim.cuadro(0.0, 0.0, 0.0,
                                   [pr.Pose(t=0.0, altura=600, cx=540, base=1700),
                                    pr.Pose(t=9.9, altura=600, cx=540, base=1700)]))
        area = int((quieto > 128).sum())
        for t in (1.1, 2.7, 4.3):
            a = _alfa(anim.cuadro(t, 1.0, 0.0,
                                  [pr.Pose(t=0.0, altura=600, cx=540, base=1700),
                                   pr.Pose(t=9.9, altura=600, cx=540, base=1700)]))
            assert abs(int((a > 128).sum()) - area) < 0.06 * area

    def test_solo_la_cabeza_responde_a_la_silaba_fuerte(self, anim):
        """Si el cuadro entero anduviera junto, seria pegatina deslizandose.

        Entre abertura 0 y 1 en el MISMO instante solo cambia `giro_cabeza` (el
        acento de la cabeza) y la boca -- las dos cosas de la cabeza. La base del
        busto, debajo de la costura, tiene que salir identica.
        """
        poses = [pr.Pose(t=0.0, altura=900, cx=540, base=1700),
                 pr.Pose(t=9.9, altura=900, cx=540, base=1700)]
        tranquilo = np.asarray(anim.cuadro(3.0, 0.0, 0.0, poses)).astype(int)
        fuerte = np.asarray(anim.cuadro(3.0, 1.0, 0.0, poses)).astype(int)
        d = np.abs(tranquilo - fuerte).mean(axis=2)
        lineas = np.flatnonzero(
            (_alfa(anim.cuadro(3.0, 0.0, 0.0, poses)) > 128).any(axis=1))
        y0, y1 = int(lineas.min()), int(lineas.max())
        alto = y1 - y0
        cabeza = d[y0:y0 + int(0.25 * alto)].mean()
        pecho = d[y1 - int(0.12 * alto):y1].mean()
        assert cabeza > 4 * max(pecho, 0.05), f"cabeza {cabeza:.2f} x pecho {pecho:.2f}"


class TestPuertaEnElCuadro:
    """La puerta que impide que el MP4 sin presentador pase callado.

    La capa puede salir perfecta y no llegar al cuadro: basta una etiqueta de
    flujo equivocada en el `overlay`. Nada de eso levanta error -- sale un MP4
    valido, con la duracion correcta, sin avatar. Mismo defecto del MP4 mudo.

    La escena es sintetica a proposito: el rostro es un patron con estructura, el
    material de apoyo es otro, y la posproduccion entra como eq + vineta encima.
    """

    ALTO, ANCHO = 600, 400
    X, Y = 100, 300
    AN, AL = 200, 200

    @classmethod
    def _piezas(cls, con_avatar: bool, tarjeta_fuerte: bool = False):
        """Devuelve (final, color de la capa, mascara de la capa).

        `tarjeta_fuerte` enciende la tarjeta del gancho justo encima del
        presentador -- el caso que derribaba la medida vieja.
        """
        rng = np.random.default_rng(7)
        # Rostro: estructura propia. Material de apoyo: otra estructura, sin
        # relacion.
        rostro = rng.integers(40, 220, (cls.AL, cls.AN)).astype(np.uint8)
        apoyo = rng.integers(30, 200, (cls.ALTO, cls.ANCHO)).astype(np.uint8)

        mascara = np.zeros((cls.AL, cls.AN), dtype=np.uint8)
        mascara[40:170, 60:150] = 255
        color = np.where(mascara > 0, rostro, 0).astype(np.uint8)

        final = apoyo.astype(np.int16)
        if con_avatar:
            recorte = final[cls.Y:cls.Y + cls.AL, cls.X:cls.X + cls.AN]
            recorte[mascara > 0] = rostro[mascara > 0]
        # posproduccion: eq (ganancia + offset) y vineta (oscurece los bordes)
        final = final * 1.12 + 10
        final[:80] -= 26
        final[-80:] -= 26
        if tarjeta_fuerte:
            final[cls.Y - 120:cls.Y - 10, 20:cls.ANCHO - 20] = 235
        return (np.clip(final, 0, 255).astype(np.uint8), color, mascara)

    def _correr(self, con_avatar, tmp_path, monkeypatch, tarjeta_fuerte=False):
        final, color, mascara = self._piezas(con_avatar, tarjeta_fuerte)
        cuadros = iter([final, color, mascara])
        monkeypatch.setattr(pr, "_un_cuadro", lambda *a, **k: next(cuadros))
        capa = pr.Capa(path=tmp_path / "apr.mp4", x=self.X, y=self.Y,
                       width=self.AN, height=self.AL, subtitle_y=940,
                       subtitle_start=4.0, card_s=4.0, frames=120, fps=30,
                       seconds=4.0)
        return pr.conferir_presentador(tmp_path / "f.mp4", capa, tmp_path)

    def test_avatar_en_cuadro_pasa(self, tmp_path, monkeypatch):
        assert "OK" in self._correr(True, tmp_path, monkeypatch)

    def test_avatar_que_sumio_se_acusa(self, tmp_path, monkeypatch):
        assert "NO LLEGO AL CUADRO" in self._correr(False, tmp_path, monkeypatch)

    def test_la_tarjeta_del_gancho_no_derriba_el_veredicto(self, tmp_path, monkeypatch):
        """El falso negativo que jubilo la medida vieja, bloqueado.

        La medida vieja comparaba cuadro final x cuadro crudo dentro de la
        mascara contra un anillo alrededor. Con el presentador fotorrealista y la
        tarjeta del gancho justo encima de el, el anillo cambiaba tanto como la
        mascara y el veredicto salia NO LLEGO en video que tenian al presentador
        en el cuadro -- comprobado a ojo en el artefacto del 20/09/2026. Comparar
        con la propia capa en vez de con el crudo no se importa con lo que pasa
        fuera.
        """
        assert "OK" in self._correr(True, tmp_path, monkeypatch, tarjeta_fuerte=True)

    def test_la_medida_es_contra_la_capa_y_no_contra_el_crudo(self, tmp_path,
                                                              monkeypatch):
        texto = self._correr(True, tmp_path, monkeypatch)
        assert "correlacion con la capa" in texto

    def test_mascara_vacia_se_vuelve_aviso(self, tmp_path, monkeypatch):
        vacia = np.zeros((self.AL, self.AN), dtype=np.uint8)
        final, color, _ = self._piezas(True)
        cuadros = iter([final, color, vacia])
        monkeypatch.setattr(pr, "_un_cuadro", lambda *a, **k: next(cuadros))
        capa = pr.Capa(path=tmp_path / "apr.mp4", x=self.X, y=self.Y,
                       width=self.AN, height=self.AL, subtitle_y=940,
                       subtitle_start=4.0, card_s=4.0, frames=120, fps=30,
                       seconds=4.0)
        assert "mascara vacia" in pr.conferir_presentador(
            tmp_path / "f.mp4", capa, tmp_path)

    def test_cuadro_indisponible_se_vuelve_aviso_y_no_excepcion(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pr, "_un_cuadro", lambda *a, **k: None)
        capa = pr.Capa(path=tmp_path / "apr.mp4", x=0, y=660, width=1080,
                       height=1160, subtitle_y=940, subtitle_start=4.0,
                       card_s=4.0, frames=120, fps=30, seconds=4.0)
        texto = pr.conferir_presentador(tmp_path / "f.mp4", capa, tmp_path)
        assert "indisponible" in texto


class TestBordeSuave:
    """Serrucho y cuantizacion -- lo que la gente veia como 'pixelado'.

    Los tres defectos de aqui pasaban en cualquier test de codigo de salida: el
    mosaico salia, la boca abria, el MP4 corria. Solo midiendo el pixel del borde
    y la posicion fraccionaria entre dos cuadros aparecen.
    """

    def test_la_boca_tiene_borde_suave_y_no_escalón(self, anim):
        """`ImageDraw` de Pillow no tiene antialiasing: la media era CERO px de borde."""
        for abertura in (0.3, 0.6, 1.0):
            losa, _ = pr.losa_boca(anim.maxilar, abertura)
            alfa = _alfa(losa)
            parcial = int(((alfa > 0) & (alfa < 255)).sum())
            assert parcial > 300, f"abertura {abertura}: solo {parcial} px de borde parcial"

    def test_la_boca_anda_en_fraccion_de_pixel(self, anim):
        """Antes la esquina se truncaba: saltaba 1 px entero y se quedaba 3 cuadros quieta."""
        centros = []
        for abertura in (0.30, 0.33, 0.36, 0.39, 0.42):
            losa, esquina = pr.losa_boca(anim.maxilar, abertura)
            alfa = _alfa(losa).astype(np.float32)
            ys = np.arange(alfa.shape[0], dtype=np.float32)[:, None]
            centros.append(esquina[1] + float((alfa * ys).sum() / alfa.sum()))
        pasos = np.diff(centros)
        assert (pasos > 0).all(), "la boca se paro entre dos cuadros"
        assert pasos.max() < 0.99, f"salto {pasos.max():.2f} px -- volvio a cuantizar"

    def test_la_boca_nace_en_degradado_en_vez_de_aparecer(self, anim):
        """Por debajo de BOCA_MIN_PX el alfa entra en ease cubico; antes surgia opaca."""
        piezas = [pr.losa_boca(anim.maxilar, a) for a in (0.06, 0.10, 0.16)]
        picos = [int(_alfa(t).max()) for t, _ in piezas if t is not None]
        assert len(picos) >= 2
        assert picos == sorted(picos) and picos[0] < 250

    def test_el_parpado_baja_en_fraccion_de_pixel(self, anim):
        """Era `int(round(linea))`: saltaba 31 px de un cuadro al otro.

        La medida es la **continuidad**: un paso minimo de cierre tiene que
        producir un cambio minimo en el pixel. Con la franja atada a una linea
        entera, 1% de cierre o no cambiaba nada o cambiaba una linea entera de
        una vez -- y ese es el escalon que se veia en el parpado.
        """
        x0, y0, x1, y1 = anim.ojos[0].caja

        def con(cierre: float) -> np.ndarray:
            cuadro = anim._rgb.copy()
            pr.aplicar_parpadeo(cuadro, anim.ojos, cierre)
            return cuadro[y0:y1, x0:x1].astype(int)

        fino = np.abs(con(0.50) - con(0.51)).mean()
        grueso = np.abs(con(0.50) - con(0.62)).mean()
        assert fino > 0, "1% de cierre no movio un pixel: sigue cuantizado"
        assert fino < grueso / 4, (
            f"1% de cierre cambio {fino:.2f}/255 contra {grueso:.2f} de 12% -- "
            "la respuesta no es proporcional, sigue habiendo salto de linea")

    def test_nada_se_dobla_en_la_reamuestreo_del_ojo(self, anim):
        """Doblar = la imagen vuelve atras = pliegue. `maximum.accumulate` lo impide."""
        for cierre in (0.3, 0.6, 0.9, 1.0):
            antes = anim._rgb.copy()
            pr.aplicar_parpadeo(antes, anim.ojos, cierre)
            assert np.isfinite(antes).all()
        # Y el borron vertical del parpado inferior (5 px pegados a la ultima
        # linea, medido el 20/09) no puede volver: con el ojo cerrado la mejilla
        # sigue teniendo que mostrar textura propia, no una linea estirada.
        cerrado = anim._rgb.copy()
        pr.aplicar_parpadeo(cerrado, anim.ojos, 1.0)
        x0, y0, x1, y1 = anim.ojos[0].caja
        pie = cerrado[y1 - 8:y1, x0:x1].astype(int)
        variacion = np.abs(np.diff(pie, axis=0)).mean()
        assert variacion > 0.5, "las ultimas lineas se volvieron la misma linea estirada"


class TestParpadeoHumano:
    def test_cierra_rapido_y_abre_despacio(self):
        """El parpadeo humano es asimetrico; el anterior era `sin(pi*u)**0.6`, simetrico."""
        assert pr.PARPADEO_ABRE_S > 1.8 * pr.PARPADEO_CIERRA_S
        ventana = [(0.0, pr.PARPADEO_S)]
        ts = np.arange(0, pr.PARPADEO_S, 0.001)
        v = np.array([pr.cierre_en(ventana, float(t)) for t in ts])
        cierra = float(ts[v >= 0.995][0])
        abre = float(pr.PARPADEO_S - ts[v >= 0.995][-1])
        assert abre > 1.8 * cierra, f"cierra {cierra:.3f}s, abre {abre:.3f}s"
        # 1-2 cuadros cerrando, 3-4 abriendo, a 30 fps.
        assert 1.0 <= cierra * 30 <= 2.0
        assert 3.0 <= abre * 30 <= 4.5

    def test_la_curva_no_tiene_esquina(self):
        """Ease cubico: llega y sale con velocidad cero, si no se lee como mecanismo."""
        ventana = [(0.0, pr.PARPADEO_S)]
        v = np.array([pr.cierre_en(ventana, float(t))
                      for t in np.arange(0, pr.PARPADEO_S, 0.001)])
        assert np.abs(np.diff(v)).max() < 0.10, "la curva tiene escalon"

    def test_la_caja_del_ojo_no_deja_costura_vertical(self, anim):
        """La deformacion valia lleno hasta la ultima columna y cero en la siguiente.

        Medido antes de la pluma: salto de 9,4/255 en el borde contra 2,8 tipico
        -- 3,4x, que en la piel lisa de la sien se lee como raya vertical. La
        mandibula ya lo resolvia con `peso_x`; el ojo no tenia pluma ninguna. Este
        test mide el salto entre las dos columnas que se tocan en el borde.
        """
        ojo = anim.ojos[0]
        x0, y0, x1, y1 = ojo.caja
        cuadro = anim._rgb.copy()
        pr.aplicar_parpadeo(cuadro, anim.ojos, 0.55)

        def salto(x: int) -> float:
            columna = cuadro[y0:y1, x].astype(int)
            return float(np.abs(columna - cuadro[y0:y1, x - 1].astype(int)).mean())

        tipico = float(np.median([salto(x) for x in range(x0 + 6, x1 - 6)]))
        for borde in (x0, x1):
            assert salto(borde) < 1.6 * tipico, (
                f"borde en x={borde}: salto {salto(borde):.2f}/255 contra "
                f"{tipico:.2f} tipico -- la costura vertical volvio")

    def test_la_mejilla_y_el_parpado_de_abajo_reaccionan(self, anim, retrato):
        """Parpadear tira de lo que hay alrededor; solo el párpado superior
        se lee como una persiana."""
        ojo = anim.ojos[0]
        cerrado = anim._rgb.copy()
        pr.aplicar_parpadeo(cerrado, anim.ojos, 1.0)
        x0, _, x1, y1 = ojo.caja
        # Franja debajo del cilio inferior: es mejilla, y tiene que haber cambiado.
        tope = int(ojo.base_y) + 2
        antes = anim._rgb[tope:tope + 14, x0:x1].astype(int)
        despues = cerrado[tope:tope + 14, x0:x1].astype(int)
        assert np.abs(antes - despues).mean() > 1.0, "la mejilla se quedo quieta"
        # Y la caja tiene que alcanzar la mejilla, no parar en el cilio.
        assert y1 > ojo.base_y + 1.5 * ojo.alto

    def test_el_parpadeo_busca_la_frontera_de_frase(self):
        """La gente parpadea en la puntuacion. `tiempos.sentencias` ya existe -- gratis."""
        sentencias = tuple(round(v, 2) for v in np.arange(4.1, 65, 4.6))
        suelto = pr.parpadeos(65.0, "Bonsai 2")
        atado = pr.parpadeos(65.0, "Bonsai 2", sentencias)
        assert len(atado) == len(suelto), "el ancla cambio el ritmo"

        def cerca(ventanas, desplaza):
            return float(np.median([min(abs(i + desplaza - m) for m in sentencias)
                                    for i, _ in ventanas]))

        assert cerca(atado, pr.PARPADEO_CIERRA_S) < cerca(suelto, 0.0) * 0.75
        assert atado == pr.parpadeos(65.0, "Bonsai 2", sentencias), "dejo de ser reproducible"

    def test_sin_tiempo_de_palabra_el_sorteo_es_el_de_antes(self):
        """El camino del MPT no tiene sentencia: no puede romper."""
        assert pr.parpadeos(30.0, "x", ()) == pr.parpadeos(30.0, "x")


class TestEnvolventeSuave:
    def test_el_ataque_filtra_de_verdad(self):
        """Ataque menor que un cuadro no filtra nada -- y la boca abre de una vez."""
        assert pr.ATAQUE_S * pr.FPS * pr.ENV_SUB > 3.0, "el ataque cabe en 1 muestra"
        assert pr.RELAJACION_S > pr.ATAQUE_S

    def test_la_curva_baja_al_cuadro_por_media_y_no_por_muestra(self):
        """Filtrar en la tasa alta y solo entonces diezmar: supermuestreo en el tiempo."""
        assert pr.ENV_SUB >= 2
