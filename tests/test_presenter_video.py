"""Presentador de video: visemas de grafema castellano y boca sobre rostro filmado.

Los test de aqui miden el **artefacto** (el campo de deformacion, los pixels
del mosaico, la pista de abertura), nunca el codigo de salida. Fue asi como
aparecieron el MP4 mudo y la boca escancarada, y los dos defectos que estos
test bloquean -- la raya horizontal en la mejilla y la vocal final nasalizada
-- tambien pasaban por cualquier test que solo comprobase que la funcion corrio.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.render import presenter_video as pv
from agent.render.visemes import REPOSO, VISEMAS, fonemas, pista, visemas


class TiempoFalso:
    """Lo que `voice/edge.WordTiming` entrega, sin TTS ninguno."""

    def __init__(self, text: str, start_s: float, end_s: float) -> None:
        self.text, self.start_s, self.end_s = text, start_s, end_s


# ===========================================================================
# grafema -> visema
# ===========================================================================


class TestFonemas:
    def test_vocal_final_no_nasaliza(self):
        """`"" in "mn"` es True en Python, y eso nasalizaba TODA palabra.

        La regla quiere decir "vocal seguida de m o n"; escrita como
        `sig in "mn"`, tambien casaba con la string vacia del final de la
        palabra. Efecto medido: "bonito" salia `P o T i T o~`, o sea, toda
        palabra terminada en vocal ganaba boca de vocal nasal. Se arregla solo
        cuando la comparacion es contra una tupla de caracteres.
        """
        assert fonemas("bonito")[-1] == "u"
        assert fonemas("papai")[-1] == "i"
        assert fonemas("chuva")[-1] == "a"
        # la nasal de verdad sigue nasal
        assert fonemas("bom") == ["P", "o~"]
        assert fonemas("um") == ["u~"]

    def test_c_y_g_se_ablandan_antes_de_vocal_cerrada(self):
        """`sig in "ei"` no pega "e" ni "i": "vez" salia con [k] duro."""
        assert "s" in fonemas("você")
        assert fonemas("você") == ["F", "o", "s", "e"]
        assert "S" in fonemas("inteligencia")
        assert fonemas("gigante")[0] == "S"     # g antes de i
        assert fonemas("gato")[0] == "K"        # g antes de a sigue duro

    def test_m_antes_de_vocal_pega_el_labio_y_al_final_no(self):
        """La diferencia por la que el modulo entero existe."""
        assert fonemas("mamae")[0] == "P"
        assert VISEMAS[fonemas("mamae")[0]].cerrado
        # "bom" acaba en vocal nasal: ningun visema de labio pegado al final
        assert not any(VISEMAS[f].cerrado for f in fonemas("bom")[1:])

    def test_e_o_atonas_al_final_se_reducen(self):
        assert fonemas("modelo")[-1] == "u"
        assert fonemas("que")[-1] == "i"

    def test_l_al_final_de_silaba_se_vuelve_semivocal(self):
        assert fonemas("sal") == ["s", "a", "u"]
        assert fonemas("lado")[0] == "T"      # l antes de vocal sigue alveolar

    def test_palabra_vacia_o_solo_puntuacion_no_rompe(self):
        assert fonemas("") == []
        assert fonemas("...") == []
        assert visemas("—") == []


# ===========================================================================
# pista de visemas
# ===========================================================================


class TestPista:
    def test_el_labio_pega_en_la_consonante_bilabial(self):
        """La /m/ tiene que LLEGAR a cero, no pasar cerca.

        Es el defecto que la envolvente de energia sola nunca resolvio: en la
        /m/ el audio tiene energia (es un sonido sonoro), asi que la boca
        quedaba abierta justo donde el espectador espera ver el labio pegado.
        """
        n, fps = 60, 30
        abertura, _ = pista([TiempoFalso("mamae", 0.2, 1.4)], n, fps)
        assert abertura.min() < 0.02
        assert abertura.max() > 0.4          # y la vocal sigue abriendo

    def test_vocal_abierta_abre_mas_que_vocal_cerrada(self):
        n, fps = 90, 30
        a_abierta, _ = pista([TiempoFalso("ah", 0.3, 1.2)], n, fps)
        a_cerrada, _ = pista([TiempoFalso("iii", 0.3, 1.2)], n, fps)
        assert a_abierta.max() > a_cerrada.max() + 0.3

    def test_redondeada_y_estirada_salen_con_signos_opuestos(self):
        n, fps = 90, 30
        _, anchura_u = pista([TiempoFalso("uuu", 0.3, 1.2)], n, fps)
        _, anchura_i = pista([TiempoFalso("iii", 0.3, 1.2)], n, fps)
        assert anchura_u.min() < -0.4          # /u/ recoge
        assert anchura_i.max() > 0.4           # /i/ estira

    def test_la_boca_descansa_en_el_silencio_entre_palabras(self):
        n, fps = 120, 30
        abertura = pista([TiempoFalso("ana", 0.1, 0.6), TiempoFalso("ana", 3.0, 3.5)],
                         n, fps)[0]
        # en medio (t ~ 1,8s) no hay palabra ninguna
        assert abertura[54] < 0.05

    def test_pista_sin_palabras_no_rompe(self):
        abertura, anchura = pista([], 30, 30)
        assert abertura.shape == (30,) and anchura.shape == (30,)
        assert float(abertura.max()) == 0.0

    def test_la_pista_cubre_todos_los_cuadros_pedidos(self):
        abertura, anchura = pista([TiempoFalso("teste", 0.0, 0.5)], 45, 30)
        assert len(abertura) == 45 and len(anchura) == 45


class TestModular:
    def test_energia_modula_pero_no_abre_lo_que_el_visema_cerro(self):
        """Audio alto en una /m/ no puede despegar el labio."""
        visema = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        energia = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        assert pv.modular(visema, energia)[0] == 0.0

    def test_el_suelo_evita_que_el_silencio_borre_la_forma(self):
        visema = np.array([1.0], dtype=np.float32)
        mudo = pv.modular(visema, np.array([0.0], dtype=np.float32))
        alto = pv.modular(visema, np.array([1.0], dtype=np.float32))
        assert mudo[0] == pytest.approx(pv.ENERGIA_PISO)
        assert alto[0] > mudo[0]


# ===========================================================================
# el vaiven del clip base
# ===========================================================================


class BaseFalsa:
    frames = 10
    fps = 10.0


class TestVaiven:
    def test_va_y_vuelve_sin_saltarse_un_cuadro(self):
        """La union del bucle es el cuadro vecino, nunca un corte.

        Medido en el clip de THEO: el mejor par de cuadros no vecinos difiere
        6,74/255 contra 1,13 entre vecinos. Un corte costaria seis veces el
        movimiento normal -- visible en cada vuelta.
        """
        b = BaseFalsa()
        secuencia = [pv.indice_vaiven(k / 10, b) for k in range(40)]
        assert max(abs(a - c) for a, c in zip(secuencia, secuencia[1:], strict=False)) <= 1
        assert max(secuencia) == b.frames - 1 and min(secuencia) == 0
        assert secuencia[:10] == list(range(10))          # ida
        assert secuencia[10] < secuencia[9]               # vuelta

    def test_clip_de_un_cuadro_no_divide_por_cero(self):
        class Uno:
            frames = 1
            fps = 24.0

        assert pv.indice_vaiven(3.7, Uno()) == 0


# ===========================================================================
# el campo de la mandibula
# ===========================================================================


def _rostro_sintetico(h: int = 900, w: int = 700):
    """Un rostro inventado: solo los puntos, que es todo lo que la boca necesita."""
    cuadro = np.zeros((h, w, 4), dtype=np.uint8)
    # Tablero de ajedrez: varía en los DOS ejes, si no, un desplazamiento
    # horizontal sobre un patron constante en x no mueve pixel ninguno y el
    # test pasa de vano.
    ys, xs = np.mgrid[0:h, 0:w]
    cuadro[:, :, 0] = ((ys // 7 + xs // 7) % 2 * 180 + 40).astype(np.uint8)
    cuadro[:, :, 1] = (xs % 251).astype(np.uint8)
    cuadro[:, :, 3] = 255
    pts = {
        "labio_sup": (350.0, 500.0), "labio_inf": (350.0, 502.0),
        "boca_izq": (280.0, 501.0), "boca_der": (420.0, 501.0),
        "menton": (350.0, 620.0), "subnasal": (350.0, 450.0),
        "ojo_izq_abajo": (300.0, 300.0), "ojo_der_abajo": (400.0, 300.0),
        "mandibula_izq": (230.0, 560.0), "mandibula_der": (470.0, 560.0),
        "rostro_izq": (200.0, 420.0), "rostro_der": (500.0, 420.0),
        "nariz_punta": (350.0, 430.0),
        "labio_sup_ext": (350.0, 478.0), "labio_inf_ext": (350.0, 528.0),
    }
    return cuadro, pts


def _contorno(pts, n: int = 11):
    """El labio interno de una boca CERRADA, tal como sale del clip base.

    Los dos arcos casi se tocan (la boca del clip esta cerrada todo el tiempo)
    y corren en sentidos opuestos, como en la malla del mediapipe: el de abajo
    de izquierda a derecha, el de arriba de vuelta.
    """
    izq, der = pts["boca_izq"][0], pts["boca_der"][0]
    y = (pts["labio_sup"][1] + pts["labio_inf"][1]) / 2
    u = np.linspace(0.0, 1.0, n, dtype=np.float32)
    arco = np.sin(np.pi * u)
    inferior = np.column_stack([izq + (der - izq) * u, y + 2.0 * arco])
    superior = np.column_stack([der - (der - izq) * u, y - 1.0 * arco[::-1]])
    return inferior.astype(np.float32), superior.astype(np.float32)


def _campo(abertura: float = 1.0):
    """El campo de desplazamiento en pixel, en el cuadro entero.

    Mide el campo directo de `campo_maxilar` en vez de intentar recuperar el
    desplazamiento de los pixels despues de la deformacion. La primera version
    de este test ponia una rampa `valor = y` en la imagen y leia el
    desplazamiento de vuelta; parecia listo y estaba mal, porque la rampa
    satura en 255 y el rostro tiene 900 lineas -- media cero en todo lugar que
    importaba.
    """
    _, pts = _rostro_sintetico()
    caja, campo, caida = pv.campo_maxilar(pts, 570.0, abertura, (900, 700))
    entero = np.zeros((900, 700), dtype=np.float32)
    x0, y0, x1, y1 = caja
    entero[y0:y1, x0:x1] = campo
    return entero, caida


class TestMaxilar:
    def test_el_menton_baja_y_la_altura_de_los_ojos_no(self):
        campo, caida = _campo(1.0)
        assert campo[618, 350] > 0.8 * caida    # el menton baja de verdad
        assert campo[302, 350] < 0.5            # la bisagra no se mueve

    def test_fuera_de_la_boca_el_campo_no_tiene_escalón(self):
        """La raya horizontal atravesando la mejilla, bloqueada.

        La primera version bajaba en bloque todo lo que estaba debajo de la
        linea de los labios. En el centro el hueco pintado escondia la union;
        desde la comisura hacia fuera sobraba un escalon desnudo de hasta 61 px
        de una linea a la siguiente, que en pantalla es una raya atravesando el
        rostro. El hueso gira alrededor de la oreja, y girando el campo crece
        despacio donde no hay boca que abrir.

        Mide el salto entre lineas vecinas en las columnas FUERA del hueco.
        """
        campo, _ = _campo(1.0)
        fuera = np.r_[np.arange(205, 262), np.arange(438, 495)]   # mejillas
        salto = float(np.abs(np.diff(campo[290:700, fuera], axis=0)).max())
        assert salto < 4.0, f"escalón de {salto:.1f} px en la mejilla"

    def test_en_el_centro_de_la_boca_el_labio_de_abajo_baja_con_la_mandibula(self):
        """El opuesto del test de arriba, y por eso los dos andan en par.

        Si la rampa central fuera suave la boca no abria: el labio de abajo
        descenderia una fraccion de lo que el menton baja y el hueco se
        volveria una raya. Justo debajo de la linea de los labios el campo ya
        tiene que estar casi lleno.
        """
        campo, caida = _campo(1.0)
        assert campo[508, 350] > 0.8 * caida

    def test_el_campo_crece_de_la_comisura_de_la_mandibula_al_menton(self):
        """Amplitud por distancia a la bisagra: es el giro del hueso.

        El menton baja todo, la comisura de la mandibula baja poco, y entre los
        dos la caida cae sin vuelta. Traslacion en bloque (la version vieja)
        daria el mismo valor en las tres columnas.
        """
        campo, caida = _campo(1.0)
        centro, medio, comisura = campo[600, 350], campo[600, 280], campo[600, 215]
        assert centro > medio > comisura
        assert centro > 0.9 * caida
        assert comisura < 0.5 * caida

    def test_boca_cerrada_no_toca_un_pixel(self):
        cuadro, pts = _rostro_sintetico()
        copia = cuadro.copy()
        assert pv.abrir_maxilar(cuadro, pts, 570.0, 0.0) == 0.0
        assert np.array_equal(cuadro, copia)

    def test_el_campo_muere_antes_del_borde_de_la_caja(self):
        """Costura vertical en el lateral: defecto pagado en el presentador quieto."""
        campo, _ = _campo(1.0)
        assert float(np.abs(campo[:, :180]).max()) < 0.5
        assert float(np.abs(campo[:, 525:]).max()) < 0.5

    def test_abrir_de_verdad_mueve_los_pixels(self):
        cuadro, pts = _rostro_sintetico()
        copia = cuadro.copy()
        caida = pv.abrir_maxilar(cuadro, pts, 570.0, 0.9)
        assert caida > 0
        assert not np.array_equal(cuadro, copia)


class TestHuecoDeLaBoca:
    """El hueco entre los labios -- recortado por el contorno REAL, no por elipse."""

    def _tonos(self):
        return pv.Tonos(interior=(42.0, 20.0, 20.0), diente=(180.0, 178.0, 172.0),
                        lengua=(96.0, 52.0, 52.0))

    def _hueco(self, abertura=0.9, anchura=0.0, caida=50.0):
        _, pts = _rostro_sintetico()
        return pv.hueco_boca(pts, _contorno(pts), caida, abertura, anchura,
                             self._tonos())

    def test_el_borde_del_hueco_es_suave(self):
        """El `ImageDraw` de Pillow no tiene antialiasing: la mascara sale en 4x.

        Sin la supermuestreo el poligono sale con CERO pixel de borde parcial
        -- escalon duro, que en pantalla se lee como serrucho en el labio.
        """
        mosaico, _ = self._hueco()
        alfa = np.asarray(mosaico)[:, :, 3]
        parciales = int(((alfa > 12) & (alfa < 243)).sum())
        assert parciales > 200, f"solo {parciales} pixels de borde parcial"

    def test_el_hueco_cierra_en_punta_en_las_comisuras(self):
        """Comisura de boca no abre: es donde los dos labios se encuentran.

        Sin el perfil de abertura el labio de abajo bajaba lo mismo en el medio
        y en la punta, y el hueco salia como una **barra rectangular** de
        esquina en escuadra -- defecto visto en la ampliacion 2x del 20/09/2026.
        """
        mosaico, _ = self._hueco()
        alfa = np.asarray(mosaico)[:, :, 3]
        alturas = (alfa > 128).sum(axis=0)
        con_hueco = np.flatnonzero(alturas > 0)
        assert con_hueco.size > 20
        medio = int(alturas.max())
        # en los bordes del hueco la altura tiene que ser una fraccion pequena
        # del medio
        punta = max(int(alturas[con_hueco[0] + 2]), int(alturas[con_hueco[-1] - 2]))
        assert punta < 0.35 * medio, f"punta {punta} contra medio {medio}"

    def test_el_diente_aparece_arriba_del_hueco_y_no_en_el_medio(self):
        """El diente cuelga del craneo, no del hueso que baja."""
        mosaico, _ = self._hueco(abertura=1.0)
        a = np.asarray(mosaico)
        dentro = a[:, :, 3] > 200
        lineas = np.flatnonzero(dentro.any(axis=1))
        alto, bajo = lineas[0], lineas[-1]
        cx = a.shape[1] // 2
        columna = a[alto:bajo + 1, cx, 0].astype(int)
        pico = int(np.argmax(columna)) / max(len(columna) - 1, 1)
        assert 0.02 < pico < 0.45, f"pico del diente en {pico:.2f} de la altura"
        assert columna.max() > 120

    def test_el_fondo_del_hueco_no_es_negro(self):
        """Sin lengua el hueco se lee como agujero recortado en el rostro."""
        mosaico, _ = self._hueco(abertura=1.0)
        a = np.asarray(mosaico)
        dentro = a[:, :, 3] > 200
        lineas = np.flatnonzero(dentro.any(axis=1))
        fondo = a[lineas[-3], a.shape[1] // 2, :3].astype(int)
        assert fondo[0] > fondo[2] + 8, "el fondo debia tirar hacia el rojo"
        assert fondo[0] > 50

    def test_boca_casi_cerrada_no_dibuja_hueco(self):
        assert self._hueco(abertura=0.02, caida=0.4) is None

    def test_el_hueco_nunca_toca_el_alfa_de_la_silueta(self):
        """Tocar el alfa ahi abriria un agujero en mitad del rostro."""
        cuadro, pts = _rostro_sintetico()
        alfa_antes = cuadro[:, :, 3].copy()
        pv.hablar(cuadro, pts, _contorno(pts), 570.0, 0.9, 0.2, self._tonos())
        assert np.array_equal(cuadro[:, :, 3], alfa_antes)

    def test_hablar_de_verdad_pinta_el_hueco(self):
        cuadro, pts = _rostro_sintetico()
        copia = cuadro.copy()
        pv.hablar(cuadro, pts, _contorno(pts), 570.0, 0.9, 0.0, self._tonos())
        assert not np.array_equal(cuadro, copia)


class TestExtenderLabios:
    def test_redondear_y_estirar_mueven_y_el_neutro_no(self):
        cuadro, pts = _rostro_sintetico()
        copia = cuadro.copy()
        pv.extender_labios(cuadro, pts, 0.0)
        assert np.array_equal(cuadro, copia)
        pv.extender_labios(cuadro, pts, -0.9)
        assert not np.array_equal(cuadro, copia)

    def test_el_campo_se_queda_en_la_franja_del_labio(self):
        """Con la caja vieja (121 px de altura) el estirar ondulaba la barba.

        Labio es lo que estira; mejilla y menton alrededor, no. Bloqueado
        porque el defecto solo aparecia en la ampliacion 2x, en movimiento.
        """
        cuadro, pts = _rostro_sintetico()
        antes = cuadro[:, :, 0].astype(int).copy()
        pv.extender_labios(cuadro, pts, 0.9)
        cambio = np.abs(cuadro[:, :, 0].astype(int) - antes) > 1
        lineas = np.flatnonzero(cambio.any(axis=1))
        assert lineas.size
        alto = pts["labio_sup_ext"][1] - 14
        bajo = pts["labio_inf_ext"][1] + 14
        assert lineas[0] >= alto and lineas[-1] <= bajo, (
            f"movio de y={lineas[0]} a {lineas[-1]}, fuera de la franja "
            f"{alto:.0f}-{bajo:.0f}")


class TestVisemasBasicos:
    def test_reposo_es_boca_cerrada(self):
        assert REPOSO.abertura == 0.0

    def test_todo_visema_del_mapa_es_plausible(self):
        for nombre, v in VISEMAS.items():
            assert 0.0 <= v.abertura <= 1.0, nombre
            assert -1.0 <= v.anchura <= 1.0, nombre
            assert v.peso > 0, nombre
