"""Vector de marca: una fuente, puertas medidas, conformidad probada.

Si la guía cambia, cambia `brand/brand.json` y estos tests dicen dónde el código
divergió -- en vez de que la identidad se pudra en silencio.
"""

from __future__ import annotations

import re

from agent.brand.brand import (
    avatar_prompt,
    load,
    suggest_content_pillar,
    voice_brief,
)
from agent.brand.checks import (
    check_emoji_muletilla,
    check_hook,
    check_numbers,
)


class TestVector:
    def test_siete_pilares_un_acento_por_pieza(self):
        """Los seis de la guía + historia (20/09/2026, petición del autor)."""
        brand = load()
        assert sorted(brand.pillars) == ["analisis", "dato", "futuro", "historia",
                                         "news", "tutorial", "vs"]
        for p in brand.pillars.values():
            assert re.fullmatch(r"#[0-9A-F]{6}", p.accent)
        assert brand.accent_for("dato") == "#00E0FF"
        assert brand.accent_for("news") == "#39FF88"
        assert brand.accent_for("inexistente") == "#39FF88"

    def test_hashtags_y_bio(self):
        brand = load()
        assert brand.hashtags == ("#ia", "#inteligenciaartificial", "#tecnologia",
                                  "#ai", "#circuitocero")
        assert "@circuitocero" in brand.handle and "Señales del futuro" in brand.tagline

    def test_sugerencia_por_asunto(self):
        assert suggest_content_pillar("Cómo usar prompts en el día a día") == "tutorial"
        assert suggest_content_pillar("Modelo contra modelo: cuál gana") == "vs"
        assert suggest_content_pillar("Anuncio de modelo nuevo") == "news"

    def test_voz_tiene_las_reglas(self):
        texto = voice_brief()
        assert "12 palabras" in texto and "emoji" in texto


class TestPresentadores:
    def test_elenco_con_seeds_y_formatos(self):
        brand = load()
        assert sorted(brand.presenters) == ["iris", "theo"]
        assert brand.presenters["iris"].seed == 481502
        assert brand.presenters["theo"].seed == 907314
        # Desde la noche del 20/09/2026 ATLAS presenta TODOS los pilares: solo
        # él tiene clip base filmado, y mezclar un presentador filmado con uno
        # sintetizado en el mismo canal sería una diferencia de calidad visible.
        for pilar in ("analisis", "tutorial", "news", "futuro", "historia",
                      "dato", "vs"):
            assert brand.presenter_for(pilar).id == "theo", pilar
        # Nova no se eliminó, se quedó sin formato -- vuelve a disputar pilar
        # sola cuando `nova_base.json` exista.
        assert brand.presenters["iris"].formats == ()

    def test_voces_mapeadas_sin_invencion(self):
        brand = load()
        assert brand.presenters["theo"].library_voice == "atlas"
        assert brand.presenters["iris"].library_voice is None

    def test_prompt_travado_con_variables(self):
        texto = avatar_prompt("iris", expresion="concentrada")
        assert "seed: 481502" in texto and "concentrada" in texto
        assert "fondo transparente" in texto

    def test_variable_fuera_de_lista_falla(self):
        import pytest
        with pytest.raises(ValueError):
            avatar_prompt("theo", gesto="bailando")


class TestPuertas:
    def test_gancho_hasta_12(self):
        assert check_hook(" ".join(["palabra"] * 12)) is None
        fallo = check_hook(" ".join(["palabra"] * 13))
        assert fallo is not None and "12" in fallo

    def test_nombre_propio_no_cuenta_como_numero(self):
        assert check_numbers("El Bonsai 27B corre en la RTX 5090.") == []
        assert check_numbers("Escala FP16 con 1.76 bits.") == []

    def test_intervalo_cuenta_como_uno(self):
        assert check_numbers("Soporta de 25 a 300 cuentas por clave.") == []

    def test_dos_numeros_piden_dos_frases(self):
        texto = "Ocupa 5,9 GB y retiene 98,2% del rendimiento."
        (fallo,) = check_numbers(texto)
        assert "uno por frase" in fallo
        assert check_numbers("Ocupa 5,9 GB en disco. Retiene casi todo.") == []

    def test_emoji_y_muletilla(self):
        assert any("emoji" in p for p in check_emoji_muletilla("Mira esto \U0001F600"))
        assert any("que pasa gente" in p for p in check_emoji_muletilla("¿Qué pasa gente!"))
        assert check_emoji_muletilla("Texto limpio.") == []


class TestDiapositivasEnLaMarca:
    def test_leyenda_lleva_hashtags(self, tmp_path):
        from agent.models import Carousel
        from agent.render.carousel import render_carousel
        from tests.test_carousel import slides
        carrusel = Carousel.model_validate_json(slides())
        render_carousel(carrusel, tmp_path / "car", "dato")
        caption = (tmp_path / "car" / "caption.txt").read_text(encoding="utf-8")
        assert "#circuitocero" in caption
