"""Biblioteca de vozes: cada entrada diz o que e, de onde veio e o que pode.

Regra de honestidade deste arquivo: campo nao verificado fica marcado, nunca
inventado. Genero aparente, por exemplo, so entra quando a fonte (model card
ou escuta) diz -- as 4 amostras em `output/vozes/` existem justamente para a
escuta calibrar estes metadados.

Lacunas declaradas (19/09/2026):
- **Voz da Íris**: a apresentadora pede pt-BR feminino medio-grave, e voz
  feminina pt-BR aberta nao existe nos catalogos Piper. Sem ela, Íris tem
  rosto e prompt, mas nao tem voz -- e sintetiza-la em timbre masculino
  quebraria a personagem. Théo usa `jeff` (estilo `theo`).
- **Sotaque regional**: nenhum modelo aberto reproduz sotaque (paulista,
  carioca, nordestino...). Variedade aqui e troca de locutor, nao de sotaque.
  Fingir sotaque com pitch/velocidade seria caricatura, nao recurso.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    id: str
    nome: str
    genero: str
    idade_aparente: str
    timbre: str
    sotaque: str
    usos: tuple[str, ...] = ()
    modelo_url: str = ""
    config_url: str = ""
    arquivo: str = ""
    licenca: str = ""
    notas: str = ""
    ativo: bool = True


VOICES: dict[str, Voice] = {
    "razo": Voice(
        id="razo",
        nome="Razo (tecnologia)",
        genero="masculino",
        idade_aparente="30-40",
        timbre="medio, articulado",
        sotaque="brasileiro neutro",
        usos=("noticia", "curiosidade", "documental", "historia"),
        modelo_url=("https://huggingface.co/Lucasllfs/Razo-piper-voice/"
                    "resolve/main/pt-BR-razo-medium.onnx"),
        config_url=("https://huggingface.co/Lucasllfs/Razo-piper-voice/"
                    "resolve/main/config.json"),
        arquivo="razo.onnx",
        licenca="MIT (modelo e codigo)",
        notas=("Fine-tune p/ vocabulario de tecnologia; RTF 0.08 medido em "
               "i5-1334U (60s de audio em ~5s). Sem warnings de fonema."),
    ),
    "faber": Voice(
        id="faber",
        nome="Faber (neutro)",
        genero="a confirmar pela escuta",
        idade_aparente="a confirmar pela escuta",
        timbre="neutro",
        sotaque="brasileiro neutro",
        usos=("documental", "noticia", "calmo"),
        modelo_url=("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                    "pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx"),
        arquivo="faber.onnx",
        licenca="MIT (repo piper-voices); rehost Trelis marca CC0",
        notas="RTF 0.098 medido; sem warnings de fonema.",
    ),
    "jeff": Voice(
        id="jeff",
        nome="Jeff (grave, pausado)",
        genero="a confirmar pela escuta",
        idade_aparente="a confirmar pela escuta",
        timbre="grave, ritmo mais lento",
        sotaque="brasileiro neutro",
        usos=("suspense", "misterio", "dramatico", "emocional"),
        modelo_url=("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                    "pt/pt_BR/jeff/medium/pt_BR-jeff-medium.onnx"),
        arquivo="jeff.onnx",
        licenca="MIT (repo piper-voices)",
        notas=("RTF 0.071 medido; ~2,4 palavras/s (mais lento que os 2,5 "
               "assumidos -- roteiro rende audio ~20% mais longo)."),
    ),
}

# Fora da biblioteca, com motivo gravado (mesma regra do ledger: rejeicao com
# motivo calibra, rejeicao silenciosa vira re-teste).
REJEITADAS: dict[str, str] = {
    "edresson-low": ("fonemas de nasal (ã/õ) ausentes no mapa -- 60+ warnings e "
                     "audio arrastado (90s p/ 176 palavras). Tier low nao serve "
                     "p/ narracao."),
    "F5-TTS-pt-br": "checkpoint CC-BY-NC 4.0: sem uso comercial. Canal monetiza.",
    "XTTS-v2": "CPML: gratis so p/ nao-comercial. Canal monetiza.",
    "Bark": "MIT, mas RTF 10-20x em CPU: 60s de audio = 10-20min. Inviavel.",
    "Kokoro-82M": "Apache-2.0, mas sem voz pt-BR verificada ate 19/09/2026.",
    "edge-tts": ("ja usado no MPT: gratis sem contrato, pode quebrar; vozes "
                 "femininas existem la, mas externas e sem SLA -- fallback, "
                 "nao base."),
}


@dataclass(frozen=True)
class Style:
    id: str
    nome: str
    voz: str
    velocidade: float
    ruido: float
    pausa_frase_s: float
    descricao: str = ""


# Estilo e camada interpretativa sobre 3 timbres: muda ritmo, pausa e
# variacao -- nao finge ser locutor novo. Velocidade perto de 1.0 de
# proposito: naturalidade > efeito.
STYLES: dict[str, Style] = {
    "documental": Style("documental", "Narrador documental", "faber", 1.0, 0.5, 0.30,
                        "serio, seguro, informativo"),
    "suspense": Style("suspense", "Suspense", "jeff", 0.92, 0.4, 0.55,
                      "baixo, controlado, pausas estrategicas"),
    "misterio": Style("misterio", "Misterio", "jeff", 0.95, 0.6, 0.45,
                      "intrigante, descoberta"),
    "historia": Style("historia", "Historia", "razo", 0.98, 0.6, 0.35,
                      "natural, progressiva"),
    "curiosidade": Style("curiosidade", "Curiosidade", "razo", 1.05, 0.7, 0.25,
                         "energetico sem exagero"),
    "noticia": Style("noticia", "Noticia", "razo", 1.08, 0.4, 0.25,
                     "objetivo, claro, profissional"),
    "dramatico": Style("dramatico", "Dramatico", "jeff", 0.95, 0.8, 0.40,
                       "variacao de intensidade"),
    "emocional": Style("emocional", "Emocional", "jeff", 0.92, 0.8, 0.50,
                       "sensivel, ritmo variado"),
    "energetico": Style("energetico", "Energetico", "razo", 1.12, 0.7, 0.20,
                        "rapido, dinamico"),
    "calmo": Style("calmo", "Calmo", "faber", 0.90, 0.4, 0.50,
                   "lento, suave"),
    # Voz do Théo (guia de marca: grave, ritmo 1.0, ênfase nos números).
    # Íris não tem estilo: sem voz feminina aberta, sintetizá-la em voz
    # masculina quebraria a personagem -- gap declarado, não gambiarra.
    "theo": Style("theo", "Théo (tutorial/comparação)", "jeff", 1.0, 0.6, 0.35,
                  "didático, direto; números com micro-pausa via *enfase*"),
}

NAMES = sorted(VOICES)
STYLE_NAMES = sorted(STYLES)

__all__ = ["NAMES", "REJEITADAS", "STYLES", "STYLE_NAMES", "VOICES", "Style", "Voice"]
