"""Encaixe no nicho: tech, IA e ciencia.

Lexico bilingue porque as fontes sao bilingues -- o Hacker News entrega titulo em
ingles, o Google Trends e a Wikipedia entregam termo em portugues. Um lexico so
em pt descartaria a fonte primaria do nicho.

O encaixe e um **portao**, nao um tempero do score: termo fora do nicho e
descartado mesmo com velocidade altissima. Sem isso, os assuntos mais quentes do
Brasil (que raramente sao tech) dominariam toda coleta, e a memoria de
performance do M5 ficaria com temas incomparaveis entre si.

Duas coisas foram aprendidas medindo contra titulos reais, nao no papel:

1. `content_tokens` descartava tokens com menos de 3 caracteres, o que matava
   "ai" e "ia" -- os dois termos mais centrais do lexico -- antes da comparacao.
   O nicho usa piso 2; a deduplicacao mantem 3, onde tokens curtos so somam ruido.

2. A fonte carrega informacao sobre o nicho. O Hacker News e uma comunidade de
   tecnologia: um titulo que chegou la ja passou por curadoria humana de assunto.
   Exigir dele a mesma prova lexical que de uma fonte generalista joga fora essa
   evidencia. Por isso existe um prior por fonte -- deliberadamente abaixo do
   limiar, para dar vantagem e nao passe livre: "Warren Buffett deixa a Berkshire"
   tambem esta no Hacker News e continua sendo rejeitado.
"""

from __future__ import annotations

from agent.text import content_tokens

# Termos que sozinhos ja caracterizam o nicho.
NUCLEO = frozenset("""
ia ai llm llms gpt chatbot bot agente agentes agent agents copilot
inteligencia artificial modelo modelos model models machine learning aprendizado
neural neurais algoritmo algoritmos dataset embedding embeddings transformer
inference inferencia treinamento training quantizacao quantization compressao
compression comprimido benchmark benchmarks prompt tokens parametros parameters
openai anthropic claude gemini qwen llama deepseek mistral gpt4 chatgpt copilot
huggingface nvidia amd intel tsmc arm apple microsoft google meta tesla waymo
chip chips gpu gpus cpu npu tpu semicondutor semicondutores processador
risc x86 arm64 silicio wafer litografia nanometro
software codigo code coding programacao programming developer desenvolvedor
linguagem language compiler compilador interpretador kernel linux unix bsd
opensource api sdk framework biblioteca library runtime allocator
database banco dados servidor server cloud nuvem container kubernetes docker
devops backend frontend browser navegador protocolo protocol
criptografia cryptography ciberseguranca cybersecurity vulnerabilidade
vulnerability exploit malware ransomware phishing overflow injection
privacidade seguranca security hacker breach passkey passkeys senha password
autenticacao authentication sso encryption criptografado
quantum quantica quantico qubit qubits supercomputador computacao computador
computing
robo robos robotica robot robotics drone drones autonomo autonoma autonomous
espaco space foguete rocket satelite satelites nasa spacex esa orbita orbital
marte mars lua moon telescopio telescope astronomia astronomo galaxia galaxy
planeta exoplaneta asteroide sonda cometa nebulosa
ciencia science cientista cientistas pesquisa research estudo study experimento
laboratorio fisica physics quimica chemistry biologia biology genoma genome
genetica dna rna proteina protein celula neurociencia neuroscience microbioma
clima climate climatico energia energy fusao fusion nuclear solar eolica
bateria baterias battery hidrogenio hydrogen fotovoltaico supercondutor
internet web wifi bluetooth 5g 6g android ios iphone ipad macos windows smartphone
smartphones celular celulares aplicativo aplicativos notebook laptop hardware firmware
transistor microprocessador microchip semicondutor eniac arpanet criptografica
huggingface transformers deepfake deepfakes chatbots
""".split())

# Sozinhos nao dizem nada ("lancamento" pode ser de qualquer coisa), mas somados
# a um termo do nucleo reforcam que o assunto e mesmo do nicho.
APOIO = frozenset("""
lancamento lancou anuncia anuncio anunciou descoberta descobriram avanco recorde
versao update atualizacao beta release open source desempenho performance
eficiencia eficiente bilhoes bilhao milhoes trilhoes bits bytes gigabytes
terabytes latencia throughput cache memoria memory escala escalabilidade
startup empresa laboratorio universidade instituto paper artigo preprint
launch launched announces announced discovery breakthrough record version
billion million research study engineering engineer build built tool tooling
""".split())

# Fontes cujo escopo ja e o nicho. O valor fica ABAIXO do limiar de proposito:
# e vantagem inicial, nao aprovacao automatica.
PRIOR_POR_FONTE: dict[str, float] = {
    "hacker_news": 0.22,
    # Redacoes de tech tambem cobrem console, show e celebridade: vantagem,
    # nao passe livre -- mesmo raciocinio do HN.
    "rss_tech_br": 0.22,
    "rss_tech": 0.22,
    "rss_ciencia": 0.22,
    # Estes dois SAO o nicho por construcao: o Hub so lista modelos de IA, e o
    # arquivo e curado a mao para historia da computacao e ciencia. Aqui o
    # prior aprova sozinho, de proposito.
    "huggingface": 0.6,
    "arquivo": 0.6,
}

# Abaixo disso o termo e descartado. Calibrado contra os 20 titulos reais em
# tests/fixtures/radar/hacker_news.json -- ver test_niche_calibracao.
LIMIAR_PADRAO = 0.34

# O nicho precisa enxergar "ai" e "ia"; a deduplicacao nao ganha nada com isso.
_MIN_TOKEN_NICHO = 2


def fit(termo: str, source: str | None = None) -> float:
    """Quanto o termo pertence ao nicho, de 0 a 1.

    Um termo do nucleo ja garante 0,6 -- suficiente para passar o portao sozinho,
    porque "NASA" ou "Linux" sao assunto do canal mesmo sem mais nenhuma pista.
    Termos de apoio somam pouco e nunca aprovam sozinhos.

    O prior da fonte entra como piso, nao como soma: uma fonte de nicho garante
    o minimo, e o lexico so tem como melhorar a partir dai.
    """
    palavras = content_tokens(termo, min_len=_MIN_TOKEN_NICHO)
    prior = PRIOR_POR_FONTE.get(source or "", 0.0)
    if not palavras:
        return prior

    nucleo = len(palavras & NUCLEO)
    apoio = len(palavras & APOIO)

    if nucleo == 0:
        # So apoio nao caracteriza nicho: "lancamento recorde" pode ser de futebol.
        lexico = min(0.15 * apoio, 0.25)
    else:
        # Satura rapido: dois termos do nucleo ja e sinal forte, e o terceiro nao
        # deveria valer mais que a diferenca entre ter e nao ter nicho.
        lexico = min(0.6 + 0.2 * min(nucleo - 1, 2) + 0.05 * min(apoio, 2), 1.0)

    return round(max(lexico, prior), 4)


def matched_terms(termo: str) -> tuple[set[str], set[str]]:
    """Quais termos casaram. Usado para justificar a decisao no registro."""
    palavras = content_tokens(termo, min_len=_MIN_TOKEN_NICHO)
    return palavras & NUCLEO, palavras & APOIO
