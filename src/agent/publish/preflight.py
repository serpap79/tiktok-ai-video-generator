"""Preflight: a ultima porta antes de producao.

Junta o que cada estagio mediu sozinho numa unica sentenca publicavel:
parecer aprovado existe, MP4 tem dimensao/audio/duracao da faixa, fatos com
fonte. Nao julga nada de novo -- so confere que os artefatos combinam entre
si. O que so o humano confere (rotulo AIGC, legenda) sai como checklist, nao
como portao: portao que ninguem pode passar sozinho e teatro.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BANDS = {"long": (60, 90), "short": (10, 30)}


@dataclass
class Gate:
    label: str
    passed: bool
    detail: str = ""


@dataclass
class PreflightReport:
    gates: list[Gate] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return bool(self.gates) and all(g.passed for g in self.gates)


def _review_match(script: dict, scripts: list[dict],
                  reviews: list[dict]) -> dict | None:
    """Parecer aprovado ligado a linha do roteiro (via script_id).

    Tema sozinho mente: o roteiro pode ter sido reescrito depois do parecer.
    O elo roteiro->parecer ja existe no banco (produce grava os dois
    ligados); o preflight so confere o elo, nao rejulga o texto.
    """
    topic = script.get("topic", "")
    hook = script.get("hook", "")
    body = script.get("body", "")
    closing = script.get("closing", "")
    palavras = len(f"{hook} {body} {closing}".split())
    linha = next((s for s in scripts
                  if s.get("topic") == topic and s.get("word_count") == palavras),
                 None)
    if linha is None:
        return None
    return next((r for r in reviews
                 if r.get("script_id") == linha["id"] and r.get("approved")),
                None)


def preflight_video(script: dict, scripts: list[dict], reviews: list[dict],
                    probe: dict | None) -> PreflightReport:
    """Pacote video (long/short): roteiro + parecer + MP4 medido."""
    gates: list[Gate] = []
    formato = script.get("format", "long")
    faixa = BANDS.get(formato, BANDS["long"])

    fatos = script.get("facts", [])
    gates.append(Gate(
        "fatos com fonte no roteiro", bool(fatos),
        f"{len(fatos)} fato(s)" if fatos else "roteiro sem facts"))

    parecer = _review_match(script, scripts, reviews)
    gates.append(Gate(
        "parecer aprovado para este texto", parecer is not None,
        f"parecer #{parecer['id']} ({parecer['model']})" if parecer
        else "rode judge/produce neste roteiro antes"))

    if probe is None:
        gates.append(Gate("mp4 medido", False, "arquivo ausente ou ilegivel"))
        return PreflightReport(gates)

    gates.append(Gate(
        "9:16 em 1080x1920",
        (probe.get("width"), probe.get("height")) == (1080, 1920),
        f"{probe.get('width')}x{probe.get('height')}"))
    gates.append(Gate(
        "trilha de audio presente", bool(probe.get("has_audio")),
        "" if probe.get("has_audio") else "mp4 mudo"))
    dur = probe.get("duration_s")
    gates.append(Gate(
        f"duracao na faixa {formato} ({faixa[0]}-{faixa[1]}s)",
        dur is not None and faixa[0] <= dur <= faixa[1],
        f"{dur}s" if dur is not None else "sem duracao"))
    return PreflightReport(gates)


def preflight_carousel(carousel: dict, approved: bool,
                       slides_ok: bool) -> PreflightReport:
    """Pacote carrossel: parecer + 5 slides na marca."""
    return PreflightReport([
        Gate("parecer aprovado (corte 8/10)", approved,
             "" if approved else "rode produce --mode carousel antes"),
        Gate("5 slides 1080x1920", slides_ok,
             "" if slides_ok else "slides fora do aceite"),
        Gate("fatos com fonte", bool(carousel.get("facts")),
             "" if carousel.get("facts") else "carrossel sem facts"),
    ])


CHECKLIST = [
    "LIGUE o rotulo de conteudo gerado por IA (obrigatorio, manual).",
    "Legenda cita as fontes do roteiro/dossie.",
    "Slide 1 paga a promessa nos seguintes (sem clickbait).",
]


__all__ = ["CHECKLIST", "Gate", "PreflightReport", "preflight_carousel",
           "preflight_video"]
