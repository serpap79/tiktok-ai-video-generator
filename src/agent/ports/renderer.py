"""Porta Renderer: transforma um Script em MP4 vertical.

Existe como porta porque a primeira implementacao delega a um projeto externo
(MoneyPrinterTurbo). Se um dia quisermos renderizar por conta propria, troca-se
o adaptador sem tocar em nenhum estagio do agente.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent.models import RenderResult, Script


class RendererError(RuntimeError):
    """Falha na comunicacao com o renderizador, nao na renderizacao em si.

    Renderizacao que falha do lado do renderizador volta como
    RenderResult(state=failed), porque e resultado do dominio e precisa ser
    gravada na memoria. Esta excecao e para o que impede chegar a um resultado:
    servico fora do ar, autenticacao, timeout.
    """


@runtime_checkable
class Renderer(Protocol):
    def render(self, script: Script) -> RenderResult:
        """Produz o MP4 e devolve o resultado com duracao e dimensoes medidas.

        Bloqueia ate terminar ou estourar o timeout. Nao levanta excecao quando
        a renderizacao falha por motivo de dominio (material nao encontrado,
        TTS indisponivel): isso vem em RenderResult.state.
        """
        ...

    def health(self) -> bool:
        """True se o renderizador responde. Usado antes de gastar trabalho."""
        ...
