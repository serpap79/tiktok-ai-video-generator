"""Porta Renderer: transforma um Script em MP4 vertical.

Existe como puerto porque la primera implementación delega en un proyecto externo
(MoneyPrinterTurbo). Se um dia quisermos renderizar por conta propria, troca-se
o adaptador sem tocar em nenhum estagio do agente.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent.models import RenderResult, Script


class RendererError(RuntimeError):
    """Fallo en la comunicación con el renderizador, no en la renderización.

    Una renderización que falla en el renderizador vuelve como
    RenderResult(state=failed), porque e resultado do dominio e precisa ser
    un resultado guardado en memoria. Esta excepción representa lo que impide obtener un resultado:
    servico fora do ar, autenticacao, timeout.
    """


@runtime_checkable
class Renderer(Protocol):
    def render(self, script: Script) -> RenderResult:
        """Produz o MP4 e devolve o resultado com duracao e dimensoes medidas.

        Bloquea hasta terminar o agotar el tiempo de espera. No lanza una excepción cuando
        la renderización falla por un motivo de dominio (material no encontrado,
        TTS indisponivel): isso vem em RenderResult.state.
        """
        ...

    def health(self) -> bool:
        """True si el renderizador responde. Se usa antes de gastar trabajo."""
        ...
