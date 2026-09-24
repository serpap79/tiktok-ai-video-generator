"""Porta Publisher: leva um MP4 pronto para a inbox do TikTok.

Existe como porta porque a primeira implementacao fala com a Content Posting
API oficial (inbox, escopo `video.upload`). Se um dia o destino mudar, troca-se
o adaptador sem tocar na CLI nem na memoria.

El flujo de bandeja es deliberado: **no exige auditoría** de la app, a diferencia del
Direct Post (que posta direto mas so em modo privado ate passar na auditoria).
El precio es que el endpoint de bandeja no recibe título, descripción ni `is_aigc`
so `source_info`. Legenda e rotulo AIGC sao aplicados pelo criador no app, ao
y hay que completar la publicación desde la notificación. El adaptador no finge
contrario: ele sobe o video e registra o `publish_id`; rotular como AIGC e
etapa manual documentada na CLI, sem flag para desligar.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent.models import PublishResult


class PublisherError(RuntimeError):
    """Falha que impede chegar a um resultado: rede, auth, cota, rejeicao da API.

    Publicacao recusada pela API com motivo de dominio (ex. video invalido)
    volta como PublishResult(state=failed), porque e resultado e precisa ficar
    registrado. Esta excepción representa lo que impide completar la llamada.
    """


class PublisherAuthError(PublisherError):
    """Token invalido, expirado ou sem o escopo `video.upload`."""


class PublisherRateLimited(PublisherError):
    """A API barrou por cota: 6 req/min por token, ou teto diario de shares."""


@runtime_checkable
class Publisher(Protocol):
    def upload(self, video_path: str, *, access_token: str) -> PublishResult:
        """Sobe o MP4 para a inbox. Bloqueia ate concluir ou falhar."""
        ...

    def fetch_status(self, publish_id: str, *, access_token: str) -> str:
        """Estado atual do post na API do TikTok (string opaca, sem validacao).

        Los valores posibles no se validan aquí de forma intencionada: afirmar un
        lista fechada sem ter exercitado a API real seria documentar chute.
        """
        ...
