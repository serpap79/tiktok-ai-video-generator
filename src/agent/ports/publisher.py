"""Porta Publisher: leva um MP4 pronto para a inbox do TikTok.

Existe como porta porque a primeira implementacao fala com a Content Posting
API oficial (inbox, escopo `video.upload`). Se um dia o destino mudar, troca-se
o adaptador sem tocar na CLI nem na memoria.

O fluxo inbox e deliberado: ele **nao exige auditoria** do app, ao contrario do
Direct Post (que posta direto mas so em modo privado ate passar na auditoria).
O preco e que o endpoint inbox nao recebe titulo, descricao nem `is_aigc` --
so `source_info`. Legenda e rotulo AIGC sao aplicados pelo criador no app, ao
concluir o post a partir da notificacao da inbox. O adaptador nao finge o
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
    gravado. Esta excecao e para o que impede concluir a chamada.
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

        Os valores possiveis nao sao validados aqui de proposito: afirmar uma
        lista fechada sem ter exercitado a API real seria documentar chute.
        """
        ...
