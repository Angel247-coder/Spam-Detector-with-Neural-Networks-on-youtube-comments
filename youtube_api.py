"""
youtube_api.py
==============
Integración con la YouTube Data API v3.

IMPORTANTE: el documento original (sección "Exclusiones del Producto")
indica explícitamente que la aplicación NO se conectará a las APIs de
YouTube. El usuario ha pedido conectar la aplicación a la API a pesar
de esa exclusión, así que este módulo cubre ese requisito ampliado.

Uso:
    from youtube_api import fetch_comments
    comments = fetch_comments("https://youtube.com/watch?v=XXXX",
                              api_key=os.environ["YOUTUBE_API_KEY"],
                              max_comments=500)

Requisitos:
    pip install google-api-python-client

La clave de API se obtiene en Google Cloud Console habilitando
"YouTube Data API v3". No se almacena en el código.
"""

from __future__ import annotations

import re
from typing import List
from urllib.parse import parse_qs, urlparse

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:                                  # pragma: no cover
    build = None                                     # se valida en runtime
    HttpError = Exception


VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Acepta una URL de YouTube en cualquiera de los formatos comunes
    o directamente un ID de 11 caracteres."""
    s = url_or_id.strip()
    if VIDEO_ID_RE.match(s):
        return s

    parsed = urlparse(s)
    host = (parsed.hostname or "").lower()

    if host in {"youtu.be"}:
        candidate = parsed.path.lstrip("/")
        if VIDEO_ID_RE.match(candidate):
            return candidate

    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if parsed.path == "/watch":
            v = parse_qs(parsed.query).get("v", [None])[0]
            if v and VIDEO_ID_RE.match(v):
                return v
        # /embed/<id>, /shorts/<id>, /v/<id>
        parts = parsed.path.split("/")
        for i, part in enumerate(parts):
            if part in {"embed", "shorts", "v"} and i + 1 < len(parts):
                candidate = parts[i + 1]
                if VIDEO_ID_RE.match(candidate):
                    return candidate

    raise ValueError(
        f"No se pudo extraer un ID de vídeo de YouTube válido de: {url_or_id!r}"
    )


def fetch_comments(
    url_or_id: str,
    api_key: str,
    *,
    max_comments: int = 500,
    include_replies: bool = False,
) -> List[dict]:
    """Devuelve una lista de dicts con claves:
        comment_id, author, date, content
    listos para pasarlos por el pipeline de features.
    """
    if build is None:
        raise ImportError(
            "Falta google-api-python-client. Instala con:\n"
            "    pip install google-api-python-client"
        )
    if not api_key:
        raise ValueError("Se requiere una API key de YouTube Data API v3.")

    video_id = extract_video_id(url_or_id)
    youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    results: List[dict] = []
    page_token: str | None = None

    try:
        while len(results) < max_comments:
            request = youtube.commentThreads().list(
                part="snippet,replies" if include_replies else "snippet",
                videoId=video_id,
                pageToken=page_token,
                maxResults=min(100, max_comments - len(results)),
                textFormat="plainText",
            )
            response = request.execute()

            for item in response.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                results.append({
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "author":     top.get("authorDisplayName", ""),
                    "date":       top.get("publishedAt", ""),
                    "content":    top.get("textDisplay", ""),
                })
                if include_replies and item.get("replies"):
                    for reply in item["replies"]["comments"]:
                        rsnip = reply["snippet"]
                        results.append({
                            "comment_id": reply["id"],
                            "author":     rsnip.get("authorDisplayName", ""),
                            "date":       rsnip.get("publishedAt", ""),
                            "content":    rsnip.get("textDisplay", ""),
                        })
                        if len(results) >= max_comments:
                            break
                if len(results) >= max_comments:
                    break

            page_token = response.get("nextPageToken")
            if not page_token:
                break
    except HttpError as exc:
        raise RuntimeError(
            f"Error en YouTube Data API: {exc}. "
            "Comprueba que la API key sea válida y que el vídeo permita comentarios."
        ) from exc

    return results[:max_comments]
