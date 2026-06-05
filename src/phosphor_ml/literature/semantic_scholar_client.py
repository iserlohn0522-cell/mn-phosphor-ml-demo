from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class SemanticScholarClient:
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: str = "", timeout: int = 30) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int = 25) -> dict:
        params = {
            "query": query,
            "limit": str(max(1, min(max_results, 100))),
            "fields": ",".join(
                [
                    "paperId",
                    "externalIds",
                    "title",
                    "year",
                    "authors",
                    "venue",
                    "abstract",
                    "url",
                    "publicationTypes",
                    "publicationDate",
                ]
            ),
        }
        return self._get_json(params)

    def _get_json(self, params: dict[str, str]) -> dict:
        headers = {"User-Agent": "mn-phosphor-ml/0.1"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        request = Request(
            f"{self.base_url}?{urlencode(params)}",
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Semantic Scholar request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Semantic Scholar request failed: {exc.reason}") from exc
