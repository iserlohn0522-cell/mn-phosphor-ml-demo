from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class CrossrefClient:
    base_url = "https://api.crossref.org/works"

    def __init__(self, mailto: str = "", timeout: int = 30) -> None:
        self.mailto = mailto
        self.timeout = timeout

    def search(self, query: str, max_results: int = 25) -> dict:
        params = {
            "query": query,
            "rows": str(max(1, min(max_results, 1000))),
        }
        if self.mailto:
            params["mailto"] = self.mailto
        return self._get_json(params)

    def _get_json(self, params: dict[str, str]) -> dict:
        user_agent = "mn-phosphor-ml/0.1"
        if self.mailto:
            user_agent = f"{user_agent} (mailto:{self.mailto})"
        request = Request(
            f"{self.base_url}?{urlencode(params)}",
            headers={"User-Agent": user_agent},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Crossref request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Crossref request failed: {exc.reason}") from exc
