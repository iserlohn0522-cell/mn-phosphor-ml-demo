from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class OpenAlexClient:
    base_url = "https://api.openalex.org/works"

    def __init__(self, email: str = "", api_key: str = "", timeout: int = 30) -> None:
        self.email = email
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int = 25) -> dict:
        params = {
            "search": query,
            "per-page": str(max(1, min(max_results, 200))),
        }
        if self.email:
            params["mailto"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        return self._get_json(params)

    def _get_json(self, params: dict[str, str]) -> dict:
        user_agent = "mn-phosphor-ml/0.1"
        if self.email:
            user_agent = f"{user_agent} (mailto:{self.email})"
        request = Request(
            f"{self.base_url}?{urlencode(params)}",
            headers={"User-Agent": user_agent},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"OpenAlex request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAlex request failed: {exc.reason}") from exc
