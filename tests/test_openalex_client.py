from phosphor_ml.literature.openalex_client import OpenAlexClient


class CapturingOpenAlexClient(OpenAlexClient):
    def __init__(self):
        super().__init__(email="demo@example.org", api_key="oa_demo_key")
        self.params = None

    def _get_json(self, params):
        self.params = params
        return {"results": []}


def test_openalex_client_sends_email_and_api_key_parameters():
    client = CapturingOpenAlexClient()

    client.search("Mn4+ fluoride phosphor", max_results=5)

    assert client.params["mailto"] == "demo@example.org"
    assert client.params["api_key"] == "oa_demo_key"
    assert client.params["per-page"] == "5"
