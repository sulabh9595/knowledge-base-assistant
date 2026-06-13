from app.loaders.confluence_loader import ConfluenceLoader, html_to_text


def test_html_to_text_strips_tags() -> None:
    html_content = "<h1>Title</h1><p>Hello <strong>world</strong></p>"
    assert html_to_text(html_content) == "Title Hello world"


class DummyClient:
    def __init__(self, response):
        self.response = response

    def get(self, url, params=None):
        class Response:
            def __init__(self, data):
                self._data = data

            def raise_for_status(self):
                pass

            def json(self):
                return self._data

        return Response(self.response)


def test_loader_fetch_space_pages_monkeypatch(monkeypatch):
    response = {
        "results": [
            {
                "id": "123",
                "title": "Test Page",
                "_links": {"webui": "/pages/123"},
                "body": {"storage": {"value": "<p>Test</p>"}},
                "version": {"number": 1},
                "space": {"key": "TEST"},
            }
        ],
        "size": 1,
        "start": 0,
        "totalSize": 1,
    }

    loader = ConfluenceLoader(base_url="https://example.atlassian.net", email="user@example.com", api_token="token")
    monkeypatch.setattr(loader, "client", DummyClient(response))

    pages = loader.fetch_space_pages("TEST")
    assert len(pages) == 1
    assert pages[0]["page_id"] == "123"
    assert pages[0]["title"] == "Test Page"
    assert pages[0]["text"] == "Test"
