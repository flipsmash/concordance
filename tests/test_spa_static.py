from starlette.applications import Starlette
from starlette.testclient import TestClient

from webapp.backend.main import SPAStaticFiles


def _client(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><div id=root></div>")
    (tmp_path / "assets" / "Home-abc.v2.js").write_text("export default 1")
    app = Starlette()
    app.mount("/", SPAStaticFiles(directory=str(tmp_path), html=True))
    return TestClient(app)


def test_client_routes_get_the_page_always_revalidated(tmp_path):
    r = _client(tmp_path).get("/app/visualizations")
    assert r.status_code == 200 and "id=root" in r.text
    assert r.headers["cache-control"] == "no-cache"


def test_hashed_assets_cache_for_good(tmp_path):
    r = _client(tmp_path).get("/assets/Home-abc.v2.js")
    assert r.status_code == 200 and "immutable" in r.headers["cache-control"]


def test_missing_asset_is_an_uncached_404_not_the_page(tmp_path):
    # Chrome cached a mid-deploy failure for a chunk and broke on that page
    r = _client(tmp_path).get("/assets/Home-gone.css")
    assert r.status_code == 404 and r.headers["cache-control"] == "no-store"
    assert "id=root" not in r.text
