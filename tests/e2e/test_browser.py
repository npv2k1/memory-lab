import json
import os
from uuid import uuid4


def test_document_lifecycle(page):
    from playwright.sync_api import expect
    content = "Memory Lab preserves original evidence. Every research chunk cites its source blocks."
    filename = f"e2e-{uuid4().hex}.txt"
    page.goto(os.getenv("E2E_BASE_URL", "http://localhost:4321"))
    expect(page.locator("#status")).to_have_text("Hoàn tất.")
    page.locator("#file").set_input_files({"name": filename, "mimeType": "text/plain", "buffer": content.encode()})
    page.locator("#upload button").click()
    document = page.locator("#documents article").filter(has_text=filename)
    expect(document).to_be_visible()
    page.locator("#parser").select_option("text")
    with page.expect_response(lambda response: response.url.endswith("/process") and response.request.method == "POST") as processed:
        document.get_by_role("button", name="Process", exact=True).click()
    assert processed.value.status == 200
    run = processed.value.json()
    assert run["status"] == "completed"
    expect(page.locator("#status")).to_have_text("Hoàn tất.")
    row = page.locator("#runs article").filter(has_text=run["pipeline_run_id"][:12])
    row.get_by_role("button", name="normalized", exact=True).click()
    expect(page.locator("#artifact")).to_contain_text(content)
    normalized = json.loads(page.locator("#artifact").inner_text())
    assert normalized["version"] == run["version"]
    row.get_by_role("button", name="chunks", exact=True).click()
    expect(page.locator("#artifact")).to_contain_text('"block_ids"')
    chunks = json.loads(page.locator("#artifact").inner_text())
    assert chunks[0]["block_ids"] == ["text:0"]
    page.locator("#query").fill("research evidence")
    page.locator("#search button").click()
    expect(page.locator("#results")).to_contain_text(content)
    with page.expect_download() as downloaded:
        page.locator("#results a").first.click()
    assert downloaded.value.failure() is None
    assert open(downloaded.value.path(), "rb").read() == content.encode()
    with page.expect_response(lambda response: response.url.endswith("/process")) as retried:
        document.get_by_role("button", name="Process", exact=True).click()
    assert retried.value.json()["pipeline_run_id"] == run["pipeline_run_id"]


def test_processing_error_is_visible(page):
    from playwright.sync_api import expect
    page.goto(os.getenv("E2E_BASE_URL", "http://localhost:4321"))
    expect(page.locator("#status")).to_have_text("Hoàn tất.")
    filename = f"invalid-{uuid4().hex}.txt"
    page.locator("#file").set_input_files({"name": filename, "mimeType": "text/plain", "buffer": b"Error fixture"})
    page.locator("#upload button").click()
    document = page.locator("#documents article").filter(has_text=filename)
    expect(document).to_be_visible()
    page.locator("#parser").select_option("text")
    page.locator("#chunker").select_option("hybrid")
    document.get_by_role("button", name="Process", exact=True).click()
    expect(page.locator("#status")).to_contain_text("Hybrid chunking requires Docling")


def test_memory_and_experimentation_ui(page):
    from playwright.sync_api import expect
    marker = uuid4().hex[:8]
    page.goto(os.getenv("E2E_BASE_URL", "http://localhost:4321"))
    expect(page.locator("#status")).to_have_text("Hoàn tất.")

    # Switch to Memory Playground tab
    page.locator("button.tab[data-tab='memory']").click()
    page.locator("#mem-content").fill(f"Memory Lab uses Neo4j and BM25 for marker-{marker} on 2026-11-25.")
    page.locator("#mem-store-strategy").select_option("hybrid")
    page.locator("#mem-store-form button[type='submit']").click()
    expect(page.locator("#status")).to_have_text("Hoàn tất.")
    expect(page.locator("#mem-records")).to_contain_text(f"marker-{marker}")
    expect(page.locator("#mem-graph")).to_contain_text("neo4j")

    page.locator("#mem-query").fill(f"Neo4j marker-{marker}")
    page.locator("#mem-search-strategy").select_option("zero_token")
    page.locator("#mem-search-form button[type='submit']").click()
    expect(page.locator("#mem-results")).to_contain_text(f"marker-{marker}")

    # Switch to Experimentation & Comparison tab
    page.locator("button.tab[data-tab='experiments']").click()
    page.locator("#exp-ds-name").fill(f"eval-{marker}")
    page.locator("#exp-ds-cases").fill(
        json.dumps([{"query": f"Neo4j marker-{marker}", "relevant_substrings": [f"marker-{marker}"]}])
    )
    page.locator("#exp-dataset-form button[type='submit']").click()
    expect(page.locator("#exp-datasets")).to_contain_text(f"eval-{marker}")

    page.locator("#exp-run-name").fill(f"zero-token-{marker}")
    page.locator("#exp-run-target").select_option("memory")
    page.locator("#exp-run-strategy").select_option("zero_token")
    page.locator("#exp-run-form button[type='submit']").click()
    expect(page.locator("#exp-reports")).to_contain_text(f"zero-token-{marker}")

    page.locator("#btn-compare").click()
    expect(page.locator("#exp-comparison table")).to_contain_text(f"zero-token-{marker}")


def test_knowledge_explorer_graph_table_and_search(page):
    from playwright.sync_api import expect
    marker = uuid4().hex[:8]
    page.goto(os.getenv("E2E_BASE_URL", "http://localhost:4321"))
    expect(page.locator("#status")).to_have_text("Hoàn tất.")

    # Upload & process a document to seed knowledge
    page.locator("#file").set_input_files(
        {
            "name": f"kb-{marker}.txt",
            "mimeType": "text/plain",
            "buffer": f"Memory Lab uses Qdrant and BM25 for marker-{marker} since 2026-11-25.".encode(),
        }
    )
    page.locator("#upload button[type='submit']").click()
    doc = page.locator("#documents article").filter(has_text=f"kb-{marker}.txt")
    expect(doc).to_be_visible()
    page.locator("#parser").select_option("text")
    doc.get_by_role("button", name="Process", exact=True).click()
    expect(page.locator("#status")).to_have_text("Hoàn tất.")

    # Use Global Search Bar to jump into Knowledge Explorer (default: Embedding Atlas)
    page.locator("#global-search-input").fill(f"Qdrant marker-{marker}")
    page.locator("#global-search-form button[type='submit']").click()
    expect(page.locator("#tab-knowledge")).to_be_visible()
    expect(page.locator("#kb-search-results")).to_contain_text(f"marker-{marker}")
    expect(page.locator("#embedding-atlas-canvas")).to_be_visible()
    expect(page.locator("#atlas-linked-table table")).to_contain_text(f"kb-{marker}.txt")

    # Switch to Interactive Knowledge Graph view
    page.locator("#kb-viz-switch button[data-viz='graph']").click()
    expect(page.locator("#knowledge-graph-svg g[data-node-id]")).not_to_have_count(0)

    # Switch to Table view and check entities + relations
    page.locator("#kb-viz-switch button[data-viz='table']").click()
    expect(page.locator("#kb-table-container table")).to_contain_text("Qdrant")
    page.locator("#kb-table-views button[data-view='relations']").click()
    expect(page.locator("#kb-table-container table")).to_contain_text("uses")


def test_embedding_settings_background_progress_and_delete(page):
    from playwright.sync_api import expect
    marker = uuid4().hex[:8]
    filename = f"bg-del-{marker}.txt"
    page.goto(os.getenv("E2E_BASE_URL", "http://localhost:4321"))
    expect(page.locator("#status")).to_have_text("Hoàn tất.")

    # 1. Open Embedding Settings tab, verify presets + Test Connection + Save & Apply
    page.locator("button.tab[data-tab='settings']").click()
    expect(page.locator("#tab-settings")).to_be_visible()
    expect(page.locator("#emb-presets")).to_contain_text("bge-m3")
    page.locator("#btn-emb-test").click()
    expect(page.locator("#emb-test-result")).to_contain_text("Kết nối thành công")
    page.locator("#btn-emb-save").click()
    expect(page.locator("#emb-test-result")).to_contain_text("Đã lưu & áp dụng")
    expect(page.locator("#emb-active-source")).to_have_text("runtime saved")

    # 2. Switch back to Documents tab, upload, run Process (Background) and watch 5-stage progress complete
    page.locator("button.tab[data-tab='documents']").click()
    page.locator("#file").set_input_files(
        {
            "name": filename,
            "mimeType": "text/plain",
            "buffer": f"Background processing and deletion test for {marker}.".encode(),
        }
    )
    page.locator("#upload button[type='submit']").click()
    doc = page.locator("#documents article").filter(has_text=filename)
    expect(doc).to_be_visible()
    page.locator("#parser").select_option("text")
    doc.get_by_role("button", name="Process (Background)", exact=True).click()
    expect(page.locator("#pipeline-progress")).to_contain_text("completed (100%)")
    expect(page.locator("#status")).to_have_text("Hoàn tất.")

    # 3. Delete the document from UI and confirm removal
    doc.get_by_role("button", name="Delete", exact=True).click()
    expect(page.locator("#status")).to_have_text("Hoàn tất.")
    expect(page.locator("#documents article").filter(has_text=filename)).to_have_count(0)


