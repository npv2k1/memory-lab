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

