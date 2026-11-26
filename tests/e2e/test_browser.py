import json
import os
from uuid import uuid4


def test_document_upload_autoprocess_detail_and_ask_drawer(page):
    from playwright.sync_api import expect

    marker = uuid4().hex[:8]
    content = (
        f"# Quy trình vận hành {marker}\n\n"
        f"Memory Lab sử dụng PostgreSQL và Qdrant để lưu trữ chứng cứ gốc cho dự án {marker}. "
        f"Alice quản lý Memory Lab và kiểm tra dữ liệu hàng tuần.\n\n"
        f"## Hạ tầng triển khai\n\n"
        f"Hệ thống chạy trên Docker với quy trình nạp nguyên liệu tự động cho {marker}."
    )
    filename = f"tai-lieu-{marker}.md"

    page.goto(os.getenv("E2E_BASE_URL", "http://localhost:4321"))
    expect(page.locator("#conn-label")).to_have_text("Đã kết nối", timeout=15000)
    # Default landing screen is Hỏi (no 6 KPI clutter boxes)
    expect(page.locator("#screen-ask")).to_be_visible()

    # Switch to Tài liệu and upload file — auto-processes in background without clicking Process
    page.locator("button.nav-item[data-screen='docs']").click()
    expect(page.locator("#screen-docs")).to_be_visible()
    page.locator("#doc-file-input").set_input_files(
        {"name": filename, "mimeType": "text/markdown", "buffer": content.encode("utf-8")}
    )

    row = page.locator("#docs-list .doc-row").filter(has_text=filename)
    expect(row).to_be_visible(timeout=15000)
    expect(row).to_contain_text("Sẵn sàng", timeout=30000)

    # Open Document Reading Detail view (TOC + readable prose + entities)
    row.locator(".doc-main").click()
    expect(page.locator("#docs-detail-view")).to_be_visible()
    expect(page.locator("#doc-detail-title")).to_have_text(filename)
    expect(page.locator("#doc-detail-toc")).to_contain_text(f"Quy trình vận hành {marker}")
    expect(page.locator("#doc-detail-chunks")).to_contain_text("PostgreSQL")
    expect(page.locator("#doc-detail-entities")).to_contain_text("PostgreSQL")

    # Download original file from detail view
    with page.expect_download() as downloaded:
        page.locator("#doc-detail-download").click()
    assert downloaded.value.failure() is None
    assert open(downloaded.value.path(), "rb").read() == content.encode("utf-8")

    # Go to Hỏi, search unified knowledge, inspect highlighted snippet + Reading Drawer
    page.locator("button.nav-item[data-screen='ask']").click()
    page.locator("#ask-input").fill(f"PostgreSQL {marker}")
    page.locator("#ask-form button[type='submit']").click()

    first_card = page.locator("#ask-results-list .result-card").first
    expect(first_card).to_be_visible(timeout=15000)
    expect(first_card).to_contain_text(filename)
    expect(first_card.locator("mark").first).to_be_visible()
    expect(first_card.locator(".relevance-bars")).to_be_visible()

    # Click result card to open Right-hand Reading Drawer
    first_card.click()
    expect(page.locator("#reading-drawer")).to_be_visible()
    expect(page.locator("#drawer-body")).to_contain_text(f"dự án {marker}")
    page.locator("#btn-close-drawer").click()
    expect(page.locator("#reading-drawer")).to_be_hidden()


def test_memory_timeline_and_explore_topics_ego_table(page):
    from playwright.sync_api import expect

    marker = uuid4().hex[:8]
    page.goto(os.getenv("E2E_BASE_URL", "http://localhost:4321"))
    expect(page.locator("#conn-label")).to_have_text("Đã kết nối", timeout=15000)

    # Switch to Ký ức and store a memory using the clean top composer (options collapsed by default)
    page.locator("button.nav-item[data-screen='memory']").click()
    expect(page.locator("#memory-options-panel")).to_be_hidden()
    page.locator("#memory-input").fill(
        f"Alice quản lý Memory Lab và dùng PostgreSQL cho chiến dịch {marker}."
    )
    page.locator("#memory-form button[type='submit']").click()

    # Visual timeline grouped by Hôm nay (no raw JSON!)
    expect(page.locator("#memory-timeline")).to_contain_text("Hôm nay", timeout=15000)
    expect(page.locator("#memory-timeline")).to_contain_text(marker)

    # Click an entity chip in the memory card to jump directly into Khám phá -> Ego Graph
    card = page.locator("#memory-timeline .memory-card").filter(has_text=marker)
    card.locator("button.entity-chip").filter(has_text="Memory Lab").first.click()

    expect(page.locator("#screen-explore")).to_be_visible()
    expect(page.locator("#explore-view-ego")).to_be_visible()
    expect(page.locator("#ego-stage svg.ego-svg g.ego-node-group")).not_to_have_count(0, timeout=15000)
    expect(page.locator("#ego-inspector")).to_contain_text("Memory Lab")

    # Switch to Bản đồ chủ đề (Treemap)
    page.locator("[data-explore-view='topics']").click()
    expect(page.locator("#explore-view-topics")).to_be_visible()
    expect(page.locator("#topic-treemap-container .tm-tile").first).to_be_visible(timeout=15000)
    expect(page.locator("#topic-detail-panel")).to_contain_text("5 thực thể chính")

    # Switch to Bảng thực thể and check quoted evidence sentence
    page.locator("[data-explore-view='table']").click()
    expect(page.locator("#explore-view-table")).to_be_visible()
    # Uncheck clean filter in case marker-only test DB has 1 mention
    page.locator("#ent-table-clean").uncheck()
    expect(page.locator("#ent-table-body")).to_contain_text("PostgreSQL", timeout=15000)
    expect(page.locator("#ent-table-body")).to_contain_text("“")


def test_settings_embedding_dev_mode_and_document_delete(page):
    from playwright.sync_api import expect

    marker = uuid4().hex[:8]
    filename = f"del-{marker}.txt"
    page.goto(os.getenv("E2E_BASE_URL", "http://localhost:4321"))
    expect(page.locator("#conn-label")).to_have_text("Đã kết nối", timeout=15000)

    # Developer section is hidden by default
    expect(page.locator("#nav-dev-section")).to_be_hidden()

    # Open Cài đặt -> test Embedding connection + Save & Apply
    page.locator("button.nav-item[data-screen='settings']").click()
    expect(page.locator("#screen-settings")).to_be_visible()
    page.locator("#btn-emb-test").click()
    expect(page.locator("#emb-test-result")).to_contain_text("Kết nối thành công", timeout=15000)
    page.locator("#emb-settings-form button[type='submit']").click()
    expect(page.locator("#toast-rack")).to_contain_text("Embedding", timeout=10000)

    # Enable Developer Mode -> reveals Pipeline Inspector, Embedding Atlas, Eval
    page.locator(".toggle-switch").click()
    expect(page.locator("#nav-dev-section")).to_be_visible()

    # Upload a document, wait for Sẵn sàng, then inspect in Pipeline Inspector and delete via ⋯ menu
    page.locator("button.nav-item[data-screen='docs']").click()
    page.locator("#doc-file-input").set_input_files(
        {
            "name": filename,
            "mimeType": "text/plain",
            "buffer": f"Kiểm thử xóa tài liệu và Pipeline Inspector cho {marker}.".encode("utf-8"),
        }
    )
    row = page.locator("#docs-list .doc-row").filter(has_text=filename)
    expect(row).to_contain_text("Sẵn sàng", timeout=30000)

    # Inspect chunks artifact in Pipeline Inspector
    page.locator("button.nav-item[data-screen='dev-pipeline']").click()
    expect(page.locator("#dev-runs-list")).to_contain_text(filename, timeout=15000)
    run_card = page.locator("#dev-runs-list .card").filter(has_text=filename).first
    run_card.get_by_role("button", name="chunks", exact=True).click()
    expect(page.locator("#dev-artifact-viewer")).to_contain_text(marker, timeout=10000)

    # Open Developer Embedding Atlas
    page.locator("button.nav-item[data-screen='dev-atlas']").click()
    expect(page.locator("#dev-atlas-canvas")).to_be_visible()

    # Open Developer Eval and create + run dataset
    page.locator("button.nav-item[data-screen='dev-eval']").click()
    page.locator("#eval-ds-name").fill(f"eval-{marker}")
    page.locator("#eval-ds-cases").fill(
        json.dumps([{"query": f"Pipeline Inspector {marker}", "relevant_substrings": [marker]}])
    )
    page.locator("#eval-dataset-form button[type='submit']").click()
    expect(page.locator("#eval-run-dataset")).to_contain_text(f"eval-{marker}", timeout=10000)
    page.locator("#eval-run-name").fill(f"run-{marker}")
    page.locator("#eval-run-form button[type='submit']").click()
    expect(page.locator("#eval-reports-list")).to_contain_text(f"run-{marker}", timeout=15000)

    # Delete the uploaded document via ⋯ overflow menu
    page.locator("button.nav-item[data-screen='docs']").click()
    row = page.locator("#docs-list .doc-row").filter(has_text=filename)
    expect(row).to_be_visible(timeout=10000)
    page.on("dialog", lambda dialog: dialog.accept())
    row.locator("button[aria-label='Thao tác']").click()
    row.get_by_role("button", name="Xoá tài liệu").click()
    expect(page.locator("#docs-list .doc-row").filter(has_text=filename)).to_have_count(0, timeout=15000)
