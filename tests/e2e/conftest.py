import os
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def browser():
    if os.getenv("E2E") != "1":
        pytest.skip("Set E2E=1 and start the E2E stack")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        executable = os.getenv("E2E_CHROMIUM_PATH")
        browser = playwright.chromium.launch(executable_path=executable or None)
        yield browser
        browser.close()


@pytest.fixture
def page(browser, request):
    context = browser.new_context(accept_downloads=True)
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    page.set_default_timeout(30000)
    yield page
    destination = Path("test-results") / request.node.name
    destination.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(destination / "screenshot.png"), full_page=True)
    context.tracing.stop(path=str(destination / "trace.zip"))
    context.close()
