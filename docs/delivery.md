# Docker, CI/CD và E2E

## Local stack

```bash
cp .env.example .env
# Điền EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_API_KEY.
docker compose up -d --build --wait
```

UI: http://localhost:4321. Frontend dùng `/api` cùng origin; Nginx proxy vào API. Astro dev cũng proxy `/api` tới localhost:8000; tùy chỉnh bằng `API_PROXY_TARGET`.
Backend có healthcheck API + Qdrant, web chờ API healthy. Raw data/model/index dùng named volumes; `docker compose down` giữ dữ liệu. `down -v` xóa volumes, chỉ dùng với stack test nếu muốn reset.

## CI và image delivery

`.github/workflows/ci.yml` chạy khi push main, mở/cập nhật PR, hoặc chạy thủ công:

1. Python integration tests, Astro production build, kiểm tra cấu hình Compose.
2. Stack E2E độc lập: web Nginx → API HTTP → embedding fixture HTTP → Qdrant server.
3. Playwright Chromium kiểm tra upload → process TXT → inspector → search → tải đúng source → retry và báo lỗi cấu hình.
4. Khi cả hai job đạt trên main: build production images, publish GHCR với tags `sha-<full commit SHA>` và `main`.

Images:

- `ghcr.io/npv2k1/memory-lab/api:sha-<commit>` (có Docling).
- `ghcr.io/npv2k1/memory-lab/web:sha-<commit>`.

Publish dùng `GITHUB_TOKEN`, permission `packages: write`; không cần PAT cho workflow. Repository/organization có thể cần bật quyền Actions/Packages theo policy. Kiến trúc image mặc định linux/amd64 của GitHub-hosted runner.

E2E artifacts gồm screenshot, Playwright trace và container logs, giữ 7 ngày. Không dùng LLM/embedding API bên ngoài. Test xác nhận orchestration/provenance, không đo chất lượng semantic và chưa bao phủ parse PDF/Hybrid Docling. E2E chạy image backend `base` nhẹ; image `production` chứa Docling được build trong job publish.

## Chạy E2E

```bash
uv sync --locked --extra dev --extra e2e
uv run --no-sync playwright install --with-deps chromium
docker compose -f docker-compose.e2e.yml up -d --build --wait --wait-timeout 180
E2E=1 uv run --no-sync pytest tests/e2e -q
# Xem log trước khi dọn:
docker compose -f docker-compose.e2e.yml logs
# Chỉ xóa dữ liệu của stack test:
docker compose -f docker-compose.e2e.yml down -v --remove-orphans
```

Port 4321 phải trống. `docker-compose.e2e.yml` là file độc lập, không ghép với stack development; dùng project/volumes riêng. `pytest` mặc định skip browser tests nếu chưa đặt `E2E=1`.

Nếu máy không có Docker, có thể kiểm tra browser với Qdrant local và embedding HTTP fixture (không thay thế kiểm tra container/network):

```bash
# Terminal 1
uv run --no-sync python tests/e2e/support/embeddings.py
# Terminal 2, chọn một DATA_DIR mới cho mỗi lần test
DATA_DIR=/tmp/memory-lab-e2e EMBEDDING_BASE_URL=http://localhost:8001/v1 EMBEDDING_MODEL=e2e-fixture \
  uv run --no-sync uvicorn local_api:create_app --app-dir tests/e2e/support --factory --port 8000
# Terminal 3
cd apps/web && bun run dev
# Terminal 4, từ root repository
E2E=1 uv run --no-sync pytest tests/e2e -q
```

## Deploy (cần cấu hình máy đích)

`.github/workflows/deploy.yml` là manual deployment; chưa có server/runner được provision trong source.

Trên máy Linux có Docker Compose v2 hỗ trợ `--wait`:

1. Đăng ký GitHub Actions self-hosted runner cho repo, thêm label `memory-lab-deploy` (runner phải có labels `self-hosted`, `linux`). Không dùng runner này cho code PR không tin cậy.
2. Tạo `.env` ngoài checkout, ví dụ `/opt/memory-lab/.env`; điền embedding provider/model/key. Runner phải đọc được file.
3. Trong GitHub Environment `production`, tạo variable `DEPLOY_ENV_FILE=/opt/memory-lab/.env`. Có thể cấu hình required reviewers tại environment nếu muốn.
4. Chờ workflow CI and images hoàn tất; chạy workflow Deploy trên main, nhập `sha-<full commit SHA>` đã publish.

Deploy đăng nhập GHCR bằng job token, pull cả hai image trước khi thay containers, giữ volumes, chờ healthchecks. Web bind `127.0.0.1:4321`; dùng reverse proxy/tunnel trên máy đích nếu cần truy cập từ xa. API/Qdrant không publish port trong deployment stack. Chưa thiết lập domain/TLS vì chưa có cấu hình đích.

Rollback: chạy lại Deploy với tag SHA cũ. Không có automatic rollback; deployment lỗi trả job failed, giữ volumes. Không chạy `down -v` trên deployment.

Tự chạy trên máy đích thay cho runner (cần đăng nhập GHCR nếu package private):

```bash
export REGISTRY_PREFIX=ghcr.io/npv2k1/memory-lab
export IMAGE_TAG=sha-<full-commit-sha>
export DEPLOY_ENV_FILE=/opt/memory-lab/.env
docker compose -f docker-compose.deploy.yml pull
docker compose -f docker-compose.deploy.yml up -d --wait --wait-timeout 180
```

## Kiểm chứng trong phiên setup

- Python integration: 2 tests passed.
- Astro production build: passed.
- YAML workflows/Compose: parse thành công; chưa thay thế `docker compose config`.
- Môi trường setup không có Docker; Chromium bị chặn Unix socket bởi sandbox. Browser/container E2E và publish/deploy cần được xác nhận bởi Actions/runner thực tế.
