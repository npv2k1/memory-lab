# Docker, CI/CD và E2E

## Production stack (chạy ngay từ GHCR)

Với repository public, GitHub Actions tự động build và publish image tại `ghcr.io/npv2k1/memory-lab/{api,web}` với các tags `main`, `latest`, `sha-<commit>` (và `v*` khi gắn git tag).

```bash
# Chạy ngay không cần cấu hình thêm (chế độ sparse/graph/zero_token chạy lập tức):
docker compose -f docker-compose.prod.yml up -d --wait

# Hoặc cấu hình Embedding API / Neo4j / Domain HTTPS:
cp .env.prod.example .env.prod
docker compose -f docker-compose.prod.yml up -d --wait

# Bật thêm Neo4j Graph Store:
docker compose -f docker-compose.prod.yml --profile graph up -d --wait

# Bật thêm HTTPS tự động qua Caddy (cần điền DOMAIN=... trong .env.prod):
docker compose -f docker-compose.prod.yml --profile tls up -d --wait
```

`docker-compose.prod.yml` đã cấu hình sẵn mạng nội bộ `internal` (không mở port Qdrant/API/Neo4j ra ngoài), healthchecks cho mọi dịch vụ, log rotation (`10m × 3`), `no-new-privileges: true`, Nginx nén gzip + header bảo mật và volume bền vững (`artifacts`, `models`, `qdrant`).

## Local build stack

```bash
cp .env.example .env
# Điền EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_API_KEY.
docker compose up -d --build --wait
```

UI: http://localhost:4321. Frontend dùng `/api` cùng origin; Nginx proxy vào API. Astro dev cũng proxy `/api` tới localhost:8000; tùy chỉnh bằng `API_PROXY_TARGET`.
Backend có healthcheck API + Qdrant, web chờ API healthy. Raw data/model/index dùng named volumes; `docker compose down` giữ dữ liệu. `down -v` xóa volumes, chỉ dùng với stack test nếu muốn reset.

## CI và image delivery

`.github/workflows/ci.yml` chạy khi push `main` / tag `v*`, mở/cập nhật PR, hoặc chạy thủ công:

1. Toàn bộ Python unit/integration tests, kiểm tra cú pháp Colab notebook, Astro production build, và kiểm tra cấu hình cả 4 file Compose (`docker-compose.yml`, `docker-compose.e2e.yml`, `docker-compose.prod.yml`, `docker-compose.deploy.yml`).
2. Stack E2E độc lập: web Nginx → API HTTP → embedding fixture HTTP → Qdrant server.
3. Playwright Chromium kiểm tra toàn bộ luồng Documents, Memory Playground, Experimentation/Comparison, Embedding Atlas, Knowledge Graph và Knowledge Table.
4. Khi cả hai job đạt trên `main` (hoặc tag `v*`): build production images, publish GHCR với tags `sha-<full commit SHA>`, `main`, `latest` (và `v*`).

Images:

- `ghcr.io/npv2k1/memory-lab/api:latest` / `:main` / `:sha-<commit>` (có Docling).
- `ghcr.io/npv2k1/memory-lab/web:latest` / `:main` / `:sha-<commit>` (Nginx + Astro SaaS UI).

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
