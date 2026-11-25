# Memory Lab Enterprise SaaS — Kiến trúc & Vận hành Production Quy mô lớn

Bản **Enterprise Production SaaS** ([docker-compose.enterprise.yml](../docker-compose.enterprise.yml)) mở rộng Memory Lab từ công cụ đơn người dùng thành nền tảng **Multi-Tenant SaaS** chịu tải cao, tách bạch tầng xử lý bất đồng bộ, có giám sát tập trung và cô lập dữ liệu tuyệt đối theo từng tổ chức (`tenant_id`).

## 1. Kiến trúc hệ thống (10 thành phần cốt lõi)

```
Internet / HTTPS (HTTP/2 & HTTP/3)
        │
   ┌────▼─────┐
   │  Caddy 2 │  Edge Load Balancer (Round-Robin) + Auto TLS
   └─┬──────┬─┘
     │      │
┌────▼───┐ ┌▼───────────────┐        ┌──────────────────────┐
│ Web x2 │ │ Enterprise API │───────▶│ Redis 7.4 (AOF)      │
│ Nginx  │ │ FastAPI x2+    │        │ Rate Limit + Job Q   │
└────────┘ └──┬───┬───┬───┬─┘        └──────────┬───────────┘
              │   │   │   │                     │ BRPOP
     ┌────────┘   │   │   └────────┐     ┌──────▼──────────────┐
     ▼            ▼   ▼            ▼     │ Async Worker x2+    │
┌──────────┐ ┌──────┐ ┌──────┐ ┌───────┐ │ Docling + Embedding │
│Postgres16│ │Qdrant│ │Neo4j │ │ MinIO │ └─────────────────────┘
│Tenants/  │ │Vector│ │Graph │ │  S3   │
│Keys/Audit│ │Index │ │Store │ │Objects│
└──────────┘ └──────┘ └──────┘ └───────┘
        ▲
        │ scrapes /metrics
┌───────┴──────┐      ┌─────────┐
│ Prometheus   │─────▶│ Grafana │  Pre-provisioned SaaS Dashboard
└──────────────┘      └─────────┘
```

| Tầng | Công nghệ | Vai trò ở quy mô lớn |
|---|---|---|
| **Edge Proxy & LB** | **Caddy 2** (`deploy/enterprise/Caddyfile`) | Cân bằng tải round-robin qua nhiều replica `api` & `web`, tự cấp chứng chỉ Let's Encrypt HTTPS, nén gzip/zstd |
| **Web Frontend** | **Nginx + Astro** (`WEB_REPLICAS=2`) | Giao diện SaaS nhẹ (14 KB gzipped), hỗ trợ nhập `Tenant ID` và `API Key / Bearer JWT` |
| **Stateless API Tier** | **FastAPI + Uvicorn multi-worker** (`API_REPLICAS=2 × UVICORN_WORKERS=2`) | Xác thực JWT/API Key, RBAC (`admin`/`editor`/`viewer`), rate limiting, audit log, MCP, Embedding Atlas |
| **Async Worker Tier** | **Dedicated Worker** (`apps/worker/main.py`, `WORKER_REPLICAS=2`) | Tách toàn bộ tác vụ nặng (parse PDF Docling, chunk, enrich, embed) khỏi luồng HTTP, scale ngang độc lập |
| **Metadata & Audit DB** | **PostgreSQL 16** | Lưu bảng `tenants`, `api_keys` (băm SHA-256), `jobs`, `audit_logs` chuẩn ACID |
| **Cache & Broker** | **Redis 7.4 (AOF)** | Rate limiting phân tán theo tenant (`X-RateLimit-*`) và hàng đợi job `BRPOP` độ trễ thấp |
| **Vector Search** | **Qdrant v1.14.1** | Cô lập collection theo từng tenant (`documents_t_<tenant>_<ns>`) với HNSW + payload filtering |
| **Knowledge Graph** | **Neo4j 5.26** | Lưu và duyệt đồ thị thực thể / quan hệ multi-hop |
| **Object Storage** | **MinIO (S3 API)** | Nhân bản bền vững file gốc PDF/DOCX dung lượng lớn theo khóa `tenants/<tenant_id>/...` (AWS SigV4) |
| **Observability** | **Prometheus + Grafana** | Thu thập `/metrics` (req/s, latency, active tenants, job queue, rate limits) + dashboard tự nạp sẵn |

---

## 2. Khởi chạy Enterprise Stack

```bash
# 1. Tạo cấu hình môi trường và đổi toàn bộ mật khẩu / JWT_SECRET / ADMIN_BOOTSTRAP_KEY
cp .env.enterprise.example .env.enterprise
# Gợi ý tạo secret ngẫu nhiên:
# openssl rand -hex 32

# 2. Khởi chạy toàn bộ 11 services (Caddy, Web x2, API x2, Worker x2, Postgres, Redis, Qdrant, Neo4j, MinIO, Prometheus, Grafana)
docker compose -f docker-compose.enterprise.yml up -d --build --wait

# 3. Kiểm tra trạng thái & readiness
docker compose -f docker-compose.enterprise.yml ps
curl -s http://localhost:8080/api/ready
```

- **SaaS Web UI**: `http://localhost:8080` (hoặc `https://<SITE_ADDRESS>` khi đặt domain thực trong `.env.enterprise`)
- **Grafana Dashboard**: `http://127.0.0.1:3000` (tài khoản cấu hình trong `.env.enterprise`, mở sẵn dashboard **Memory Lab Enterprise SaaS Overview**)
- **Prometheus Metrics**: `/api/metrics`

---

## 3. Quy trình Onboarding Tenant & Phân quyền RBAC

1. **Tạo tổ chức (Tenant) mới** với hạn mức riêng (`max_documents`, `max_memories`, `rate_limit_rpm`):

```bash
curl -X POST http://localhost:8080/api/admin/tenants \
  -H "X-Admin-Secret: $ADMIN_BOOTSTRAP_KEY" \
  -H "Content-Type: application/json" \
  -d '{"id":"acme-corp","name":"Acme Corporation","max_documents":50000,"max_memories":500000,"rate_limit_rpm":1200}'
```

2. **Cấp API Key theo vai trò (`admin`, `editor`, `viewer`)**:

```bash
curl -X POST http://localhost:8080/api/admin/api-keys \
  -H "X-Admin-Secret: $ADMIN_BOOTSTRAP_KEY" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"acme-corp","name":"prod-agent-writer","role":"editor"}'
```

3. **Đổi API Key lấy JWT ngắn hạn (tùy chọn cho Web/Mobile)**:

```bash
curl -X POST http://localhost:8080/api/auth/token \
  -H "X-API-Key: mlab_..."
```

4. **Đẩy tác vụ xử lý PDF lớn vào Async Worker (`202 Accepted`)**:

```bash
curl -X POST "http://localhost:8080/api/documents/<doc_id>/versions/<version>/process?async_mode=true" \
  -H "X-API-Key: mlab_..." \
  -H "Content-Type: application/json" \
  -d '{"parser":"docling","chunker":"hybrid","enrich":true}'

# Theo dõi tiến độ job:
curl -s http://localhost:8080/api/jobs/<job_id> -H "X-API-Key: mlab_..."
```

5. **Kiểm tra nhật ký kiểm toán (`Audit Logs`)**:

```bash
curl -s http://localhost:8080/api/admin/audit \
  -H "X-Admin-Secret: $ADMIN_BOOTSTRAP_KEY" \
  -H "X-Tenant-ID: acme-corp"
```

---

## 4. Mở rộng quy mô ngang (Horizontal Scaling)

Khi lượng người dùng hoặc khối lượng PDF tăng mạnh:

```bash
# Tăng lên 4 API replicas và 6 Docling Worker replicas không cần dừng dịch vụ:
docker compose -f docker-compose.enterprise.yml up -d --scale api=4 --scale worker=6 --no-recreate
```

> **Lưu ý khi scale đa máy chủ (Multi-Node / Kubernetes)**: thay named volume `shared_artifacts` bằng storage chia sẻ RWX (NFS / EFS / CephFS) hoặc chuyển hoàn toàn sang đọc/ghi trực tiếp qua MinIO/S3 (`S3ObjectMirror`), chạy Qdrant ở chế độ distributed cluster, và cấu hình PostgreSQL Primary + Read Replica.
