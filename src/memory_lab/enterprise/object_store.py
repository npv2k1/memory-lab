from datetime import datetime, timezone
import hashlib
import hmac
import logging
from urllib.parse import quote, urlparse
import httpx

logger = logging.getLogger(__name__)


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


class S3ObjectMirror:
    """Zero-dependency AWS SigV4 S3/MinIO object mirror for durable raw document replication."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint_url and self.bucket and self.access_key and self.secret_key)

    def put_object(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        if not self.enabled:
            return False
        now = datetime.now(timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(data).hexdigest()

        parsed = urlparse(self.endpoint_url)
        host = parsed.netloc
        canonical_uri = f"/{self.bucket}/{quote(key.lstrip('/'), safe='/')}"
        canonical_headers = (
            f"content-type:{content_type}\n"
            f"host:{host}\n"
            f"x-amz-content-sha256:{payload_hash}\n"
            f"x-amz-date:{amz_date}\n"
        )
        signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
        canonical_request = f"PUT\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

        credential_scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        string_to_sign = (
            f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n"
            f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        )

        k_date = _sign(("AWS4" + self.secret_key).encode("utf-8"), date_stamp)
        k_region = _sign(k_date, self.region)
        k_service = _sign(k_region, "s3")
        k_signing = _sign(k_service, "aws4_request")
        signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        auth_header = (
            f"AWS4-HMAC-SHA256 Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        url = f"{self.endpoint_url}{canonical_uri}"
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.put(
                    url,
                    content=data,
                    headers={
                        "Content-Type": content_type,
                        "x-amz-content-sha256": payload_hash,
                        "x-amz-date": amz_date,
                        "Authorization": auth_header,
                    },
                )
                resp.raise_for_status()
            return True
        except Exception:
            logger.warning("s3_mirror_put_failed", extra={"key": key})
            return False
