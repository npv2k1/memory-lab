import os
from uuid import uuid4
import pytest
import httpx


def test_http_document_lifecycle():
    if os.getenv('E2E') != '1':
        pytest.skip('Set E2E=1 and start the E2E stack')
    origin = os.getenv('E2E_BASE_URL', 'http://localhost:4321').rstrip('/')
    content = f'Memory Lab HTTP evidence {uuid4().hex}'.encode()
    with httpx.Client(base_url=origin, timeout=30) as client:
        uploaded = client.post('/api/documents', files={'file': ('http-e2e.txt', content)})
        assert uploaded.status_code == 201, uploaded.text
        doc = uploaded.json()
        route = f"/api/documents/{doc['id']}/versions/{doc['version']}/process"
        response = client.post(route, json={'parser': 'text'})
        assert response.status_code == 200, response.text
        run = response.json()
        assert run['status'] == 'completed'
        retried = client.post(route, json={'parser': 'text'})
        assert retried.json()['pipeline_run_id'] == run['pipeline_run_id']
        result = client.get('/api/search', params={'q': content.decode(), 'limit': 50})
        assert result.status_code == 200, result.text
        evidence = next(hit for hit in result.json() if hit['chunk']['document_id'] == doc['id'])
        assert evidence['chunk']['block_ids'] == ['text:0']
        assert client.get('/api' + evidence['source_url']).content == content
        chunks = client.get(f"/api/runs/{run['pipeline_run_id']}/chunks")
        assert chunks.json()[0]['text'].strip() == content.decode()
        invalid = client.post(route, json={'parser': 'text', 'chunker': 'hybrid'})
        assert invalid.status_code == 422
