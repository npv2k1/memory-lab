import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from apps.api.main import create_app
from memory_lab.core.config import Settings
from memory_lab.core.models import PipelineConfig
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.documents.parsers.adapters import TextParser
from memory_lab.documents.parsers.service import ParsingService
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex


class Embeddings:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_upload_process_retry_search_source(tmp_path):
    store = FileStore(tmp_path)
    embeddings = Embeddings()
    client = QdrantClient(':memory:')
    index = QdrantIndex(client, 'test')
    class CountingParser(TextParser):
        calls = 0

        def parse(self, path):
            self.calls += 1
            return super().parse(path)

    parser = CountingParser()
    parsing = ParsingService(store, {"text": parser})
    pipeline = DocumentPipeline(store, embeddings, index, 'test-model', parsing=parsing)
    with TestClient(create_app(Settings(data_dir=tmp_path), pipeline)) as api:
        response = api.post('/documents', files={'file': ('notes.txt', b'Memory Lab preserves original content.\n\nEvery chunk cites source blocks.')})
        assert response.status_code == 201
        doc = response.json()
        assert api.post('/documents', files={'file': ('notes.txt', b'Memory Lab preserves original content.\n\nEvery chunk cites source blocks.')}).json() == doc
        url = f"/documents/{doc['id']}/versions/{doc['version']}"
        run = api.post(url + '/process', json={'parser': 'text'}).json()
        assert run['status'] == 'completed'
        assert api.post(url + '/process', json={'parser': 'text'}).json() == run
        assert embeddings.calls == 1
        assert parser.calls == 1
        assert client.count(index.name(pipeline.namespace)).count == 1
        evidence = api.get('/search', params={'q': 'original'}).json()[0]
        assert evidence['chunk']['block_ids'] == ['text:0', 'text:1']
        assert api.get(evidence['source_url']).content.startswith(b'Memory Lab')
        assert api.get(f"/runs/{run['pipeline_run_id']}/normalized").json()['document_id'] == doc['id']
        # Rebuild an emptied index from saved artifacts without embedding again.
        client.delete_collection(index.name(pipeline.namespace))
        assert api.post(url + '/process', json={'parser': 'text'}).status_code == 200
        assert parser.calls == 1  # retry/rebuild reuse the parsing artifact
        assert embeddings.calls == 2  # one process + one search
        assert client.count(index.name(pipeline.namespace)).count == 1
        assert api.post(url + '/process', json={'parser': 'text', 'chunker': 'hybrid'}).status_code == 422
        assert api.get('/runs/missing/chunks').status_code == 404
    client.close()


def test_versions_and_failed_retry(tmp_path):
    store = FileStore(tmp_path)
    first = store.ingest('one.txt', b'first', 'family')
    second = store.ingest('one.txt', b'second', 'family')
    assert first.id == second.id and first.version != second.version
    class BrokenIndex:
        def upsert(self, *args):
            raise RuntimeError('offline')
    embeddings = Embeddings()
    pipeline = DocumentPipeline(store, embeddings, BrokenIndex(), 'test')
    with pytest.raises(RuntimeError):
        pipeline.process(first.id, first.version, PipelineConfig(parser='text'))
    manifests = list(tmp_path.glob('runs/*/manifest.json'))
    assert store.get(str(manifests[0].relative_to(tmp_path)))['status'] == 'failed'
    client = QdrantClient(':memory:')
    pipeline.index = QdrantIndex(client, 'retry')
    assert pipeline.process(first.id, first.version, PipelineConfig(parser='text'))['status'] == 'completed'
    assert embeddings.calls == 1
    with pytest.raises(ValueError):
        store.path('../escape')
    client.close()
