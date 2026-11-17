"""Local browser-test fallback when Docker is unavailable; Qdrant runs in-process."""
from qdrant_client import QdrantClient
from apps.api.main import create_app as api
from memory_lab.core.config import Settings
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.retrieval.embedding import OpenAIEmbeddings
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex


def create_app():
    settings = Settings()
    store = FileStore(settings.data_dir)
    embeddings = OpenAIEmbeddings(settings.embedding_base_url, "", settings.embedding_model)
    index = QdrantIndex(QdrantClient(location=":memory:"), settings.qdrant_collection)
    pipeline = DocumentPipeline(store, embeddings, index, settings.embedding_model)
    return api(settings, pipeline)
