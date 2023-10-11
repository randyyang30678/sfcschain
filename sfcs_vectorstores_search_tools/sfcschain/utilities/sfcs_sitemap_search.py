import json
from typing import List, Tuple
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
from langchain.vectorstores.milvus import Milvus
from langchain.schema.embeddings import Embeddings
import requests
from langchain.utils import get_from_env

DEFAULT_MILVUS_CONNECTION = {
    "host": "localhost",
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
}


class SfcsSiteMapSearchWrapper(BaseModel):
    """Wrapper for the SfcsDutySearch engine."""

    search_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the search request."""

    embedding_url: str = Field(default="http://localhost:8000/api/v1/search")
    """The base URL for the Brave search engine."""

    def run(self, query: str) -> str:
        self.embedding_url = get_from_env("embedding_url", "EMBEDDING_URL")
        vectorstores_search_results = self._search_sfcsduty_milvus(query=query)
        final_results = [
            {
                "id": item[0].metadata["id"],
                "text": item[0].metadata["text"],
                "score": item[1],
            }
            for item in vectorstores_search_results
        ]
        return json.dumps(final_results)

    def _search_sfcsduty_milvus(self, query: str) -> List[Tuple[Document, float]]:
        try:
            milvus = Milvus(
                embedding_function=BgeEmbedding(),
                collection_name="WebSiteBGE",
                connection_args=DEFAULT_MILVUS_CONNECTION,
                consistency_level="Strong",
                index_params=None,
                search_params={"HNSW": {"metrics_type": "L2", "params": {"ef": 256}}},
                drop_old=False,
                primary_field="id",
                text_field="text",
                vector_field="embedding",
            )
            # List[Tuple[Document, float]]
            # doc = Document(page_content=meta.pop(self._text_field), metadata=meta)

            search_result = milvus.similarity_search_with_score_by_vector(
                embedding=BgeEmbedding().embed_query(text=query),
                k=5,
                param=milvus.search_params,
                expr=None,
                timeout=None,
            )
            return search_result
        except Exception as e:
            raise e


class BgeEmbedding(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        req = requests.PreparedRequest()
        params = {**self.search_kwargs, **{"q": text}}
        req.prepare_url(self.base_url, params)

        if req.url is None:
            raise ValueError("prepared url is None, this should not happen")

        response = requests.post(req.url, headers=headers)
        if not response.ok:
            raise Exception(f"HTTP error {response.status_code}")

        return response.json().get("embedding", [])
