import os
import re
import math
from collections import Counter
from src.interfaces.vector_store import BaseVectorStore
from src.interfaces.embeddings import BaseEmbeddingModel

class InMemoryBM25Index:
    """
    Fast in-memory BM25 Okapi search index for local banking policy text chunks.
    Executes in < 2ms without external dependencies.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_lengths = []
        self.avg_doc_length = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def index_documents(self, documents: list[dict]) -> None:
        self.docs = documents
        self.doc_freqs = []
        self.doc_lengths = []
        df = Counter()

        for doc in documents:
            tokens = self._tokenize(doc['text'])
            self.doc_lengths.append(len(tokens))
            term_freq = Counter(tokens)
            self.doc_freqs.append(term_freq)
            for term in term_freq:
                df[term] += 1

        num_docs = len(documents)
        self.avg_doc_length = (sum(self.doc_lengths) / num_docs) if num_docs > 0 else 1.0

        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, k: int = 4, filter_metadata: dict = None) -> list[dict]:
        if not self.docs:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, (doc, doc_len, tf) in enumerate(zip(self.docs, self.doc_lengths, self.doc_freqs)):
            if filter_metadata:
                match = all(doc['metadata'].get(key) == val for key, val in filter_metadata.items())
                if not match:
                    continue

            score = 0.0
            for token in query_tokens:
                if token not in tf:
                    continue
                freq = tf[token]
                idf = self.idf.get(token, 0.0)
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
                score += idf * (numerator / denominator)

            if score > 0.0:
                scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [{'text': d['text'], 'metadata': d['metadata'], 'score': float(s)} for d, s in scores[:k]]

def reciprocal_rank_fusion(dense_results: list[dict], sparse_results: list[dict], k_rrf: int = 60, top_k: int = 4) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) combining Dense Vector and Sparse BM25 candidate ranks.
    RRF_Score(d) = sum(1 / (k_rrf + rank))
    """
    doc_map = {}
    rrf_scores = {}

    for rank, item in enumerate(dense_results):
        doc_id = item['metadata'].get('chunk_id') or item['text'][:80]
        doc_map[doc_id] = item
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + rank + 1))

    for rank, item in enumerate(sparse_results):
        doc_id = item['metadata'].get('chunk_id') or item['text'][:80]
        if doc_id not in doc_map:
            doc_map[doc_id] = item
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + rank + 1))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids[:top_k]:
        doc_item = dict(doc_map[doc_id])
        doc_item['score'] = float(rrf_scores[doc_id])
        fused_results.append(doc_item)

    return fused_results

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_directory: str, embedding_model: BaseEmbeddingModel, collection_name="banking_policies"):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.bm25_index = InMemoryBM25Index()

        from langchain_chroma import Chroma
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
        self._init_bm25()

    def _init_bm25(self):
        """Pre-index existing Chroma documents into the BM25 sparse index."""
        try:
            data = self.db.get()
            if data and data.get('documents'):
                docs = []
                for text, meta in zip(data['documents'], data['metadatas']):
                    docs.append({'text': text, 'metadata': meta or {}})
                self.bm25_index.index_documents(docs)
        except Exception as e:
            pass

    def add_documents(self, documents: list[dict]) -> None:
        if not documents:
            return

        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [doc['metadata']['chunk_id'] for doc in documents]

        self.db.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        self._init_bm25()

    def similarity_search(self, query: str, k: int = 4, filter_metadata: dict = None) -> list[dict]:
        # 1. Dense Semantic Search
        fetch_k = max(k * 2, 6)
        dense_raw = self.db.similarity_search_with_score(
            query=query,
            k=fetch_k,
            filter=filter_metadata
        )
        dense_results = []
        for doc, score in dense_raw:
            dense_results.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })

        # 2. Sparse BM25 Keyword Search
        sparse_results = self.bm25_index.search(
            query=query,
            k=fetch_k,
            filter_metadata=filter_metadata
        )

        # 3. Reciprocal Rank Fusion (RRF)
        if sparse_results:
            return reciprocal_rank_fusion(dense_results, sparse_results, k_rrf=60, top_k=k)
        return dense_results[:k]

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, persist_directory: str, embedding_model: BaseEmbeddingModel, index_name="faiss_index"):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.index_name = index_name

        from langchain_community.vectorstores import FAISS
        self.faiss_class = FAISS
        self.db = None

        # Load local index if exists
        index_path = os.path.join(persist_directory, f"{index_name}.faiss")
        if os.path.exists(index_path):
            self.db = FAISS.load_local(
                folder_path=persist_directory,
                embeddings=embedding_model,
                index_name=index_name,
                allow_dangerous_deserialization=True
            )

    def add_documents(self, documents: list[dict]) -> None:
        if not documents:
            return

        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [doc['metadata']['chunk_id'] for doc in documents]

        if self.db is None:
            self.db = self.faiss_class.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas,
                ids=ids
            )
        else:
            self.db.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )

        # Save to disk
        self.db.save_local(
            folder_path=self.persist_directory,
            index_name=self.index_name
        )

    def similarity_search(self, query: str, k: int = 4, filter_metadata: dict = None) -> list[dict]:
        if self.db is None:
            return []

        filter_fn = None
        if filter_metadata:
            filter_fn = lambda metadata: all(metadata.get(key) == val for key, val in filter_metadata.items())

        results = self.db.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_fn
        )

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })
        return formatted_results

def get_vector_store(config: dict, embedding_model: BaseEmbeddingModel) -> BaseVectorStore:
    """Factory function to get vector store from config settings."""
    provider = config.get("vector_store", {}).get("provider", "chroma").lower()
    persist_dir = config.get("vector_store", {}).get("persist_directory", "data/chroma_db")

    if provider == "chroma":
        return ChromaVectorStore(persist_dir, embedding_model)
    elif provider == "faiss":
        return FAISSVectorStore(persist_dir, embedding_model)
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")
