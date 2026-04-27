from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .query_rewrite import (
    QueryRewrite,
    QueryRewriter,
    get_default_query_rewriter,
    infer_intent_expert,
)
from .records import QueryRecord


DEFAULT_EMBEDDING_MODEL = os.getenv("ROUTER_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
DEFAULT_LEXICAL_WEIGHT = float(os.getenv("ROUTER_LEXICAL_WEIGHT", "0.15"))
DEFAULT_QUERY_SEARCH_MULTIPLIER = int(os.getenv("ROUTER_SEARCH_MULTIPLIER", "8"))
DEFAULT_INTENT_PRIOR = float(os.getenv("ROUTER_INTENT_PRIOR", "12.0"))


@dataclass(frozen=True)
class NeighborMatch:
    query_id: str
    expert: str
    text: str
    score: float
    metadata: dict[str, str]


@dataclass(frozen=True)
class RoutingDecision:
    query: str
    selected_expert: str
    confidence: float
    expert_scores: dict[str, float]
    neighbors: list[NeighborMatch]
    rewrites: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "selected_expert": self.selected_expert,
            "confidence": self.confidence,
            "expert_scores": self.expert_scores,
            "neighbors": [asdict(neighbor) for neighbor in self.neighbors],
            "rewrites": self.rewrites,
        }


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize: bool = True
    batch_size: int = 64


class DenseQueryEncoder:
    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.model = _load_embedding_model(self.config.model_name, self.config.device)

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        prompts = [_format_passage(text, self.config.model_name) for text in texts]
        return self._encode(prompts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prompts = [_format_query(text, self.config.model_name) for text in texts]
        return self._encode(prompts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


@lru_cache(maxsize=4)
def _load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def _format_query(text: str, model_name: str) -> str:
    if "e5" in model_name.casefold():
        return f"query: {text}"
    return text


def _format_passage(text: str, model_name: str) -> str:
    if "e5" in model_name.casefold():
        return f"passage: {text}"
    return text


class MoERouter:
    def __init__(
        self,
        vectorizer: TfidfVectorizer | None = None,
        encoder: DenseQueryEncoder | None = None,
        query_rewriter: QueryRewriter | None = None,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    ) -> None:
        self.vectorizer = vectorizer or TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.encoder = encoder or DenseQueryEncoder()
        self.query_rewriter = query_rewriter or get_default_query_rewriter()
        self.lexical_weight = lexical_weight
        self.records: list[QueryRecord] = []
        self.matrix = None
        self.cpu_index: faiss.Index | None = None
        self.search_index: faiss.Index | None = None
        self.embedding_dim: int | None = None

    def fit(self, records: list[QueryRecord]) -> "MoERouter":
        if not records:
            raise ValueError("At least one query record is required to fit the router.")
        self.records = records
        texts = [record.text for record in records]
        self.matrix = self.vectorizer.fit_transform(texts)

        corpus_embeddings = self.encoder.encode_corpus(texts)
        if corpus_embeddings.ndim != 2 or corpus_embeddings.shape[0] != len(records):
            raise ValueError("Dense embeddings must be a 2D array aligned with the records.")

        self.embedding_dim = int(corpus_embeddings.shape[1])
        self.cpu_index = faiss.IndexFlatIP(self.embedding_dim)
        self.cpu_index.add(np.ascontiguousarray(corpus_embeddings, dtype=np.float32))
        self.search_index = None
        return self

    def route(self, query: str, top_k: int = 7) -> RoutingDecision:
        if not query.strip():
            raise ValueError("Query must be non-empty.")
        if not self.records:
            raise RuntimeError("Router has not been fitted.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        rewrites = self.query_rewriter.rewrite(query)
        if not rewrites:
            rewrites = [QueryRewrite(text=query, weight=1.0, source="original")]

        dense_scores = self._dense_scores(rewrites, top_k=top_k)
        lexical_scores = self._lexical_scores(rewrites)
        anchor_boosts = self._anchor_boosts(query)
        intent_expert = self._resolve_intent_expert(infer_intent_expert(query))

        combined_scores = dense_scores + (self.lexical_weight * lexical_scores) + anchor_boosts
        if not np.any(combined_scores > 0):
            combined_scores = lexical_scores + anchor_boosts

        if intent_expert is not None:
            selected_expert = intent_expert
            expert_scores = {selected_expert: DEFAULT_INTENT_PRIOR}
            top_indices = self._top_indices_for_expert(
                combined_scores,
                selected_expert=selected_expert,
                top_k=top_k,
            )
            for index in top_indices.tolist():
                score = float(max(combined_scores[index], 0.0))
                if score <= 0:
                    continue
                expert_scores[selected_expert] = expert_scores.get(selected_expert, 0.0) + score
            confidence = 1.0
        else:
            expert_scores: dict[str, float] = {}
            vote_indices = self._top_vote_indices(combined_scores, top_k=top_k)
            for index in vote_indices.tolist():
                score = float(max(combined_scores[index], 0.0))
                if score <= 0:
                    continue
                record = self.records[index]
                expert_scores[record.expert] = expert_scores.get(record.expert, 0.0) + score

            if not expert_scores:
                raise RuntimeError("Failed to compute expert scores for the query.")

            selected_expert = max(expert_scores.items(), key=lambda item: item[1])[0]
            confidence = expert_scores[selected_expert] / max(sum(expert_scores.values()), 1e-8)

            top_indices = self._top_indices_for_expert(
                combined_scores,
                selected_expert=selected_expert,
                top_k=top_k,
            )

        neighbors: list[NeighborMatch] = []
        for index in top_indices.tolist():
            score = float(max(combined_scores[index], 0.0))
            record = self.records[index]
            neighbors.append(
                NeighborMatch(
                    query_id=record.query_id,
                    expert=record.expert,
                    text=record.text,
                    score=score,
                    metadata=record.metadata,
                )
            )

        return RoutingDecision(
            query=query,
            selected_expert=selected_expert,
            confidence=float(confidence),
            expert_scores={key: float(value) for key, value in expert_scores.items()},
            neighbors=neighbors,
            rewrites=[rewrite.text for rewrite in rewrites],
        )

    def _dense_scores(self, rewrites: list[QueryRewrite], top_k: int) -> np.ndarray:
        if self.cpu_index is None or self.embedding_dim is None:
            return np.zeros(len(self.records), dtype=np.float32)

        search_index = self._get_search_index()
        search_k = min(len(self.records), max(top_k * DEFAULT_QUERY_SEARCH_MULTIPLIER, 32))
        query_embeddings = self.encoder.encode_queries([rewrite.text for rewrite in rewrites])
        scores = np.zeros(len(self.records), dtype=np.float32)

        for rewrite, embedding in zip(rewrites, query_embeddings, strict=True):
            distances, indices = search_index.search(embedding[None, :], search_k)
            for distance, record_index in zip(distances[0].tolist(), indices[0].tolist(), strict=True):
                if record_index < 0:
                    continue
                scores[record_index] += rewrite.weight * float(max(distance, 0.0))
        return scores

    def _lexical_scores(self, rewrites: list[QueryRewrite]) -> np.ndarray:
        if self.matrix is None:
            return np.zeros(len(self.records), dtype=np.float32)

        scores = np.zeros(len(self.records), dtype=np.float32)
        for rewrite in rewrites[:2]:
            query_vector = self.vectorizer.transform([rewrite.text])
            similarities = cosine_similarity(query_vector, self.matrix).ravel()
            scores += rewrite.weight * similarities.astype(np.float32)
        return scores

    def _top_indices_for_expert(
        self,
        combined_scores: np.ndarray,
        selected_expert: str,
        top_k: int,
    ) -> np.ndarray:
        candidate_indices = [
            index
            for index, record in enumerate(self.records)
            if record.expert == selected_expert
        ]
        if not candidate_indices:
            k = min(top_k, len(self.records))
            return np.argsort(combined_scores)[::-1][:k]

        candidate_array = np.array(candidate_indices, dtype=np.int64)
        candidate_scores = combined_scores[candidate_array]
        order = np.argsort(candidate_scores)[::-1]
        k = min(top_k, len(candidate_array))
        return candidate_array[order[:k]]

    def _top_vote_indices(
        self,
        combined_scores: np.ndarray,
        top_k: int,
    ) -> np.ndarray:
        vote_k = min(len(self.records), max(top_k * DEFAULT_QUERY_SEARCH_MULTIPLIER, 48))
        return np.argsort(combined_scores)[::-1][:vote_k]

    def _anchor_boosts(self, query: str) -> np.ndarray:
        boosts = np.zeros(len(self.records), dtype=np.float32)
        lowered = query.casefold()

        target_expert = None
        if any(token in lowered for token in ("预测", "predict", "forecast")):
            target_expert = "predict"
        elif any(token in lowered for token in ("统计", "数量", "多少", "count")):
            target_expert = "count"
        elif any(token in lowered for token in ("总和", "合计", "sum", "total", "aggregate", "总计")):
            target_expert = "sum"

        if target_expert is None:
            return boosts
        target_expert = self._resolve_intent_expert(target_expert)
        if target_expert is None:
            return boosts

        for index, record in enumerate(self.records):
            if (
                record.expert == target_expert
                and record.metadata.get("source_dataset") == "router_anchors"
            ):
                boosts[index] = 0.2
        return boosts

    def _resolve_intent_expert(self, intent_expert: str | None) -> str | None:
        if intent_expert is None:
            return None
        experts = {record.expert for record in self.records}
        if intent_expert in experts:
            return intent_expert
        if intent_expert == "predict" and "forecast" in experts:
            return "forecast"
        return None

    def _get_search_index(self) -> faiss.Index:
        if self.search_index is not None:
            return self.search_index
        if self.cpu_index is None:
            raise RuntimeError("Dense FAISS index is not available.")

        if torch.cuda.is_available() and hasattr(faiss, "index_cpu_to_all_gpus"):
            try:
                self.search_index = faiss.index_cpu_to_all_gpus(self.cpu_index)
                return self.search_index
            except Exception:
                self.search_index = self.cpu_index
                return self.search_index

        self.search_index = self.cpu_index
        return self.search_index

    def save(self, output_dir: str | Path) -> Path:
        if self.cpu_index is None or not self.records or self.matrix is None:
            raise RuntimeError("Router has not been fitted.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        records_path = output_path / "training_queries.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")

        faiss.write_index(self.cpu_index, str(output_path / "router.faiss"))
        joblib.dump(
            {
                "backend": "dense_faiss",
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "embedding_config": asdict(self.encoder.config),
                "lexical_weight": self.lexical_weight,
            },
            output_path / "router.joblib",
        )
        (output_path / "metadata.json").write_text(
            json.dumps(
                {
                    "backend": "dense_faiss",
                    "num_records": len(self.records),
                    "experts": sorted({record.expert for record in self.records}),
                    "embedding_model": self.encoder.config.model_name,
                    "embedding_dim": self.embedding_dim,
                    "lexical_weight": self.lexical_weight,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, input_dir: str | Path) -> "MoERouter":
        input_path = Path(input_dir)
        payload = joblib.load(input_path / "router.joblib")
        records_path = input_path / "training_queries.jsonl"
        records: list[QueryRecord] = []
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                records.append(
                    QueryRecord(
                        query_id=row["query_id"],
                        expert=row["expert"],
                        text=row["text"],
                        metadata=row["metadata"],
                    )
                )

        embedding_config = payload.get("embedding_config")
        encoder = DenseQueryEncoder(EmbeddingConfig(**embedding_config)) if embedding_config else DenseQueryEncoder()
        router = cls(
            vectorizer=payload["vectorizer"],
            encoder=encoder,
            lexical_weight=float(payload.get("lexical_weight", DEFAULT_LEXICAL_WEIGHT)),
        )
        router.records = records
        router.matrix = payload.get("matrix")

        faiss_path = input_path / "router.faiss"
        if faiss_path.exists():
            router.cpu_index = faiss.read_index(str(faiss_path))
            router.embedding_dim = router.cpu_index.d
        else:
            router.cpu_index = None
            router.embedding_dim = None
        router.search_index = None
        return router
