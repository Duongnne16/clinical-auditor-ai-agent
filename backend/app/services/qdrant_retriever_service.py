from __future__ import annotations

import hashlib
from typing import Any, Iterable

from backend.app.core.config import get_settings
from backend.app.services.embedding_model_cache import get_sentence_transformer


SECTION_PRIORITIES: dict[str, list[str]] = {
    "interaction": ["tuong_tac_thuoc", "tuong_ky", "than_trong"],
    "contraindication": ["chong_chi_dinh", "than_trong"],
    "precaution": ["than_trong", "canh_bao", "luu_y"],
    "pregnancy_lactation": [
        "thai_ky_cho_con_bu",
        "thoi_ky_mang_thai",
        "than_trong",
    ],
    "adverse_effect": [
        "tac_dung_khong_mong_muon",
        "thuong_gap",
        "it_gap",
        "hiem_gap",
        "rat_hiem_gap",
        "rat_thuong_gap",
        "chua_xac_dinh_duoc_tan_suat",
    ],
    "dose": ["lieu_luong_va_cach_dung", "lieu_luong", "nguoi_lon", "tre_em"],
    "renal_hepatic": [
        "lieu_luong_va_cach_dung",
        "lieu_luong",
        "than_trong",
        "duoc_dong_hoc",
        "thai_tru",
        "suy_than",
        "suy_gan",
    ],
    "overdose": ["qua_lieu_va_xu_tri", "qua_lieu", "trieu_chung", "xu_tri"],
    "general": [],
}

QUERY_TYPE_KEYWORDS: dict[str, str] = {
    "interaction": "tương tác thuốc phối hợp dùng chung thận trọng",
    "contraindication": "chống chỉ định không dùng cấm dùng",
    "precaution": "thận trọng cảnh báo lưu ý khi dùng",
    "pregnancy_lactation": "phụ nữ có thai thai kỳ cho con bú",
    "adverse_effect": (
        "tác dụng không mong muốn tác dụng phụ phản ứng bất lợi"
    ),
    "dose": "liều lượng cách dùng người lớn trẻ em",
    "renal_hepatic": "suy thận suy gan hiệu chỉnh liều thận trọng",
    "overdose": "quá liều ngộ độc xử trí giải độc triệu chứng",
    "general": (
        "thông tin thuốc chỉ định chống chỉ định thận trọng tương tác"
    ),
}

DEFAULT_BUNDLE_QUERY_TYPES = [
    "interaction",
    "contraindication",
    "precaution",
    "pregnancy_lactation",
    "renal_hepatic",
    "adverse_effect",
    "overdose",
]


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _normalize_query_type(query_type: str | None) -> str:
    if query_type in SECTION_PRIORITIES:
        return str(query_type)
    return "general"


def detect_query_type(query: str) -> str:
    q = (query or "").lower()
    if any(
        phrase in q
        for phrase in [
            "chống chỉ định",
            "không dùng",
            "cấm dùng",
            "không được dùng",
        ]
    ):
        return "contraindication"
    if any(
        phrase in q
        for phrase in ["tương tác", "dùng chung", "phối hợp", "uống chung"]
    ):
        return "interaction"
    if any(
        phrase in q
        for phrase in [
            "mang thai",
            "bà bầu",
            "thai kỳ",
            "cho con bú",
            "đang cho bú",
        ]
    ):
        return "pregnancy_lactation"
    if any(
        phrase in q
        for phrase in [
            "tác dụng phụ",
            "adr",
            "phản ứng",
            "quá mẫn",
            "dị ứng",
            "sốc phản vệ",
        ]
    ):
        return "adverse_effect"
    if any(phrase in q for phrase in ["suy thận", "suy gan", "child-pugh", "egfr"]):
        return "renal_hepatic"
    if any(
        phrase in q
        for phrase in ["quá liều", "ngộ độc", "xử trí quá liều", "giải độc"]
    ):
        return "overdose"
    if any(
        phrase in q
        for phrase in ["liều", "cách dùng", "uống bao nhiêu", "dùng bao nhiêu"]
    ):
        return "dose"
    return "general"


def get_preferred_sections(query_type: str = "general") -> list[str]:
    return list(SECTION_PRIORITIES[_normalize_query_type(query_type)])


def build_query_text(
    evidence_slugs: list[str],
    query_type: str = "general",
    query_text: str | None = None,
) -> str:
    text = (query_text or "").strip()
    if not text:
        normalized_query_type = _normalize_query_type(query_type)
        slug_text = " ".join(evidence_slugs)
        keyword_text = QUERY_TYPE_KEYWORDS[normalized_query_type]
        text = f"{slug_text} {keyword_text}".strip()
    if not text.startswith("query:"):
        text = f"query: {text}"
    return text


class QdrantRetrieverService:
    def __init__(
        self,
        qdrant_client: Any | None = None,
        embedding_model: Any | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.settings = get_settings()
        self.client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = (
            collection_name or self.settings.qdrant_medical_evidence_collection
        )
        self.embedding_model_name = self.settings.embedding_model
        self._qdrant_backend = (
            type(qdrant_client).__name__ if qdrant_client is not None else "lazy"
        )
        self._embedding_backend = (
            type(embedding_model).__name__
            if embedding_model is not None
            else "lazy"
        )

    def _get_client(self) -> Any:
        if self.client is None:
            from backend.app.vectorstore.qdrant_client import create_qdrant_client

            self.client = create_qdrant_client(self.settings)
            self._qdrant_backend = type(self.client).__name__
        return self.client

    def _get_embedding_model(self) -> Any:
        if self.embedding_model is None:
            self.embedding_model = get_sentence_transformer(self.embedding_model_name)
            self._embedding_backend = type(self.embedding_model).__name__
        return self.embedding_model

    @staticmethod
    def _vector_list(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            return list(vector.tolist())
        return list(vector)

    def _embed(self, query_text: str) -> list[float]:
        model = self._get_embedding_model()
        vector = model.encode(query_text, normalize_embeddings=True)
        return self._vector_list(vector)

    @staticmethod
    def _match_any(values: list[str]) -> Any:
        try:
            from qdrant_client.http import models

            return models.MatchAny(any=values)
        except ImportError:
            return {"any": values}

    @classmethod
    def _field_condition(cls, key: str, values: list[str]) -> Any:
        try:
            from qdrant_client.http import models

            return models.FieldCondition(key=key, match=cls._match_any(values))
        except ImportError:
            return {"key": key, "match": cls._match_any(values)}

    @classmethod
    def _build_filter(
        cls, evidence_slugs: list[str], sections: list[str] | None = None
    ) -> Any:
        conditions = [cls._field_condition("slug", evidence_slugs)]
        if sections:
            conditions.append(cls._field_condition("section", sections))
        try:
            from qdrant_client.http import models

            return models.Filter(must=conditions)
        except ImportError:
            return {"must": conditions}

    def _query_points(
        self,
        query_vector: list[float],
        evidence_slugs: list[str],
        limit: int,
        sections: list[str] | None = None,
    ) -> list[Any]:
        response = self._get_client().query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._build_filter(evidence_slugs, sections),
            limit=limit,
            with_payload=True,
        )
        return list(getattr(response, "points", response or []))

    def _scroll_points(
        self,
        evidence_slugs: list[str],
        limit: int,
        sections: list[str] | None = None,
    ) -> list[Any]:
        scroll_kwargs = {
            "collection_name": self.collection_name,
            "scroll_filter": self._build_filter(evidence_slugs, sections),
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
        }
        attempts = [
            scroll_kwargs,
            {
                **{
                    key: value
                    for key, value in scroll_kwargs.items()
                    if key not in {"scroll_filter", "with_vectors"}
                },
                "query_filter": scroll_kwargs["scroll_filter"],
            },
            {
                key: value
                for key, value in scroll_kwargs.items()
                if key not in {"scroll_filter", "with_vectors"}
            },
        ]
        for attempt in attempts:
            try:
                response = self._get_client().scroll(**attempt)
                points = response[0] if isinstance(response, tuple) else response
                return list(points or [])
            except Exception:
                continue
        return []

    @staticmethod
    def _nested_payload(payload: dict[str, Any]) -> dict[str, Any]:
        metadata = payload.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    @classmethod
    def _payload_value(cls, payload: dict[str, Any], key: str) -> Any:
        if payload.get(key) is not None:
            return payload.get(key)
        return cls._nested_payload(payload).get(key)

    @classmethod
    def _payload_text(cls, payload: dict[str, Any]) -> str:
        text = (
            payload.get("content")
            or payload.get("text")
            or cls._nested_payload(payload).get("content")
            or cls._nested_payload(payload).get("text")
            or ""
        )
        return str(text)

    @classmethod
    def _chunk_key(cls, chunk: dict[str, Any]) -> str:
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        text_hash = hashlib.sha1(
            str(chunk.get("text") or "")[:200].encode("utf-8")
        ).hexdigest()
        return "|".join(
            [
                str(chunk.get("slug") or ""),
                str(chunk.get("section") or ""),
                text_hash,
            ]
        )

    @staticmethod
    def _point_score(point: Any) -> float:
        try:
            return float(getattr(point, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _public_chunk(
        self,
        point: Any,
        evidence_slugs: list[str],
        preferred_sections: list[str],
    ) -> dict[str, Any]:
        payload = getattr(point, "payload", None) or {}
        vector_score = self._point_score(point)
        slug = self._payload_value(payload, "slug")
        section = self._payload_value(payload, "section")
        source = (
            self._payload_value(payload, "source")
            or self._payload_value(payload, "source_name")
        )
        rerank_score = self._rerank_score(
            vector_score=vector_score,
            slug=str(slug or ""),
            section=str(section or ""),
            source=str(source or ""),
            evidence_slugs=evidence_slugs,
            preferred_sections=preferred_sections,
        )
        return {
            "chunk_id": self._payload_value(payload, "chunk_id"),
            "slug": slug,
            "entity_name": self._payload_value(payload, "entity_name"),
            "section": section,
            "section_title": self._payload_value(payload, "section_title"),
            "source": source,
            "source_type": self._payload_value(payload, "source_type"),
            "url": self._payload_value(payload, "url"),
            "title": self._payload_value(payload, "title"),
            "text": self._payload_text(payload),
            "vector_score": round(vector_score, 6),
            "rerank_score": round(rerank_score, 6),
            "rank": 0,
        }

    @staticmethod
    def _rerank_score(
        vector_score: float,
        slug: str,
        section: str,
        source: str,
        evidence_slugs: list[str],
        preferred_sections: list[str],
    ) -> float:
        score = vector_score
        if section in preferred_sections:
            score += 0.15
        if preferred_sections and section == preferred_sections[0]:
            score += 0.10
        if source == "trungtamthuoc":
            score += 0.05
        if slug in evidence_slugs:
            score += 0.05
        return score

    @classmethod
    def _deduplicate_chunks(
        cls, chunks: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        output: list[dict[str, Any]] = []
        for chunk in chunks:
            key = cls._chunk_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            output.append(chunk)
        return output

    @staticmethod
    def _rank_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks.sort(
            key=lambda chunk: (
                chunk.get("rerank_score") or 0,
                chunk.get("vector_score") or 0,
                chunk.get("chunk_id") or "",
            ),
            reverse=True,
        )
        for index, chunk in enumerate(chunks, start=1):
            chunk["rank"] = index
        return chunks

    def retrieve(
        self,
        evidence_slugs: list[str],
        query_text: str | None = None,
        query_type: str = "general",
        top_k: int = 10,
        sections: list[str] | None = None,
    ) -> dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        normalized_query_type = _normalize_query_type(query_type)
        slugs = _deduplicate([slug for slug in evidence_slugs if slug])
        preferred_sections = (
            _deduplicate(sections)
            if sections is not None
            else get_preferred_sections(normalized_query_type)
        )

        if not slugs:
            return {
                "query_type": normalized_query_type,
                "query_text": None,
                "evidence_slugs": [],
                "preferred_sections": preferred_sections,
                "total_results": 0,
                "chunks": [],
                "warnings": ["no_evidence_slugs_provided"],
            }

        final_query_text = build_query_text(
            slugs, normalized_query_type, query_text=query_text
        )
        fetch_limit = max(top_k * 3, 20)
        warnings: list[str] = []

        use_payload_filter_fallback = (
            self.settings.disable_local_embeddings and self.embedding_model is None
        )
        if use_payload_filter_fallback:
            warnings.append("local_embeddings_disabled_payload_filter_fallback")
            points = self._scroll_points(
                evidence_slugs=slugs,
                sections=preferred_sections or None,
                limit=fetch_limit,
            )
        else:
            query_vector = self._embed(final_query_text)
            points = self._query_points(
                query_vector=query_vector,
                evidence_slugs=slugs,
                sections=preferred_sections or None,
                limit=fetch_limit,
            )
        if preferred_sections and not points:
            warnings.append("section_filter_no_results_fallback_to_slug_only")
            if use_payload_filter_fallback:
                points = self._scroll_points(
                    evidence_slugs=slugs,
                    sections=None,
                    limit=fetch_limit,
                )
            else:
                points = self._query_points(
                    query_vector=query_vector,
                    evidence_slugs=slugs,
                    sections=None,
                    limit=fetch_limit,
                )

        chunks = [
            self._public_chunk(point, slugs, preferred_sections) for point in points
        ]
        chunks = self._deduplicate_chunks(chunks)
        chunks = self._rank_chunks(chunks)[:top_k]
        for index, chunk in enumerate(chunks, start=1):
            chunk["rank"] = index

        return {
            "query_type": normalized_query_type,
            "query_text": final_query_text,
            "evidence_slugs": slugs,
            "preferred_sections": preferred_sections,
            "total_results": len(chunks),
            "chunks": chunks,
            "warnings": warnings,
        }

    @staticmethod
    def _slugs_from_normalized_result(normalized_result: dict[str, Any]) -> list[str]:
        explicit_slugs = normalized_result.get("unique_evidence_slugs")
        if isinstance(explicit_slugs, list):
            return _deduplicate(
                [slug for slug in explicit_slugs if isinstance(slug, str) and slug]
            )

        slugs: list[str] = []
        medications = normalized_result.get("medications")
        if not isinstance(medications, list):
            return []
        for medication in medications:
            if not isinstance(medication, dict):
                continue
            if medication.get("mapping_status") == "unmatched":
                continue
            ingredients = medication.get("active_ingredients")
            if not isinstance(ingredients, list):
                continue
            for ingredient in ingredients:
                if not isinstance(ingredient, dict):
                    continue
                slug = ingredient.get("evidence_slug")
                if slug:
                    slugs.append(str(slug))
        return _deduplicate(slugs)

    def retrieve_for_normalized_result(
        self,
        normalized_result: dict[str, Any],
        query_type: str = "general",
        query_text: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        return self.retrieve(
            evidence_slugs=self._slugs_from_normalized_result(normalized_result),
            query_text=query_text,
            query_type=query_type,
            top_k=top_k,
        )

    def build_prescription_evidence_bundle(
        self,
        normalized_result: dict[str, Any],
        query_types: list[str] | None = None,
        top_k_per_type: int = 8,
    ) -> dict[str, Any]:
        if top_k_per_type <= 0:
            raise ValueError("top_k_per_type must be greater than 0")

        slugs = self._slugs_from_normalized_result(normalized_result)
        selected_query_types = query_types or DEFAULT_BUNDLE_QUERY_TYPES
        query_results: dict[str, Any] = {}
        warnings: list[str] = []
        all_chunks: list[dict[str, Any]] = []

        for query_type in selected_query_types:
            result = self.retrieve(
                evidence_slugs=slugs,
                query_type=query_type,
                top_k=top_k_per_type,
            )
            query_results[_normalize_query_type(query_type)] = result
            warnings.extend(result.get("warnings", []))
            all_chunks.extend(result.get("chunks", []))

        return {
            "evidence_slugs": slugs,
            "query_results": query_results,
            "all_chunks": all_chunks,
            "unique_chunks": self._deduplicate_chunks(all_chunks),
            "warnings": _deduplicate(warnings),
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "service": "QdrantRetrieverService",
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "qdrant_backend": self._qdrant_backend,
            "embedding_backend": self._embedding_backend,
            "query_types": sorted(SECTION_PRIORITIES),
        }
