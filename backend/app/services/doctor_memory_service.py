from __future__ import annotations

import itertools
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterable

from backend.app.core.config import get_settings


DOCTOR_MEMORY_LABEL = "Ghi chú riêng của bác sĩ"
DOCTOR_MEMORY_VECTOR_SIZE = 768
DOCTOR_MEMORY_DISTANCE = "Cosine"
DEFAULT_NOTE_STATUS = "active"
DEFAULT_NOTE_PRIORITY = "normal"
DEFAULT_NOTE_TYPE = "clinical_experience"
DEFAULT_SOURCE_CONTEXT = "prescription_audit"


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fold_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFD", _normalize_text(value).casefold())
    return "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


def _normalized_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return _deduplicate(_fold_text(value) for value in values if _normalize_text(value))


def _normalize_drug_pair_key(value: Any) -> str:
    parts = [_fold_text(part) for part in str(value or "").split("|")]
    parts = [part for part in parts if part]
    if len(parts) != 2:
        return _fold_text(value)
    return "|".join(parts)


def _normalized_pair_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return _deduplicate(_normalize_drug_pair_key(value) for value in values)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


class DoctorMemoryService:
    """Qdrant-backed private note retrieval for a single doctor's experience."""

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
            collection_name or self.settings.qdrant_doctor_memory_collection
        )
        self.embedding_model_name = self.settings.embedding_model
        self._collection_ensured = False
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
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for doctor memory"
                ) from exc
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self._embedding_backend = type(self.embedding_model).__name__
        return self.embedding_model

    @staticmethod
    def _vector_list(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            return list(vector.tolist())
        return list(vector)

    def _embed(self, text: str) -> list[float]:
        model = self._get_embedding_model()
        vector = model.encode(text, normalize_embeddings=True)
        return self._vector_list(vector)

    @staticmethod
    def _distance_model() -> Any:
        try:
            from qdrant_client.http import models

            return models.Distance.COSINE
        except ImportError:
            return DOCTOR_MEMORY_DISTANCE

    @classmethod
    def _vector_params(cls) -> Any:
        try:
            from qdrant_client.http import models

            return models.VectorParams(
                size=DOCTOR_MEMORY_VECTOR_SIZE,
                distance=cls._distance_model(),
            )
        except ImportError:
            return {
                "size": DOCTOR_MEMORY_VECTOR_SIZE,
                "distance": DOCTOR_MEMORY_DISTANCE,
            }

    def ensure_collection(self) -> None:
        if self._collection_ensured:
            return
        client = self._get_client()
        exists = False
        if hasattr(client, "collection_exists"):
            exists = bool(client.collection_exists(collection_name=self.collection_name))
        elif hasattr(client, "get_collection"):
            try:
                client.get_collection(collection_name=self.collection_name)
                exists = True
            except Exception:
                exists = False
        if not exists:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._vector_params(),
            )
        self._collection_ensured = True

    @staticmethod
    def _point_id(note_id: str) -> str:
        try:
            uuid.UUID(str(note_id))
            return str(note_id)
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doctor-memory:{note_id}"))

    @classmethod
    def _point_struct(cls, point_id: str, vector: list[float], payload: dict[str, Any]) -> Any:
        try:
            from qdrant_client.http import models

            return models.PointStruct(id=cls._point_id(point_id), vector=vector, payload=payload)
        except ImportError:
            return {"id": cls._point_id(point_id), "vector": vector, "payload": payload}

    @staticmethod
    def _match_value(value: Any) -> Any:
        try:
            from qdrant_client.http import models

            return models.MatchValue(value=value)
        except ImportError:
            return {"value": value}

    @classmethod
    def _field_condition(cls, key: str, value: Any) -> Any:
        try:
            from qdrant_client.http import models

            return models.FieldCondition(key=key, match=cls._match_value(value))
        except ImportError:
            return {"key": key, "match": cls._match_value(value)}

    @classmethod
    def _doctor_filter(cls, doctor_id: str) -> Any:
        conditions = [
            cls._field_condition("doctor_id", doctor_id),
            cls._field_condition("status", DEFAULT_NOTE_STATUS),
        ]
        try:
            from qdrant_client.http import models

            return models.Filter(must=conditions)
        except ImportError:
            return {"must": conditions}

    @staticmethod
    def build_drug_pair_keys(active_ingredients: list[str]) -> list[str]:
        ingredients = _deduplicate(sorted(_fold_text(item) for item in active_ingredients))
        return [
            "|".join(pair)
            for pair in itertools.combinations(ingredients, 2)
            if pair[0] and pair[1]
        ]

    @classmethod
    def build_vector_text(cls, payload: dict[str, Any]) -> str:
        parts = [
            DOCTOR_MEMORY_LABEL,
            payload.get("title"),
            payload.get("note_text"),
            " ".join(_as_list(payload.get("active_ingredients"))),
            " ".join(_as_list(payload.get("drug_pair_keys"))),
            " ".join(_as_list(payload.get("diagnosis_keywords"))),
            " ".join(_as_list(payload.get("patient_tags"))),
            payload.get("note_type"),
        ]
        return "passage: " + _normalize_text(" ".join(str(part or "") for part in parts))

    @staticmethod
    def _search_query_text(query: str) -> str:
        return "query: " + _normalize_text(query)

    @staticmethod
    def _payload_from_note(
        *,
        doctor_id: str,
        note_id: str,
        title: str | None,
        note_text: str,
        note_type: str | None,
        source_context: str | None,
        active_ingredients: list[str] | None,
        drug_pair_keys: list[str] | None,
        diagnosis_keywords: list[str] | None,
        patient_tags: list[str] | None,
        applicability: dict[str, Any] | None,
        priority: str | None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        ingredients = _normalized_list(active_ingredients or [])
        pairs = _normalized_pair_list(drug_pair_keys or [])
        if not pairs and ingredients:
            pairs = DoctorMemoryService.build_drug_pair_keys(ingredients)
        return {
            "note_id": note_id,
            "doctor_id": doctor_id,
            "title": _normalize_text(title),
            "note_text": _normalize_text(note_text),
            "note_type": _normalize_text(note_type) or DEFAULT_NOTE_TYPE,
            "source_context": _normalize_text(source_context) or DEFAULT_SOURCE_CONTEXT,
            "active_ingredients": ingredients,
            "drug_pair_keys": pairs,
            "diagnosis_keywords": _normalized_list(diagnosis_keywords or []),
            "patient_tags": _normalized_list(patient_tags or []),
            "applicability": applicability or {},
            "priority": _normalize_text(priority) or DEFAULT_NOTE_PRIORITY,
            "status": DEFAULT_NOTE_STATUS,
            "created_at": created_at or now,
            "updated_at": updated_at or now,
        }

    def save_note(
        self,
        *,
        doctor_id: str,
        title: str | None = None,
        note_text: str,
        note_type: str | None = None,
        source_context: str | None = None,
        active_ingredients: list[str] | None = None,
        drug_pair_keys: list[str] | None = None,
        diagnosis_keywords: list[str] | None = None,
        patient_tags: list[str] | None = None,
        applicability: dict[str, Any] | None = None,
        priority: str | None = None,
        note_id: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        final_note_id = note_id or str(uuid.uuid4())
        payload = self._payload_from_note(
            doctor_id=doctor_id,
            note_id=final_note_id,
            title=title,
            note_text=note_text,
            note_type=note_type,
            source_context=source_context,
            active_ingredients=active_ingredients,
            drug_pair_keys=drug_pair_keys,
            diagnosis_keywords=diagnosis_keywords,
            patient_tags=patient_tags,
            applicability=applicability,
            priority=priority,
            created_at=created_at,
            updated_at=updated_at,
        )
        vector = self._embed(self.build_vector_text(payload))
        self.ensure_collection()
        self._get_client().upsert(
            collection_name=self.collection_name,
            points=[self._point_struct(final_note_id, vector, payload)],
        )
        return payload

    @staticmethod
    def _point_payload(point: Any) -> dict[str, Any]:
        payload = getattr(point, "payload", None)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _scroll_point_payload(point: Any) -> dict[str, Any]:
        if isinstance(point, tuple) and point:
            point = point[0]
        if isinstance(point, dict):
            payload = point.get("payload")
            return payload if isinstance(payload, dict) else {}
        payload = getattr(point, "payload", None)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _point_score(point: Any) -> float:
        try:
            return float(getattr(point, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _public_note(cls, payload: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "note_id": payload.get("note_id"),
            "title": payload.get("title"),
            "note_text": payload.get("note_text"),
            "note_type": payload.get("note_type"),
            "source_context": payload.get("source_context"),
            "active_ingredients": _normalized_list(payload.get("active_ingredients") or []),
            "drug_pair_keys": _normalized_pair_list(payload.get("drug_pair_keys") or []),
            "diagnosis_keywords": _normalized_list(payload.get("diagnosis_keywords") or []),
            "patient_tags": _normalized_list(payload.get("patient_tags") or []),
            "applicability": payload.get("applicability") or {},
            "priority": payload.get("priority"),
            "score": round(score, 6),
            "match_reason": "semantic_match",
            "created_at": payload.get("created_at"),
        }

    @staticmethod
    def _query_tokens(query: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", _fold_text(query)))

    @classmethod
    def _metadata_match_score(cls, payload: dict[str, Any], query: str) -> float:
        tokens = cls._query_tokens(query)
        if not tokens:
            return 0.0
        score = 0.0
        ingredients = _normalized_list(payload.get("active_ingredients") or [])
        pairs = _normalized_pair_list(payload.get("drug_pair_keys") or [])
        for pair in pairs:
            pair_parts = set(part for part in pair.split("|") if part)
            if pair_parts and pair_parts <= tokens:
                score += 5.0
        score += 2.0 * len(set(ingredients) & tokens)
        return score

    @staticmethod
    def _deduplicate_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        output: list[dict[str, Any]] = []
        for note in notes:
            key = str(note.get("note_id") or "")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            output.append(note)
        return output

    def _metadata_fallback_notes(
        self,
        *,
        doctor_id: str,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        client = self._get_client()
        if not hasattr(client, "scroll"):
            return []
        scroll_kwargs = {
            "collection_name": self.collection_name,
            "scroll_filter": self._doctor_filter(doctor_id),
            "limit": max(limit * 10, 50),
            "with_payload": True,
            "with_vectors": False,
        }
        response = None
        attempts = [
            scroll_kwargs,
            {key: value for key, value in scroll_kwargs.items() if key != "with_vectors"},
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
                response = client.scroll(**attempt)
                break
            except Exception:
                continue
        if response is None:
            return []
        points = response[0] if isinstance(response, tuple) else response
        notes: list[dict[str, Any]] = []
        for point in list(points or []):
            payload = self._scroll_point_payload(point)
            if payload.get("doctor_id") != doctor_id:
                continue
            if payload.get("status") != DEFAULT_NOTE_STATUS:
                continue
            metadata_score = self._metadata_match_score(payload, query)
            if metadata_score <= 0:
                continue
            note = self._public_note(payload, metadata_score)
            note["match_reason"] = "metadata_match"
            notes.append(note)
        notes.sort(key=lambda item: item.get("score") or 0, reverse=True)
        return notes[:limit]

    def search_notes(
        self,
        *,
        doctor_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        self.ensure_collection()
        points: list[Any] = []
        try:
            response = self._get_client().query_points(
                collection_name=self.collection_name,
                query=self._embed(self._search_query_text(query)),
                query_filter=self._doctor_filter(doctor_id),
                limit=top_k,
                with_payload=True,
            )
            points = list(getattr(response, "points", response or []))
        except Exception:
            points = []
        notes: list[dict[str, Any]] = []
        for point in points:
            payload = self._point_payload(point)
            if payload.get("doctor_id") != doctor_id:
                continue
            if payload.get("status") != DEFAULT_NOTE_STATUS:
                continue
            note = self._public_note(payload, self._point_score(point))
            metadata_score = self._metadata_match_score(payload, query)
            if metadata_score > 0:
                note["score"] = round(max(float(note.get("score") or 0.0), metadata_score), 6)
                note["match_reason"] = "metadata_match"
            notes.append(note)
        notes.extend(
            self._metadata_fallback_notes(
                doctor_id=doctor_id,
                query=query,
                limit=top_k,
            )
        )
        notes = self._deduplicate_notes(notes)
        notes.sort(key=lambda item: item.get("score") or 0, reverse=True)
        return notes[:top_k]

    @staticmethod
    def _active_ingredients_from_normalized(normalized_result: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for medication in _as_list(normalized_result.get("medications")):
            if not isinstance(medication, dict):
                continue
            for ingredient in _as_list(medication.get("active_ingredients")):
                if not isinstance(ingredient, dict):
                    continue
                values.append(
                    str(
                        ingredient.get("evidence_slug")
                        or ingredient.get("normalized_name")
                        or ingredient.get("name")
                        or ""
                    )
                )
        if not values:
            values = [str(value) for value in _as_list(normalized_result.get("unique_evidence_slugs"))]
        return _normalized_list(values)

    @staticmethod
    def _diagnosis_keywords(patient_context: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("diagnoses", "diagnosis", "comorbidities"):
            value = patient_context.get(key)
            if isinstance(value, list):
                values.extend(str(item) for item in value if item)
            elif value:
                values.append(str(value))
        return _normalized_list(values)

    @staticmethod
    def _patient_tags(patient_context: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        pregnancy_text = _fold_text(
            patient_context.get("pregnancy_status")
            or patient_context.get("pregnancy_lactation")
        )
        if pregnancy_text:
            if any(term in pregnancy_text for term in ("not pregnant", "khong mang thai", "khong")):
                tags.append("not_pregnant")
            elif any(term in pregnancy_text for term in ("pregnant", "mang thai", "thai")):
                tags.append("pregnancy")

        renal_text = _fold_text(patient_context.get("renal_function"))
        if any(term in renal_text for term in ("suy than", "egfr", "renal impairment")):
            tags.append("renal_impairment")

        hepatic_text = _fold_text(patient_context.get("hepatic_function"))
        if any(term in hepatic_text for term in ("suy gan", "hepatic impairment", "child-pugh")):
            tags.append("hepatic_impairment")

        diagnosis_text = " ".join(DoctorMemoryService._diagnosis_keywords(patient_context))
        if "soi than" in diagnosis_text:
            tags.append("renal_stone")
        return _deduplicate(tags)

    @staticmethod
    def _risk_types(risk_analysis: dict[str, Any] | None) -> list[str]:
        if not isinstance(risk_analysis, dict):
            return []
        return _normalized_list(
            [
                str(item.get("risk_type") or "")
                for item in _as_list(risk_analysis.get("risk_items"))
                if isinstance(item, dict)
            ]
        )

    @classmethod
    def build_audit_memory_context(
        cls,
        *,
        doctor_id: str | None,
        normalized_result: dict[str, Any],
        patient_context: dict[str, Any] | None,
        risk_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = patient_context if isinstance(patient_context, dict) else {}
        active_ingredients = cls._active_ingredients_from_normalized(normalized_result)
        diagnosis_keywords = cls._diagnosis_keywords(context)
        patient_tags = cls._patient_tags(context)
        drug_pair_keys = cls.build_drug_pair_keys(active_ingredients)
        risk_types = cls._risk_types(risk_analysis)
        return {
            "doctor_id": doctor_id,
            "active_ingredients": active_ingredients,
            "drug_pair_keys": drug_pair_keys,
            "diagnosis_keywords": diagnosis_keywords,
            "patient_tags": patient_tags,
            "risk_types": risk_types,
            "source_context": DEFAULT_SOURCE_CONTEXT,
        }

    @staticmethod
    def _query_from_audit_context(context: dict[str, Any]) -> str:
        return " ".join(
            [
                DOCTOR_MEMORY_LABEL,
                " ".join(context.get("active_ingredients") or []),
                " ".join(context.get("drug_pair_keys") or []),
                " ".join(context.get("diagnosis_keywords") or []),
                " ".join(context.get("patient_tags") or []),
                " ".join(context.get("risk_types") or []),
            ]
        )

    @staticmethod
    def _applicability_conflicts(
        note: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        applicability = _as_dict(note.get("applicability"))
        pregnancy = _fold_text(applicability.get("pregnancy_status"))
        patient_tags = set(context.get("patient_tags") or [])
        if pregnancy in {"pregnant", "mang thai", "pregnancy"} and "not_pregnant" in patient_tags:
            return True
        if pregnancy in {"not_pregnant", "khong mang thai"} and "pregnancy" in patient_tags:
            return True
        return False

    @staticmethod
    def _overlap(left: list[str], right: list[str]) -> int:
        return len(set(left) & set(right))

    @classmethod
    def _audit_rerank_score(cls, note: dict[str, Any], context: dict[str, Any]) -> float:
        score = float(note.get("score") or 0.0)
        if set(note.get("drug_pair_keys") or []) & set(context.get("drug_pair_keys") or []):
            score += 5
        if cls._overlap(note.get("active_ingredients") or [], context.get("active_ingredients") or []):
            score += 3
        if cls._overlap(note.get("patient_tags") or [], context.get("patient_tags") or []):
            score += 3
        if cls._overlap(note.get("diagnosis_keywords") or [], context.get("diagnosis_keywords") or []):
            score += 2
        if note.get("source_context") == context.get("source_context"):
            score += 1
        return score

    def retrieve_for_audit_context(
        self,
        *,
        doctor_id: str | None,
        normalized_result: dict[str, Any],
        patient_context: dict[str, Any] | None,
        risk_analysis: dict[str, Any] | None = None,
        max_notes: int = 3,
    ) -> dict[str, Any]:
        if not doctor_id:
            return {"matched_notes": []}
        context = self.build_audit_memory_context(
            doctor_id=doctor_id,
            normalized_result=normalized_result,
            patient_context=patient_context,
            risk_analysis=risk_analysis,
        )
        if not (
            context["active_ingredients"]
            or context["diagnosis_keywords"]
            or context["patient_tags"]
        ):
            return {"matched_notes": []}
        notes = self.search_notes(
            doctor_id=doctor_id,
            query=self._query_from_audit_context(context),
            top_k=max(max_notes * 5, 10),
        )
        reranked: list[dict[str, Any]] = []
        for note in notes:
            if self._applicability_conflicts(note, context):
                continue
            final_score = self._audit_rerank_score(note, context)
            if final_score < 3:
                continue
            copied = dict(note)
            copied["score"] = round(final_score, 6)
            copied["match_reason"] = "audit_context_match"
            reranked.append(copied)
        reranked.sort(key=lambda item: item.get("score") or 0, reverse=True)
        return {"matched_notes": reranked[:max_notes]}

    def get_stats(self) -> dict[str, Any]:
        return {
            "service": "DoctorMemoryService",
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "qdrant_backend": self._qdrant_backend,
            "embedding_backend": self._embedding_backend,
        }


@lru_cache
def get_doctor_memory_service() -> DoctorMemoryService:
    return DoctorMemoryService()
