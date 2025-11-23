"""
Hybrid retrieval helpers that make it easier to resolve property references.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.data_layer import load_assets
from app.config import load_address_aliases

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover - optional dependency
    OpenAIEmbeddings = None  # type: ignore

logger = logging.getLogger(__name__)
_EMBED_MODEL_ENV = "OPENAI_EMBEDDINGS_MODEL"
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class PropertyRecord:
    address: str
    property_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    tenant_name: Optional[str] = None
    price: Optional[float] = None
    pnl: Optional[float] = None
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    document: str = ""


@dataclass(frozen=True)
class PropertyMatch:
    address: str
    property_name: Optional[str]
    confidence: float
    rank: int
    reason: str
    metadata: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class PropertyMemoryResult:
    query: str
    matches: List[PropertyMatch]
    candidate_terms: List[str] = field(default_factory=list)
    unresolved_terms: List[str] = field(default_factory=list)


class PropertyMemory:
    """
    Cached view over the asset dataset that supports semantic property lookups.
    """

    def __init__(self) -> None:
        records = self._build_records()
        records = self._inject_alias_map(records)
        self.records: List[PropertyRecord] = records
        self._documents = [record.document for record in records]
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._alias_index: Dict[str, int] = {}
        self._embedding_client = None
        self._document_embeddings: Optional[np.ndarray] = None
        self._use_embeddings = False

        if self._documents:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = self._vectorizer.fit_transform(self._documents)
            self._build_alias_index()
            self._maybe_warm_embeddings()
        else:
            logger.warning("PropertyMemory initialized with an empty dataset")

    # ----------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------

    def search(self, text: str, top_k: int = 5) -> List[PropertyMatch]:
        if not text or not self.records:
            return []
        scores = self._tfidf_scores(text)
        scores = self._blend_with_embeddings(text, scores)
        return self._build_matches(scores, top_k=top_k, reason=self._reason_label())

    def resolve_mentions(
        self,
        text: str,
        expected: int = 2,
        min_confidence: float = 0.12,
    ) -> PropertyMemoryResult:
        if not text:
            return PropertyMemoryResult(query=text, matches=[])

        candidate_terms = self._extract_candidate_terms(text)
        matches: List[PropertyMatch] = []
        unresolved: List[str] = []
        seen_addresses: set[str] = set()

        for candidate in candidate_terms:
            ranked = self.search(candidate, top_k=1)
            if not ranked:
                unresolved.append(candidate)
                continue
            match = ranked[0]
            if match.confidence < min_confidence or match.address in seen_addresses:
                unresolved.append(candidate)
                continue
            seen_addresses.add(match.address)
            reason = f"{match.reason} (from '{candidate}')"
            matches.append(replace(match, rank=len(matches) + 1, reason=reason))
            if len(matches) >= expected:
                break

        if candidate_terms and len(matches) < expected:
            filler = self.search(text, top_k=expected - len(matches))
            for match in filler:
                if match.address in seen_addresses:
                    continue
                if match.confidence < min_confidence:
                    continue
                seen_addresses.add(match.address)
                reason = f"{match.reason} (fallback from full query)"
                matches.append(replace(match, rank=len(matches) + 1, reason=reason))
                if len(matches) >= expected:
                    break

        return PropertyMemoryResult(
            query=text,
            matches=matches,
            candidate_terms=candidate_terms,
            unresolved_terms=unresolved,
        )

    # ----------------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------------

    def _build_records(self) -> List[PropertyRecord]:
        df = load_assets()
        if df is None or df.empty:
            return []
        working = df.copy()
        if "address" not in working.columns and "property_name" in working.columns:
            working["address"] = working["property_name"]
        if "property_name" not in working.columns and "address" in working.columns:
            working["property_name"] = working["address"]

        group_key = "address" if "address" in working.columns else "property_name"
        grouped = working.groupby(group_key, dropna=False).first().reset_index(drop=False)

        records: List[PropertyRecord] = []
        for _, row in grouped.iterrows():
            address = str(row.get(group_key) or "").strip()
            if not address:
                continue
            property_name = str(row.get("property_name") or "").strip() or None
            city = str(row.get("city") or "").strip() or None
            state = str(row.get("state") or "").strip() or None
            tenant = str(row.get("tenant_name") or "").strip() or None
            price = self._safe_float(row.get("price"))
            pnl = self._safe_float(row.get("pnl"))
            document = self._compose_document(
                address=address,
                property_name=property_name,
                city=city,
                state=state,
                tenant=tenant,
                price=price,
                pnl=pnl,
            )
            aliases = self._build_aliases(address, property_name, tenant)
            records.append(
                PropertyRecord(
                    address=address,
                    property_name=property_name,
                    city=city,
                    state=state,
                    tenant_name=tenant,
                    price=price,
                    pnl=pnl,
                    aliases=aliases,
                    document=document,
                )
            )
        logger.info("PropertyMemory loaded %s property records", len(records))
        return records

    def _inject_alias_map(self, records: List[PropertyRecord]) -> List[PropertyRecord]:
        alias_map = load_address_aliases()
        if not alias_map or not records:
            return records
        index: Dict[str, int] = {}
        for idx, record in enumerate(records):
            keys = {record.address.lower()}
            if record.property_name:
                keys.add(record.property_name.lower())
            for alias in record.aliases:
                keys.add(alias.lower())
            for key in keys:
                index.setdefault(key, idx)
        updated = list(records)
        for alias, canonical in alias_map.items():
            if not canonical:
                continue
            idx = index.get(str(canonical).lower())
            if idx is None:
                continue
            record = updated[idx]
            alias_set = set(record.aliases)
            if alias in alias_set:
                continue
            alias_set.add(alias)
            updated[idx] = replace(record, aliases=tuple(sorted(alias_set)))
            index.setdefault(alias.lower(), idx)
        return updated

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        if value in (None, "", "nan"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _compose_document(
        *,
        address: str,
        property_name: Optional[str],
        city: Optional[str],
        state: Optional[str],
        tenant: Optional[str],
        price: Optional[float],
        pnl: Optional[float],
    ) -> str:
        parts: List[str] = []
        if property_name and property_name.lower() != address.lower():
            parts.append(property_name)
        parts.append(address)
        if city or state:
            city_state = ", ".join(part for part in (city, state) if part)
            parts.append(city_state)
        if tenant:
            parts.append(f"Tenant {tenant}")
        if price is not None:
            parts.append(f"Price {price}")
        if pnl is not None:
            parts.append(f"PNL {pnl}")
        return " | ".join(parts)

    @staticmethod
    def _build_aliases(address: str, property_name: Optional[str], tenant: Optional[str]) -> Tuple[str, ...]:
        aliases = {address}
        if property_name:
            aliases.add(property_name)
        building_match = re.findall(r"(building\s+\d+)", address, flags=re.IGNORECASE)
        for match in building_match:
            aliases.add(match.title())
        if property_name:
            building_match = re.findall(r"(building\s+\d+)", property_name, flags=re.IGNORECASE)
            for match in building_match:
                aliases.add(match.title())
        if tenant:
            aliases.add(tenant)
        normalized = tuple(sorted({alias.strip() for alias in aliases if alias.strip()}))
        return normalized

    def _build_alias_index(self) -> None:
        alias_index: Dict[str, int] = {}
        for idx, record in enumerate(self.records):
            for alias in record.aliases:
                key = alias.lower()
                alias_index.setdefault(key, idx)
        self._alias_index = alias_index

    def _maybe_warm_embeddings(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAIEmbeddings is None:
            return
        model = os.getenv(_EMBED_MODEL_ENV, _DEFAULT_EMBED_MODEL)
        try:
            client = OpenAIEmbeddings(model=model)
            embeddings = client.embed_documents(self._documents)
            self._embedding_client = client
            self._document_embeddings = np.array(embeddings)
            self._use_embeddings = True
            logger.info("PropertyMemory enabled OpenAI embeddings via model %s", model)
        except Exception as exc:  # pragma: no cover - optional network path
            logger.warning("Failed to initialize OpenAI embeddings (%s). Falling back to TF-IDF.", exc)
            self._embedding_client = None
            self._document_embeddings = None
            self._use_embeddings = False

    def _tfidf_scores(self, text: str) -> np.ndarray:
        if not self._vectorizer or self._matrix is None:
            return np.zeros(len(self.records))
        query_vec = self._vectorizer.transform([text])
        similarities = cosine_similarity(self._matrix, query_vec).reshape(-1)
        return similarities

    def _blend_with_embeddings(self, text: str, tfidf_scores: np.ndarray) -> np.ndarray:
        if not self._use_embeddings or self._document_embeddings is None or not self._embedding_client:
            return tfidf_scores
        try:
            query_embedding = np.array(self._embedding_client.embed_query(text))
        except Exception as exc:  # pragma: no cover - optional network path
            logger.warning("OpenAI embedding lookup failed (%s). Using TF-IDF only.", exc)
            self._use_embeddings = False
            return tfidf_scores

        doc_matrix = self._document_embeddings
        query_norm = np.linalg.norm(query_embedding) or 1.0
        doc_norms = np.linalg.norm(doc_matrix, axis=1)
        denom = np.clip(doc_norms * query_norm, 1e-9, None)
        cosine_scores = (doc_matrix @ query_embedding) / denom
        normalized = (cosine_scores + 1.0) / 2.0  # scale to 0-1
        blended = (0.6 * tfidf_scores) + (0.4 * normalized)
        return blended

    def _build_matches(self, scores: np.ndarray, *, top_k: int, reason: str) -> List[PropertyMatch]:
        if scores.size == 0:
            return []
        order = np.argsort(scores)[::-1]
        matches: List[PropertyMatch] = []
        for position, idx in enumerate(order[:top_k], start=1):
            score = float(scores[idx])
            if score <= 0:
                break
            record = self.records[idx]
            metadata = {
                "city": record.city,
                "state": record.state,
                "tenant_name": record.tenant_name,
            }
            confidence = max(0.0, min(1.0, round(score, 4)))
            matches.append(
                PropertyMatch(
                    address=record.address,
                    property_name=record.property_name,
                    confidence=confidence,
                    rank=position,
                    reason=reason,
                    metadata=metadata,
                )
            )
        return matches

    def _reason_label(self) -> str:
        return "Hybrid similarity (tf-idf + embeddings)" if self._use_embeddings else "TF-IDF similarity"

    def _extract_candidate_terms(self, text: str) -> List[str]:
        lowered = text.lower()
        alias_terms = self._alias_hits(text, lowered)
        building_terms = re.findall(r"(building\s+\d+)", text, flags=re.IGNORECASE)
        address_terms = self._address_like_terms(text)

        ordered: List[str] = []
        for collection in (alias_terms, building_terms, address_terms):
            for term in collection:
                normalized = term.strip()
                if normalized and normalized not in ordered:
                    ordered.append(normalized)
        return ordered

    def _alias_hits(self, original: str, lowered: str) -> List[str]:
        hits: List[Tuple[int, str]] = []
        for alias, idx in self._alias_index.items():
            position = lowered.find(alias)
            if position == -1:
                continue
            snippet = original[position : position + len(alias)]
            hits.append((position, snippet))
        hits.sort(key=lambda item: item[0])
        ordered: List[str] = []
        for _, snippet in hits:
            if snippet not in ordered:
                ordered.append(snippet)
        return ordered

    @staticmethod
    def _address_like_terms(text: str) -> List[str]:
        pattern = re.findall(
            r"\b\d{2,6}\s+[A-Za-z0-9.\s]+?"
            r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct|way|parkway|pkwy)\b",
            text,
            flags=re.IGNORECASE,
        )
        return [match.strip() for match in pattern]


@lru_cache(maxsize=1)
def get_property_memory() -> PropertyMemory:
    """
    Singleton accessor used by the supervisor & tools module.
    """

    return PropertyMemory()


