from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import Any
from .config import Settings
from .ingest import _load_manifest, _save_manifest
from .vectorstore import get_vector_store

logger = logging.getLogger(__name__)


def get_kb_stats(settings: Settings) -> dict:
    """Reads the manifest and calculates knowledge base statistics."""
    manifest_path = Path(settings.manifest_path)
    manifest = _load_manifest(manifest_path)
    
    sources = manifest.get("sources", {})
    
    total_sources = len(sources)
    total_chunks = 0
    origins = {"local": 0, "upload": 0, "website": 0}
    
    source_names = []
    
    for source_id, metadata in sources.items():
        chunk_ids = metadata.get("chunk_ids", [])
        total_chunks += len(chunk_ids)
        
        origin = metadata.get("origin", "local")
        if origin in origins:
            origins[origin] += 1
        else:
            origins[origin] = 1
            
        source_names.append(f"{source_id} ({origin})")
        
    last_sync = "Never"
    if manifest_path.exists():
        mtime = manifest_path.stat().st_mtime
        last_sync = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "total_sources": total_sources,
        "total_chunks": total_chunks,
        "origins": origins,
        "last_sync": last_sync,
        "source_names": sorted(source_names)
    }


def list_all_sources(settings: Settings) -> list[dict[str, Any]]:
    """Returns a detailed list of all sources in the manifest."""
    manifest_path = Path(settings.manifest_path)
    manifest = _load_manifest(manifest_path)
    sources = manifest.get("sources", {})
    results = []
    for source_id, metadata in sources.items():
        chunk_ids = metadata.get("chunk_ids", [])
        results.append({
            "id": source_id,
            "origin": metadata.get("origin", "local"),
            "chunk_count": len(chunk_ids),
            "hash": metadata.get("hash", "")[:12],
        })
    results.sort(key=lambda r: r["origin"])
    return results


def delete_source(settings: Settings, source_id: str) -> bool:
    """Deletes a source and its chunks from Pinecone and the manifest."""
    manifest_path = Path(settings.manifest_path)
    manifest = _load_manifest(manifest_path)
    sources = manifest.get("sources", {})

    metadata = sources.get(source_id)
    if not metadata:
        logger.warning("Source %s not found in manifest.", source_id)
        return False

    chunk_ids = metadata.get("chunk_ids", [])
    if chunk_ids:
        try:
            vector_store = get_vector_store(settings)
            vector_store.delete(ids=chunk_ids)
            logger.info("Deleted %d chunks for source %s", len(chunk_ids), source_id)
        except Exception as e:
            logger.error("Failed to delete chunks from vector store: %s", e)
            return False

    sources.pop(source_id, None)
    _save_manifest(manifest_path, manifest)
    return True


def search_chunks(settings: Settings, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Searches vector store for chunks matching the query."""
    if not query.strip():
        return []
    try:
        vector_store = get_vector_store(settings)
        results = vector_store.similarity_search_with_score(query, k=top_k)
        chunks = []
        for doc, score in results:
            chunks.append({
                "score": round(score, 4),
                "source": doc.metadata.get("source", "unknown"),
                "filename": doc.metadata.get("filename", "unknown"),
                "content": doc.page_content,
                "preview": doc.page_content[:200],
            })
        return chunks
    except Exception as e:
        logger.error("Search failed: %s", e)
        return []


def get_source_chunks(settings: Settings, source_id: str, max_chunks: int = 20) -> list[dict[str, Any]]:
    """Returns chunk content for a given source (from manifest chunk_ids)."""
    manifest_path = Path(settings.manifest_path)
    manifest = _load_manifest(manifest_path)
    sources = manifest.get("sources", {})
    metadata = sources.get(source_id)
    if not metadata:
        return []
    chunk_ids = metadata.get("chunk_ids", [])[:max_chunks]
    if not chunk_ids:
        return []
    try:
        vector_store = get_vector_store(settings)
        results = vector_store.get(ids=chunk_ids)
        chunks = []
        for doc in results:
            chunks.append({
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "content": doc.page_content,
                "preview": doc.page_content[:150],
            })
        return chunks
    except Exception as e:
        logger.error("Failed to fetch chunks: %s", e)
        return []
