from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pptx import Presentation

from .config import Settings
from .vectorstore import get_vector_store

SUPPORTED_EXTENSIONS = {".pdf", ".ppt", ".pptx"}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"sources": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def _extract_ppt_text(path: Path) -> str:
    presentation = Presentation(str(path))
    blocks: list[str] = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            text = getattr(shape, "text", "")
            if text:
                blocks.append(text)
    return "\n".join(blocks).strip()


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    if suffix in {".ppt", ".pptx"}:
        return _extract_ppt_text(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def _build_documents(path: Path, source_id: str, text: str) -> list[Document]:
    return [
        Document(
            page_content=text,
            metadata={
                "source": source_id,
                "filename": path.name,
                "suffix": path.suffix.lower(),
            },
        )
    ]


def ingest_files(settings: Settings, file_paths: list[str], origin: str) -> dict[str, int]:
    vector_store = get_vector_store(settings)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    manifest_path = Path(settings.manifest_path)
    manifest = _load_manifest(manifest_path)
    sources = manifest.setdefault("sources", {})

    indexed = 0
    skipped = 0
    deleted = 0

    for raw_path in file_paths:
        path = Path(raw_path)
        if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue

        source_id = str(path.resolve())
        file_hash = _sha256_file(path)
        existing = sources.get(source_id)
        if existing and existing.get("hash") == file_hash:
            skipped += 1
            continue

        text = _extract_text(path)
        if not text:
            skipped += 1
            continue

        documents = _build_documents(path, source_id, text)
        chunks = splitter.split_documents(documents)
        chunk_ids: list[str] = []
        for idx, chunk in enumerate(chunks):
            chunk_id = hashlib.sha1(f"{source_id}:{file_hash}:{idx}".encode("utf-8")).hexdigest()
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["origin"] = origin
            chunk.metadata["hash"] = file_hash
            chunk_ids.append(chunk_id)

        if existing and existing.get("chunk_ids"):
            vector_store.delete(ids=existing["chunk_ids"])
            deleted += len(existing["chunk_ids"])

        vector_store.add_documents(documents=chunks, ids=chunk_ids)
        sources[source_id] = {
            "hash": file_hash,
            "chunk_ids": chunk_ids,
            "origin": origin,
        }
        indexed += 1

    _save_manifest(manifest_path, manifest)
    return {"indexed": indexed, "skipped": skipped, "deleted": deleted}


def sync_local_docs(settings: Settings) -> dict[str, int]:
    search_roots = [Path(settings.docs_pdf_dir), Path(settings.docs_ppt_dir)]
    all_files: list[str] = []
    for root in search_roots:
        if not root.exists():
            continue
        for file in root.rglob("*"):
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                all_files.append(str(file))

    result = ingest_files(settings, all_files, origin="local")

    manifest_path = Path(settings.manifest_path)
    manifest = _load_manifest(manifest_path)
    sources = manifest.setdefault("sources", {})

    active_sources = {str(Path(path).resolve()) for path in all_files}
    vector_store = get_vector_store(settings)
    delete_count = 0
    removed_sources = []

    for source_id, meta in list(sources.items()):
        if meta.get("origin") != "local":
            continue
        if source_id in active_sources:
            continue
        chunk_ids = meta.get("chunk_ids", [])
        if chunk_ids:
            vector_store.delete(ids=chunk_ids)
            delete_count += len(chunk_ids)
        removed_sources.append(source_id)

    for source_id in removed_sources:
        sources.pop(source_id, None)

    _save_manifest(manifest_path, manifest)
    result["deleted"] += delete_count
    result["total_files"] = len(all_files)
    return result
