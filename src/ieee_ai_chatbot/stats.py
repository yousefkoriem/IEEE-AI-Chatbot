from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from .config import Settings
from .ingest import _load_manifest

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
