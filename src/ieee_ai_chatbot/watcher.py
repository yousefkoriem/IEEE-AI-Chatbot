from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import Settings
from .ingest import sync_local_docs

logger = logging.getLogger(__name__)


class DocChangeHandler(FileSystemEventHandler):
    def __init__(self, settings: Settings, callback=None) -> None:
        self.settings = settings
        self.callback = callback
        self._last_trigger = 0.0
        self._debounce_seconds = 2.0

    def on_any_event(self, event) -> None:
        if event.is_directory:
            return
        now = time.time()
        if now - self._last_trigger < self._debounce_seconds:
            return
        self._last_trigger = now
        ext = Path(event.src_path).suffix.lower()
        if ext in (".pdf", ".ppt", ".pptx", ".doc", ".docx", ".md", ".html", ".htm"):
            logger.info("File change detected: %s", event.src_path)
            try:
                result = sync_local_docs(self.settings)
                msg = f"Auto-sync: {result.get('total_files', 0)} files | Indexed: {result['indexed']} | Deleted: {result['deleted']}"
                logger.info(msg)
                if self.callback:
                    self.callback(msg)
            except Exception as e:
                logger.error("Auto-sync failed: %s", e)
                if self.callback:
                    self.callback(f"Auto-sync failed: {e}")


class DocWatcher:
    def __init__(self, settings: Settings, callback=None) -> None:
        self.settings = settings
        self.callback = callback
        self._observer: Observer | None = None
        self._handler = DocChangeHandler(settings, callback=callback)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        dirs = []
        for attr in ("docs_pdf_dir", "docs_ppt_dir", "docs_doc_dir"):
            d = getattr(self.settings, attr, None)
            if d and Path(d).is_dir():
                dirs.append(d)
        if not dirs:
            logger.warning("No docs directories found to watch.")
            return
        self._observer = Observer()
        for d in dirs:
            self._observer.schedule(self._handler, d, recursive=False)
            logger.info("Watching: %s", d)
        self._observer.start()

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("File watcher stopped.")
