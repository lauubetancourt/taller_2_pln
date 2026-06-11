"""Procesamiento local de PDFs y preparacion de fragmentos de texto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def discover_pdf_files(pdf_dir: str | Path) -> list[Path]:
    """Encontrar automaticamente todos los archivos PDF de un directorio local."""

    raise NotImplementedError("Pendiente: descubrir PDFs locales.")


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extraer texto de un PDF usando ``pymupdf4llm``."""

    raise NotImplementedError("Pendiente: extraer texto de PDF.")


def extract_texts_from_pdfs(pdf_dir: str | Path) -> dict[Path, str]:
    """Extraer texto de todos los PDFs encontrados en ``pdf_dir``."""

    raise NotImplementedError("Pendiente: extraer textos de multiples PDFs.")


def build_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Any:
    """Crear un ``RecursiveCharacterTextSplitter`` con parametros del taller."""

    raise NotImplementedError("Pendiente: crear text splitter.")


def chunk_documents(texts_by_file: dict[Path, str], splitter: Any) -> list[dict[str, Any]]:
    """Dividir textos extraidos de PDFs en fragmentos con metadatos."""

    raise NotImplementedError("Pendiente: fragmentar documentos.")
