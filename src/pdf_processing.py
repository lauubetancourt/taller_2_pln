"""Procesamiento local de PDFs y preparacion de fragmentos de texto."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import CHUNK_OVERLAP, CHUNK_SIZE

Chunk = dict[str, Any]


def discover_pdf_files(pdf_dir: str | Path) -> list[Path]:
    """Encontrar automaticamente todos los archivos PDF de un directorio local."""

    directory = Path(pdf_dir)
    if not directory.exists():
        raise FileNotFoundError(f"No existe el directorio de PDFs: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"La ruta de PDFs no es un directorio: {directory}")

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extraer texto de un PDF usando ``pymupdf4llm`` en formato Markdown."""

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo PDF: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"El archivo no tiene extension .pdf: {path}")

    import pymupdf4llm

    markdown_text = pymupdf4llm.to_markdown(str(path))
    return str(markdown_text).strip()


def extract_texts_from_pdfs(pdf_dir: str | Path) -> dict[Path, str]:
    """Extraer texto de todos los PDFs encontrados en ``pdf_dir``."""

    pdf_files = discover_pdf_files(pdf_dir)
    if not pdf_files:
        raise FileNotFoundError(
            f"No se encontraron archivos PDF en {Path(pdf_dir)}. "
            "Agrega los documentos del taller en data/pdfs/."
        )

    texts_by_file: dict[Path, str] = {}
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        if text:
            texts_by_file[pdf_path] = text

    if not texts_by_file:
        raise ValueError(
            "Se encontraron PDFs, pero no se pudo extraer texto util de ninguno."
        )
    return texts_by_file


def build_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Any:
    """Crear un ``RecursiveCharacterTextSplitter`` con parametros del taller."""

    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que cero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap no puede ser negativo.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size.")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError as exc:  # pragma: no cover - depende del entorno local
            raise ImportError(
                "No se encontro LangChain ni langchain-text-splitters. "
                "Instala las dependencias con: pip install -r requirements.txt"
            ) from exc

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_documents(texts_by_file: dict[Path, str], splitter: Any) -> list[Chunk]:
    """Dividir textos extraidos de PDFs en fragmentos con metadatos."""

    if not texts_by_file:
        raise ValueError("No hay textos de PDFs para fragmentar.")

    documents = []
    for file_path, text in texts_by_file.items():
        clean_text = str(text).strip()
        if not clean_text:
            continue
        documents.append(
            _make_document(
                page_content=clean_text,
                metadata={
                    "source": str(file_path),
                    "source_file": Path(file_path).name,
                },
            )
        )

    if not documents:
        raise ValueError("Todos los textos extraidos estaban vacios.")

    split_documents = splitter.split_documents(documents)
    chunks: list[Chunk] = []
    per_file_counts: dict[str, int] = {}

    for global_index, document in enumerate(split_documents):
        metadata = dict(getattr(document, "metadata", {}) or {})
        source_path = str(metadata.get("source", ""))
        source_file = str(metadata.get("source_file") or Path(source_path).name)
        local_index = per_file_counts.get(source_path, 0)
        per_file_counts[source_path] = local_index + 1
        text = str(getattr(document, "page_content", "")).strip()

        chunks.append(
            {
                "chunk_id": _build_chunk_id(source_file, local_index),
                "global_chunk_index": global_index,
                "chunk_index": local_index,
                "source_file": source_file,
                "source_path": source_path,
                "text": text,
                "n_characters": len(text),
            }
        )

    return chunks


def save_chunks_jsonl(chunks: Iterable[Chunk], output_path: str | Path) -> int:
    """Guardar fragmentos de PDFs en JSONL y retornar cuantos se escribieron."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_chunks_jsonl(input_path: str | Path) -> list[Chunk]:
    """Cargar fragmentos de PDFs desde un archivo JSONL."""

    path = Path(input_path)
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("text"):
                chunks.append(row)
    return chunks


def _make_document(page_content: str, metadata: dict[str, Any]) -> Any:
    try:
        from langchain_core.documents import Document
    except ImportError:
        try:
            from langchain.schema import Document
        except ImportError as exc:  # pragma: no cover - depende del entorno local
            raise ImportError(
                "No se encontro LangChain. Instala las dependencias con: "
                "pip install -r requirements.txt"
            ) from exc

    return Document(page_content=page_content, metadata=metadata)


def _build_chunk_id(source_file: str, chunk_index: int) -> str:
    stem = Path(source_file).stem or "documento"
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_").lower()
    return f"{safe_stem}_chunk_{chunk_index:04d}"
