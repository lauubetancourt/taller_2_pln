"""Procesar PDFs locales y generar fragmentos persistidos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PDF_CHUNKS_PATH,
    PDFS_DIR,
    SENTENCE_EMBEDDINGS_REPORTS_DIR,
)
from src.pdf_processing import (
    build_text_splitter,
    chunk_documents,
    extract_texts_from_pdfs,
    save_chunks_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=PDFS_DIR)
    parser.add_argument("--output-file", type=Path, default=PDF_CHUNKS_PATH)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--preview-chars", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    """Extraer texto con PyMuPDF4LLM, fragmentar con LangChain y guardar JSONL."""

    args = parse_args()
    report_file = args.report_file or (
        SENTENCE_EMBEDDINGS_REPORTS_DIR / "pdf_chunks_report.csv"
    )

    print("=== Procesamiento de PDFs ===")
    print(f"Directorio de PDFs: {args.pdf_dir}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"chunk_overlap: {args.chunk_overlap}")

    texts_by_file = extract_texts_from_pdfs(args.pdf_dir)
    print(f"PDFs con texto extraido: {len(texts_by_file)}")

    splitter = build_text_splitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = chunk_documents(texts_by_file, splitter)
    if not chunks:
        raise ValueError("No se generaron fragmentos a partir de los PDFs.")

    n_saved = save_chunks_jsonl(chunks, args.output_file)
    report_df = _build_chunks_report(chunks, preview_chars=args.preview_chars)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_file, index=False)

    print("\n=== Resumen ===")
    print(f"Fragmentos generados: {len(chunks)}")
    print(f"Fragmentos guardados: {n_saved}")
    print(f"JSONL de chunks: {args.output_file}")
    print(f"Reporte CSV: {report_file}")


def _build_chunks_report(chunks: list[dict], preview_chars: int) -> pd.DataFrame:
    rows = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        rows.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "global_chunk_index": chunk.get("global_chunk_index"),
                "source_file": chunk.get("source_file"),
                "chunk_index": chunk.get("chunk_index"),
                "n_characters": chunk.get("n_characters"),
                "text_preview": _preview_text(text, preview_chars),
            }
        )
    return pd.DataFrame(rows)


def _preview_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text).split())
    return compact[:max_chars]


if __name__ == "__main__":
    main()
