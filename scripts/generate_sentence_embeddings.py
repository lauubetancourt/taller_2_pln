"""Generar embeddings semanticos, similitud coseno y visualizaciones."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    BATCH_SIZE_SENTENCE_EMBEDDINGS,
    PDF_CHUNKS_PATH,
    SENTENCE_EMBEDDINGS_FIGURES_DIR,
    SENTENCE_EMBEDDINGS_OUTPUT_DIR,
    SENTENCE_EMBEDDINGS_REPORTS_DIR,
    SENTENCE_EMBEDDINGS_SIMILARITY_DIR,
)
from src.pdf_processing import load_chunks_jsonl
from src.sentence_embeddings import (
    SENTENCE_EMBEDDING_MODELS,
    encode_query_and_chunks,
    load_sentence_embedding_model,
    model_name_to_slug,
)
from src.similarity import cosine_similarity_scores, top_k_similar_chunks
from src.visualization import (
    plot_query_match_projection,
    reduce_embeddings_pca,
    reduce_embeddings_tsne,
)

DEFAULT_QUERY = "¿Cuál es la idea principal de los documentos procesados?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-file", type=Path, default=PDF_CHUNKS_PATH)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--models", default=None)
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE_SENTENCE_EMBEDDINGS
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--projection-k", type=int, default=10)
    parser.add_argument("--skip-tsne", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Evaluar todos los modelos definidos en la actividad 3."""

    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k debe ser mayor o igual a 1.")
    if args.projection_k < 1:
        raise ValueError("--projection-k debe ser mayor o igual a 1.")
    if not args.chunks_file.exists():
        raise FileNotFoundError(
            f"No existe el archivo de chunks: {args.chunks_file}. "
            "Ejecuta primero: python3 scripts/process_pdfs.py"
        )

    chunks = load_chunks_jsonl(args.chunks_file)
    if not chunks:
        raise ValueError(
            f"No se encontraron chunks en {args.chunks_file}. "
            "Ejecuta primero: python3 scripts/process_pdfs.py"
        )

    model_names = _parse_models(args.models)
    if not model_names:
        raise ValueError("No se especificaron modelos para evaluar.")
    _ensure_output_dirs()

    print("=== Embeddings semanticos y similitud coseno ===")
    print(f"Chunks cargados: {len(chunks)}")
    print(f"Consulta: {args.query}")
    print(f"Modelos: {len(model_names)}")

    top_result_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        top_result_rows.append(
            _process_model(
                model_name=model_name,
                query=args.query,
                chunks=chunks,
                batch_size=args.batch_size,
                top_k=args.top_k,
                projection_k=args.projection_k,
                run_tsne=not args.skip_tsne,
            )
        )

    top_results_df = pd.DataFrame(top_result_rows)
    top_results_path = SENTENCE_EMBEDDINGS_REPORTS_DIR / "top_results.csv"
    top_results_json_path = SENTENCE_EMBEDDINGS_REPORTS_DIR / "top_results.json"
    top_results_df.to_csv(top_results_path, index=False)
    top_results_df.to_json(
        top_results_json_path, orient="records", force_ascii=False, indent=2
    )

    print("\n=== Resultados principales ===")
    print(
        top_results_df[
            ["model_name", "similarity_score", "source_file", "chunk_index"]
        ].to_string(index=False)
    )
    print(f"Top resultados CSV: {top_results_path}")
    print(f"Top resultados JSON: {top_results_json_path}")


def _process_model(
    model_name: str,
    query: str,
    chunks: list[dict[str, Any]],
    batch_size: int,
    top_k: int,
    projection_k: int,
    run_tsne: bool,
) -> dict[str, Any]:
    print(f"\n--- Modelo: {model_name} ---")
    slug = model_name_to_slug(model_name)
    model = load_sentence_embedding_model(model_name)

    query_embedding, chunk_embeddings = encode_query_and_chunks(
        query,
        chunks,
        model,
        batch_size=batch_size,
    )
    scores = cosine_similarity_scores(query_embedding, chunk_embeddings)
    top_chunks = top_k_similar_chunks(chunks, scores, k=top_k)
    best_chunk = top_chunks[0]

    _save_embeddings(slug, query_embedding, chunk_embeddings, scores)
    _save_similarity_outputs(slug, model_name, query, chunks, scores, top_chunks)
    _save_projection_outputs(
        slug=slug,
        model_name=model_name,
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        top_chunks=top_chunks,
        projection_k=projection_k,
        run_tsne=run_tsne,
    )

    print(
        "Mejor fragmento: "
        f"score={best_chunk['similarity_score']:.4f}, "
        f"archivo={best_chunk.get('source_file')}, "
        f"chunk={best_chunk.get('chunk_index')}"
    )

    return {
        "model_name": model_name,
        "model_slug": slug,
        "query": query,
        "similarity_score": best_chunk["similarity_score"],
        "rank": best_chunk["rank"],
        "chunk_id": best_chunk.get("chunk_id"),
        "chunk_position": best_chunk.get("chunk_position"),
        "source_file": best_chunk.get("source_file"),
        "source_path": best_chunk.get("source_path"),
        "chunk_index": best_chunk.get("chunk_index"),
        "n_characters": best_chunk.get("n_characters"),
        "text": best_chunk.get("text"),
    }


def _save_embeddings(
    slug: str,
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    scores: np.ndarray,
) -> None:
    SENTENCE_EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(
        SENTENCE_EMBEDDINGS_OUTPUT_DIR / f"{slug}_query_embedding.npy", query_embedding
    )
    np.save(
        SENTENCE_EMBEDDINGS_OUTPUT_DIR / f"{slug}_chunk_embeddings.npy",
        chunk_embeddings,
    )
    np.save(SENTENCE_EMBEDDINGS_OUTPUT_DIR / f"{slug}_cosine_scores.npy", scores)
    print(f"Embeddings guardados en: {SENTENCE_EMBEDDINGS_OUTPUT_DIR}")


def _save_similarity_outputs(
    slug: str,
    model_name: str,
    query: str,
    chunks: list[dict[str, Any]],
    scores: np.ndarray,
    top_chunks: list[dict[str, Any]],
) -> None:
    SENTENCE_EMBEDDINGS_SIMILARITY_DIR.mkdir(parents=True, exist_ok=True)
    SENTENCE_EMBEDDINGS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ranking_df = _build_similarity_ranking_dataframe(model_name, query, chunks, scores)
    ranking_path = SENTENCE_EMBEDDINGS_SIMILARITY_DIR / f"{slug}_similarity.csv"
    ranking_df.to_csv(ranking_path, index=False)

    top_df = pd.DataFrame(top_chunks)
    top_path = SENTENCE_EMBEDDINGS_REPORTS_DIR / f"{slug}_top_chunks.csv"
    top_df.to_csv(top_path, index=False)

    print(f"Ranking de similitud: {ranking_path}")
    print(f"Top-{len(top_chunks)} fragmentos: {top_path}")


def _save_projection_outputs(
    slug: str,
    model_name: str,
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    top_chunks: list[dict[str, Any]],
    projection_k: int,
    run_tsne: bool,
) -> None:
    if not top_chunks:
        return

    selected_chunks = top_chunks[: max(1, projection_k)]
    selected_positions = [int(chunk["chunk_position"]) for chunk in selected_chunks]
    selected_embeddings = chunk_embeddings[selected_positions]
    projection_embeddings = np.vstack(
        [query_embedding.reshape(1, -1), selected_embeddings]
    )
    labels = ["Consulta"] + [
        f"Top {chunk['rank']} ({chunk.get('source_file')}, chunk {chunk.get('chunk_index')})"
        for chunk in selected_chunks
    ]
    roles = ["query"] + ["best_chunk"] + ["chunk"] * max(0, len(selected_chunks) - 1)
    similarity_scores: list[float | None] = [None]
    for chunk in selected_chunks:
        score = chunk.get("similarity_score")
        similarity_scores.append(float(score) if score is not None else None)

    _save_single_projection(
        slug=slug,
        model_name=model_name,
        method="pca",
        projection_embeddings=projection_embeddings,
        labels=labels,
        roles=roles,
        similarity_scores=similarity_scores,
    )
    if run_tsne:
        _save_single_projection(
            slug=slug,
            model_name=model_name,
            method="tsne",
            projection_embeddings=projection_embeddings,
            labels=labels,
            roles=roles,
            similarity_scores=similarity_scores,
        )


def _save_single_projection(
    slug: str,
    model_name: str,
    method: str,
    projection_embeddings: np.ndarray,
    labels: list[str],
    roles: list[str],
    similarity_scores: list[float | None],
) -> None:
    SENTENCE_EMBEDDINGS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SENTENCE_EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if method == "pca":
        points = reduce_embeddings_pca(projection_embeddings)
    elif method == "tsne":
        points = reduce_embeddings_tsne(projection_embeddings)
    else:
        raise ValueError(f"Metodo de reduccion no soportado: {method}")

    title = f"{model_name} - {method.upper()} consulta vs fragmentos"
    figure_path = SENTENCE_EMBEDDINGS_FIGURES_DIR / f"{slug}_{method}_query_match.png"
    coordinates_path = (
        SENTENCE_EMBEDDINGS_OUTPUT_DIR / f"{slug}_{method}_query_match.csv"
    )

    plot_query_match_projection(
        points,
        labels=labels,
        roles=roles,
        title=title,
        output_path=figure_path,
    )
    coordinates_df = pd.DataFrame(
        {
            "model_name": model_name,
            "method": method,
            "label": labels,
            "role": roles,
            "similarity_score": similarity_scores,
            "x": points[:, 0],
            "y": points[:, 1],
        }
    )
    coordinates_df.to_csv(coordinates_path, index=False)
    print(f"Figura {method.upper()}: {figure_path}")
    print(f"Coordenadas {method.upper()}: {coordinates_path}")


def _build_similarity_ranking_dataframe(
    model_name: str,
    query: str,
    chunks: list[dict[str, Any]],
    scores: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for position, (chunk, score) in enumerate(zip(chunks, scores, strict=True)):
        text = str(chunk.get("text", ""))
        rows.append(
            {
                "model_name": model_name,
                "query": query,
                "chunk_position": position,
                "chunk_id": chunk.get("chunk_id"),
                "source_file": chunk.get("source_file"),
                "source_path": chunk.get("source_path"),
                "chunk_index": chunk.get("chunk_index"),
                "n_characters": chunk.get("n_characters"),
                "similarity_score": float(score),
                "text_preview": " ".join(text.split())[:500],
            }
        )
    return pd.DataFrame(rows).sort_values("similarity_score", ascending=False)


def _parse_models(value: str | None) -> list[str]:
    if not value:
        return SENTENCE_EMBEDDING_MODELS
    return [model.strip() for model in value.split(",") if model.strip()]


def _ensure_output_dirs() -> None:
    for directory in (
        SENTENCE_EMBEDDINGS_OUTPUT_DIR,
        SENTENCE_EMBEDDINGS_REPORTS_DIR,
        SENTENCE_EMBEDDINGS_FIGURES_DIR,
        SENTENCE_EMBEDDINGS_SIMILARITY_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
