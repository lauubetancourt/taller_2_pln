"""Analizar resultados de tokenizacion ya generados."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import OUTPUTS_DIR
from src.tokenization_analysis import (
    compare_tokenizers_sentence_level,
    compute_enriched_tokenization_metrics,
    get_most_fragmented_sentences,
    summarize_fragmentation_by_tokenizer,
)


TOKENIZATION_OUTPUT_DIR = OUTPUTS_DIR / "tokenization"


def main() -> None:
    """Cargar CSV base, enriquecer metricas y guardar analisis."""

    conll_path = TOKENIZATION_OUTPUT_DIR / "conll2002_tokenization_comparison.csv"
    ancora_path = TOKENIZATION_OUTPUT_DIR / "ancora_tokenization_comparison.csv"
    _ensure_input_files_exist(conll_path, ancora_path)

    conll_df = pd.read_csv(conll_path)
    ancora_df = pd.read_csv(ancora_path)
    combined_df = pd.concat([conll_df, ancora_df], ignore_index=True)

    enriched_df = compute_enriched_tokenization_metrics(combined_df)
    enriched_df = compare_tokenizers_sentence_level(enriched_df)
    global_summary_df, by_dataset_summary_df = summarize_fragmentation_by_tokenizer(
        enriched_df
    )
    most_fragmented_wordpiece_df = get_most_fragmented_sentences(
        enriched_df,
        tokenizer="wordpiece",
        top_n=5,
    )
    most_fragmented_sentencepiece_df = get_most_fragmented_sentences(
        enriched_df,
        tokenizer="sentencepiece",
        top_n=5,
    )

    TOKENIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_dataframe(
        enriched_df,
        TOKENIZATION_OUTPUT_DIR / "tokenization_enriched_analysis",
        save_json=True,
    )
    _save_dataframe(
        global_summary_df,
        TOKENIZATION_OUTPUT_DIR / "tokenization_summary_global",
    )
    _save_dataframe(
        by_dataset_summary_df,
        TOKENIZATION_OUTPUT_DIR / "tokenization_summary_by_dataset",
    )
    _save_dataframe(
        most_fragmented_wordpiece_df,
        TOKENIZATION_OUTPUT_DIR / "most_fragmented_wordpiece",
    )
    _save_dataframe(
        most_fragmented_sentencepiece_df,
        TOKENIZATION_OUTPUT_DIR / "most_fragmented_sentencepiece",
    )

    _print_console_summary(enriched_df)


def _ensure_input_files_exist(*paths: Path) -> None:
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "No se encontraron los CSV base de tokenizacion:\n"
            f"{missing}\n\n"
            "Ejecuta primero: python scripts/run_tokenization.py"
        )


def _save_dataframe(dataframe: pd.DataFrame, output_base_path: Path, save_json: bool = False) -> None:
    csv_path = output_base_path.with_suffix(".csv")
    dataframe.to_csv(csv_path, index=False)
    print(f"Guardado CSV:  {csv_path}")

    if save_json:
        json_path = output_base_path.with_suffix(".json")
        dataframe.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"Guardado JSON: {json_path}")


def _print_console_summary(enriched_df: pd.DataFrame) -> None:
    counts = enriched_df["tokenizador_menos_fragmentado"].value_counts()
    print("\n=== Resumen interpretativo de tokenizacion ===")
    print(f"Sentencias analizadas: {len(enriched_df)}")
    print(
        "Sentencias informativas: "
        f"{int(enriched_df['es_sentencia_informativa'].sum())}"
    )
    print(
        "Promedio fragmentacion WordPiece sin especiales: "
        f"{enriched_df['razon_fragmentacion_wordpiece_sin_especiales'].mean():.3f}"
    )
    print(
        "Promedio fragmentacion SentencePiece sin especiales: "
        f"{enriched_df['razon_fragmentacion_sentencepiece_sin_especiales'].mean():.3f}"
    )
    print(f"Total <unk> WordPiece: {int(enriched_df['n_unknown_wordpiece'].sum())}")
    print(
        "Total <unk> SentencePiece: "
        f"{int(enriched_df['n_unknown_sentencepiece'].sum())}"
    )
    print(f"WordPiece menos fragmentado: {int(counts.get('wordpiece', 0))}")
    print(f"SentencePiece menos fragmentado: {int(counts.get('sentencepiece', 0))}")
    print(f"Empates: {int(counts.get('empate', 0))}")


if __name__ == "__main__":
    main()
