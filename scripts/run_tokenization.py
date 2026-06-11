"""Ejecutar comparacion de tokenizacion sobre datasets locales."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR
from src.dataset_loading import (
    get_first_sentences_by_split,
    load_ancora_local,
    load_conll2002_local,
    split_ancora_70_15_15,
)
from src.tokenization import (
    build_tokenization_comparison_dataframe,
    load_sentencepiece_tokenizer,
    load_wordpiece_tokenizer,
)


TOKENIZATION_OUTPUT_DIR = OUTPUTS_DIR / "tokenization"


def main() -> None:
    """Cargar datos locales, tokenizar ejemplos y guardar comparaciones."""

    TOKENIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Carga local de datasets ===")
    conll2002_splits = load_conll2002_local()
    ancora_sentences = load_ancora_local()
    ancora_splits = split_ancora_70_15_15(ancora_sentences)

    print("\n=== Seleccion de ejemplos ===")
    conll2002_examples = get_first_sentences_by_split(conll2002_splits, n=3)
    ancora_examples = get_first_sentences_by_split(ancora_splits, n=3)
    _print_selected_counts("Conll2002", conll2002_examples)
    _print_selected_counts("Ancora", ancora_examples)

    print("\n=== Carga de tokenizadores ===")
    print("Cargando WordPiece-BETO: dccuchile/bert-base-spanish-wwm-cased")
    beto_tokenizer = load_wordpiece_tokenizer()
    print("Cargando SentencePiece-T5: google-t5/t5-small")
    t5_tokenizer = load_sentencepiece_tokenizer()

    print("\n=== Tokenizacion y comparacion ===")
    conll2002_comparison = build_tokenization_comparison_dataframe(
        "conll2002",
        conll2002_examples,
        beto_tokenizer=beto_tokenizer,
        t5_tokenizer=t5_tokenizer,
    )
    ancora_comparison = build_tokenization_comparison_dataframe(
        "ancora",
        ancora_examples,
        beto_tokenizer=beto_tokenizer,
        t5_tokenizer=t5_tokenizer,
    )

    _save_comparison(
        conll2002_comparison,
        TOKENIZATION_OUTPUT_DIR / "conll2002_tokenization_comparison",
    )
    _save_comparison(
        ancora_comparison,
        TOKENIZATION_OUTPUT_DIR / "ancora_tokenization_comparison",
    )

    print("\nListo. Comparaciones de tokenizacion guardadas correctamente.")


def _print_selected_counts(
    dataset_name: str,
    examples_by_split: dict[str, list[dict]],
) -> None:
    total = sum(len(examples) for examples in examples_by_split.values())
    print(f"[{dataset_name}] {total} sentencias usadas para comparacion")
    for split_name, examples in examples_by_split.items():
        print(f"  - {split_name}: {len(examples)}")


def _save_comparison(dataframe, output_base_path: Path) -> None:
    csv_path = output_base_path.with_suffix(".csv")
    json_path = output_base_path.with_suffix(".json")

    dataframe.to_csv(csv_path, index=False)
    dataframe.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print(f"Guardado CSV:  {csv_path}")
    print(f"Guardado JSON: {json_path}")


if __name__ == "__main__":
    main()
