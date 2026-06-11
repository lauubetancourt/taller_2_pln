"""Analisis interpretativo de resultados de tokenizacion.

Este modulo trabaja sobre los CSV/JSON ya generados por el bloque de
tokenizacion. No modifica datos originales ni vuelve a ejecutar tokenizadores.
"""

from __future__ import annotations

import ast
import csv
import math
from io import StringIO
from typing import Any

import pandas as pd


WORDPIECE_SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}
SENTENCEPIECE_SPECIAL_TOKENS = {"</s>", "<pad>"}


def parse_token_list(value: Any) -> list[str]:
    """Convertir una celda de tokens a ``list[str]`` de forma segura.

    Acepta listas reales, tuplas y strings con representacion de lista. Si el
    CSV contiene una lista poco estricta, por ejemplo ``[a, b, c]``, se intenta
    una separacion conservadora por comas.
    """

    if isinstance(value, list):
        return [str(token) for token in value]
    if isinstance(value, tuple):
        return [str(token) for token in value]
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, list | tuple):
        return [str(token) for token in parsed]
    if isinstance(parsed, str):
        return [parsed]

    if text.startswith("[") and text.endswith("]"):
        return _parse_loose_list(text[1:-1])

    return [text]


def remove_special_tokens(tokens: list[str], tokenizer_type: str) -> list[str]:
    """Eliminar tokens especiales de un tokenizador sin remover ``<unk>``."""

    tokenizer_key = tokenizer_type.lower()
    if tokenizer_key in {"wordpiece", "beto", "wordpiece-beto"}:
        special_tokens = WORDPIECE_SPECIAL_TOKENS
    elif tokenizer_key in {"sentencepiece", "t5", "sentencepiece-t5"}:
        special_tokens = SENTENCEPIECE_SPECIAL_TOKENS
    else:
        raise ValueError(
            "tokenizer_type debe ser 'wordpiece'/'beto' o 'sentencepiece'/'t5'."
        )

    return [token for token in tokens if token not in special_tokens]


def count_unknown_tokens(tokens: list[str]) -> int:
    """Contar apariciones de ``<unk>`` en una lista de tokens."""

    return sum(1 for token in tokens if token == "<unk>")


def get_wordpiece_continuations(tokens: list[str]) -> list[str]:
    """Retornar subtokens WordPiece que continuan una palabra con prefijo ``##``."""

    return [token for token in tokens if token.startswith("##")]


def get_sentencepiece_space_markers(tokens: list[str]) -> list[str]:
    """Retornar tokens SentencePiece que empiezan con la marca ``▁``."""

    return [token for token in tokens if token.startswith("▁")]


def compute_enriched_tokenization_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Agregar metricas interpretativas a una tabla de comparacion.

    Calcula conteos con y sin tokens especiales, desconocidos, continuaciones
    WordPiece, marcas SentencePiece y banderas para sentencias cortas e
    informativas.
    """

    enriched = _normalize_comparison_columns(df).copy()

    enriched["tokens_originales"] = enriched["tokens_originales"].apply(parse_token_list)
    enriched["tokens_wordpiece_beto"] = enriched["tokens_wordpiece_beto"].apply(
        parse_token_list
    )
    enriched["tokens_sentencepiece_t5"] = enriched["tokens_sentencepiece_t5"].apply(
        parse_token_list
    )

    enriched["n_tokens_originales"] = enriched["tokens_originales"].apply(len)
    enriched["n_subtokens_wordpiece"] = enriched["tokens_wordpiece_beto"].apply(len)
    enriched["n_subtokens_sentencepiece"] = enriched["tokens_sentencepiece_t5"].apply(len)

    enriched["tokens_wordpiece_sin_especiales"] = enriched[
        "tokens_wordpiece_beto"
    ].apply(lambda tokens: remove_special_tokens(tokens, "wordpiece"))
    enriched["tokens_sentencepiece_sin_especiales"] = enriched[
        "tokens_sentencepiece_t5"
    ].apply(lambda tokens: remove_special_tokens(tokens, "sentencepiece"))

    enriched["n_subtokens_wordpiece_sin_especiales"] = enriched[
        "tokens_wordpiece_sin_especiales"
    ].apply(len)
    enriched["n_subtokens_sentencepiece_sin_especiales"] = enriched[
        "tokens_sentencepiece_sin_especiales"
    ].apply(len)

    enriched["razon_fragmentacion_wordpiece"] = _safe_ratio_series(
        enriched["n_subtokens_wordpiece"],
        enriched["n_tokens_originales"],
    )
    enriched["razon_fragmentacion_sentencepiece"] = _safe_ratio_series(
        enriched["n_subtokens_sentencepiece"],
        enriched["n_tokens_originales"],
    )
    enriched["razon_fragmentacion_wordpiece_sin_especiales"] = _safe_ratio_series(
        enriched["n_subtokens_wordpiece_sin_especiales"],
        enriched["n_tokens_originales"],
    )
    enriched["razon_fragmentacion_sentencepiece_sin_especiales"] = _safe_ratio_series(
        enriched["n_subtokens_sentencepiece_sin_especiales"],
        enriched["n_tokens_originales"],
    )

    enriched["n_unknown_wordpiece"] = enriched["tokens_wordpiece_beto"].apply(
        count_unknown_tokens
    )
    enriched["n_unknown_sentencepiece"] = enriched["tokens_sentencepiece_t5"].apply(
        count_unknown_tokens
    )
    enriched["subtokens_con_prefijo_wordpiece"] = enriched[
        "tokens_wordpiece_beto"
    ].apply(get_wordpiece_continuations)
    enriched["n_continuaciones_wordpiece"] = enriched[
        "subtokens_con_prefijo_wordpiece"
    ].apply(len)
    enriched["tokens_con_marca_sentencepiece"] = enriched[
        "tokens_sentencepiece_t5"
    ].apply(get_sentencepiece_space_markers)
    enriched["n_marcas_sentencepiece"] = enriched["tokens_con_marca_sentencepiece"].apply(
        len
    )

    enriched["es_sentencia_corta"] = enriched["n_tokens_originales"] <= 1
    enriched["es_sentencia_informativa"] = enriched["n_tokens_originales"] > 3
    return enriched


def summarize_fragmentation_by_tokenizer(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resumir fragmentacion global y por dataset."""

    enriched = df.copy()
    if "es_sentencia_informativa" not in enriched.columns:
        enriched = compute_enriched_tokenization_metrics(enriched)

    global_summary = _summarize_group(enriched, group_label="global")
    by_dataset_summary = pd.DataFrame(
        [
            {"dataset": dataset, **_summary_values(group)}
            for dataset, group in enriched.groupby("dataset", dropna=False)
        ]
    )
    return global_summary, by_dataset_summary


def get_most_fragmented_sentences(
    df: pd.DataFrame,
    tokenizer: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Obtener las sentencias mas fragmentadas por tokenizador."""

    tokenizer_key = tokenizer.lower()
    if tokenizer_key in {"wordpiece", "beto", "wordpiece-beto"}:
        ratio_column = "razon_fragmentacion_wordpiece_sin_especiales"
    elif tokenizer_key in {"sentencepiece", "t5", "sentencepiece-t5"}:
        ratio_column = "razon_fragmentacion_sentencepiece_sin_especiales"
    else:
        raise ValueError("tokenizer debe ser 'wordpiece' o 'sentencepiece'.")

    enriched = df.copy()
    if ratio_column not in enriched.columns:
        enriched = compute_enriched_tokenization_metrics(enriched)

    columns = [
        "dataset",
        "split",
        "indice_sentencia",
        "texto_reconstruido",
        "n_tokens_originales",
        ratio_column,
        "tokens_wordpiece_beto",
        "tokens_sentencepiece_t5",
    ]
    available_columns = [column for column in columns if column in enriched.columns]
    return enriched.sort_values(ratio_column, ascending=False).head(top_n)[
        available_columns
    ]


def compare_tokenizers_sentence_level(df: pd.DataFrame) -> pd.DataFrame:
    """Agregar que tokenizador fragmenta menos por sentencia."""

    compared = df.copy()
    if "razon_fragmentacion_wordpiece_sin_especiales" not in compared.columns:
        compared = compute_enriched_tokenization_metrics(compared)

    def choose_tokenizer(row: pd.Series) -> str:
        wordpiece_ratio = row["razon_fragmentacion_wordpiece_sin_especiales"]
        sentencepiece_ratio = row["razon_fragmentacion_sentencepiece_sin_especiales"]
        if pd.isna(wordpiece_ratio) or pd.isna(sentencepiece_ratio):
            return "empate"
        if wordpiece_ratio < sentencepiece_ratio:
            return "wordpiece"
        if sentencepiece_ratio < wordpiece_ratio:
            return "sentencepiece"
        return "empate"

    compared["tokenizador_menos_fragmentado"] = compared.apply(choose_tokenizer, axis=1)
    return compared


def _parse_loose_list(text: str) -> list[str]:
    if not text.strip():
        return []

    try:
        reader = csv.reader(StringIO(text), skipinitialspace=True)
        values = next(reader)
    except csv.Error:
        values = text.split(",")

    return [
        value.strip().strip("'\"")
        for value in values
        if value.strip().strip("'\"")
    ]


def _normalize_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "example_index": "indice_sentencia",
        "text": "texto_reconstruido",
        "original_tokens": "tokens_originales",
        "beto_tokens": "tokens_wordpiece_beto",
        "t5_tokens": "tokens_sentencepiece_t5",
        "beto_token_count": "n_subtokens_wordpiece",
        "t5_token_count": "n_subtokens_sentencepiece",
    }
    normalized = df.rename(columns=rename_map).copy()
    required_columns = {
        "dataset",
        "split",
        "indice_sentencia",
        "tokens_originales",
        "texto_reconstruido",
        "tokens_wordpiece_beto",
        "tokens_sentencepiece_t5",
    }
    missing_columns = sorted(required_columns - set(normalized.columns))
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas para el analisis de tokenizacion: "
            f"{missing_columns}."
        )
    return normalized


def _safe_ratio_series(numerators: pd.Series, denominators: pd.Series) -> pd.Series:
    ratios = numerators / denominators.replace(0, pd.NA)
    return ratios.astype("Float64").round(3)


def _summarize_group(group: pd.DataFrame, group_label: str) -> pd.DataFrame:
    values = _summary_values(group)
    return pd.DataFrame([{"grupo": group_label, **values}])


def _summary_values(group: pd.DataFrame) -> dict[str, float | int]:
    not_short = group[~group["es_sentencia_corta"]]
    informative = group[group["es_sentencia_informativa"]]

    return {
        "n_sentencias": int(len(group)),
        "n_sentencias_informativas": int(group["es_sentencia_informativa"].sum()),
        "promedio_wordpiece_con_especiales": _mean_or_na(
            group["razon_fragmentacion_wordpiece"]
        ),
        "promedio_sentencepiece_con_especiales": _mean_or_na(
            group["razon_fragmentacion_sentencepiece"]
        ),
        "promedio_wordpiece_sin_especiales": _mean_or_na(
            group["razon_fragmentacion_wordpiece_sin_especiales"]
        ),
        "promedio_sentencepiece_sin_especiales": _mean_or_na(
            group["razon_fragmentacion_sentencepiece_sin_especiales"]
        ),
        "promedio_wordpiece_sin_sentencias_cortas": _mean_or_na(
            not_short["razon_fragmentacion_wordpiece_sin_especiales"]
        ),
        "promedio_sentencepiece_sin_sentencias_cortas": _mean_or_na(
            not_short["razon_fragmentacion_sentencepiece_sin_especiales"]
        ),
        "promedio_wordpiece_sentencias_informativas": _mean_or_na(
            informative["razon_fragmentacion_wordpiece_sin_especiales"]
        ),
        "promedio_sentencepiece_sentencias_informativas": _mean_or_na(
            informative["razon_fragmentacion_sentencepiece_sin_especiales"]
        ),
        "total_unknown_wordpiece": int(group["n_unknown_wordpiece"].sum()),
        "total_unknown_sentencepiece": int(group["n_unknown_sentencepiece"].sum()),
        "promedio_continuaciones_wordpiece": _mean_or_na(
            group["n_continuaciones_wordpiece"]
        ),
        "promedio_marcas_sentencepiece": _mean_or_na(group["n_marcas_sentencepiece"]),
    }


def _mean_or_na(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return round(float(series.mean()), 3)
