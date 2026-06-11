"""Carga local y particion de datasets para el taller.

El flujo principal del bloque de tokenizacion trabaja con archivos descargados
manualmente por el estudiante en ``data/raw/conll2002`` y ``data/raw/ancora``.
Las funciones de Hugging Face quedan como alternativa opcional, pero no son el
camino usado por defecto.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import ast
import random
import re
from typing import Any

import pandas as pd

from .config import ANCORA_RAW_DIR, CONLL2002_RAW_DIR, SEED


Sentence = dict[str, Any]
DatasetSplits = dict[str, list[Sentence]]

SUPPORTED_TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
SUPPORTED_CONLL_EXTENSIONS = {".txt", ".conll", ".iob", ".bio"}
SUPPORTED_DATA_EXTENSIONS = SUPPORTED_TABLE_EXTENSIONS | SUPPORTED_CONLL_EXTENSIONS

SPLIT_ALIASES = {
    "train": ("train", "training", "esp.train", "spanish.train"),
    "validation": ("validation", "valid", "val", "dev", "development"),
    "test": ("test", "testing"),
}

GROUPED_TOKEN_COLUMN_CANDIDATES = (
    "tokens",
    "tokenized_sentence",
    "tokenized_sentences",
    "words",
    "sentence_tokens",
    "oracion_tokens",
    "oración_tokens",
)

TEXT_COLUMN_CANDIDATES = (
    "sentence",
    "sentencia",
    "oracion",
    "oración",
    "text",
    "texto",
)

TOKEN_COLUMN_CANDIDATES = (
    "word",
    "token",
    "form",
    "palabra",
    "term",
)

SENTENCE_ID_COLUMN_CANDIDATES = (
    "sentence #",
    "sentence_id",
    "sentence id",
    "sent_id",
    "sentid",
    "id_sentence",
    "id_sentencia",
    "oracion_id",
    "oración_id",
    "sentence",
)


def load_conll2002_es(split: str | None = None) -> Any:
    """Cargar Conll2002 desde Hugging Face Datasets como alternativa opcional.

    El taller usa por defecto ``load_conll2002_local``. Esta funcion permanece
    disponible para comparar o recuperar datos desde Hugging Face cuando el
    usuario lo decida explicitamente.
    """

    from datasets import load_dataset

    if split is None:
        return load_dataset("conll2002", "es")
    return load_dataset("conll2002", "es", split=split)


def load_ancora_dataset(split: str | None = None) -> Any:
    """Cargar Ancora desde Hugging Face Datasets como alternativa opcional."""

    from datasets import load_dataset

    if split is None:
        return load_dataset("ancora")
    return load_dataset("ancora", split=split)


def load_conll2002_local(data_dir: str | Path = CONLL2002_RAW_DIR) -> DatasetSplits:
    """Cargar Conll2002 desde archivos locales separados por split.

    La funcion busca archivos con nombres aproximados de train, validation/dev y
    test. Para archivos tipo CoNLL usa un parser basico donde el primer campo de
    cada linea no vacia es el token y una linea vacia separa sentencias.
    """

    data_path = Path(data_dir)
    files = _list_supported_files(data_path)
    print(f"[Conll2002] Archivos encontrados en {data_path}:")
    for file_path in files:
        print(f"  - {file_path.name}")

    files_by_split = _detect_split_files(files)
    missing = [split for split in ("train", "validation", "test") if split not in files_by_split]
    if missing:
        found = ", ".join(path.name for path in files)
        raise FileNotFoundError(
            "No se encontraron archivos para los splits "
            f"{missing} en {data_path}. Archivos disponibles: {found or 'ninguno'}."
        )

    dataset_splits: DatasetSplits = {}
    for split_name in ("train", "validation", "test"):
        file_path = files_by_split[split_name]
        dataset_splits[split_name] = _load_sentences_from_file(file_path)
        print(
            f"[Conll2002] {split_name}: {len(dataset_splits[split_name])} "
            f"sentencias cargadas desde {file_path.name}"
        )

    return dataset_splits


def load_ancora_local(data_dir: str | Path = ANCORA_RAW_DIR) -> list[Sentence]:
    """Cargar Ancora desde un archivo local ``.xlsx`` o ``.csv``.

    Si hay columnas de tokens agrupados se usan directamente. Si hay columna de
    token y columna de identificador de sentencia, se agrupa por sentencia.
    """

    data_path = Path(data_dir)
    files = _list_supported_files(data_path, extensions=SUPPORTED_TABLE_EXTENSIONS)
    print(f"[Ancora] Archivos encontrados en {data_path}:")
    for file_path in files:
        print(f"  - {file_path.name}")

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos .xlsx, .xls, .csv o .tsv en {data_path}."
        )

    file_path = _select_ancora_file(files)
    dataframe = _read_table_file(file_path)
    inspect_ancora_dataframe(dataframe, source_name=file_path.name)
    sentences = build_ancora_sentences_from_dataframe(dataframe)
    print(f"[Ancora] {len(sentences)} sentencias cargadas desde {file_path.name}")
    return sentences


def inspect_ancora_dataframe(
    dataframe: pd.DataFrame,
    source_name: str | None = None,
    n_rows: int = 5,
) -> None:
    """Imprimir columnas disponibles y primeras filas de un archivo Ancora."""

    label = f" de {source_name}" if source_name else ""
    print(f"[Ancora] Columnas disponibles{label}: {list(dataframe.columns)}")
    print(f"[Ancora] Primeras {min(n_rows, len(dataframe))} filas:")
    print(dataframe.head(n_rows).to_string(index=False))


def build_ancora_sentences_from_dataframe(
    dataframe: pd.DataFrame,
    token_column: str | None = None,
    sentence_id_column: str | None = None,
    grouped_tokens_column: str | None = None,
) -> list[Sentence]:
    """Construir sentencias de Ancora desde columnas inferidas o declaradas.

    Parameters
    ----------
    dataframe:
        Tabla leida desde CSV o Excel.
    token_column:
        Columna que contiene un token por fila.
    sentence_id_column:
        Columna que identifica a que sentencia pertenece cada token.
    grouped_tokens_column:
        Columna que contiene tokens ya agrupados o texto de sentencia.

    Raises
    ------
    ValueError
        Si no se puede inferir automaticamente como reconstruir las sentencias.
    """

    dataframe = dataframe.dropna(how="all").copy()
    columns = list(dataframe.columns)

    grouped_column = grouped_tokens_column or _find_column(
        columns, GROUPED_TOKEN_COLUMN_CANDIDATES
    )
    if grouped_column:
        return _sentences_from_grouped_column(dataframe, grouped_column)

    token_col = token_column or _find_column(columns, TOKEN_COLUMN_CANDIDATES)
    sentence_id_col = sentence_id_column or _find_column(
        columns, SENTENCE_ID_COLUMN_CANDIDATES
    )
    if token_col and sentence_id_col:
        return _sentences_from_token_rows(dataframe, token_col, sentence_id_col)

    text_column = _find_column(columns, TEXT_COLUMN_CANDIDATES)
    if text_column and not _looks_like_identifier_column(dataframe[text_column]):
        return _sentences_from_text_column(dataframe, text_column)

    raise ValueError(
        "No se pudo inferir automaticamente como construir sentencias de Ancora. "
        f"Columnas encontradas: {columns}. Define manualmente token_column y "
        "sentence_id_column, o grouped_tokens_column."
    )


def split_ancora_70_15_15(sentences: Sequence[Sentence], seed: int = SEED) -> DatasetSplits:
    """Dividir sentencias de Ancora en train/validation/test con proporcion 70/15/15."""

    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)

    n_sentences = len(shuffled)
    train_end = int(n_sentences * 0.70)
    validation_end = train_end + int(n_sentences * 0.15)

    splits = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    print(
        "[Ancora] Split 70/15/15: "
        f"train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}"
    )
    return splits


def get_first_sentences_by_split(
    dataset_splits: DatasetSplits,
    n: int = 3,
) -> DatasetSplits:
    """Obtener las primeras ``n`` sentencias de cada split."""

    selected = {
        split_name: list(sentences[:n])
        for split_name, sentences in dataset_splits.items()
    }
    for split_name, sentences in selected.items():
        print(f"[{split_name}] {len(sentences)} sentencias usadas para comparacion")
    return selected


def split_ancora_dataset(
    dataset: Sequence[Sentence],
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> DatasetSplits:
    """Compatibilidad con el nombre anterior para dividir Ancora.

    El flujo nuevo usa ``split_ancora_70_15_15``. Solo se aceptan proporciones
    70/15/15 para evitar ambiguedad en esta fase del taller.
    """

    if (train_size, validation_size, test_size) != (0.70, 0.15, 0.15):
        raise ValueError("En esta fase solo esta configurado el split 70/15/15.")
    return split_ancora_70_15_15(dataset, seed=seed)


def get_first_sentences(dataset_split: Sequence[Sentence], n: int = 3) -> list[Sentence]:
    """Compatibilidad con el nombre anterior para obtener primeras sentencias."""

    return list(dataset_split[:n])


def _list_supported_files(
    data_dir: Path,
    extensions: set[str] | None = None,
) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe el directorio local: {data_dir}")

    allowed_extensions = extensions or SUPPORTED_DATA_EXTENSIONS
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )


def _detect_split_files(files: Sequence[Path]) -> dict[str, Path]:
    files_by_split: dict[str, Path] = {}
    for split_name in ("train", "validation", "test"):
        matching_files = [
            file_path
            for file_path in files
            if _filename_matches_split(file_path, split_name)
        ]
        if matching_files:
            files_by_split[split_name] = sorted(
                matching_files,
                key=lambda path: (len(path.name), path.name.lower()),
            )[0]
    return files_by_split


def _filename_matches_split(file_path: Path, split_name: str) -> bool:
    normalized_name = _normalize_name(file_path.name)
    normalized_stem = _normalize_name(file_path.stem)
    aliases = [_normalize_name(alias) for alias in SPLIT_ALIASES[split_name]]

    return any(
        alias == normalized_stem
        or alias in normalized_name.split()
        or normalized_stem.endswith(f" {alias}")
        or normalized_stem.startswith(f"{alias} ")
        for alias in aliases
    )


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _load_sentences_from_file(file_path: Path) -> list[Sentence]:
    if file_path.suffix.lower() in SUPPORTED_TABLE_EXTENSIONS:
        dataframe = _read_table_file(file_path)
        return build_ancora_sentences_from_dataframe(dataframe)
    return parse_conll_file(file_path)


def parse_conll_file(file_path: str | Path) -> list[Sentence]:
    """Parsear archivo tipo CoNLL tomando el primer campo como token."""

    path = Path(file_path)
    sentences: list[Sentence] = []
    current_tokens: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                _append_sentence(sentences, current_tokens, source_file=path.name)
                current_tokens = []
                continue
            if line.startswith("#"):
                continue

            token = re.split(r"\s+", line, maxsplit=1)[0]
            if token:
                current_tokens.append(token)

    _append_sentence(sentences, current_tokens, source_file=path.name)
    return sentences


def _append_sentence(
    sentences: list[Sentence],
    tokens: list[str],
    source_file: str | None = None,
) -> None:
    if tokens:
        sentences.append(
            {
                "tokens": list(tokens),
                "text": " ".join(tokens),
                "source_file": source_file,
            }
        )


def _select_ancora_file(files: Sequence[Path]) -> Path:
    excel_files = [path for path in files if path.suffix.lower() in {".xlsx", ".xls"}]
    if excel_files:
        return sorted(excel_files)[0]
    return sorted(files)[0]


def _read_table_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    return pd.read_csv(file_path)


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    normalized_columns = {_normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        normalized_candidate = _normalize_column_name(candidate)
        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]
    return None


def _normalize_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")


def _looks_like_identifier_column(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique(dropna=True) / len(non_null)
    return pd.api.types.is_numeric_dtype(non_null) or unique_ratio < 0.5


def _sentences_from_grouped_column(
    dataframe: pd.DataFrame,
    grouped_column: str,
) -> list[Sentence]:
    sentences: list[Sentence] = []
    for row_index, value in dataframe[grouped_column].dropna().items():
        tokens = _value_to_tokens(value)
        if tokens:
            sentences.append(
                {
                    "tokens": tokens,
                    "text": " ".join(tokens),
                    "sentence_id": row_index,
                    "source_column": grouped_column,
                }
            )
    return sentences


def _sentences_from_text_column(
    dataframe: pd.DataFrame,
    text_column: str,
) -> list[Sentence]:
    sentences: list[Sentence] = []
    for row_index, value in dataframe[text_column].dropna().items():
        tokens = str(value).split()
        if tokens:
            sentences.append(
                {
                    "tokens": tokens,
                    "text": str(value),
                    "sentence_id": row_index,
                    "source_column": text_column,
                }
            )
    return sentences


def _sentences_from_token_rows(
    dataframe: pd.DataFrame,
    token_column: str,
    sentence_id_column: str,
) -> list[Sentence]:
    sentences: list[Sentence] = []
    grouped = dataframe.dropna(subset=[token_column, sentence_id_column]).groupby(
        sentence_id_column,
        sort=False,
    )
    for sentence_id, group in grouped:
        tokens = [str(token) for token in group[token_column].tolist() if str(token).strip()]
        if tokens:
            sentences.append(
                {
                    "tokens": tokens,
                    "text": " ".join(tokens),
                    "sentence_id": sentence_id,
                    "token_column": token_column,
                    "sentence_id_column": sentence_id_column,
                }
            )
    return sentences


def _value_to_tokens(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(token) for token in value if str(token).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list | tuple):
            return [str(token) for token in parsed if str(token).strip()]

    return text.split()
