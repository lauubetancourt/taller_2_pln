# Taller PLN

Proyecto base para desarrollar localmente un taller de Procesamiento del Lenguaje Natural desde Jupyter Notebook, manteniendo la logica pesada en scripts y modulos reutilizables de Python.

El taller cubre tres bloques:

1. Tokenizacion de datasets con Conll2002, Ancora, WordPiece con BETO y SentencePiece con `t5-small`.
2. Entrenamiento y analisis de embeddings distribucionales con Word2Vec y FastText sobre Spanish Billion Clean en modo `streaming=True`.
3. Procesamiento de PDFs, chunking, embeddings semanticos y busqueda por similitud coseno.

## Estructura

```text
taller_pln/
├── notebooks/
│   └── taller_pln.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset_loading.py
│   ├── preprocessing.py
│   ├── tokenization.py
│   ├── word_embeddings.py
│   ├── pdf_processing.py
│   ├── sentence_embeddings.py
│   ├── similarity.py
│   └── visualization.py
├── scripts/
│   ├── run_tokenization.py
│   ├── train_word2vec.py
│   ├── train_fasttext.py
│   ├── process_pdfs.py
│   └── generate_sentence_embeddings.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── pdfs/
├── outputs/
│   ├── tokenization/
│   ├── embeddings/
│   ├── figures/
│   ├── similarity/
│   └── reports/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalacion

Desde la raiz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name taller-pln --display-name "Python (taller-pln)"
```

En Windows PowerShell, la activacion del entorno cambia a:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Ejecutar el notebook

```bash
jupyter notebook notebooks/taller_pln.ipynb
```

El notebook debe funcionar como orquestador: probar funciones, mostrar ejemplos pequenos, generar tablas, graficar y redactar analisis. Las funciones costosas deben vivir en `src/` y ejecutarse desde scripts cuando sea necesario.

## Ejecutar scripts desde terminal

Tokenizacion local:

```bash
python scripts/run_tokenization.py
python scripts/analyze_tokenization.py
```

Punto 2, embeddings distribucionales:

```bash
# 1. Preparar muestras con streaming
python3 scripts/prepare_spanish_billion_sample.py --max-sentences 100 --output-name sample_100
python3 scripts/prepare_spanish_billion_sample.py --max-sentences 5000 --output-name sample_5000
python3 scripts/prepare_spanish_billion_sample.py --max-sentences 10000 --output-name sample_10000

# 2. Entrenar Word2Vec y FastText
python3 scripts/train_word_embeddings.py --sentences-file data/processed/sentences/spanish_billion_clean_sample_100.jsonl --output-suffix sample_100 --workers 4
python3 scripts/train_word_embeddings.py --sentences-file data/processed/sentences/spanish_billion_clean_sample_5000.jsonl --output-suffix sample_5000 --workers 4
python3 scripts/train_word_embeddings.py --sentences-file data/processed/sentences/spanish_billion_clean_sample_10000.jsonl --output-suffix sample_10000 --workers 4

# 3. Analizar similaridad y OOV
python3 scripts/analyze_word_similarity.py \
  --word2vec-model models/word2vec_sample_10000.model \
  --fasttext-model models/fasttext_sample_10000.model \
  --words gobierno,presidente,ciudad,equipo,mercado \
  --oov-words hiperconectividad,futbolísticamente

# 4. Visualizar PCA y t-SNE
python3 scripts/visualize_word_embeddings.py \
  --word2vec-model models/word2vec_sample_10000.model \
  --fasttext-model models/fasttext_sample_10000.model \
  --training-size-label sample_10000 \
  --max-words 200 \
  --run-pca \
  --run-tsne
```

Punto 3, PDFs y embeddings semanticos:

```bash
# 1. Copiar o descargar los PDFs del taller en data/pdfs/

# 2. Extraer texto con PyMuPDF4LLM y generar chunks con LangChain
python3 scripts/process_pdfs.py \
  --pdf-dir data/pdfs \
  --output-file data/processed/pdf_chunks.jsonl

# 3. Generar embeddings, calcular similitud coseno y crear PCA/t-SNE
python3 scripts/generate_sentence_embeddings.py \
  --chunks-file data/processed/pdf_chunks.jsonl \
  --query "¿Cuál es la idea principal de los documentos procesados?"
```

El segundo script evalua por defecto los cuatro modelos solicitados en el enunciado. Para probar solo uno durante desarrollo:

```bash
python3 scripts/generate_sentence_embeddings.py \
  --models sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --skip-tsne
```

## Nota sobre memoria

No cargues datasets grandes completos en memoria si no es necesario. Para corpus masivos como Spanish Billion Clean, usa `streaming=True`, procesamiento por lotes e iteradores. Ajusta `N_WORKERS`, tamanos de lote y subconjuntos de visualizacion segun la memoria y CPU de la maquina local.

Durante el entrenamiento, Gensim usa `workers` para aprovechar varios nucleos. No se aplica multiprocessing adicional sobre el streaming completo para evitar saturar la RAM.
