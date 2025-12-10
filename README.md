# Sistema de Recuperación de Información

**Autor:** (Angel Falcon, Inti Poaquiza) — Código: [`proyectoRI.py`](https://github.com/intiinside/rip1b/blob/main/proyectoRI.py)


**Resumen**

Este proyecto implementa un sistema de Recuperación de Información (IR) orientado a un corpus de noticias (BBC). Está diseñado como una implementación didáctica y reproducible que abarca desde el preprocesamiento de texto y la construcción de un índice invertido hasta tres modelos clásicos de recuperación (Jaccard, TF–IDF y BM25), además de un módulo de evaluación que calcula métricas estándares en IR (Precisión, Exhaustividad / Recall y MAP).

El propósito es proporcionar una base práctica para experimentar con conceptos fundamentales de IR, comparar modelos y estudiar el impacto de decisiones de preprocesamiento en la recuperación.

---

## Tabla de contenido

1. [Características principales](#caracter%C3%ADsticas-principales)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Requisitos y dependencias](#requisitos-y-dependencias)
4. [Instalación y ejecución](#instalaci%C3%B3n-y-ejecuci%C3%B3n)
5. [Descripción del código y funciones principales](#descripci%C3%B3n-del-c%C3%B3digo-y-funciones-principales)
6. [Modo de uso / Ejemplos](#modo-de-uso--ejemplos)
7. [Evaluación del sistema](#evaluaci%C3%B3n-del-sistema)
8. [Buenas prácticas y recomendaciones](#buenas-pr%C3%A1cticas-y-recomendaciones)
9. [Extensiones sugeridas / Trabajo futuro](#extensiones-sugeridas--trabajo-futuro)
10. [Contacto](#contacto)

---

## Características principales

* Carga de un corpus en formato CSV (se asume que la última columna contiene la `description` de la noticia).
* Pipeline de preprocesamiento que incluye: tokenización, normalización (minusculización, eliminación de no-alfabéticos), eliminación de stopwords y stemming (Porter).
* Construcción de un índice invertido simple con frecuencias por documento.
* Implementación de tres modelos de recuperación:

  * Jaccard (similitud entre conjuntos de tokens).
  * TF–IDF con similitud coseno (scikit-learn: `TfidfVectorizer` + `cosine_similarity`).
  * BM25 (implementación mediante la librería `rank_bm25`).
* Interfaz de consola con un menú para inspeccionar el dataset, índice, realizar búsquedas por cualquiera de los tres modelos y evaluar el sistema usando queries y qrels predefinidas.
* Módulo de evaluación que calcula Precisión, Exhaustividad (Recall), Average Precision (AP) por consulta y Mean Average Precision (MAP) por modelo.

---

## Estructura del proyecto

```
rip1b/
├── proyectoRI.py        # Código principal (implementación completa)
├── bbc_news.csv         # Corpus de noticias BBC (CSV)
├── README.md            # (este archivo)
```

> Nota: en el archivo `proyectoRI.py` se encuentran todas las funciones y el menú principal `Sistema_RI()` que coordina la ejecución.

---

## Requisitos y dependencias

Recomendado usar un entorno virtual (`venv`, `conda`) para reproducir los resultados.

* Python 3.8+ (probado en 3.8 — 3.11)

Paquetes necesarios (ejemplo para `pip`):

```
pip install nltk pandas scikit-learn tabulate rank_bm25
```

**Recursos NLTK**

El script descarga automáticamente los recursos NLTK necesarios (`punkt`, `stopwords`) si no están presentes. Sin embargo, puede descargarlos manualmente ejecutando en Python:

```py
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Instalación y ejecución

1. Clonar el repositorio y situarse en la carpeta del proyecto.
2. Crear y activar un entorno virtual (opcional, recomendado):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install nltk pandas scikit-learn tabulate rank_bm25
```

4. Colocar el `bbc_news.csv` en la ruta indicada o modificar la variable `ruta` en la función `Sistema_RI()` para apuntar al archivo.

5. Ejecutar el sistema:

```bash
python proyectoRI.py
```

El sistema mostrará un menú interactivo en la consola con las opciones para inspeccionar el dataset, construir índices, realizar búsquedas y ejecutar la evaluación.

---

## Descripción del código y funciones principales

A continuación se documentan las funciones y componentes más relevantes dentro de `proyectoRI.py` (breve resumen técnico):
```bash
proyectoRI.py
├── Sistema_RI()                            # Menú principal del sistema
│
├── Carga del Corpus
│   ├── cargar_corpus()                     # Lee corpus BBC desde CSV
│   └── mostrar_info_dataset()              # Muestra estadísticas del dataset
│
├── Preprocesamiento
│   ├── inicializar_preprocesamiento()      # Carga stopwords, stemmer, lemmatizer
│   ├── tokenizar()                         # Tokenización
│   ├── normalizar()                        # Limpieza de tokens
│   ├── remover_stopwords()                 # Filtrado de stopwords
│   ├── stemming()                          # Stemming con PorterStemmer
│   └── procesamiento_doc()                 # Pipeline completo para un documento
│
├── Índice Invertido
│   ├── construir_indice()                  # Crea índice invertido
│   └── mostrar__indice_invertido()         # Imprime resumen del índice
│
├── Visualización / Debug
│   └── mostrar_tabla_documentos()          # Original vs preprocesado
│
├── Modelos de Recuperación
│   ├── TF-IDF
│   │   ├── construir_modelo_tfidf()        # Entrena vectorizador TF-IDF
│   │   └── buscar_tfidf()                  # Búsqueda por similitud coseno
│   │
│   ├── BM25
│   │   ├── construir_modelo_bm25()         # Crea instancia BM25Okapi
│   │   └── buscar_bm25()                   # Búsqueda BM25
│   │
│   └── Jaccard
│       └── buscar_jaccard()                # Búsqueda por similitud Jaccard
│
├── Ejecución de Consultas
│   └── ejecutar_consultas()                # Módulo unificado para TF-IDF, BM25 y Jaccard
│
├── Evaluación (IR Metrics)
│   ├── cargar_consultas_evaluacion()       # Queries y Qrels internos
│   ├── calcular_precision_recall()         # P, R y Average Precision
│   └── evaluar_sistema()                   # Evalúa MAP para cada modelo
│
└── Utilidades de Evaluación
    └── mostrar_queries_qrels()             # Visualiza Queries y Qrels usados en evaluación

```

### PARTE 0 — Carga del corpus

* `cargar_corpus(ruta)`: Lee un CSV y devuelve una lista `corpus` con el texto (asume que la `description` está en la última columna). Imprime el número de documentos cargados.
* `mostrar_info_dataset(corpus, n=10)`: Muestra información general del dataset y previsualiza los primeros y últimos `n` documentos.

### PARTE 1 — Preprocesamiento

* Inicialización de recursos NLTK y descarga de `punkt` y `stopwords` si fueran necesarios.
* `inicializar_preprocesamiento()`: Devuelve un diccionario con `stemmer`, `lemmatizer` y `stop_words` para usar en el pipeline.
* `tokenizar(documento)`: Tokeniza y pasa a minúsculas usando `word_tokenize` de NLTK.
* `normalizar(tokenizacion)`: Filtra tokens para quedarse sólo con cadenas alfabéticas (`token.isalpha()`).
* `remover_stopwords(tokenizacion, stop_words)`: Elimina stopwords en inglés.
* `stemming(tokenizacion, stemmer)`: Aplica `PorterStemmer` a los tokens.
* `procesamiento_doc(documento, preprocesamientod)`: Pipeline completo que integra los pasos anteriores para devolver una lista de tokens procesados.

### Construcción del índice invertido

* `construir_indice(doc_preprocesados)`: Crea un diccionario `{term: {doc_id: freq}}` con la frecuencia de término por documento.
* `mostrar__indice_invertido(indice_invertido, num_terms=15)`: Presenta un resumen tabulado (con `tabulate` y `pandas`) de los términos más y menos frecuentes en el índice.

### Visualización comparativa

* `mostrar_tabla_documentos(corpus, doc_procesados, num_docs=20)`: Muestra una tabla comparativa (texto original vs texto preprocesado) para `num_docs` documentos.

### PARTE 2 — Modelos de Recuperación

* **TF–IDF**

  * `construir_modelo_tfidf(docs_procesados)`: Ajusta `TfidfVectorizer` sobre los documentos (listas de tokens convertidas a cadenas) y devuelve el vectorizador y la matriz TF–IDF.
  * `buscar_tfidf(vectorizador_tfidf, matriz_tfidf, consulta_procesada_str)`: Vectoriza la consulta y calcula similitud coseno con la matriz de documentos; devuelve una lista ordenada `(doc_id, score)`.

* **BM25**

  * `construir_modelo_bm25(docs_procesados)`: Ajusta `rank_bm25.BM25Okapi` sobre los documentos tokenizados.
  * `buscar_bm25(modelo_bm25, consulta_procesada_str)`: Tokeniza la consulta y devuelve los scores BM25 para cada documento.

* **Jaccard**

  * `buscar_jaccard(docs_procesados, consulta_procesada_str)`: Calcula la similitud Jaccard (intersección / unión) entre la consulta y cada documento.

* `ejecutar_consultas(...)`: Interfaz que pide una consulta por teclado, la procesa, ejecuta la búsqueda con el método seleccionado y muestra los `Top-N` resultados por consola.

### PARTE 3 — Evaluación del sistema

* `cargar_consultas_evaluacion()`: Retorna un conjunto predefinido de queries y sus QRELS (documentos relevantes por query).
* `calcular_precision_recall(resultados_ranking, documentos_relevantes)`: Calcula Precisión, Recall y Average Precision (AP) para un ranking dado.
* `evaluar_sistema(modelos_busqueda, procesamiento)`: Ejecuta la evaluación sobre las queries predefinidas para cada modelo (TF–IDF, BM25, Jaccard), calcula AP por consulta y MAP por modelo y muestra tablas con resultados.
* `mostrar_queries_qrels()`: Muestra en pantalla las queries y los Qrels asociados.

---

## Modo de uso / Ejemplos

1. Inicie el script: `python proyectoRI.py`.
2. En el menú seleccione la opción `1` para verificar que el corpus ha sido cargado correctamente.
```bash
================================================================================
                     SISTEMA DE RECUPERACIÓN DE INFORMACIÓN                     
================================================================================
1. Ver información del dataset
2. Ver índice invertido
3. Mostrar tabla de documentos (original vs procesado)
4. Búsqueda TF-IDF
5. Búsqueda BM25
6. Búsqueda Jaccard
7. Evaluación de resultados
8. Ver queries y QRELs
9. Salir

Eliga una opción: 
```
3. Seleccione `4`, `5` o `6` para realizar búsquedas con TF–IDF, BM25 o Jaccard respectivamente. Se le solicitará una consulta por consola.
4. Para evaluación automática ejecute la opción `7` (presentará tablas con Precisión, Recall, AP y MAP).

**Ejemplo de consulta**: `Ukraine's youngest cabinet minister` (cargado en las queries de evaluación como `Q1`).

---

## Evaluación del sistema

* El sistema calcula Precisión (P), Exhaustividad (Recall), Average Precision (AP) por consulta y Mean Average Precision (MAP) por modelo, ejemplo:
```bash
======================================================================
--- EVALUANDO MODELO: TF-IDF ---
======================================================================
╒═══════════════╤═════════════╤══════════════════════════╤══════════╕
│ Consulta ID   │ Precisión   │ Exhaustividad (Recall)   │       AP │
╞═══════════════╪═════════════╪══════════════════════════╪══════════╡
│ Q1            │ 0.0027      │ 1.0000                   │ 0.811111 │
├───────────────┼─────────────┼──────────────────────────┼──────────┤
│ Q2            │ 0.0033      │ 0.8000                   │ 0.8      │
├───────────────┼─────────────┼──────────────────────────┼──────────┤
│ Q3            │ 0.0006      │ 1.0000                   │ 0.412844 │
├───────────────┼─────────────┼──────────────────────────┼──────────┤
│ Q4            │ 0.0053      │ 1.0000                   │ 0.711667 │
├───────────────┼─────────────┼──────────────────────────┼──────────┤
│ Q5            │ 0.0044      │ 1.0000                   │ 0.605455 │
├───────────────┼─────────────┼──────────────────────────┼──────────┤
│ --- MAP ---   │             │                          │ 0.6682   │
╘═══════════════╧═════════════╧══════════════════════════╧══════════╛
```
* Limitaciones a tener en cuenta:

  * Las QRELS son un conjunto fijo y limitado: la evaluación será válida sólo para ese subconjunto.
  * El corpus puede contener ruido (duplicados, textos cortos) y la representación actual es sensible al stemming y a la eliminación de stopwords.
  * TF–IDF y BM25 requieren una buena tokenización y normalización; modificaciones en el preprocesamiento (por ejemplo, lematización o manejo de entidades nombradas) impactarán fuertemente los resultados.

---

## Buenas prácticas y recomendaciones

* Versionar el dataset o indicar su procedencia y checksum (MD5/SHA256) para garantizar reproducibilidad.
* Registrar la versión de Python y de las librerías críticas (scikit-learn, rank_bm25, nltk).
* Añadir tests unitarios para funciones clave: preprocesamiento, construcción de índice, cálculo de métricas.

---

## Extensiones sugeridas / Trabajo futuro

* Añadir manejo de sinónimos y expansión de consulta (thesaurus/WordNet).
* Relevancia y ponderación por campos (si el CSV contiene título, fecha, categoría).
* Interfaz web sencilla (Flask/FastAPI) para facilitar pruebas y demos.


---

### Contacto

Si desea reportar errores, proponer mejoras o compartir resultados, abra un *issue* en el repositorio o envíe un correo al autor inti.poaquiza@epn.edu.ec
