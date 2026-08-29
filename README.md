# Normalización, lematización y vectorización de texto

Actividad de los bloques 3 y 4 de Procesamiento de Lenguaje Natural.

Este proyecto procesa el libro *Lazarillo de Tormes* mediante spaCy y su modelo en español. Posteriormente, convierte el texto en representaciones numéricas utilizando Bag-of-Words y TF-IDF.

## Procesos aplicados

* Carga del libro en formato TXT.
* Eliminación de los avisos de Project Gutenberg.
* Tokenización del texto.
* Conversión de las palabras a minúsculas.
* Eliminación de stop words.
* Eliminación de signos de puntuación y espacios.
* Lematización.
* Generación del texto normalizado.
* Vectorización mediante Bag-of-Words.
* Vectorización mediante TF-IDF.
* Reducción de dimensiones mediante PCA.
* Generación de gráficas en 3D.

## Archivos principales

* `libro.txt`: libro original.
* `normalizar.py`: realiza la limpieza y lematización.
* `libro_normalizado.txt`: contiene el texto procesado.
* `vectorizar.py`: aplica Bag-of-Words, TF-IDF y PCA.
* `resultados_vectorizacion.txt`: resumen de los resultados.
* `grafica_vectorizacion_3d.png`: comparación gráfica de BoW y TF-IDF.
* `requirements.txt`: dependencias necesarias para ejecutar el proyecto.

## Instalación

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

Instalar el modelo de spaCy en español:

```bash
python -m spacy download es_core_news_sm
```

## Ejecución

Para realizar la normalización y lematización:

```bash
python normalizar.py
```

Para aplicar la representación vectorial:

```bash
python vectorizar.py
```

## Resultados

Bag-of-Words representa el texto mediante la cantidad de veces que aparece cada palabra.

TF-IDF asigna mayor importancia a las palabras que destacan dentro de una oración o documento.

PCA reduce las dimensiones de las matrices obtenidas para permitir su representación mediante gráficas en 3D.
