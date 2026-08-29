import spacy
import numpy as np
import matplotlib

# Permite guardar la gráfica aunque Ubuntu no abra una ventana.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA


ARCHIVO_LIBRO = "libro.txt"
ARCHIVO_RESULTADOS = "resultados_vectorizacion.txt"
ARCHIVO_GRAFICA = "grafica_vectorizacion_3d.png"

# Limita el vocabulario para que el procesamiento sea más rápido.
MAX_CARACTERISTICAS = 2000

# Cantidad de palabras que aparecerán en cada gráfica.
PALABRAS_EN_GRAFICA = 30


def quitar_metadatos_gutenberg(texto):
    """
    Elimina el encabezado y el cierre agregados por Project Gutenberg.
    """

    marca_inicio = "*** START OF THE PROJECT GUTENBERG EBOOK"
    marca_final = "*** END OF THE PROJECT GUTENBERG EBOOK"

    if marca_inicio in texto:
        texto = texto.split(marca_inicio, 1)[1]

        if "\n" in texto:
            texto = texto.split("\n", 1)[1]

    if marca_final in texto:
        texto = texto.split(marca_final, 1)[0]

    return texto.strip()


def obtener_palabras_principales(vocabulario, valores, cantidad=15):
    """
    Obtiene las palabras con los valores más altos.
    """

    cantidad = min(cantidad, len(vocabulario))
    indices = np.argsort(valores)[-cantidad:][::-1]

    return [(vocabulario[indice], valores[indice]) for indice in indices]


def reducir_palabras_a_3d(
    matriz,
    vocabulario,
    importancia,
    cantidad=30
):
    """
    Selecciona las palabras principales y utiliza PCA para
    reducir sus vectores a tres dimensiones.
    """

    cantidad = min(cantidad, len(vocabulario))
    indices = np.argsort(importancia)[-cantidad:][::-1]

    palabras = vocabulario[indices]

    # Filas = palabras y columnas = oraciones del libro.
    matriz_palabras = matriz[:, indices].T.toarray()

    pca = PCA(n_components=3)
    coordenadas = pca.fit_transform(matriz_palabras)

    return palabras, coordenadas


def agregar_grafica_3d(
    eje,
    palabras,
    coordenadas,
    titulo,
    color
):
    """
    Agrega las palabras y sus coordenadas a una gráfica 3D.
    """

    x = coordenadas[:, 0]
    y = coordenadas[:, 1]
    z = coordenadas[:, 2]

    eje.scatter(
        x,
        y,
        z,
        c=color,
        s=70,
        edgecolors="black",
        alpha=0.8,
        depthshade=True
    )

    for indice, palabra in enumerate(palabras):
        eje.text(
            x[indice],
            y[indice],
            z[indice],
            palabra,
            fontsize=7
        )

    eje.set_title(titulo)
    eje.set_xlabel("Componente 1")
    eje.set_ylabel("Componente 2")
    eje.set_zlabel("Componente 3")


print("Cargando el libro...")

with open(ARCHIVO_LIBRO, "r", encoding="utf-8") as archivo:
    texto = archivo.read()

texto = quitar_metadatos_gutenberg(texto)

print(f"Texto cargado correctamente: {len(texto)} caracteres.")
print("Cargando el modelo de español...")

nlp = spacy.load("es_core_news_sm")
nlp.max_length = max(nlp.max_length, len(texto) + 1000)

print("Normalizando y lematizando las oraciones...")

doc = nlp(texto)

corpus_lematizado = []

for oracion in doc.sents:
    lemas_oracion = [
        token.lemma_.lower()
        for token in oracion
        if token.is_alpha
        and not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

    # Conserva solamente oraciones con al menos tres palabras útiles.
    if len(lemas_oracion) >= 3:
        corpus_lematizado.append(" ".join(lemas_oracion))

if len(corpus_lematizado) < 3:
    raise ValueError(
        "No se obtuvieron suficientes oraciones para realizar la vectorización."
    )

print(f"Oraciones procesadas: {len(corpus_lematizado)}")
print("Aplicando Bag-of-Words...")

vectorizador_bow = CountVectorizer(
    max_features=MAX_CARACTERISTICAS
)

matriz_bow = vectorizador_bow.fit_transform(corpus_lematizado)
vocabulario_bow = vectorizador_bow.get_feature_names_out()

print("Aplicando TF-IDF...")

vectorizador_tfidf = TfidfVectorizer(
    max_features=MAX_CARACTERISTICAS
)

matriz_tfidf = vectorizador_tfidf.fit_transform(corpus_lematizado)
vocabulario_tfidf = vectorizador_tfidf.get_feature_names_out()

# Suma de apariciones de cada palabra en Bag-of-Words.
frecuencias_bow = np.asarray(
    matriz_bow.sum(axis=0)
).ravel()

# Promedio de importancia de cada palabra en TF-IDF.
importancia_tfidf = np.asarray(
    matriz_tfidf.mean(axis=0)
).ravel()

principales_bow = obtener_palabras_principales(
    vocabulario_bow,
    frecuencias_bow
)

principales_tfidf = obtener_palabras_principales(
    vocabulario_tfidf,
    importancia_tfidf
)

print("\nProceso terminado correctamente.")
print(f"Matriz BoW: {matriz_bow.shape}")
print(f"Matriz TF-IDF: {matriz_tfidf.shape}")

print("\nPalabras más frecuentes con Bag-of-Words:")

for palabra, frecuencia in principales_bow:
    print(f"{palabra}: {int(frecuencia)}")

print("\nPalabras con mayor importancia según TF-IDF:")

for palabra, importancia in principales_tfidf:
    print(f"{palabra}: {importancia:.4f}")

# ---------------------------------------------------------
# REDUCCIÓN A TRES DIMENSIONES CON PCA
# ---------------------------------------------------------

palabras_bow, coordenadas_bow = reducir_palabras_a_3d(
    matriz_bow,
    vocabulario_bow,
    frecuencias_bow,
    PALABRAS_EN_GRAFICA
)

palabras_tfidf, coordenadas_tfidf = reducir_palabras_a_3d(
    matriz_tfidf,
    vocabulario_tfidf,
    importancia_tfidf,
    PALABRAS_EN_GRAFICA
)

figura = plt.figure(figsize=(18, 8))

eje_bow = figura.add_subplot(121, projection="3d")
agregar_grafica_3d(
    eje_bow,
    palabras_bow,
    coordenadas_bow,
    "Espacio BoW 3D (conteos)",
    "orange"
)

eje_tfidf = figura.add_subplot(122, projection="3d")
agregar_grafica_3d(
    eje_tfidf,
    palabras_tfidf,
    coordenadas_tfidf,
    "Espacio TF-IDF 3D (importancia)",
    "teal"
)

plt.tight_layout()

plt.savefig(
    ARCHIVO_GRAFICA,
    dpi=180,
    bbox_inches="tight"
)

plt.close(figura)

# ---------------------------------------------------------
# GUARDAR RESUMEN DE RESULTADOS
# ---------------------------------------------------------

lineas_resultado = [
    "RESULTADOS DE VECTORIZACIÓN",
    "",
    f"Caracteres procesados: {len(texto)}",
    f"Oraciones procesadas: {len(corpus_lematizado)}",
    f"Forma de la matriz BoW: {matriz_bow.shape}",
    f"Forma de la matriz TF-IDF: {matriz_tfidf.shape}",
    "",
    "PALABRAS MÁS FRECUENTES CON BAG-OF-WORDS:"
]

for palabra, frecuencia in principales_bow:
    lineas_resultado.append(
        f"{palabra}: {int(frecuencia)}"
    )

lineas_resultado.extend([
    "",
    "PALABRAS CON MAYOR IMPORTANCIA SEGÚN TF-IDF:"
])

for palabra, importancia in principales_tfidf:
    lineas_resultado.append(
        f"{palabra}: {importancia:.4f}"
    )

with open(
    ARCHIVO_RESULTADOS,
    "w",
    encoding="utf-8"
) as archivo:
    archivo.write("\n".join(lineas_resultado))

print(f"\nResultados guardados en: {ARCHIVO_RESULTADOS}")
print(f"Gráfica guardada en: {ARCHIVO_GRAFICA}")