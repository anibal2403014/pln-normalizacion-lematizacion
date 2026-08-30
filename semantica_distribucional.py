from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import spacy

from gensim.models import Word2Vec
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# UBICACIÓN DE LOS ARCHIVOS
# ---------------------------------------------------------

CARPETA = Path(__file__).resolve().parent

ARCHIVO_LIBRO = CARPETA / "libro.txt"
ARCHIVO_RESULTADOS = CARPETA / "resultados_semantica.txt"
ARCHIVO_HEATMAP = CARPETA / "heatmap_embeddings.png"
ARCHIVO_3D = CARPETA / "espacio_semantico_3d.png"


# ---------------------------------------------------------
# ELIMINAR AVISOS DE PROJECT GUTENBERG
# ---------------------------------------------------------

def quitar_metadatos_gutenberg(texto):
    marca_inicio = "*** START OF THE PROJECT GUTENBERG EBOOK"
    marca_final = "*** END OF THE PROJECT GUTENBERG EBOOK"

    if marca_inicio in texto:
        texto = texto.split(marca_inicio, 1)[1]

        if "\n" in texto:
            texto = texto.split("\n", 1)[1]

    if marca_final in texto:
        texto = texto.split(marca_final, 1)[0]

    return texto.strip()


# ---------------------------------------------------------
# CARGAR EL LIBRO
# ---------------------------------------------------------

print("Cargando el libro...")

if not ARCHIVO_LIBRO.exists():
    raise FileNotFoundError(
        "No se encontró libro.txt dentro de la carpeta del proyecto."
    )

texto = ARCHIVO_LIBRO.read_text(encoding="utf-8")
texto = quitar_metadatos_gutenberg(texto)

print(f"Texto cargado correctamente: {len(texto)} caracteres.")


# ---------------------------------------------------------
# NORMALIZAR Y SEPARAR LAS ORACIONES
# ---------------------------------------------------------

print("Cargando el modelo de spaCy...")

nlp = spacy.load("es_core_news_sm")
nlp.max_length = max(nlp.max_length, len(texto) + 1000)

print("Procesando y lematizando las oraciones...")

doc = nlp(texto)

oraciones = []

for oracion in doc.sents:
    palabras = [
        token.lemma_.lower()
        for token in oracion
        if token.is_alpha
        and not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.lemma_) > 1
    ]

    if len(palabras) >= 3:
        oraciones.append(palabras)

if len(oraciones) < 3:
    raise ValueError(
        "No se obtuvieron suficientes oraciones para entrenar Word2Vec."
    )

print(f"Total de oraciones procesadas: {len(oraciones)}")
print(f"Ejemplo de una oración procesada: {oraciones[0]}")


# ---------------------------------------------------------
# ENTRENAR WORD2VEC
# ---------------------------------------------------------

print("\nEntrenando el modelo Word2Vec...")

modelo = Word2Vec(
    sentences=oraciones,
    vector_size=50,
    window=5,
    min_count=2,
    workers=1,
    sg=1,
    epochs=100,
    seed=42
)

vocabulario = list(modelo.wv.index_to_key)

if len(vocabulario) < 3:
    raise ValueError(
        "El vocabulario es demasiado pequeño para crear las gráficas."
    )

print("Modelo Word2Vec entrenado correctamente.")
print(f"Palabras dentro del vocabulario: {len(vocabulario)}")
print("Cada palabra tiene un vector de 50 dimensiones.")


# ---------------------------------------------------------
# BUSCAR PALABRAS SEMÁNTICAMENTE CERCANAS
# ---------------------------------------------------------

palabras_para_probar = [
    "lázaro",
    "amo",
    "señor",
    "casa",
    "hambre"
]

palabras_disponibles = [
    palabra
    for palabra in palabras_para_probar
    if palabra in modelo.wv
]

if not palabras_disponibles:
    palabras_disponibles = vocabulario[:3]

lineas_resultados = [
    "RESULTADOS DE SEMÁNTICA DISTRIBUCIONAL",
    "",
    f"Oraciones procesadas: {len(oraciones)}",
    f"Palabras en el vocabulario: {len(vocabulario)}",
    "Dimensiones de cada vector: 50",
    "Método utilizado: Word2Vec Skip-gram",
    ""
]

for palabra in palabras_disponibles:
    similares = modelo.wv.most_similar(
        palabra,
        topn=5
    )

    titulo = (
        f"Palabras más cercanas semánticamente a '{palabra}':"
    )

    print(f"\n{titulo}")
    lineas_resultados.append(titulo)

    for palabra_similar, puntuacion in similares:
        resultado = (
            f"- {palabra_similar}: {puntuacion:.4f}"
        )

        print(resultado)
        lineas_resultados.append(resultado)

    lineas_resultados.append("")


# ---------------------------------------------------------
# IMAGEN 1: HEATMAP DE EMBEDDINGS
# ---------------------------------------------------------

cantidad_heatmap = min(12, len(vocabulario))
palabras_heatmap = vocabulario[:cantidad_heatmap]
vectores_heatmap = modelo.wv[palabras_heatmap]

figura, eje = plt.subplots(figsize=(14, 6))

imagen = eje.imshow(
    vectores_heatmap,
    cmap="RdBu_r",
    aspect="auto"
)

eje.set_yticks(range(len(palabras_heatmap)))
eje.set_yticklabels(
    palabras_heatmap,
    fontsize=11
)

eje.set_xlabel("Dimensiones del vector")
eje.set_ylabel("Palabras")
eje.set_title(
    "Embeddings Word2Vec del Lazarillo de Tormes"
)

figura.colorbar(
    imagen,
    ax=eje,
    label="Valor"
)

plt.tight_layout()

plt.savefig(
    ARCHIVO_HEATMAP,
    dpi=180,
    bbox_inches="tight"
)

plt.close(figura)

print(f"\nPrimera imagen guardada en: {ARCHIVO_HEATMAP.name}")


# ---------------------------------------------------------
# IMAGEN 2: ESPACIO SEMÁNTICO EN 3D
# ---------------------------------------------------------

cantidad_3d = min(40, len(vocabulario))
palabras_3d = vocabulario[:cantidad_3d]
vectores = modelo.wv[palabras_3d]

pca = PCA(n_components=3)
vectores_reducidos = pca.fit_transform(vectores)

figura = plt.figure(figsize=(14, 10))
eje = figura.add_subplot(111, projection="3d")

x = vectores_reducidos[:, 0]
y = vectores_reducidos[:, 1]
z = vectores_reducidos[:, 2]

eje.scatter(
    x,
    y,
    z,
    c="crimson",
    s=75,
    edgecolors="black",
    alpha=0.8,
    depthshade=True
)

for indice, palabra in enumerate(palabras_3d):
    eje.text(
        x[indice],
        y[indice],
        z[indice],
        palabra,
        fontsize=8
    )

eje.set_title(
    "Espacio semántico Word2Vec - Lazarillo de Tormes",
    fontsize=14
)

eje.set_xlabel("Dimensión latente 1")
eje.set_ylabel("Dimensión latente 2")
eje.set_zlabel("Dimensión latente 3")

plt.tight_layout()

plt.savefig(
    ARCHIVO_3D,
    dpi=180,
    bbox_inches="tight"
)

plt.close(figura)

print(f"Segunda imagen guardada en: {ARCHIVO_3D.name}")


# ---------------------------------------------------------
# GUARDAR RESULTADOS
# ---------------------------------------------------------

ARCHIVO_RESULTADOS.write_text(
    "\n".join(lineas_resultados),
    encoding="utf-8"
)

print(f"Resultados guardados en: {ARCHIVO_RESULTADOS.name}")
print("\nProceso terminado correctamente.")