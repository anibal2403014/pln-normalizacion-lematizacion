from pathlib import Path

import spacy


# Ubicación de los archivos.
CARPETA = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = CARPETA / "libro.txt"
ARCHIVO_SALIDA = CARPETA / "libro_normalizado.txt"


print("Cargando el modelo en español...")

# Cargamos el modelo de spaCy en español.
# Desactivamos componentes que no utilizaremos.
nlp = spacy.load(
    "es_core_news_sm",
    disable=["parser", "ner"]
)


# Comprobamos que exista el libro.
if not ARCHIVO_ENTRADA.exists():
    raise FileNotFoundError(
        "No se encontró libro.txt dentro de la carpeta del proyecto."
    )


# Cargamos el contenido del libro.
texto = ARCHIVO_ENTRADA.read_text(
    encoding="utf-8"
)


# Eliminamos los avisos de Project Gutenberg.
# Conservamos únicamente el contenido del libro.
lineas = texto.splitlines()
inicio = 0
fin = len(lineas)

for posicion, linea in enumerate(lineas):
    if linea.startswith(
        "*** START OF THE PROJECT GUTENBERG EBOOK"
    ):
        inicio = posicion + 1

    if linea.startswith(
        "*** END OF THE PROJECT GUTENBERG EBOOK"
    ):
        fin = posicion
        break

texto = "\n".join(lineas[inicio:fin])


# Permitimos que spaCy procese textos largos.
nlp.max_length = max(
    nlp.max_length,
    len(texto) + 1000
)


print(
    f"Texto cargado correctamente: "
    f"{len(texto)} caracteres."
)

print(
    "Aplicando tokenización, limpieza "
    "y lematización..."
)


# spaCy divide y analiza el texto.
doc = nlp(texto)


# Tokens antes de la limpieza.
tokens_originales = [
    token.text
    for token in doc
    if not token.is_space
]


# Aplicamos normalización y lematización.
tokens_normalizados = [
    token.lemma_.lower()
    for token in doc
    if not token.is_stop
    and not token.is_punct
    and not token.is_space
]


# Unimos los tokens para formar el texto normalizado.
texto_normalizado = " ".join(
    tokens_normalizados
)


# Guardamos el resultado.
ARCHIVO_SALIDA.write_text(
    texto_normalizado,
    encoding="utf-8"
)


print("\nProceso terminado correctamente.")

print(
    f"Tokens originales: "
    f"{len(tokens_originales)}"
)

print(
    f"Tokens normalizados: "
    f"{len(tokens_normalizados)}"
)


print("\nPrimeros 15 tokens originales:")

print(
    tokens_originales[:15]
)


print("\nPrimeros 15 tokens normalizados:")

print(
    tokens_normalizados[:15]
)


print("\nEjemplos de lematización:")

cambios_mostrados = 0

for token in doc:
    original = token.text.lower()
    lema = token.lemma_.lower()

    if (
        not token.is_stop
        and not token.is_punct
        and not token.is_space
        and original != lema
    ):
        print(
            f"{token.text} -> {lema}"
        )

        cambios_mostrados += 1

    if cambios_mostrados == 15:
        break


print(
    f"\nResultado guardado en: "
    f"{ARCHIVO_SALIDA}"
)