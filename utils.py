import pandas as pd

def analyze_data(files):
    dataframes = []
    courses = set()
    students = set()

    for file in files:
        # Leer archivo
        df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

        # Convertir nombres de columnas a string de forma segura
        clean_columns = []
        for col in df.columns:
            try:
                col_str = str(col).lower().strip()
            except Exception:
                col_str = "columna_sin_nombre"
            clean_columns.append(col_str)

        df.columns = clean_columns

        # Verificación antes de procesar
        if "curso" not in df.columns or "nombre" not in df.columns:
            continue

        # Convertir columnas clave
        df["curso"] = df["curso"].astype(str)
        df["nombre"] = df["nombre"].astype(str)

        courses.update(df["curso"].unique())
        students.update(df["nombre"].unique())
        dataframes.append(df)

    if not dataframes:
        raise ValueError("No se encontraron columnas válidas ('curso' y 'nombre') en los archivos.")

    # Concatenar todos
    combined = pd.concat(dataframes, ignore_index=True)
    return combined, courses, len(students)
