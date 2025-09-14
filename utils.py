import pandas as pd

def analyze_data(files):
    dataframes = []
    courses = set()
    students = set()

    for file in files:
        # Leer archivo original
        df_raw = pd.read_excel(file, header=None) if file.name.endswith("xlsx") else pd.read_csv(file, header=None)

        # Buscar fila que contiene "nombre" o "estudiante"
        start_row = None
        for i, row in df_raw.iterrows():
            row_str = row.astype(str).str.lower()
            if row_str.str.contains("nombre").any() or row_str.str.contains("estudiante").any():
                start_row = i
                break

        if start_row is None:
            continue  # no se encontró encabezado válido

        # Volver a leer con encabezado desde fila detectada
        df = pd.read_excel(file, header=start_row) if file.name.endswith("xlsx") else pd.read_csv(file, header=start_row)

        # Normalizar nombres de columnas
        clean_columns = []
        for col in df.columns:
            try:
                col_str = str(col).lower().strip()
            except Exception:
                col_str = "columna_sin_nombre"
            clean_columns.append(col_str)
        df.columns = clean_columns

        # Verificación de columnas mínimas necesarias
        if "curso" not in df.columns or "nombre" not in df.columns:
            continue

        # Convertir valores clave a texto
        df["curso"] = df["curso"].astype(str)
        df["nombre"] = df["nombre"].astype(str)

        courses.update(df["curso"].unique())
        students.update(df["nombre"].unique())
        dataframes.append(df)

    if not dataframes:
        raise ValueError("No se encontraron columnas válidas ('curso' y 'nombre') en los archivos.")

    combined = pd.concat(dataframes, ignore_index=True)
    return combined, courses, len(students)
