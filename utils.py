import pandas as pd

def analyze_data(files):
    dataframes = []
    courses = set()
    students = set()

    for file in files:
        # Leer archivo Excel o CSV
        df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

        # Asegurar que los nombres de columnas sean strings
        df.columns = [str(c).lower().strip() for c in df.columns]

        if "curso" in df.columns and "nombre" in df.columns:
            df["curso"] = df["curso"].astype(str)
            df["nombre"] = df["nombre"].astype(str)
            courses.update(df["curso"].unique())
            students.update(df["nombre"].unique())
            dataframes.append(df)

    if not dataframes:
        raise ValueError("No se encontraron columnas válidas ('curso' y 'nombre') en los archivos.")

    combined = pd.concat(dataframes, ignore_index=True)
    return combined, courses, len(students)
