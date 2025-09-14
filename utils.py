import pandas as pd

def analyze_data(files):
    dataframes = []
    courses = set()
    students = set()

    for file in files:
        # Intentar leer el archivo desde varias filas posibles (3 a 6)
        found_valid_structure = False
        for header_row in range(3, 7):
            try:
                df = pd.read_excel(file, header=header_row) if file.name.endswith("xlsx") else pd.read_csv(file, header=header_row)
                df.columns = [str(col).lower().strip() for col in df.columns]

                # Verificar que tenga las columnas necesarias
                if "curso" in df.columns and "nombre" in df.columns:
                    found_valid_structure = True
                    break
            except Exception:
                continue

        if not found_valid_structure:
            continue  # saltar archivos sin estructura clara

        # Convertir columnas clave
        df["curso"] = df["curso"].astype(str)
        df["nombre"] = df["nombre"].astype(str)

        courses.update(df["curso"].unique())
        students.update(df["nombre"].unique())
        dataframes.append(df)

    if not dataframes:
        raise ValueError("No se encontraron columnas válidas ('curso' y 'nombre') en los archivos.")

    combined = pd.concat(dataframes, ignore_index=True)
    return combined, courses, len(students)
