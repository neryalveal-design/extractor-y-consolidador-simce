import pandas as pd

def analyze_data(files):
    dataframes = []
    courses = set()
    students = set()

    for file in files:
        df_raw = pd.read_excel(file, header=None) if file.name.endswith("xlsx") else pd.read_csv(file, header=None)

        header_row = None
        for i, row in df_raw.iterrows():
            row_str = row.astype(str).str.lower()
            if row_str.str.contains("nombre estudiante").any():
                header_row = i
                break

        if header_row is None:
            continue

        df = pd.read_excel(file, header=header_row) if file.name.endswith("xlsx") else pd.read_csv(file, header=header_row)

        # Limpieza de columnas
        clean_columns = []
        for idx, col in enumerate(df.columns):
            try:
                col_str = str(col).lower().strip()
                if col_str == "" or col_str == "nan":
                    col_str = f"columna_{idx}"
            except:
                col_str = f"columna_{idx}"
            clean_columns.append(col_str)
        df.columns = clean_columns

        # Verificar existencia de columnas necesarias
        if "curso" not in df.columns or "nombre estudiante" not in df.columns:
            continue

        df["curso"] = df["curso"].astype(str)
        df["nombre estudiante"] = df["nombre estudiante"].astype(str)

        courses.update(df["curso"].unique())
        students.update(df["nombre estudiante"].unique())
        dataframes.append(df)

    if not dataframes:
        raise ValueError("No se encontraron columnas válidas ('curso' y 'nombre estudiante') en los archivos.")

    combined = pd.concat(dataframes, ignore_index=True)
    return combined, courses, len(students)
