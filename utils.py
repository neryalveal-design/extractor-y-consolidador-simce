import pandas as pd
import streamlit as st

def analyze_data(files):
    all_data = []

    for file in files:
        # Leer desde la fila 9 (índice 8)
        df = pd.read_excel(file, header=8)

        # Intentar extraer el nombre del curso desde la celda C3
        try:
            metadata = pd.read_excel(file, header=None, nrows=5, usecols="C")
            curso_value = metadata.iloc[2, 0] if not pd.isna(metadata.iloc[2, 0]) else "Curso desconocido"
        except:
            curso_value = "Curso desconocido"

        df["curso"] = curso_value

        # Normalizar nombres de columnas
        df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")

        # Renombrar nombre de estudiante si es necesario
        if "nombre_estudiante" not in df.columns:
            posibles = [col for col in df.columns if "estudiante" in col]
            if posibles:
                df.rename(columns={posibles[0]: "nombre_estudiante"}, inplace=True)

        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    cursos = full_df["curso"].unique()
    estudiantes = full_df["nombre_estudiante"].nunique() if "nombre_estudiante" in full_df.columns else "No encontrado"

    return full_df, cursos, estudiantes
