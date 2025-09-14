
import streamlit as st
import pandas as pd
from io import BytesIO

# Función de limpieza con fix
def limpiar_dataframe(df):
    # Asegura que los nombres de columna sean strings
    df.columns = df.columns.map(str)

    # Eliminar columnas sin nombre
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]

    # Eliminar columnas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]

    # Limpiar espacios y pasar todo a minúscula
    df.columns = [col.strip().lower() for col in df.columns]

    # Filtrar solo columnas útiles
    columnas_validas = [col for col in df.columns if "nombre" in col or "puntaje" in col or "curso" in col]
    df = df[columnas_validas]

    return df

# Validación
def validar_columnas(df, columnas_requeridas):
    return all(col in df.columns for col in columnas_requeridas)

# Consolidación por curso
def consolidar_por_curso(df):
    return df.groupby("curso").agg({
        "nombre": "count",
        "puntaje": "mean"
    }).reset_index()

# Interfaz principal
st.title("Extractor de Puntajes SIMCE y Ensayos")

uploaded_files = st.file_uploader("Sube uno o varios archivos Excel", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    dataframes = []
    for file in uploaded_files:
        df = pd.read_excel(file)
        df = limpiar_dataframe(df)

        columnas_requeridas = ['nombre', 'puntaje']
        if not validar_columnas(df, columnas_requeridas):
            st.error(f"Archivo '{file.name}' no contiene columnas requeridas: {columnas_requeridas}")
            continue

        dataframes.append(df)

    if dataframes:
        df_total = pd.concat(dataframes, ignore_index=True)

        st.subheader("Vista previa del archivo limpio")
        st.dataframe(df_total.head())

        st.subheader("Consolidado por curso")
        df_consolidado = consolidar_por_curso(df_total)
        st.dataframe(df_consolidado)

        # Descarga del Excel limpio
        output = BytesIO()
        df_total.to_excel(output, index=False, engine='xlsxwriter')
        st.download_button(
            label="📥 Descargar Excel limpio",
            data=output.getvalue(),
            file_name="archivo_limpio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Descarga del consolidado
        output2 = BytesIO()
        df_consolidado.to_excel(output2, index=False, engine='xlsxwriter')
        st.download_button(
            label="📥 Descargar Consolidado por Curso",
            data=output2.getvalue(),
            file_name="consolidado_cursos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Ningún archivo válido fue procesado.")
