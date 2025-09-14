import streamlit as st
import pandas as pd
from utils import analyze_data

st.set_page_config(page_title="Consolidador de Ensayos por Curso", layout="wide")
st.title("📊 Consolidador de Ensayos por Curso")

uploaded_files = st.file_uploader("Sube archivos Excel o CSV", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files:
    try:
        dataframes = []
        courses = set()
        students = set()

        for file in uploaded_files:
            # Leer archivo
            df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

            # Normalizar columnas de forma segura
            df.columns = [str(c).lower().strip() if pd.notna(c) else f"columna_{i}" for i, c in enumerate(df.columns)]

            # Verifica existencia de columnas necesarias
            if "curso" in df.columns and ("nombre" in df.columns or "nombre estudiante" in df.columns):
                nombre_col = "nombre" if "nombre" in df.columns else "nombre estudiante"
                df["curso"] = df["curso"].astype(str)
                df[nombre_col] = df[nombre_col].astype(str)

                courses.update(df["curso"].unique())
                students.update(df[nombre_col].unique())
                dataframes.append(df)

        if not dataframes:
            raise ValueError("No se encontraron columnas válidas ('curso' y 'nombre') en los archivos.")

        data = pd.concat(dataframes, ignore_index=True)

        tab1, tab2, tab3 = st.tabs(["📌 Resumen", "📘 Por Curso", "📤 Exportar"])

        with tab1:
            st.metric("Archivos procesados", len(uploaded_files))
            st.metric("Cursos detectados", len(courses))
            st.metric("Estudiantes únicos", len(students))

            for course in sorted(courses):
                st.write(f"- {course}")

        with tab2:
            for course in sorted(courses):
                st.subheader(course)
                subset = data[data["curso"] == course]
                st.write(subset)

        with tab3:
            output = pd.ExcelWriter("consolidado_temp.xlsx", engine="openpyxl")
            data.to_excel(output, index=False)
            output.close()
            with open("consolidado_temp.xlsx", "rb") as f:
                st.download_button(
                    "📥 Descargar Excel Consolidado",
                    f,
                    file_name="consolidado.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Ocurrió un error al procesar los archivos: {e}")
else:
    st.info("Por favor, sube uno o más archivos para comenzar.")
