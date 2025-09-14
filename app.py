import streamlit as st
import pandas as pd
from utils import analyze_data

st.set_page_config(page_title="Consolidador de Ensayos por Curso", layout="wide")
st.title("📊 Consolidador de Ensayos por Curso")

uploaded_files = st.file_uploader("Sube archivos Excel o CSV", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files:
    try:
        data, courses, students = analyze_data(uploaded_files)

        tab1, tab2, tab3 = st.tabs(["📌 Resumen", "📘 Por Curso", "📤 Exportar"])

        with tab1:
            st.metric("Archivos procesados", len(uploaded_files))
            st.metric("Cursos detectados", len(courses))
            st.metric("Estudiantes únicos", students)

            for course in courses:
                st.write(f"- {course}")

        with tab2:
            for course in courses:
                st.subheader(course)
                subset = data[data["curso"] == course]
                st.write(subset)

        with tab3:
            st.download_button(
                "📥 Descargar Excel Consolidado",
                data.to_excel(index=False, engine='openpyxl'),
                file_name="consolidado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Ocurrió un error al procesar los archivos: {e}")
else:
    st.info("Por favor, sube uno o más archivos para comenzar.")
