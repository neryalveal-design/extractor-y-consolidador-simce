import streamlit as st
import pandas as pd
import unicodedata
from io import BytesIO

def normalizar_nombre(nombre):
    if pd.isna(nombre):
        return ""
    nombre = str(nombre).strip().lower()
    nombre = unicodedata.normalize('NFD', nombre).encode('ascii', 'ignore').decode('utf-8')
    return nombre

st.set_page_config(page_title="App Unificada SIMCE / Curso", layout="wide")
st.title("🧠 Consolidador de Puntajes SIMCE / Ensayos por Curso")

tab1, tab2 = st.tabs(["📘 Ensayos por Curso", "📗 Consolidar en Estructura por Grupo"])

with tab1:
    st.header("📘 Consolidación de Ensayos por Curso")

    uploaded_files = st.file_uploader("Sube archivos Excel o CSV", type=["xlsx", "csv"], accept_multiple_files=True, key="curso")

    if uploaded_files:
        try:
            dfs = []
            cursos = set()
            estudiantes = set()

            for file in uploaded_files:
                df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
                df.columns = [c.lower().strip() for c in df.columns]

                if "curso" in df.columns and "nombre" in df.columns:
                    df["curso"] = df["curso"].astype(str)
                    df["nombre"] = df["nombre"].astype(str)
                    cursos.update(df["curso"].unique())
                    estudiantes.update(df["nombre"].unique())
                    dfs.append(df)

            data = pd.concat(dfs, ignore_index=True)

            tab_resumen, tab_por_curso, tab_exportar = st.tabs(["📌 Resumen", "📘 Por Curso", "📤 Exportar"])

            with tab_resumen:
                st.metric("Archivos procesados", len(uploaded_files))
                st.metric("Cursos detectados", len(cursos))
                st.metric("Estudiantes únicos", len(estudiantes))
                for curso in sorted(cursos):
                    st.write(f"- {curso}")

            with tab_por_curso:
                for curso in sorted(cursos):
                    st.subheader(curso)
                    subset = data[data["curso"] == curso]
                    st.write(subset)

            with tab_exportar:
                output = BytesIO()
                data.to_excel(output, index=False, engine="openpyxl")
                output.seek(0)
                st.download_button("📥 Descargar Excel Consolidado", output, file_name="consolidado_por_curso.xlsx")

        except Exception as e:
            st.error(f"Ocurrió un error al procesar los archivos: {e}")
    else:
        st.info("Sube uno o más archivos para consolidar por curso.")

with tab2:
    st.header("📗 Agregar Puntaje a Estructura por Grupo")

    archivo_1 = st.file_uploader("📘 Archivo base por grupo", type=["xlsx"], key="grupo1")
    archivo_2 = st.file_uploader("📗 Archivo de nuevo ensayo por curso", type=["xlsx"], key="grupo2")

    if archivo_1 and archivo_2:
        try:
            xls1 = pd.ExcelFile(archivo_1)
            xls2 = pd.ExcelFile(archivo_2)

            df_ensayo = pd.concat([xls2.parse(sheet) for sheet in xls2.sheet_names], ignore_index=True)

            col_nombres = next((c for c in df_ensayo.columns if "nombre" in c.lower()), None)
            col_puntaje = next((c for c in df_ensayo.columns if "ensayo" in c.lower() or "puntaje" in c.lower()), None)

            if not col_nombres or not col_puntaje:
                st.error("No se encontró columna de nombres o puntaje en el archivo del nuevo ensayo.")
                st.stop()

            df_ensayo["Nombre Normalizado"] = df_ensayo[col_nombres].apply(normalizar_nombre)
            nombre_a_puntaje = df_ensayo.set_index("Nombre Normalizado")[col_puntaje].to_dict()

            ejemplo_hoja = xls1.parse(xls1.sheet_names[0])
            columnas_ensayo = [col for col in ejemplo_hoja.columns if isinstance(col, str) and col.lower().startswith("puntaje ensayo")]
            num_ensayo_nuevo = len(columnas_ensayo) + 1
            nombre_columna_nueva = f"Puntaje Ensayo {num_ensayo_nuevo}"

            consolidado = BytesIO()
            writer = pd.ExcelWriter(consolidado, engine='xlsxwriter')

            total_agregados = 0
            total_nombres = 0

            for hoja in xls1.sheet_names:
                df = xls1.parse(hoja)
                if "Nombre Estudiante" not in df.columns:
                    st.warning(f"La hoja '{hoja}' no tiene la columna 'Nombre Estudiante'. Se omite.")
                    continue

                nuevos_puntajes = []
                for nombre in df["Nombre Estudiante"]:
                    total_nombres += 1
                    nombre_norm = normalizar_nombre(nombre)
                    puntaje = nombre_a_puntaje.get(nombre_norm, None)
                    if puntaje is not None:
                        total_agregados += 1
                    nuevos_puntajes.append(puntaje)

                df[nombre_columna_nueva] = nuevos_puntajes
                df.to_excel(writer, sheet_name=hoja, index=False)

            writer.close()
            consolidado.seek(0)

            st.success("¡Archivo consolidado generado correctamente!")
            st.write(f"🧍 Total de nombres procesados: {total_nombres}")
            st.write(f"✅ Puntajes agregados: {total_agregados}")
            st.download_button("📥 Descargar archivo consolidado", data=consolidado, file_name="consolidado_simce_grupo.xlsx")

        except Exception as e:
            st.error(f"Error al procesar los archivos: {e}")
