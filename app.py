import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

def obtener_descendidos_con_nombres(data_por_curso, top_n=15):
    """
    Devuelve un diccionario con los N estudiantes de puntaje más bajo por curso.
    Detecta dinámicamente la columna de nombres y puntajes.
    """
    posibles_nombres = ["Alumno", "NOMBRE ESTUDIANTE", "Nombre", "nombre estudiante"]
    posibles_puntajes = ["Puntaje", "Puntaje Simce", "puntaje", "puntajes"]

    descendidos_por_curso = {}

    for curso, df in data_por_curso.items():
        # Buscar columna de nombres
        col_nombres = next((col for col in df.columns if any(opt.lower() in str(col).lower() for opt in posibles_nombres)), None)
        # Buscar columna de puntajes
        col_puntaje = next((col for col in df.columns if any(opt.lower() in str(col).lower() for opt in posibles_puntajes)), None)

        if col_nombres and col_puntaje:
            # Limpiar y convertir puntajes a numérico
            df_filtrado = df[[col_nombres, col_puntaje]].dropna(subset=[col_puntaje])
            df_filtrado[col_puntaje] = pd.to_numeric(df_filtrado[col_puntaje], errors='coerce')
            df_filtrado = df_filtrado.dropna(subset=[col_puntaje])

            # Ordenar de mayor a menor y tomar los N últimos (los más bajos)
            df_ordenado = df_filtrado.sort_values(by=col_puntaje, ascending=False)
            df_descendidos = df_ordenado.tail(top_n)

            # Renombrar columnas para consistencia
            df_descendidos = df_descendidos.rename(columns={col_nombres: "Alumno", col_puntaje: "Puntaje"})
            df_descendidos["Curso"] = curso

            descendidos_por_curso[curso] = df_descendidos[["Alumno", "Puntaje", "Curso"]]

    return descendidos_por_curso


def exportar_descendidos_con_nombres(descendidos_por_curso):
    """
    Exporta a Excel los 15 alumnos con puntajes más bajos por curso,
    en hojas separadas + 1 hoja general consolidada.
    """
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for curso, df in descendidos_por_curso.items():
            df[["Alumno", "Puntaje"]].to_excel(writer, sheet_name=str(curso), index=False)
        if descendidos_por_curso:
            tabla_general = pd.concat(descendidos_por_curso.values(), ignore_index=True)
            tabla_general[["Alumno", "Puntaje", "Curso"]].to_excel(writer, sheet_name="Consolidado", index=False)
    output.seek(0)
    return output.getvalue()


st.set_page_config(page_title="Extractor y Analizador SIMCE / PAES", layout="centered")

st.title("📊 Extractor, Analizador y Visualizador de Puntajes SIMCE / PAES")

st.write("""
Esta aplicación te permite:
- Extraer nóminas y puntajes desde archivos Excel complejos.
- Reutilizar nóminas ya procesadas.
- Clasificar automáticamente los rendimientos según criterios **SIMCE** o **PAES**.
- Visualizar gráficos de barras por curso.
- Exportar los gráficos a PDF (incluyendo gráfico general).
""")

analisis_tipo = st.selectbox("Selecciona el tipo de análisis:", ["SIMCE", "PAES"])

def clasificar_rendimiento(puntaje, tipo):
    if tipo == "SIMCE":
        if puntaje <= 250:
            return "Insuficiente"
        elif puntaje <= 285:
            return "Intermedio"
        else:
            return "Adecuado"
    elif tipo == "PAES":
        if puntaje <= 599:
            return "Insuficiente"
        elif puntaje <= 799:
            return "Intermedio"
        else:
            return "Adecuado"
    return "Desconocido"

uploaded_file = st.file_uploader("📂 Sube tu archivo Excel (.xlsx)", type=["xlsx"])
modo = st.radio("¿Qué tipo de archivo estás subiendo?", ["Archivo original (complejo)", "Nómina ya procesada"], horizontal=True)

resultados = {}
graficos = {}

if uploaded_file:
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names

        for sheet in sheet_names:
            df = excel_file.parse(sheet_name=sheet)

            try:
                if modo == "Archivo original (complejo)":
                    df = pd.read_excel(uploaded_file, sheet_name=sheet, header=None, skiprows=10)
                    df_nomina = df.iloc[:, [2, 166]]
                    df_nomina.columns = ["Nombre", "Puntaje"]
                    df_nomina["Puntaje"] = pd.to_numeric(df_nomina["Puntaje"], errors="coerce")
                    df_nomina = df_nomina.dropna(subset=["Nombre", "Puntaje"])
                    df_nomina = df_nomina[df_nomina["Puntaje"].between(0, 1000)]
                else:
                    df_nomina = df[["Nombre", "Puntaje SIMCE" if "Puntaje SIMCE" in df.columns else "Puntaje"]]
                    df_nomina.rename(columns={"Puntaje SIMCE": "Puntaje"}, inplace=True)
                    df_nomina = df_nomina.dropna(subset=["Nombre", "Puntaje"])
                    df_nomina["Puntaje"] = pd.to_numeric(df_nomina["Puntaje"], errors="coerce")

                df_nomina["Rendimiento"] = df_nomina["Puntaje"].apply(lambda x: clasificar_rendimiento(x, analisis_tipo))
                promedio_curso = df_nomina["Puntaje"].mean()
                df_nomina["Promedio Curso"] = promedio_curso
                resultados[sheet] = df_nomina

                conteo = df_nomina["Rendimiento"].value_counts().reindex(["Insuficiente", "Intermedio", "Adecuado"], fill_value=0)
                fig, ax = plt.subplots(figsize=(4, 3))
                conteo.plot(kind="bar", ax=ax)
                ax.set_title(f"{sheet}")
                ax.set_ylabel("Cantidad")
                ax.set_xlabel("Rendimiento")
                ax.grid(axis='y')
                graficos[sheet] = fig

            except Exception as e:
                st.warning(f"⚠️ Hoja '{sheet}' no procesada: {e}")

        if resultados:
            output_excel = BytesIO()
            with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                for sheet, df in resultados.items():
                    df.to_excel(writer, index=False, sheet_name=sheet)

            st.success("✅ Procesamiento completo.")

            st.download_button(
                label="📥 Descargar Excel con rendimientos",
                data=output_excel.getvalue(),
                file_name=f"analisis_{analisis_tipo.lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            if st.button("📊 Mostrar gráficos por curso"):
                cols = st.columns(3)
                for i, (sheet, fig) in enumerate(graficos.items()):
                    with cols[i % 3]:
                        st.pyplot(fig)
                        st.caption(f"{sheet} — Promedio: {resultados[sheet]['Promedio Curso'].iloc[0]:.2f}")

            # Mostrar gráfico general del liceo
            all_data = pd.concat(resultados.values())
            promedio_general = all_data["Puntaje"].mean()
            conteo_general = all_data["Rendimiento"].value_counts().reindex(["Insuficiente", "Intermedio", "Adecuado"], fill_value=0)

            st.subheader("📈 Rendimiento general del liceo")
            fig_all, ax = plt.subplots(figsize=(6, 4))
            conteo_general.plot(kind="bar", ax=ax)
            ax.set_title(f"Rendimiento General del Liceo (Promedio: {promedio_general:.2f})")
            ax.set_ylabel("Cantidad")
            ax.set_xlabel("Rendimiento")
            ax.grid(axis="y")
            st.pyplot(fig_all)

            if st.button("🖨️ Exportar gráficos a PDF"):
                try:
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)

                    with tempfile.TemporaryDirectory() as tempdir:
                        for sheet, fig in graficos.items():
                            img_path = os.path.join(tempdir, f"{sheet}.png")
                            fig.savefig(img_path, format='png', bbox_inches='tight')
                            pdf.add_page()
                            pdf.set_font("Arial", "B", 16)
                            pdf.cell(0, 10, f"Rendimiento - {sheet}", ln=True)
                            pdf.image(img_path, x=10, y=30, w=180)

                        # Guardar gráfico general
                        grafico_general_path = os.path.join(tempdir, "liceo_general.png")
                        fig_all.savefig(grafico_general_path, format="png", bbox_inches="tight")
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 16)
                        pdf.cell(0, 10, f"Rendimiento General del Liceo", ln=True)
                        pdf.image(grafico_general_path, x=10, y=30, w=180)

                        pdf_buffer = BytesIO()
                        pdf_output_bytes = pdf.output(dest='S').encode('latin-1')
                        pdf_buffer.write(pdf_output_bytes)
                        pdf_buffer.seek(0)

                        st.download_button(
                            label="📄 Descargar PDF de gráficos",
                            data=pdf_buffer,
                            file_name="graficos_rendimiento.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Error al exportar PDF: {e}")
        else:
            st.error("No se pudo procesar ninguna hoja.")
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

with tab4:
    st.header("📉 Alumnos Descendidos (15 puntajes más bajos por curso)")

    descendidos_por_curso = obtener_descendidos_con_nombres(data_por_curso, top_n=15)

    # Mostrar tablas por curso
    for curso, df in descendidos_por_curso.items():
        st.subheader(f"{curso}: 15 puntajes más bajos")
        st.table(df[["Alumno", "Puntaje"]])

    # Consolidado general
    if descendidos_por_curso:
        st.subheader("📊 Consolidado General de Descendidos")
        tabla_general = pd.concat(descendidos_por_curso.values(), ignore_index=True)
        st.dataframe(tabla_general[["Alumno", "Puntaje", "Curso"]])

    # Botón para descargar en Excel
    excel_data = exportar_descendidos_con_nombres(descendidos_por_curso)
    st.download_button(
        label="📥 Descargar descendidos (por curso + consolidado) en Excel",
        data=excel_data,
        file_name="descendidos_por_curso_y_consolidado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

import streamlit as st
import pandas as pd
import unicodedata
from io import BytesIO

# Función para normalizar nombres
def normalizar_nombre(nombre):
    if pd.isna(nombre):
        return ""
    nombre = str(nombre).strip().lower()
    nombre = unicodedata.normalize('NFD', nombre).encode('ascii', 'ignore').decode('utf-8')
    return nombre

st.set_page_config(page_title="Consolidación SIMCE por Grupo", layout="wide")
st.title("📊 Consolidación de Puntajes SIMCE (Estructura por Grupo)")

# Subida de archivos
archivo_1 = st.file_uploader("📘 Archivo 1 (estructura original por grupo)", type=["xlsx"])
archivo_2 = st.file_uploader("📗 Archivo 2 (nuevo ensayo por curso)", type=["xlsx"])

if archivo_1 and archivo_2:
    # Cargar archivos
    xls1 = pd.ExcelFile(archivo_1)
    xls2 = pd.ExcelFile(archivo_2)

    # Combinar todas las hojas del archivo 2 en un DataFrame
    df_ensayo = pd.concat([xls2.parse(sheet) for sheet in xls2.sheet_names], ignore_index=True)

    # Crear diccionario normalizado de nombre → puntaje
    df_ensayo["Nombre Normalizado"] = df_ensayo["Nombre"].apply(normalizar_nombre)
    nombre_a_puntaje = df_ensayo.set_index("Nombre Normalizado")["Puntaje"].to_dict()

    # Determinar número del próximo ensayo
    ejemplo_hoja = xls1.parse(xls1.sheet_names[0])
    columnas = ejemplo_hoja.columns
    columnas_ensayo = [col for col in columnas if isinstance(col, str) and col.lower().startswith("puntaje ensayo")]
    num_ensayo_nuevo = len(columnas_ensayo) + 1
    nombre_columna_nueva = f"Puntaje Ensayo {num_ensayo_nuevo}"

    # Crear archivo consolidado
    consolidado = BytesIO()
    writer = pd.ExcelWriter(consolidado, engine='xlsxwriter')

    total_agregados = 0
    total_nombres = 0

    for hoja in xls1.sheet_names:
        df = xls1.parse(hoja)

        # Verificar existencia de columna "Nombre Estudiante"
        if "Nombre Estudiante" not in df.columns:
            st.warning(f"La hoja '{hoja}' no contiene la columna 'Nombre Estudiante'. Se omitirá.")
            continue

        # Agregar puntajes si hay coincidencia
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

    # Mostrar resumen
    st.success("¡Archivo consolidado generado correctamente!")
    st.write(f"🧍‍♂️ Total de nombres procesados: {total_nombres}")
    st.write(f"✅ Puntajes nuevos agregados: {total_agregados}")
    st.download_button("📥 Descargar archivo consolidado", data=consolidado, file_name="consolidado_simce.xlsx")
