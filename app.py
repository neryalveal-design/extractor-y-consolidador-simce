
import streamlit as st
import pandas as pd
from io import BytesIO

st.title("🧠 EXTRAER PUNTAJES - Ensayos SIMCE (Longitudes sincronizadas)")

def extraer_datos(df):
    try:
        # Usar fila 10 como encabezado
        raw_columns = df.iloc[9]
        df = df[10:].reset_index(drop=True)

        # Normalizar encabezados
        normalized = raw_columns.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        df.columns = normalized

        # Detectar índices
        idx_nombre = next((i for i, col in enumerate(normalized) if "nombre estudiante" in col), None)
        idx_puntaje = next((i for i, col in enumerate(normalized) if "puntaje simce" in col), None)

        if idx_nombre is None or idx_puntaje is None:
            st.error("No se detectaron columnas válidas de nombres o puntajes.")
            return None

        # Extraer datos
        nombres = df.iloc[:, idx_nombre].dropna().astype(str).tolist()
        puntajes = df.iloc[:, idx_puntaje].dropna().astype(str).tolist()

        # Sincronizar longitudes
        min_len = min(len(nombres), len(puntajes))
        nombres = nombres[:min_len]
        puntajes = puntajes[:min_len]

        # Crear DataFrame limpio
        df_limpio = pd.DataFrame({
            "NOMBRE ESTUDIANTE": nombres,
            "SIMCE 1": puntajes
        })

        return df_limpio

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
        return None

uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    hojas_validas = xls.sheet_names

    st.write("Hojas detectadas:", hojas_validas)

    for hoja in hojas_validas:
        st.subheader(f"Procesando hoja: {hoja}")
        df = pd.read_excel(xls, sheet_name=hoja, header=None)

        df_extraido = extraer_datos(df)

        if df_extraido is not None:
            st.write("Vista previa de datos extraídos:")
            st.dataframe(df_extraido)

            # Generar archivo descargable
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_extraido.to_excel(writer, index=False, sheet_name='Resultados')
                worksheet = writer.sheets['Resultados']
                worksheet.write('A1', 'NOMBRE ESTUDIANTE')
                worksheet.write('B1', 'SIMCE 1')
            st.download_button(
                label=f"📥 Descargar resultados hoja {hoja}",
                data=output.getvalue(),
                file_name=f"resultados_{hoja}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
    # 🔄 Crear archivo combinado con todas las hojas de curso
    st.subheader("📦 Exportar todas las hojas a un único Excel normalizado")

    combined_output = BytesIO()
    with pd.ExcelWriter(combined_output, engine='xlsxwriter') as writer:
        for hoja in hojas_validas:
            df = pd.read_excel(xls, sheet_name=hoja, header=None)
            df_extraido = extraer_datos(df)

            if df_extraido is not None:
                df_extraido.to_excel(writer, index=False, sheet_name=hoja[:31])  # Máximo 31 caracteres para nombre hoja
                worksheet = writer.sheets[hoja[:31]]
                worksheet.write('A1', 'NOMBRE ESTUDIANTE')
                worksheet.write('B1', 'SIMCE 1')

    st.download_button(
        label="📥 Descargar Excel Normalizado con todas las hojas",
        data=combined_output.getvalue(),
        file_name="excel_normalizado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
# 📊 Función 2: Análisis por curso (ajustada con títulos y columnas)
import matplotlib.pyplot as plt

st.header("📈 Análisis por curso")

criterio = st.radio("Selecciona el criterio de análisis", ["SIMCE", "PAES"], key="criterio_analisis")

# Rango de criterios
if criterio == "SIMCE":
    rangos = {
        "Insuficiente": (0, 250),
        "Intermedio": (251, 285),
        "Adecuado": (286, 400)
    }
else:  # PAES
    rangos = {
        "Insuficiente": (0, 599),
        "Intermedio": (600, 799),
        "Adecuado": (800, 1000)
    }

if uploaded_file:
    xls_analisis = pd.ExcelFile(uploaded_file)
    hojas_analisis = xls_analisis.sheet_names

    total_categorias = {"Insuficiente": 0, "Intermedio": 0, "Adecuado": 0}
    total_estudiantes = 0

    col1, col2 = st.columns(2)  # 🔥 dos columnas para los gráficos
    toggle = True  # alternar entre columnas

    for hoja in hojas_analisis:
        df_raw = pd.read_excel(xls_analisis, sheet_name=hoja, header=None)
        df = extraer_datos(df_raw)

        if df is None or "SIMCE 1" not in df.columns:
            st.warning(f"La hoja '{hoja}' no pudo procesarse. Se omitirá.")
            continue

        puntajes = pd.to_numeric(df["SIMCE 1"], errors='coerce').dropna()

        categorias = {
            "Insuficiente": ((puntajes >= rangos["Insuficiente"][0]) & (puntajes <= rangos["Insuficiente"][1])).sum(),
            "Intermedio": ((puntajes >= rangos["Intermedio"][0]) & (puntajes <= rangos["Intermedio"][1])).sum(),
            "Adecuado": ((puntajes >= rangos["Adecuado"][0]) & (puntajes <= rangos["Adecuado"][1])).sum(),
        }

        suma_curso = sum(categorias.values())
        if suma_curso == 0:
            continue

        for k in categorias:
            total_categorias[k] += categorias[k]
        total_estudiantes += suma_curso

        # Gráfico circular por curso
        fig, ax = plt.subplots(figsize=(5, 5))
        valores = list(categorias.values())
        etiquetas = list(categorias.keys())
        colores = ["red", "yellow", "green"]
        explode = [0.1 if v == max(valores) else 0 for v in valores]

        ax.pie(
            valores,
            labels=[f"{etiquetas[i]} ({valores[i]})" for i in range(len(etiquetas))],
            colors=colores,
            explode=explode,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90
        )
        ax.set_title(f"Distribución por desempeño - {hoja}")

        if toggle:
            col1.pyplot(fig)
        else:
            col2.pyplot(fig)
        toggle = not toggle

    # Gráfico global
    if total_estudiantes > 0:
        fig_total, ax_total = plt.subplots(figsize=(5, 5))
        valores = list(total_categorias.values())
        etiquetas = list(total_categorias.keys())
        colores = ["red", "yellow", "green"]
        explode = [0.1 if v == max(valores) else 0 for v in valores]

        ax_total.pie(
            valores,
            labels=[f"{etiquetas[i]} ({valores[i]})" for i in range(len(etiquetas))],
            colors=colores,
            explode=explode,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90
        )
        ax_total.set_title("📊 Distribución total de desempeño")

        st.pyplot(fig_total)


