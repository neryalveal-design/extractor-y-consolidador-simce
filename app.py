
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

