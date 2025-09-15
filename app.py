
import streamlit as st
import pandas as pd
from io import BytesIO

st.title("🧠 EXTRAER PUNTAJES - Ensayos SIMCE (Tolerante a Duplicados)")

def extraer_datos(df):
    try:
        # Usar fila 10 como encabezado
        raw_columns = df.iloc[9]
        df = df[10:].reset_index(drop=True)

        # Normalizar encabezados
        normalized = raw_columns.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)

        # Detectar la primera coincidencia válida por índice
        idx_nombre = None
        idx_puntaje = None
        for i, col in enumerate(normalized):
            if "nombre estudiante" in col and idx_nombre is None:
                idx_nombre = i
            if "puntaje simce" in col and idx_puntaje is None:
                idx_puntaje = i

        if idx_nombre is None or idx_puntaje is None:
            st.error("No se detectaron columnas válidas de nombres o puntajes.")
            return None

        # Extraer columnas por índice
        nombres = df.iloc[:, idx_nombre].dropna().astype(str).tolist()
        puntajes = df.iloc[:, idx_puntaje].dropna().astype(str).tolist()

        df_limpio = pd.DataFrame({
            "NOMBRE ESTUDIANTE": nombres,
            "SIMCE 1": puntajes[:len(nombres)]
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
