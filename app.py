
import streamlit as st
import pandas as pd
from io import BytesIO

st.title("🧠 EXTRAER PUNTAJES - Ensayos SIMCE (Mejorado)")

def extraer_datos(df):
    try:
        # Usar fila 10 como encabezado
        df.columns = df.iloc[9]
        df = df[10:].reset_index(drop=True)

        # Normalizar nombres de columnas (eliminar espacios dobles, minúsculas)
        columnas = df.columns.map(str)
        columnas = columnas.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)

        df.columns = columnas  # Asignar columnas normalizadas

        # Detectar columna de nombres
        col_nombres = None
        for col in df.columns:
            if "nombre estudiante" in col:
                col_nombres = col
                break

        # Detectar columna de puntajes
        col_puntaje = None
        for col in df.columns:
            if "puntaje simce" in col:
                col_puntaje = col
                break

        if not col_nombres or not col_puntaje:
            st.error("No se pudieron detectar columnas de nombres o puntajes.")
            return None

        # Asegurar que las columnas sean Series y no DataFrames con duplicados
        if isinstance(df[col_nombres], pd.DataFrame) or isinstance(df[col_puntaje], pd.DataFrame):
            st.error("Encabezados duplicados detectados. Ajusta el archivo.")
            return None

        nombres = df[col_nombres].dropna().astype(str).tolist()
        puntajes = df[col_puntaje].dropna().astype(str).tolist()

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
