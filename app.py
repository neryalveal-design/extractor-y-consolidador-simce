
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
