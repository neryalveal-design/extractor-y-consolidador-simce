
import streamlit as st
import pandas as pd
from io import BytesIO

st.title("🧠 EXTRAER PUNTAJES - Ensayos SIMCE")

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

# ================================
# 📚 FUNCIÓN 3 (MEJORADA): CONSOLIDACIÓN DE PUNTAJES (multi-hojas + normalización robusta)
# ================================
import unicodedata, re
from io import BytesIO

st.header("📚 Consolidación de puntajes (multi-hojas + normalización)")

def _normalizar_nombre(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("\u00a0", " ")                  # NBSP -> espacio normal
    s = unicodedata.normalize("NFKD", s)          # descomponer acentos
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # quitar diacríticos
    s = re.sub(r"[^a-z0-9\s]", " ", s)            # dejar letras/números/espacios
    s = re.sub(r"\s+", " ", s).strip()            # colapsar espacios
    return s

# Reusar archivo consolidado si ya existe en memoria; si no, pedirlo.
if "uploaded_consolidado" in globals() and uploaded_consolidado is not None:
    _file_consolidado = uploaded_consolidado
else:
    _file_consolidado = st.file_uploader(
        "Sube el archivo consolidado de puntajes anteriores (todas las hojas)",
        type=["xlsx"],
        key="consolidado_v2"
    )

if _file_consolidado and uploaded_file:
    # 1) Extraer TODOS los puntajes nuevos desde el archivo complejo ya cargado
    xls_new = pd.ExcelFile(uploaded_file)
    hojas_new = xls_new.sheet_names

    df_nuevos = pd.DataFrame()
    for hoja in hojas_new:
        df_raw = pd.read_excel(xls_new, sheet_name=hoja, header=None)
        df_extraido = extraer_datos(df_raw)  # <- reutilizamos tu limpiador/selector
        if df_extraido is not None and not df_extraido.empty:
            df_nuevos = pd.concat([df_nuevos, df_extraido], ignore_index=True)

    if df_nuevos.empty:
        st.error("No se encontraron puntajes nuevos para consolidar.")
    else:
        # Normalizar nombres en df_nuevos y asegurar tipo numérico de puntajes
        df_nuevos["__key"] = df_nuevos["NOMBRE ESTUDIANTE"].map(_normalizar_nombre)
        df_nuevos["SIMCE 1"] = pd.to_numeric(df_nuevos["SIMCE 1"], errors="coerce")

        # Si hay nombres repetidos en distintos cursos, nos quedamos con el primer valor no nulo
        df_nuevos = df_nuevos.sort_index()
        df_nuevos = df_nuevos.drop_duplicates(subset="__key", keep="first")

        # 2) Abrir el consolidado original y recorrer TODAS sus hojas
        xls_consol = pd.ExcelFile(_file_consolidado)
        hojas_consol = xls_consol.sheet_names

        # Para auditar coincidencias
        resumen = []

        output_consol = BytesIO()
        with pd.ExcelWriter(output_consol, engine="xlsxwriter") as writer:
            for hoja in hojas_consol:
                df_cons = pd.read_excel(xls_consol, sheet_name=hoja)

                # Detectar la columna de nombres en el consolidado (flexible)
                col_nombres = None
                for col in df_cons.columns:
                    col_low = str(col).lower()
                    if "nombre" in col_low and "estudiante" in col_low:
                        col_nombres = col
                        break

                if col_nombres is None:
                    # Si no hay columna de nombres, escribimos tal cual
                    df_cons.to_excel(writer, index=False, sheet_name=hoja[:31])
                    resumen.append({"Hoja": hoja, "Coincidencias": 0, "Sin coincidencia": len(df_cons)})
                    continue

                # Normalizar nombres del consolidado
                df_cons["__key"] = df_cons[col_nombres].map(_normalizar_nombre)

                # Unir por clave normalizada
                df_merge = df_cons.merge(
                    df_nuevos[["__key", "SIMCE 1"]],
                    on="__key",
                    how="left"
                )

                # Renombrar columna nueva y retirar auxiliar
                if "SIMCE 1" in df_merge.columns:
                    df_merge.rename(columns={"SIMCE 1": "SIMCE Nuevo"}, inplace=True)
                if "__key" in df_merge.columns:
                    df_merge.drop(columns="__key", inplace=True)

                # Conteo de coincidencias/no coincidencias en esta hoja
                coincidencias = df_merge["SIMCE Nuevo"].notna().sum()
                sin_coinc = df_merge["SIMCE Nuevo"].isna().sum()
                resumen.append({"Hoja": hoja, "Coincidencias": int(coincidencias), "Sin coincidencia": int(sin_coinc)})

                # Escribir hoja actualizada
                df_merge.to_excel(writer, index=False, sheet_name=hoja[:31])

        # 3) Mostrar resumen de mapeo y entregar descarga
        st.subheader("📋 Resumen de consolidación")
        st.dataframe(pd.DataFrame(resumen))

        st.download_button(
            label="📥 Descargar CONSOLIDADO ACTUALIZADO (todas las hojas)",
            data=output_consol.getvalue(),
            file_name="consolidado_actualizado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 4) TIP opcional: mostrar nombres sin match por hoja (para auditoría fina)
        with st.expander("🔍 Ver nombres sin coincidencia por hoja"):
            xls_consol2 = pd.ExcelFile(_file_consolidado)
            for hoja in xls_consol2.sheet_names:
                df_cons2 = pd.read_excel(xls_consol2, sheet_name=hoja)
                col_nombres = None
                for col in df_cons2.columns:
                    col_low = str(col).lower()
                    if "nombre" in col_low and "estudiante" in col_low:
                        col_nombres = col
                        break
                if col_nombres is None:
                    continue
                df_cons2["__key"] = df_cons2[col_nombres].map(_normalizar_nombre)
                df_no_match = df_cons2[~df_cons2["__key"].isin(df_nuevos["__key"])]
                st.write(f"**{hoja}** — Sin coincidencia: {len(df_no_match)}")
                st.dataframe(df_no_match[[col_nombres]])
# ================================
# 🎯 FUNCIÓN 4: ANÁLISIS POR ESTUDIANTE
# ================================
import matplotlib.pyplot as plt

st.header("🎯 Análisis por estudiante")

uploaded_consolidado_est = st.file_uploader(
    "Sube el archivo consolidado actualizado (con todas las hojas y puntajes)",
    type=["xlsx"],
    key="consolidado_estudiantes"
)

if uploaded_consolidado_est:
    xls_est = pd.ExcelFile(uploaded_consolidado_est)
    hojas_est = xls_est.sheet_names

    # Selección de curso (hoja)
    curso_sel = st.selectbox("Elige el curso (hoja de Excel)", hojas_est)

    df_curso = pd.read_excel(xls_est, sheet_name=curso_sel)

    # Detectar columna de nombres
    col_nombres = None
    for col in df_curso.columns:
        if "nombre" in str(col).lower() and "estudiante" in str(col).lower():
            col_nombres = col
            break

    if col_nombres is None:
        st.error("No se encontró una columna de nombres de estudiantes en esta hoja.")
    else:
        # Selección de estudiante
        estudiante_sel = st.selectbox("Elige un estudiante", df_curso[col_nombres].dropna().unique())

        # Extraer fila de este estudiante
        df_est = df_curso[df_curso[col_nombres] == estudiante_sel].copy()

        # Detectar columnas de puntajes (todas las que contengan "simce" en el nombre)
        cols_puntajes = [c for c in df_est.columns if "simce" in str(c).lower()]

        if not cols_puntajes:
            st.warning("No se encontraron columnas de puntajes en esta hoja.")
        else:
            puntajes = df_est[cols_puntajes].iloc[0].tolist()

            # Crear gráfico
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(cols_puntajes, puntajes, marker="o", linestyle="-", color="blue")

            # Anotar valores en cada punto
            for i, p in enumerate(puntajes):
                ax.text(i, p + 5, str(int(p)) if pd.notna(p) else "", ha="center", fontsize=9)

            ax.set_title(f"Evolución del rendimiento - {estudiante_sel} ({curso_sel})")
            ax.set_ylabel("Puntaje")
            ax.set_xlabel("Ensayos")
            ax.grid(True)

            st.pyplot(fig)

            # Mostrar promedio
            puntajes_validos = [p for p in puntajes if pd.notna(p)]
            if puntajes_validos:
                promedio = sum(puntajes_validos) / len(puntajes_validos)
                st.success(f"📊 Puntaje promedio de {estudiante_sel}: **{promedio:.2f}**")
            else:
                st.info(f"No hay puntajes disponibles para {estudiante_sel}.")





