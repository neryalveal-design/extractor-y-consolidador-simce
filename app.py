
import streamlit as st
import pandas as pd
from io import BytesIO
# ================================
# Encabezado con logo y título centrado
# ================================
col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.image("logo.png", width=100)  # Logo local

with col2:
    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style="margin-bottom:0;">Departamento de Lenguaje</h2>
            <h4 style="margin-top:0;">Liceo Bicentenario de Excelencia Polivalente San Nicolás</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.write("")  # espacio vacío para balancear

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
# 📂 FUNCIÓN 3: CONSOLIDACIÓN DE PUNTAJES (corregida: sin nombres en la última columna)
# ================================
from io import BytesIO
import unicodedata, re

st.header("📂 Consolidación de puntajes")

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower().replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

uploaded_consolidado = st.file_uploader(
    "Sube el archivo consolidado anterior (todas las hojas de cursos)",
    type=["xlsx"], key="consolidado_fix"
)

if uploaded_file and uploaded_consolidado:
    xls_comp = pd.ExcelFile(uploaded_file)            # archivo complejo (Función 1)
    xls_cons = pd.ExcelFile(uploaded_consolidado)     # consolidado histórico

    resumen = []
    output_consol = BytesIO()
    with pd.ExcelWriter(output_consol, engine="xlsxwriter") as writer:
        for hoja in xls_cons.sheet_names:
            df_cons = pd.read_excel(xls_cons, sheet_name=hoja)

            # Detectar columna de nombres en el consolidado
            col_nombres = None
            for col in df_cons.columns:
                c = str(col).lower()
                if "nombre" in c and "estudiante" in c:
                    col_nombres = col
                    break

            if col_nombres is None:
                # No hay columna de nombres; guardar tal cual
                df_cons.to_excel(writer, index=False, sheet_name=hoja[:31])
                resumen.append({"Hoja": hoja, "Coincidencias": 0, "Sin coincidencia": len(df_cons)})
                continue

            # Intentar extraer para esta hoja los nuevos puntajes desde el archivo complejo
            try:
                df_raw = pd.read_excel(xls_comp, sheet_name=hoja, header=None)
                df_new = extraer_datos(df_raw) if 'extraer_datos' in globals() else None
            except Exception:
                df_new = None

            if df_new is None or df_new.empty or "NOMBRE ESTUDIANTE" not in df_new.columns:
                # No hay datos nuevos; agregar columna vacía (si no existe) y guardar
                if "SIMCE Nuevo" not in df_cons.columns:
                    df_cons["SIMCE Nuevo"] = pd.NA
                df_cons.to_excel(writer, index=False, sheet_name=hoja[:31])
                resumen.append({"Hoja": hoja, "Coincidencias": 0, "Sin coincidencia": len(df_cons)})
                continue

            # Normalizar claves
            df_cons["__key"] = df_cons[col_nombres].map(_norm)
            df_new["__key"]  = df_new["NOMBRE ESTUDIANTE"].map(_norm)

            # Unir por clave normalizada (solo traemos la nota)
            df_merge = df_cons.merge(
                df_new[["__key", "SIMCE 1"]], on="__key", how="left"
            )

            # Crear la nueva columna con tipo numérico
            df_merge["SIMCE Nuevo"] = pd.to_numeric(df_merge["SIMCE 1"], errors="coerce")

            # Eliminar TODAS las columnas auxiliares que podrían colarse al final
            df_merge.drop(columns=["__key", "SIMCE 1"], inplace=True, errors="ignore")
            # MUY IMPORTANTE: no dejar la "NOMBRE ESTUDIANTE" del lado derecho del merge
            if "NOMBRE ESTUDIANTE" in df_merge.columns and "NOMBRE ESTUDIANTE" != col_nombres:
                df_merge.drop(columns=["NOMBRE ESTUDIANTE"], inplace=True)

            # Contar coincidencias
            coinc = int(df_merge["SIMCE Nuevo"].notna().sum())
            sinco = int(df_merge["SIMCE Nuevo"].isna().sum())
            resumen.append({"Hoja": hoja, "Coincidencias": coinc, "Sin coincidencia": sinco})

            # Guardar hoja
            df_merge.to_excel(writer, index=False, sheet_name=hoja[:31])

    # Mostrar resumen y permitir descarga
    st.subheader("📋 Resumen de consolidación")
    st.dataframe(pd.DataFrame(resumen))

    st.download_button(
        "📥 Descargar CONSOLIDADO ACTUALIZADO (todas las hojas)",
        data=output_consol.getvalue(),
        file_name="consolidado_actualizado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Reutilizar en funciones 4 y 5 sin volver a subir
    output_consol.seek(0)
    st.session_state["consolidado_xls"] = pd.ExcelFile(output_consol)
elif uploaded_consolidado and not uploaded_file:
    st.info("⚠️ Sube también el archivo complejo en la sección 'EXTRAER PUNTAJES' para poder consolidar.")


# ================================
# 🎯 FUNCIÓN 4: ANÁLISIS POR ESTUDIANTE
# ================================
import matplotlib.pyplot as plt

st.header("🎯 Análisis por estudiante")

# Verificamos que ya exista el consolidado en memoria
if "consolidado_xls" not in st.session_state:
    st.warning("⚠️ Primero debes ejecutar la función 3 (Consolidación de puntajes).")
else:
    xls_est = st.session_state["consolidado_xls"]
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

        # Extraer fila del estudiante
        df_est = df_curso[df_curso[col_nombres] == estudiante_sel].copy()

        # Detectar columnas de puntajes
        cols_puntajes = [
            c for c in df_est.columns
            if c != col_nombres and pd.api.types.is_numeric_dtype(df_est[c])
        ]
        for c in df_est.columns:
            if c != col_nombres and ("simce" in str(c).lower() or "puntaje" in str(c).lower()):
                if c not in cols_puntajes:
                    cols_puntajes.append(c)

        if not cols_puntajes:
            st.warning("No se encontraron columnas de puntajes en esta hoja.")
        else:
            # Orden cronológico aproximado
            cols_puntajes = sorted(cols_puntajes)

            puntajes = df_est[cols_puntajes].iloc[0].tolist()

            # Filtrar solo valores válidos
            x = [c for c, p in zip(cols_puntajes, puntajes) if pd.notna(p)]
            y = [p for p in puntajes if pd.notna(p)]

            if not y:
                st.info(f"No hay puntajes disponibles para {estudiante_sel}.")
            else:
                # Crear gráfico
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(x, y, marker="o", linestyle="-", color="blue")

                # Anotar valores en cada punto
                for i, (xi, yi) in enumerate(zip(x, y)):
                    ax.text(i, yi + 5, str(int(yi)), ha="center", fontsize=9)

                ax.set_title(f"Evolución del rendimiento - {estudiante_sel} ({curso_sel})")
                ax.set_ylabel("Puntaje")
                ax.set_xlabel("Ensayos")
                ax.grid(True)

                # 🔧 Ajustar etiquetas del eje X
                ax.set_xticks(range(len(x)))
                ax.set_xticklabels(x, fontsize=8, rotation=30)

                st.pyplot(fig)

                # Promedio
                promedio = sum(y) / len(y)
                st.success(f"📊 Puntaje promedio de {estudiante_sel}: **{promedio:.2f}**")


# ================================
# 📉 FUNCIÓN 5: ESTUDIANTES CON RENDIMIENTO MÁS BAJO
# ================================
st.header("📉 Estudiantes con rendimiento más bajo")

if "consolidado_xls" not in st.session_state:
    st.warning("⚠️ Primero debes ejecutar la función 3 (Consolidación de puntajes).")
else:
    xls_bajos = st.session_state["consolidado_xls"]

    for hoja in xls_bajos.sheet_names:
        df_curso = pd.read_excel(xls_bajos, sheet_name=hoja)

        col_nombres = None
        for col in df_curso.columns:
            if "nombre" in str(col).lower() and "estudiante" in str(col).lower():
                col_nombres = col
                break

        if col_nombres:
            cols_puntajes = [c for c in df_curso.columns if c != col_nombres and pd.api.types.is_numeric_dtype(df_curso[c])]
            if cols_puntajes:
                df_curso["Promedio"] = df_curso[cols_puntajes].mean(axis=1, skipna=True)
                df_bajos = df_curso[[col_nombres, "Promedio"]].sort_values("Promedio").head(10)
                st.subheader(f"📍 Curso {hoja}")
                st.table(df_bajos)

# ================================
# 📝 FUNCIÓN 6: ANÁLISIS DE PREGUNTAS Y DISTRACTORES 
# ================================
st.header("📝 Análisis de preguntas y distractores")

if uploaded_file:  # usamos el archivo ya cargado en la función 1
    xls_preg = pd.ExcelFile(uploaded_file)
    hojas_preg = xls_preg.sheet_names

    hoja_sel = st.selectbox("Elige el curso (hoja de Excel)", hojas_preg, key="hoja_preg_final")

    df_preg = pd.read_excel(xls_preg, sheet_name=hoja_sel, header=None)

    # Extraer claves correctas y números de preguntas
    claves = df_preg.iloc[8, 3:68].tolist()      # fila 9 → índice 8, columnas D=3 a BP
    preguntas = df_preg.iloc[9, 3:68].tolist()   # fila 10 → índice 9

    # Filtrar solo preguntas con clave no vacía
    valid_idx = [i for i, c in enumerate(claves) if pd.notna(c) and str(c).strip() != ""]
    claves = [claves[i] for i in valid_idx]
    preguntas = [preguntas[i] for i in valid_idx]

    # Respuestas de estudiantes (filas 11-56 → índices 10:56)
    respuestas = df_preg.iloc[10:56, 3:68]

    # Cálculo de % de aciertos por pregunta
    resumen = []
    for j, clave in zip(valid_idx, claves):
        col = respuestas.iloc[:, j]
        total = col.notna().sum()
        aciertos = (col.astype(str).str.lower() == str(clave).lower()).sum()
        pct = aciertos / total * 100 if total > 0 else 0

        # Extraer conteos de alternativas (filas 60-64 → índices 59:64)
        conteos = df_preg.iloc[59:64, j+3]  # +3 porque D=3
        alternativas = ["A", "B", "C", "D", "E"]
        dist = dict(zip(alternativas, conteos))

        # Omitir alternativa E si es duplicada de D
        if dist["D"] == dist["E"]:
            dist.pop("E")

        # Detectar distractores
        obs = ""
        total_resps = sum(dist.values())
        if total_resps > 0:
            dist_pct = {k: v/total_resps*100 for k, v in dist.items()}
            # Quitar la alternativa correcta
            dist_incorrectas = {k: v for k, v in dist_pct.items() if k.lower() != str(clave).lower()}

            if dist_incorrectas:
                # Distractor fuerte: alternativa incorrecta con >50%
                max_alt = max(dist_incorrectas, key=dist_incorrectas.get)
                if dist_incorrectas[max_alt] > 50:
                    obs = f"Distractor fuerte: {max_alt}"

                # Alta dispersión solo si % acierto < 50
                vals = list(dist_incorrectas.values())
                if pct < 50 and max(vals) - min(vals) < 10 and len(vals) > 1:
                    obs = "Alta dispersión"

        resumen.append({
            "Pregunta": preguntas[valid_idx.index(j)],
            "Correcta": clave,
            "% Aciertos": round(pct, 2),
            "Observación": obs
        })

    df_resumen = pd.DataFrame(resumen)
    st.subheader("📊 Resumen de preguntas críticas")
    st.dataframe(df_resumen)

    # Gráfico de % de aciertos
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_resumen["Pregunta"], df_resumen["% Aciertos"], color="skyblue")
    ax.set_title(f"% de aciertos por pregunta - {hoja_sel}")
    ax.set_xlabel("Pregunta")
    ax.set_ylabel("% Aciertos")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info("⚠️ Primero debes subir un archivo en la sección 'EXTRAER PUNTAJES'.")

