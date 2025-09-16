
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
# 📂 FUNCIÓN 3: CONSOLIDACIÓN DE PUNTAJES (match robusto de nombres + 'Puntaje Ensayo n')
# ================================
from io import BytesIO
import unicodedata, re
import numpy as np

st.header("📂 Consolidación de puntajes (emparejo robusto de nombres)")

# Stopwords típicas en nombres hispanos que no aportan a la coincidencia
_STOP = {"de", "del", "la", "las", "los", "y", "e", "da", "do", "das", "dos"}

def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower().replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> set:
    toks = [t for t in _norm_text(s).split() if t and t not in _STOP]
    return set(toks)

def _bow_key(tokens: set) -> str:
    # Clave insensible al orden
    return " ".join(sorted(tokens))

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _next_ensayo_col(df_cons: pd.DataFrame, col_nombres: str) -> str:
    # Buscar "Puntaje Ensayo N" existentes para definir el siguiente N
    nums = []
    for c in df_cons.columns:
        m = re.fullmatch(r"(?i)puntaje\s+ensayo\s+(\d+)", str(c).strip())
        if m:
            try:
                nums.append(int(m.group(1)))
            except:
                pass
    if nums:
        n = max(nums) + 1
    else:
        # Estimar n inicial según columnas que parecen puntajes
        cand = [c for c in df_cons.columns
                if c != col_nombres and re.search(r"(?i)(simce|puntaje|ensayo)", str(c))]
        n = max(2, len(cand) + 1)
    name = f"Puntaje Ensayo {n}"
    while name in df_cons.columns:
        n += 1
        name = f"Puntaje Ensayo {n}"
    return name

uploaded_consolidado = st.file_uploader(
    "Sube el archivo consolidado anterior (todas las hojas de cursos)",
    type=["xlsx"], key="consol_robusto"
)

if uploaded_file and uploaded_consolidado:
    xls_comp = pd.ExcelFile(uploaded_file)        # archivo complejo (Función 1)
    xls_cons = pd.ExcelFile(uploaded_consolidado) # consolidado histórico

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
                df_cons.to_excel(writer, index=False, sheet_name=hoja[:31])
                resumen.append({"Hoja": hoja, "Coincidencias": 0, "Sin coincidencia": len(df_cons)})
                continue

            # Extraer datos nuevos desde el archivo complejo con tu extractor
            try:
                df_raw = pd.read_excel(xls_comp, sheet_name=hoja, header=None)
                df_new = extraer_datos(df_raw) if 'extraer_datos' in globals() else None
            except Exception:
                df_new = None

            if df_new is None or df_new.empty or "NOMBRE ESTUDIANTE" not in df_new.columns:
                df_cons.to_excel(writer, index=False, sheet_name=hoja[:31])
                resumen.append({"Hoja": hoja, "Coincidencias": 0, "Sin coincidencia": len(df_cons)})
                continue

            # --- Preparar claves robustas (bolsa de tokens) ---
            df_cons["__tokens"] = df_cons[col_nombres].map(_tokens)
            df_cons["__key"]    = df_cons["__tokens"].map(_bow_key)

            df_new = df_new.copy()
            df_new["__tokens"]  = df_new["NOMBRE ESTUDIANTE"].map(_tokens)
            df_new["__key"]     = df_new["__tokens"].map(_bow_key)
            df_new["SIMCE 1"]   = pd.to_numeric(df_new["SIMCE 1"], errors="coerce")

            # 1) Emparejo primario por clave BOW (insensible al orden)
            df_merge = df_cons.merge(
                df_new[["__key", "__tokens", "SIMCE 1"]],
                on="__key", how="left", suffixes=("", "_new")
            )

            # 2) Fallback: Jaccard si quedó sin match (maneja nombres incompletos/extra tokens)
            no_match_idx = df_merge.index[df_merge["SIMCE 1"].isna()].tolist()
            # Lista de candidatos (índice -> (tokens, puntaje))
            candidates = list(df_new[["__tokens", "SIMCE 1"]].itertuples(index=False, name=None))

            used = set()  # para no reutilizar el mismo registro nuevo muchas veces
            for idx in no_match_idx:
                toks_cons = df_merge.at[idx, "__tokens"]
                best_sim, best_val, best_j = 0.0, np.nan, -1
                for j, (toks_new, val) in enumerate(candidates):
                    if j in used:
                        continue
                    sim = _jaccard(toks_cons, toks_new)
                    # criterio: similitud alta o subset casi completo
                    if sim > best_sim:
                        best_sim, best_val, best_j = sim, val, j
                # Umbral razonable (≥0.75) o subset fuerte (≥0.66 con al menos 2 tokens)
                if (best_sim >= 0.75) or (len(toks_cons) >= 2 and best_sim >= 0.66):
                    df_merge.at[idx, "SIMCE 1"] = best_val
                    used.add(best_j)

            # 3) Crear nueva columna 'Puntaje Ensayo n'
            nueva_col = _next_ensayo_col(df_cons, col_nombres)
            df_merge[nueva_col] = pd.to_numeric(df_merge["SIMCE 1"], errors="coerce")

            # 4) Limpiar auxiliares
            df_merge.drop(columns=["__key", "__tokens", "SIMCE 1"], inplace=True, errors="ignore")

            # 5) Resumen y escritura
            coinc = int(df_merge[nueva_col].notna().sum())
            sinco = int(df_merge[nueva_col].isna().sum())
            resumen.append({"Hoja": hoja, "Coincidencias": coinc, "Sin coincidencia": sinco, "Nueva columna": nueva_col})

            df_merge.to_excel(writer, index=False, sheet_name=hoja[:31])

    # Mostrar resumen y habilitar descarga
    st.subheader("📋 Resumen de consolidación")
    st.dataframe(pd.DataFrame(resumen))

    st.download_button(
        "📥 Descargar CONSOLIDADO ACTUALIZADO (todas las hojas)",
        data=output_consol.getvalue(),
        file_name="consolidado_actualizado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Guardar en memoria para funciones 4/5/6
    output_consol.seek(0)
    st.session_state["consolidado_xls"] = pd.ExcelFile(output_consol)

elif uploaded_consolidado and not uploaded_file:
    st.info("⚠️ Sube también el archivo complejo en la sección 'EXTRAER PUNTAJES' para poder consolidar.")

# ================================
# 🎯 FUNCIÓN 4: ANÁLISIS POR ESTUDIANTE (con conversión robusta de puntajes)
# ================================
import matplotlib.pyplot as plt
import numpy as np
import re

st.header("🎯 Análisis por estudiante")

# Reutiliza el consolidado creado en la Función 3
if "consolidado_xls" not in st.session_state:
    st.warning("⚠️ Primero debes ejecutar la función 3 (Consolidación de puntajes).")
else:
    xls_est = st.session_state["consolidado_xls"]
    hojas_est = xls_est.sheet_names

    # Selección de curso (hoja)
    curso_sel = st.selectbox("Elige el curso (hoja de Excel)", hojas_est, key="f4_curso_sel")
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
        estudiantes_opciones = df_curso[col_nombres].dropna().unique()
        estudiante_sel = st.selectbox("Elige un estudiante", estudiantes_opciones, key="f4_estudiante_sel")

        # Fila del estudiante
        df_est = df_curso[df_curso[col_nombres] == estudiante_sel].copy()
        if df_est.empty:
            st.info("No se encontró información para el estudiante seleccionado.")
        else:
            # Detectar posibles columnas de puntajes conservando el orden original de la hoja
            cols_puntajes = []
            for c in df_est.columns:
                if c == col_nombres:
                    continue
                serie = df_est[c]
                is_num = pd.api.types.is_numeric_dtype(serie)
                has_kw = ("simce" in str(c).lower()) or ("puntaje" in str(c).lower()) or ("ensayo" in str(c).lower())
                if is_num or has_kw:
                    cols_puntajes.append(c)

            if not cols_puntajes:
                st.warning("No se encontraron columnas de puntajes en esta hoja.")
            else:
                # --- Conversión robusta a numérico (sin romper decimales ya numéricos) ---
                row_raw = df_est[cols_puntajes].iloc[0]

                def parse_val(v):
                    # Ya numérico
                    if isinstance(v, (int, float, np.number)) and not pd.isna(v):
                        return float(v)
                    # A texto normalizado
                    s = "" if pd.isna(v) else str(v).strip()
                    if s == "" or s.lower() in {"nan", "none", "-"}:
                        return np.nan
                    s = s.replace("\u00a0", " ")  # NBSP
                    # Casos con ambos separadores estilo EU: 1.234,56
                    if "." in s and "," in s:
                        s = s.replace(".", "").replace(",", ".")
                    # Solo coma decimal: 269,5652174
                    elif "," in s and "." not in s:
                        s = s.replace(",", ".")
                    # Quitar cualquier carácter no numérico relevante (mantener solo dígitos, primer punto y signo)
                    s = re.sub(r"[^0-9\.\-]", "", s)
                    # Asegurar solo un punto decimal
                    if s.count(".") > 1:
                        first = s.find(".")
                        s = s[:first+1] + s[first+1:].replace(".", "")
                    try:
                        return float(s)
                    except:
                        return np.nan

                row_num = row_raw.apply(parse_val)

                mask = row_num.notna()
                x_labels = list(row_num.index[mask])
                y_vals = list(row_num[mask].astype(float))

                if not y_vals:
                    st.info(f"No hay puntajes disponibles para {estudiante_sel}.")
                else:
                    # Gráfico líneas + puntos
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(range(len(x_labels)), y_vals, marker="o", linestyle="-")

                    # Anotar valores con desplazamiento proporcional
                    if len(y_vals) > 1:
                        offset = (max(y_vals) - min(y_vals)) * 0.03
                    else:
                        offset = 5
                    for i, yi in enumerate(y_vals):
                        ax.text(i, yi + offset, f"{yi:.2f}", ha="center", fontsize=9)

                    ax.set_title(f"Evolución del rendimiento - {estudiante_sel} ({curso_sel})")
                    ax.set_ylabel("Puntaje")
                    ax.set_xlabel("Ensayos")
                    ax.grid(True)
                    ax.set_xticks(range(len(x_labels)))
                    ax.set_xticklabels(x_labels, fontsize=8, rotation=30)

                    st.pyplot(fig)

                    # Promedio (solo válidos)
                    promedio = float(np.nanmean(np.array(y_vals, dtype=float)))
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

