# app.py
# ===================================================
# Extractor y Consolidador SIMCE / PAES
# ===================================================

import re
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Extractor y Consolidador SIMCE/PAES", layout="wide")

# ================================
# Encabezado con logo y título centrado
# ================================
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    try:
        st.image("assets/logo.png", width=90)
    except Exception:
        pass
with col2:
    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style="margin-bottom:0;">Departamento de Lenguaje</h2>
            <h4 style="margin-top:0;">Liceo Polivalente San Nicolás</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===================================================
# Utilidades comunes
# ===================================================

_STOP = {"de","del","la","las","los","y","e","da","do","das","dos"}

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
    return {t for t in _norm_text(s).split() if t and t not in _STOP}

def _bow_key(tokens: set) -> str:
    return " ".join(sorted(tokens))

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    inter = len(a & b); union = len(a | b)
    return inter/union if union else 0.0

def _next_ensayo_col(df_cons: pd.DataFrame, col_nombres: str) -> str:
    """Devuelve el próximo nombre disponible 'Puntaje Ensayo n' en df_cons."""
    nums = []
    for c in df_cons.columns:
        m = re.fullmatch(r"(?i)puntaje\s+ensayo\s+(\d+)", str(c).strip())
        if m:
            try: nums.append(int(m.group(1)))
            except: pass
    if nums:
        n = max(nums) + 1
    else:
        cand = [c for c in df_cons.columns
                if c != col_nombres and re.search(r"(?i)(simce|puntaje|ensayo)", str(c))]
        n = max(2, len(cand)+1)
    name = f"Puntaje Ensayo {n}"
    while name in df_cons.columns:
        n += 1
        name = f"Puntaje Ensayo {n}"
    return name

def _parse_numeric_series(series: pd.Series) -> pd.Series:
    """Convierte en forma robusta una serie (mezcla número/texto) a float (NaN si no se puede)."""
    s = series.astype(str).str.strip()
    s = s.str.replace("\u00a0", " ", regex=False)
    s = s.str.replace(r"[^0-9\.,\-]+", "", regex=True)
    def _fix(x: str) -> str:
        if x == "" or x.lower() in {"nan", "none", "-"}: return ""
        if "." in x and "," in x:
            x = x.replace(".", "").replace(",", ".")
        elif "," in x and "." not in x:
            x = x.replace(",", ".")
        if x.count(".") > 1:
            first = x.find(".")
            x = x[:first+1] + x[first+1:].replace(".", "")
        return x
    s = s.apply(_fix)
    return pd.to_numeric(s, errors="coerce")

def _detectar_col_puntaje(df: pd.DataFrame):
    """Devuelve el nombre de la columna de puntaje en un normalizado/consolidado."""
    preferidas = ["Puntaje Ensayo 1", "FK", "TOTAL"]
    for c in preferidas:
        if c in df.columns:
            return c
    for c in df.columns:
        n = str(c).lower()
        if any(k in n for k in ("puntaje", "simce", "ensayo")) and c != "NOMBRE ESTUDIANTE":
            return c
    for c in df.columns:
        if c != "NOMBRE ESTUDIANTE" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

# ===================================================
# FUNCIÓN 1: EXTRAER PUNTAJES (archivo complejo -> normalizado por hoja)
# ===================================================
st.header("📥 EXTRAER PUNTAJES (archivo complejo)")

def extraer_datos(
    df_raw,
    fila_encabezado: int = 9,   # 10 (humano) -> 9 (0-based)
    fila_ini: int = 10,         # 11 (humano) -> 10
    fila_fin: int = 80,         # exclusivo (por si hay más de 56 filas reales)
    fallback_nombre_col: int = 2  # C (0-based)
):
    """
    Extrae nombres y puntajes desde una hoja del archivo complejo tipo SIMCE/Ensayo.
    Devuelve DataFrame con columnas ['NOMBRE ESTUDIANTE', 'Puntaje Ensayo 1'] (float).
    Incluye un FALBACK fila-a-fila para puntajes faltantes.
    """
    headers = df_raw.iloc[fila_encabezado].astype(str).str.lower().fillna("")

    # --- Columna de nombres ---
    cand_name_cols = [i for i, v in enumerate(headers) if ("nombre" in v and "estudiante" in v)]
    col_name = cand_name_cols[0] if cand_name_cols else fallback_nombre_col

    # --- Columna de puntaje (por encabezado) ---
    cand_score_cols = [i for i, v in enumerate(headers)
                       if ("puntaje" in v and ("simce" in v or "ensayo" in v))]
    col_score = cand_score_cols[0] if cand_score_cols else None

    # --- Rango de estudiantes ---
    sub = df_raw.iloc[fila_ini:fila_fin].copy()

    # --- Nombres y filtro de válidos ---
    nombres = sub.iloc[:, col_name].astype(str).str.strip()
    invalid = {"", "nan", "nombre estudiante", "curso", "correctas", "a", "b", "c", "d", "e"}
    mask_valid = ~nombres.str.lower().isin(invalid)

    # ---------- Utilidades numéricas ----------
    def _to_num_str(s: str) -> str:
        if s is None:
            return ""
        s = str(s).strip()
        if s == "":
            return ""
        if "." in s and "," in s:
            s = s.replace(".", "").replace(",", ".")   # 1.234,56 -> 1234.56
        elif "," in s and "." not in s:
            s = s.replace(",", ".")                    # 269,56 -> 269.56
        if s.count(".") > 1:
            first = s.find(".")
            s = s[:first+1] + s[first+1:].replace(".", "")
        return s

    def _serie_to_num(sr: pd.Series) -> pd.Series:
        sr = sr.astype(str).str.strip()
        sr = sr.str.replace("\u00a0", " ", regex=False)
        sr = sr.replace({'^[oO-]$': ''}, regex=True)      # 'o'/'O'/'-' -> omitida
        sr = sr.str.replace(r"[^0-9\.,\-]+", "", regex=True)
        sr = sr.apply(_to_num_str)
        return pd.to_numeric(sr, errors="coerce")

    # --- Elegir mejor columna de puntaje si no hay encabezado claro ---
    def _col_mas_numerica(df_slice):
        best_col, best_score = None, -1.0
        for j in range(df_slice.shape[1]):
            nums = _serie_to_num(df_slice.iloc[:, j])
            valid = nums.between(0, 1000, inclusive="both")
            score = valid.mean()
            if score > best_score:
                best_score, best_col = score, j
        return best_col

    if col_score is None:
        col_score = _col_mas_numerica(sub)

    # --- Lectura inicial de puntajes ---
    puntajes = _serie_to_num(sub.iloc[:, col_score])
    # Alinear con filas válidas de nombre
    nombres_valid = nombres[mask_valid]
    puntajes_valid = puntajes[mask_valid].copy()

    # --- Fallback fila-a-fila para faltantes ---
    headers_full = headers
    sub_valid = sub[mask_valid].copy()

    def _mejor_puntaje_fila(row: pd.Series) -> float:
        mejor = np.nan
        mejor_desde_header = False
        for j, cell in enumerate(row):
            num = _serie_to_num(pd.Series([cell])).iloc[0]
            if pd.isna(num) or not (100 <= num <= 1000):
                continue
            h = str(headers_full.iloc[j]).lower()
            es_header_score = ("puntaje" in h) or ("simce" in h) or ("ensayo" in h)
            if es_header_score:
                mejor = num
                mejor_desde_header = True
            elif not mejor_desde_header:
                if pd.isna(mejor) or num > mejor:
                    mejor = num
        return mejor

    faltantes_idx = puntajes_valid.index[puntajes_valid.isna()]
    if len(faltantes_idx) > 0:
        fallback_series = sub_valid.apply(_mejor_puntaje_fila, axis=1)
        puntajes_valid = puntajes_valid.combine_first(fallback_series)

    # --- Salida ---
    out = pd.DataFrame({
        "NOMBRE ESTUDIANTE": nombres_valid.str.replace(r"\s+", " ", regex=True).str.strip().values,
        "Puntaje Ensayo 1": puntajes_valid.values
    })
    out = out[out["NOMBRE ESTUDIANTE"] != ""].reset_index(drop=True)
    return out

uploaded_file = st.file_uploader("Sube el archivo Excel complejo (múltiples hojas)", type=["xlsx"], key="archivo_complejo")

if uploaded_file:
    complex_bytes = uploaded_file.getvalue()
    st.session_state["complex_bytes"] = complex_bytes
    st.session_state["xls_complex"] = pd.ExcelFile(BytesIO(complex_bytes))

    xls = st.session_state["xls_complex"]
    hojas_validas = xls.sheet_names
    st.write("Hojas detectadas:", hojas_validas)

    df_cursos = {}
    for hoja in hojas_validas:
        st.subheader(f"Procesando hoja: {hoja}")
        df_raw = pd.read_excel(xls, sheet_name=hoja, header=None)
        df_extraido = extraer_datos(df_raw)
        if df_extraido is not None and not df_extraido.empty:
            st.dataframe(df_extraido.head(10))
            df_cursos[hoja] = df_extraido

            out = BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                df_extraido.to_excel(writer, index=False, sheet_name='Resultados')
            st.download_button(
                label=f"📥 Descargar resultados hoja {hoja}",
                data=out.getvalue(),
                file_name=f"resultados_{hoja}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning(f"No se extrajeron datos en {hoja}.")

    if df_cursos:
        st.session_state["df_cursos"] = df_cursos

    st.subheader("📦 Exportar todas las hojas a un único Excel normalizado")
    combined_output = BytesIO()
    with pd.ExcelWriter(combined_output, engine='xlsxwriter') as writer:
        for hoja, dfh in df_cursos.items():
            dfh.to_excel(writer, index=False, sheet_name=hoja[:31])
    st.download_button(
        label="📥 Descargar Excel Normalizado con todas las hojas",
        data=combined_output.getvalue(),
        file_name="excel_normalizado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===================================================
# FUNCIÓN 2: ANÁLISIS POR CURSO (SIMCE/PAES)
# ===================================================
st.header("📊 Análisis por curso")

criterio = st.radio("Elige criterio", ["SIMCE", "PAES"], horizontal=True, key="crit_curso")

if "df_cursos" in st.session_state:
    dfc = st.session_state["df_cursos"]

    def clasificar(p):
        try:
            x = float(p)
        except:
            return "Sin datos"
        if criterio == "SIMCE":
            if 0 <= x <= 250: return "Insuficiente"
            if 251 <= x <= 285: return "Intermedio"
            if 285 < x <= 400: return "Adecuado"
            return "Sin datos"
        else:  # PAES
            if 0 <= x <= 599: return "Insuficiente"
            if 600 <= x <= 799: return "Intermedio"
            if 800 <= x <= 1000: return "Adecuado"
            return "Sin datos"

    total_counts = {"Insuficiente":0, "Intermedio":0, "Adecuado":0}

    cols = st.columns(2)
    i = 0
    for hoja, dfh in dfc.items():
        col = cols[i % 2]
        with col:
            score_col = _detectar_col_puntaje(dfh)
            if not score_col:
                st.warning(f"No se encontró columna de puntajes en {hoja}.")
                continue
            serie = dfh[score_col].apply(clasificar)
            counts = serie.value_counts().reindex(["Insuficiente","Intermedio","Adecuado"], fill_value=0)
            for k in total_counts: total_counts[k] += int(counts[k])

            fig, ax = plt.subplots(figsize=(5.5, 4))
            vals = counts.values
            labels = counts.index.tolist()
            ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90, shadow=True)
            ax.set_title(f"Distribución por rendimiento - {hoja}")
            st.pyplot(fig)
        i += 1

    st.subheader("🔵 Distribución global (todos los cursos)")
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = list(total_counts.values())
    labels = list(total_counts.keys())
    ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90, shadow=True)
    ax.set_title("Distribución global por rendimiento")
    st.pyplot(fig)
else:
    st.info("Primero ejecuta 'EXTRAER PUNTAJES'.")

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
                if "Puntaje Ensayo X (obsoleto)" not in df_cons.columns:
                    df_cons["Puntaje Ensayo X (obsoleto)"] = pd.NA
                df_cons.to_excel(writer, index=False, sheet_name=hoja[:31])
                resumen.append({"Hoja": hoja, "Coincidencias": 0, "Sin coincidencia": len(df_cons)})
                continue

            # Normalizar claves
            df_cons["__key"] = df_cons[col_nombres].map(_norm)
            df_new["__key"]  = df_new["NOMBRE ESTUDIANTE"].map(_norm)

            # Determinar el nombre del nuevo puntaje (correlativo)
            n_existentes = sum("Puntaje Ensayo" in str(c) for c in df_cons.columns)
            nuevo_nombre = f"Puntaje Ensayo {n_existentes + 1}"

            # Normalizar columnas en df_new
            df_new.columns = df_new.columns.astype(str).str.strip()

            # Detectar la columna de puntajes en df_new
            col_puntaje_new = None
            if "Puntaje Ensayo 1" in df_new.columns:
                col_puntaje_new = "Puntaje Ensayo 1"
            else:
                for col in df_new.columns:
                    col_low = str(col).lower().strip()
                    if ("simce" in col_low or "puntaje" in col_low or "ensayo" in col_low 
                        or col_low == "total" or col_low == "fk"):
                        col_puntaje_new = col
                        break
            if col_puntaje_new is None:
                for col in df_new.columns:
                    if col not in ("NOMBRE ESTUDIANTE", "__key") and pd.api.types.is_numeric_dtype(df_new[col]):
                        col_puntaje_new = col
                        break

            if col_puntaje_new is None:
                st.warning(f"No se encontró columna de puntajes en los datos nuevos para {hoja}.")
                df_cons.to_excel(writer, index=False, sheet_name=hoja[:31])
                resumen.append({"Hoja": hoja, "Coincidencias": 0, "Sin coincidencia": len(df_cons)})
                continue

            # Unir por clave normalizada (solo traemos la nota y la renombramos)
            df_tmp = df_new[["__key", col_puntaje_new]].copy()
            df_tmp.rename(columns={col_puntaje_new: nuevo_nombre}, inplace=True)

            df_merge = df_cons.merge(df_tmp, on="__key", how="left")

            # Asegurar tipo numérico solo si la columna fue creada

                col_data = df_merge[nuevo_nombre]
                if isinstance(col_data, pd.DataFrame):
                    # Si hay duplicados, tomar la primera columna
                    serie = col_data.iloc[:, 0]
                else:
                    serie = col_data

                # Forzar unicidad de columnas (eliminar duplicados)
                df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()]
            else:
                st.warning(f"No se creó la columna {nuevo_nombre} en {hoja}.")

                                # Conversión segura a numérico
        if nuevo_nombre in df_merge.columns:
            col_data = df_merge[nuevo_nombre]
            if isinstance(col_data, pd.DataFrame):
                serie = col_data.iloc[:, 0]   # tomar la primera si hay duplicados
            else:
                serie = col_data
            df_merge[nuevo_nombre] = pd.to_numeric(serie, errors="coerce")
            # eliminar duplicados de columnas
            df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()]
        else:
            st.warning(f"No se creó la columna {nuevo_nombre} en {hoja}.")

# Limpiar columnas auxiliares
            df_merge.drop(columns=["__key"], inplace=True, errors="ignore")

            # Eliminar columnas duplicadas _x/_y que hayan quedado del merge
            for c in list(df_merge.columns):
                if c.endswith("_x") or c.endswith("_y"):
                    df_merge.drop(columns=[c], inplace=True, errors="ignore")

            # Extra: si quedaron duplicados de 'Puntaje Ensayo 1', normalizarlos
            cols_dup = [c for c in df_merge.columns if "Puntaje Ensayo 1" in str(c) and ("_x" in c or "_y" in c)]
            for c in cols_dup:
                df_merge.drop(columns=[c], inplace=True, errors="ignore")
            df_merge.drop(columns=["__key"], inplace=True, errors="ignore")
            for c in list(df_merge.columns):
                if c.endswith("_x") or c.endswith("_y"):
                    df_merge.drop(columns=[c], inplace=True, errors="ignore")
            df_merge.drop(columns=["__key"], inplace=True, errors="ignore")
            for c in list(df_merge.columns):
                if c.endswith("_x") or c.endswith("_y"):
                    df_merge.drop(columns=[c], inplace=True, errors="ignore")
            if col_puntaje_new in df_merge.columns:
                df_merge.rename(columns={col_puntaje_new: nuevo_nombre}, inplace=True)

                                    # Conversión segura a numérico
        if nuevo_nombre in df_merge.columns:
            col_data = df_merge[nuevo_nombre]
            if isinstance(col_data, pd.DataFrame):
                serie = col_data.iloc[:, 0]   # tomar la primera si hay duplicados
            else:
                serie = col_data
            df_merge[nuevo_nombre] = pd.to_numeric(serie, errors="coerce")
            # eliminar duplicados de columnas
            df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()]
        else:
            st.warning(f"No se creó la columna {nuevo_nombre} en {hoja}.")

# Eliminar auxiliares
            df_merge.drop(columns=["__key"], inplace=True, errors="ignore")
            if "NOMBRE ESTUDIANTE" in df_merge.columns and "NOMBRE ESTUDIANTE" != col_nombres:
                df_merge.drop(columns=["NOMBRE ESTUDIANTE"], inplace=True)

            # Calcular coincidencias

                coinc = int(df_merge[nuevo_nombre].notna().sum())
                sin_coinc = len(df_merge) - coinc
            else:
                coinc, sin_coinc = 0, len(df_merge)
            resumen.append({
                "Hoja": hoja,
                "Coincidencias": coinc,
                "Sin coincidencia": sin_coinc
            })

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

# ===================================================
# FUNCIÓN 4: ANÁLISIS POR ESTUDIANTE (reutiliza consolidado)
# ===================================================
st.header("🎯 Análisis por estudiante")

if "consolidado_bytes" not in st.session_state:
    st.warning("⚠️ Primero ejecuta la función 3 (Consolidación de puntajes).")
else:
    xls_est = pd.ExcelFile(BytesIO(st.session_state["consolidado_bytes"]))
    hojas_est = xls_est.sheet_names

    curso_sel = st.selectbox("Elige el curso (hoja de Excel)", hojas_est, key="f4_curso_sel")
    df_curso = pd.read_excel(xls_est, sheet_name=curso_sel)

    # detectar columna de nombres
    col_nombres = None
    for col in df_curso.columns:
        if "nombre" in str(col).lower() and "estudiante" in str(col).lower():
            col_nombres = col
            break

    if col_nombres is None:
        st.error("No se encontró columna de nombres en esta hoja.")
    else:
        estudiantes_opciones = df_curso[col_nombres].dropna().unique()
        estudiante_sel = st.selectbox("Elige un estudiante", estudiantes_opciones, key="f4_estudiante_sel")

        df_est = df_curso[df_curso[col_nombres] == estudiante_sel].copy()
        if df_est.empty:
            st.info("No se encontró información para el estudiante seleccionado.")
        else:
            # columnas de puntajes (mantener orden original)
            cols_puntajes = []
            for c in df_est.columns:
                if c == col_nombres: continue
                serie = df_est[c]
                is_num = pd.api.types.is_numeric_dtype(serie)
                has_kw = any(k in str(c).lower() for k in ("simce","puntaje","ensayo"))
                if is_num or has_kw:
                    cols_puntajes.append(c)

            if not cols_puntajes:
                st.warning("No se encontraron columnas de puntajes en esta hoja.")
            else:
                row_raw = df_est[cols_puntajes].iloc[0]
                row_num = _parse_numeric_series(row_raw)

                mask = row_num.notna()
                x_labels = list(row_num.index[mask])
                y_vals = list(row_num[mask].astype(float))

                if not y_vals:
                    st.info(f"No hay puntajes disponibles para {estudiante_sel}.")
                else:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(range(len(x_labels)), y_vals, marker="o", linestyle="-")
                    offset = (max(y_vals) - min(y_vals)) * 0.03 if len(y_vals) > 1 else 5
                    for i, yi in enumerate(y_vals):
                        ax.text(i, yi + offset, f"{yi:.2f}", ha="center", fontsize=9)
                    ax.set_title(f"Evolución del rendimiento - {estudiante_sel} ({curso_sel})")
                    ax.set_ylabel("Puntaje")
                    ax.set_xlabel("Ensayos")
                    ax.grid(True)
                    ax.set_xticks(range(len(x_labels)))
                    ax.set_xticklabels(x_labels, fontsize=8, rotation=30)
                    st.pyplot(fig)

                    promedio = float(np.nanmean(np.array(y_vals, dtype=float)))
                    st.success(f"📊 Puntaje promedio de {estudiante_sel}: **{promedio:.2f}**")

# ===================================================
# FUNCIÓN 5: ESTUDIANTES CON RENDIMIENTO MÁS BAJO (reutiliza consolidado)
# ===================================================
st.header("📉 Estudiantes con rendimiento más bajo")

if "consolidado_bytes" not in st.session_state:
    st.warning("⚠️ Primero ejecuta la función 3 (Consolidación de puntajes).")
else:
    xls_low = pd.ExcelFile(BytesIO(st.session_state["consolidado_bytes"]))
    hojas_low = xls_low.sheet_names

    for hoja in hojas_low:
        st.subheader(f"Curso: {hoja}")
        df_low = pd.read_excel(xls_low, sheet_name=hoja)

        # columna de nombres
        col_nombres = None
        for col in df_low.columns:
            if "nombre" in str(col).lower() and "estudiante" in str(col).lower():
                col_nombres = col
                break
        if col_nombres is None:
            st.warning(f"No se encontró columna de nombres en {hoja}")
            continue

        # detectar última columna de ensayos tipo "Puntaje Ensayo n"
        ensayos = []
        for c in df_low.columns:
            m = re.fullmatch(r"(?i)puntaje\s+ensayo\s+(\d+)", str(c).strip())
            if m:
                try: ensayos.append((int(m.group(1)), c))
                except: pass
        if ensayos:
            ensayos.sort()
            ultima_col = ensayos[-1][1]
        else:
            cand = [c for c in df_low.columns if c != col_nombres and pd.api.types.is_numeric_dtype(df_low[c])]
            if not cand:
                st.warning(f"No hay columnas de puntajes en {hoja}")
                continue
            ultima_col = cand[-1]

        df_show = df_low[[col_nombres, ultima_col]].copy()
        df_show[ultima_col] = _parse_numeric_series(df_show[ultima_col])
        df_top10 = df_show.dropna(subset=[ultima_col]).sort_values(by=ultima_col, ascending=True).head(10)
        st.table(df_top10)

# ===================================================
# FUNCIÓN 6: ANÁLISIS DE PREGUNTAS Y DISTRACTORES (reutiliza archivo complejo)
# ===================================================
st.header("📝 Análisis de preguntas y distractores")

if "xls_complex" not in st.session_state:
    st.info("⚠️ Primero sube el archivo complejo en 'EXTRAER PUNTAJES'.")
else:
    xls_preg = st.session_state["xls_complex"]
    hojas_preg = xls_preg.sheet_names
    hoja_sel = st.selectbox("Elige el curso (hoja de Excel) para analizar preguntas", hojas_preg, key="f6_hoja")

    df_preg = pd.read_excel(xls_preg, sheet_name=hoja_sel, header=None)
    # Claves y preguntas
    claves = df_preg.iloc[8, 3:68].tolist()     # D9:BP9
    preguntas = df_preg.iloc[9, 3:68].tolist()  # D10:BP10

    valid_idx = [i for i, c in enumerate(claves) if pd.notna(c) and str(c).strip() != ""]
    claves_fil = [claves[i] for i in valid_idx]
    preguntas_fil = [preguntas[i] for i in valid_idx]

    # Respuestas de estudiantes
    respuestas = df_preg.iloc[10:80, 3:68]      # D11:BP80 (por si hay más filas)

    resumen = []
    for pos, clave in zip(valid_idx, claves_fil):
        col = respuestas.iloc[:, pos]
        total = col.notna().sum()
        aciertos = (col.astype(str).str.strip().str.lower() == str(clave).strip().lower()).sum()
        pct = aciertos / total * 100 if total > 0 else 0

        # Conteos de alternativas A..E
        conteos_col = 3 + pos
        conteos = df_preg.iloc[59:64, conteos_col]  # filas 60-64 (A..E)
        alternativas = ["A", "B", "C", "D", "E"]
        dist = dict(zip(alternativas, conteos))

        # Omitir E si duplica a D
        try:
            if dist["D"] == dist["E"]:
                dist.pop("E")
        except Exception:
            pass

        # Observaciones
        obs = ""
        total_resps = float(sum([v for v in dist.values() if pd.notna(v)])) if dist else 0.0
        if total_resps > 0:
            dist_pct = {}
            for k, v in dist.items():
                try:
                    dist_pct[k] = float(v) / total_resps * 100.0
                except:
                    dist_pct[k] = 0.0
            dist_inc = {k: v for k, v in dist_pct.items() if k.lower() != str(clave).lower()}
            if dist_inc:
                max_alt = max(dist_inc, key=dist_inc.get)
                if dist_inc[max_alt] > 50:
                    obs = f"Distractor fuerte: {max_alt}"
                vals = list(dist_inc.values())
                if pct < 50 and len(vals) > 1 and (max(vals) - min(vals) < 10):
                    obs = "Alta dispersión"

        resumen.append({
            "Pregunta": preguntas_fil[valid_idx.index(pos)],
            "Correcta": clave,
            "% Aciertos": round(pct, 2),
            "Observación": obs
        })

    df_resumen = pd.DataFrame(resumen)
    st.subheader("📊 Resumen de preguntas críticas")
    st.dataframe(df_resumen)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_resumen["Pregunta"], df_resumen["% Aciertos"])
    ax.set_title(f"% de aciertos por pregunta - {hoja_sel}")
    ax.set_xlabel("Pregunta")
    ax.set_ylabel("% Aciertos")
    plt.xticks(rotation=45)
    st.pyplot(fig)
