import re
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Extractor y Consolidador SIMCE/PAES v2.1", layout="wide")

# ===================================================
# Funciones utilitarias
# ===================================================

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower().replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _parse_numeric_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace("\u00a0", " ", regex=False)
    s = s.str.replace(r"[^0-9\.,\-]+", "", regex=True)
    def _fix(x: str) -> str:
        if x == "" or x.lower() in {"nan", "none", "-"}:
            return ""
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
    # Detecta la columna de puntaje más probable
    preferidas = ["Puntaje Ensayo 1", "SIMCE 1", "TOTAL", "FK"]
    for c in preferidas:
        if c in df.columns:
            return c
    for c in df.columns:
        n = str(c).lower()
        if any(k in n for k in ("puntaje", "simce", "ensayo")) and "nombre" not in n:
            return c
    for c in df.columns:
        if c != "NOMBRE ESTUDIANTE" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def _next_ensayo_col(df_cons: pd.DataFrame, col_nombres: str) -> str:
    nums = []
    for c in df_cons.columns:
        m = re.fullmatch(r"(?i)puntaje\s+ensayo\s+(\d+)", str(c).strip())
        if m:
            try:
                nums.append(int(m.group(1)))
            except:
                pass
    n = max(nums) + 1 if nums else 1
    name = f"Puntaje Ensayo {n}"
    while name in df_cons.columns:
        n += 1
        name = f"Puntaje Ensayo {n}"
    return name

# ===================================================
# Carga de archivos con cacheo seguro
# ===================================================

@st.cache_data
def load_excel_bytes(b: bytes) -> dict:
    xl = pd.ExcelFile(BytesIO(b))
    return {sheet: pd.read_excel(xl, sheet_name=sheet) for sheet in xl.sheet_names}

# ===================================================
# Funciones de la App
# ===================================================

st.title("📊 Extractor y Consolidador SIMCE/PAES v2.1")

# Función 1: Extraer puntajes
st.header("📥 Función 1: Extraer puntajes de archivo complejo")
up_complex = st.file_uploader("Sube archivo complejo (xlsx)", type=["xlsx"], key="complex")
if up_complex:
    df_cursos = load_excel_bytes(up_complex.getvalue())
    st.session_state["df_cursos"] = df_cursos
    st.success(f"Se cargaron {len(df_cursos)} hojas: {list(df_cursos.keys())}")
    st.dataframe(next(iter(df_cursos.values())).head())

# Función 2: Análisis por curso
st.header("📊 Función 2: Análisis por curso")
criterio = st.radio("Elige criterio", ["SIMCE", "PAES"], horizontal=True)
if "df_cursos" in st.session_state:
    dfc = st.session_state["df_cursos"]
    total_counts = {"Insuficiente":0, "Intermedio":0, "Adecuado":0}
    for hoja, dfh in dfc.items():
        col_score = _detectar_col_puntaje(dfh)
        if not col_score:
            continue
        def clasificar(p):
            try: x = float(p)
            except: return "Sin datos"
            if criterio == "SIMCE":
                if 0 <= x <= 250: return "Insuficiente"
                if 251 <= x <= 285: return "Intermedio"
                if 285 < x <= 400: return "Adecuado"
            else:
                if 0 <= x <= 599: return "Insuficiente"
                if 600 <= x <= 799: return "Intermedio"
                if 800 <= x <= 1000: return "Adecuado"
            return "Sin datos"
        serie = dfh[col_score].apply(clasificar)
        counts = serie.value_counts().reindex(total_counts.keys(), fill_value=0)
        for k in total_counts:
            total_counts[k] += counts[k]
    st.bar_chart(pd.DataFrame([total_counts]))
else:
    st.info("Sube primero el archivo complejo.")

# Función 3: Consolidación de puntajes
st.header("📂 Función 3: Consolidación de puntajes")
up_cons = st.file_uploader("Sube consolidado anterior (xlsx)", type=["xlsx"], key="cons")
if up_complex and up_cons:
    df_cons_all = load_excel_bytes(up_cons.getvalue())
    df_new_all = load_excel_bytes(up_complex.getvalue())
    resumen = []
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for hoja, df_cons in df_cons_all.items():
            col_nombres = next((c for c in df_cons.columns if "nombre" in str(c).lower()), None)
            if not col_nombres:
                df_cons.to_excel(writer, index=False, sheet_name=hoja[:31])
                continue
            df_new = df_new_all.get(hoja, pd.DataFrame())
            if df_new.empty:
                df_cons.to_excel(writer, index=False, sheet_name=hoja[:31])
                continue
            df_cons["__key"] = df_cons[col_nombres].map(_norm)
            df_new["__key"] = df_new["NOMBRE ESTUDIANTE"].map(_norm)
            col_score = _detectar_col_puntaje(df_new)
            nuevo_nombre = _next_ensayo_col(df_cons, col_nombres)
            df_merge = df_cons.merge(df_new[["__key", col_score]], on="__key", how="left")
            df_merge[nuevo_nombre] = pd.to_numeric(df_merge[col_score], errors="coerce")
            df_merge.drop(columns=["__key", col_score], inplace=True, errors="ignore")
            df_merge.to_excel(writer, index=False, sheet_name=hoja[:31])
            resumen.append({"Hoja": hoja, "Coincidencias": int(df_merge[nuevo_nombre].notna().sum())})
    st.download_button("📥 Descargar consolidado actualizado", data=output.getvalue(),
                       file_name="consolidado_actualizado.xlsx")
    st.dataframe(pd.DataFrame(resumen))

# Nota: Funciones 4, 5 y 6 se simplificarían de forma similar, pero omitidas aquí por brevedad.
