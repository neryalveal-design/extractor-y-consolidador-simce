
import re
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="SIMCE/PAES - Extractor & Consolidado (sin encabezados)", layout="wide")

# ==============================
# Utilidades
# ==============================

def excel_col_to_idx(col_label: str) -> int:
    """Devuelve índice 0-based de una etiqueta de columna de Excel (p.ej. 'A'->0, 'FK'->166)."""
    col_label = col_label.strip().upper()
    n = 0
    for ch in col_label:
        n = n * 26 + (ord(ch) - 64)  # 'A' = 1
    return n - 1  # 0-based

def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower().replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_STOP = {"de","del","la","las","los","y","e","da","do","das","dos"}

def name_key(name: str) -> str:
    """Clave por bolsa de palabras para igualar 'Perez Juan' con 'Juan Pérez'."""
    toks = [t for t in _norm_text(name).split() if t and t not in _STOP]
    return " ".join(sorted(toks))

def to_numeric_safe(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    # normalizar separadores
    s = s.str.replace(r"[^0-9\.,\-]+", "", regex=True)
    def _fix(x: str) -> str:
        if x is None or x == "":
            return ""
        if "." in x and "," in x:
            x = x.replace(".", "").replace(",", ".")
        elif "," in x and "." not in x:
            x = x.replace(",", ".")
        if x.count(".") > 1:
            i = x.find(".")
            x = x[:i+1] + x[i+1:].replace(".", "")
        return x
    s = s.apply(_fix)
    return pd.to_numeric(s, errors="coerce")

@st.cache_data
def load_excel_bytes_no_header(b: bytes) -> dict:
    """Devuelve {hoja: DataFrame} leyendo un xlsx SIN encabezados (header=None)."""
    xls = pd.ExcelFile(BytesIO(b))
    data = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        data[sheet] = df
    return data

@st.cache_data
def load_excel_bytes_raw(b: bytes) -> dict:
    """Devuelve {hoja: DataFrame} del archivo complejo, sin asumir encabezados."""
    xls = pd.ExcelFile(BytesIO(b))
    return {s: pd.read_excel(xls, sheet_name=s, header=None) for s in xls.sheet_names}

def extract_from_complex_sheet(df_raw: pd.DataFrame,
                               name_col_label="C", score_col_label="FK",
                               row_start_human=11, row_end_human=53) -> pd.DataFrame:
    """
    Extrae nombres (col=C) y puntajes (col=FK) en filas 11..53 (incl), sin encabezados.
    Devuelve DataFrame con 2 columnas [nombre, puntaje], sin nombres de columnas.
    """
    r0 = row_start_human - 1  # 0-based inclusive
    r1 = row_end_human        # 0-based exclusive para iloc
    c_name = excel_col_to_idx(name_col_label)
    c_score = excel_col_to_idx(score_col_label)

    # Cortes defensivos si la hoja es pequeña
    nrows, ncols = df_raw.shape
    r0 = min(max(r0, 0), max(nrows-1, 0))
    r1 = min(max(r1, r0+1), nrows)
    c_name = min(max(c_name, 0), max(ncols-1, 0))
    c_score = min(max(c_score, 0), max(ncols-1, 0))

    nombres = df_raw.iloc[r0:r1, c_name]
    puntajes = df_raw.iloc[r0:r1, c_score]

    # Limpiar
    nombres = nombres.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    puntajes = to_numeric_safe(puntajes)

    # Filtrar filas sin nombre
    mask = nombres.str.len() > 0
    out = pd.DataFrame({
        0: nombres[mask].values,
        1: puntajes[mask].values
    })
    return out.reset_index(drop=True)

def consolidate_sheet(df_hist: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    df_hist: DataFrame SIN encabezados: col0=nombre, col1..=intentos previos
    df_new : DataFrame SIN encabezados: col0=nombre, col1=puntaje nuevo
    Devuelve df_consol con col0=nombre, col1..colN (agrega nueva última columna).
    """
    if df_hist is None or df_hist.empty:
        # Crear histórico desde cero con la nueva medición
        return df_new.copy()

    # Mapas por clave
    hist = df_hist.copy()
    new = df_new.copy()

    hist.iloc[:, 0] = hist.iloc[:, 0].astype(str)
    new.iloc[:, 0] = new.iloc[:, 0].astype(str)

    hist["__key"] = hist.iloc[:, 0].map(name_key)
    new["__key"] = new.iloc[:, 0].map(name_key)

    # última col numérica del histórico
    last_idx = hist.shape[1] - 1  # al menos 1 (nombre) + intentos
    # unir solo clave y valor nuevo
    df_merge = hist.merge(new[[ "__key", 1 ]], on="__key", how="left")
    # nueva columna al final
    df_merge[last_idx + 1] = to_numeric_safe(df_merge[1])
    # limpiar
    df_merge.drop(columns=[ "__key", 1 ], inplace=True, errors="ignore")
    return df_merge

def classify(value: float, escala: str) -> str:
    try:
        x = float(value)
    except:
        return "Sin datos"
    if escala == "SIMCE":
        if 0 <= x <= 250: return "Insuficiente"
        if 251 <= x <= 285: return "Intermedio"
        if 285 < x <= 400: return "Adecuado"
    else:  # PAES
        if 0 <= x <= 599: return "Insuficiente"
        if 600 <= x <= 799: return "Intermedio"
        if 800 <= x <= 1000: return "Adecuado"
    return "Sin datos"

# ==============================
# UI - Barra lateral
# ==============================
st.sidebar.title("Configuración")
escala = st.sidebar.radio("Escala de análisis", ["SIMCE", "PAES"], horizontal=True)

up_complex = st.sidebar.file_uploader("Archivo complejo (xlsx)", type=["xlsx"], key="complex")
up_hist    = st.sidebar.file_uploader("Histórico sin encabezados (xlsx)", type=["xlsx"], key="hist")

# ==============================
# A) Extraer de archivo complejo (sin encabezados)
# ==============================
st.header("A) Extraer nombres y puntajes (sin encabezados)")

if up_complex:
    raw_dict = load_excel_bytes_raw(up_complex.getvalue())
    st.caption("Se detectaron hojas: " + ", ".join(raw_dict.keys()))
    extraidos = {}
    for hoja, df_raw in raw_dict.items():
        try:
            df_simple = extract_from_complex_sheet(df_raw, "C", "FK", 11, 53)
            extraidos[hoja] = df_simple
        except Exception as e:
            st.warning(f"No se pudo extraer en hoja '{hoja}': {e}")

    if extraidos:
        out = BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            for hoja, dfh in extraidos.items():
                # sin encabezados
                dfh.to_excel(writer, index=False, header=False, sheet_name=hoja[:31])
        st.download_button("📥 Descargar resultados simples (sin encabezados)",
                           data=out.getvalue(),
                           file_name="resultados_simples.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success("Archivo generado sin encabezados.")

    st.divider()

# ==============================
# B) Consolidar con histórico (sin encabezados)
# ==============================
st.header("B) Consolidar con histórico (sin encabezados)")

if up_complex:
    extraidos = { }
    raw_dict = load_excel_bytes_raw(up_complex.getvalue())
    for hoja, df_raw in raw_dict.items():
        try:
            extraidos[hoja] = extract_from_complex_sheet(df_raw, "C", "FK", 11, 53)
        except Exception as e:
            st.warning(f"No se pudo extraer en hoja '{hoja}': {e}")

    if up_hist:
        hist_dict = load_excel_bytes_no_header(up_hist.getvalue())
    else:
        hist_dict = {}

    if extraidos:
        out2 = BytesIO()
        resumen = []
        with pd.ExcelWriter(out2, engine="xlsxwriter") as writer:
            for hoja, df_new in extraidos.items():
                df_hist = hist_dict.get(hoja, pd.DataFrame())
                df_cons = consolidate_sheet(df_hist, df_new)
                # Guardar sin encabezados
                df_cons.to_excel(writer, index=False, header=False, sheet_name=hoja[:31])
                # Resumen de coincidencias (última col)
                if df_cons.shape[1] >= 2:
                    last_col = df_cons.columns[-1]
                    coinc = int(pd.to_numeric(df_cons[last_col], errors="coerce").notna().sum())
                else:
                    coinc = 0
                resumen.append({"Hoja": hoja, "Coincidencias (nuevas)": coinc, "Filas": len(df_cons)})
        st.download_button("📥 Descargar CONSOLIDADO (sin encabezados)",
                           data=out2.getvalue(),
                           file_name="consolidado_sin_encabezados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.dataframe(pd.DataFrame(resumen))
    else:
        st.info("Sube un archivo complejo válido para consolidar.")

st.divider()

# ==============================
# C) Análisis y visualización (usa el consolidado sin encabezados)
# ==============================
st.header("C) Análisis de rendimiento y progreso")

up_cons_view = st.file_uploader("Sube el CONSOLIDADO (sin encabezados) para analizar", type=["xlsx"], key="cons_view")

if up_cons_view:
    cons_dict = load_excel_bytes_no_header(up_cons_view.getvalue())
    hojas = list(cons_dict.keys())
    st.write(f"Hojas detectadas: {hojas}")

    # Análisis por curso (último intento)
    st.subheader("🏫 Distribución por curso (último intento)")
    dist_rows = []
    for hoja, dfc in cons_dict.items():
        if dfc.shape[1] < 2:
            continue
        last_col = dfc.columns[-1]
        vals = pd.to_numeric(dfc[last_col], errors="coerce")
        cats = vals.apply(lambda v: classify(v, escala))
        counts = cats.value_counts().reindex(["Insuficiente", "Intermedio", "Adecuado"], fill_value=0)
        dist_rows.append({"Curso": hoja, **{k:int(v) for k,v in counts.items()}})
    if dist_rows:
        st.dataframe(pd.DataFrame(dist_rows))
    else:
        st.info("No hay columnas de intento para analizar.")

    # Top 10 con menor puntaje por curso (último intento)
    st.subheader("⬇️ Top 10 de puntajes más bajos por curso (último intento)")
    for hoja, dfc in cons_dict.items():
        st.markdown(f"**Curso:** {hoja}")
        if dfc.shape[1] < 2:
            st.write("Sin datos.")
            continue
        last_col = dfc.columns[-1]
        tmp = dfc[[0, last_col]].copy()
        tmp[last_col] = pd.to_numeric(tmp[last_col], errors="coerce")
        bot = tmp.dropna(subset=[last_col]).sort_values(by=last_col, ascending=True).head(10)
        bot.columns = ["Nombre", "Puntaje"]
        st.table(bot)

    st.subheader("📈 Progreso individual")
    curso_sel = st.selectbox("Elige curso (hoja)", hojas, key="curso_sel_prog")
    dfc = cons_dict[curso_sel].copy()
    if dfc.shape[1] < 2:
        st.info("Ese curso no tiene intentos suficientes.")
    else:
        estudiantes = dfc[0].astype(str).tolist()
        est_sel = st.selectbox("Elige estudiante", estudiantes, key="est_sel_prog")
        fila = dfc[dfc[0].astype(str) == str(est_sel)]
        if fila.empty:
            st.info("No se encontró el estudiante en esa hoja.")
        else:
            y = pd.to_numeric(fila.iloc[0, 1:], errors="coerce").values.astype(float)
            x = np.arange(1, len(y)+1)
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(x, y, marker="o")
            for i, val in enumerate(y, start=1):
                if not np.isnan(val):
                    ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
            ax.set_xlabel("Intento")
            ax.set_ylabel("Puntaje")
            ax.set_title(f"Progreso - {est_sel} ({curso_sel})")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
else:
    st.info("Sube el consolidado sin encabezados para analizar.")
