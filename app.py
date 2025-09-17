
# app.py — Versión 2.0 (rediseñada)
# ===================================================
# Extractor y Consolidador SIMCE / PAES — UI simple, robusta y rápida
# ===================================================

import io
import re
import unicodedata
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="SIMCE/PAES – Consolidador v2", layout="wide")

# =============================
# Utilidades de estilo/UI
# =============================
def _pill(txt: str, color: str = "#1259c3"):
    st.markdown(
        f'<span style="background:{color};color:white;border-radius:12px;padding:2px 10px;font-size:0.85rem;">{txt}</span>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## 🧭 Navegación")
    vista = st.radio(
        "Elige módulo",
        [
            "1) Extraer puntajes (complejo → normalizado)",
            "2) Análisis por curso",
            "3) Consolidar puntajes (agregar ensayo)",
            "4) Análisis por estudiante",
            "5) Estudiantes con menor rendimiento",
            "6) Análisis de preguntas y distractores",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown("### ℹ️ Ayuda rápida")
    st.markdown(
        "- **Sube el Excel complejo**: el que tiene muchas columnas (FK, etc.).\n"
        "- **Normalizado**: hojas con columnas `NOMBRE ESTUDIANTE` y `Puntaje Ensayo n`.\n"
        "- **Consolidado**: tu histórico por curso; al consolidar se añade `Puntaje Ensayo n+1`.\n"
    )

# =============================
# Helpers comunes
# =============================
@st.cache_data(show_spinner=False)
def load_excel_bytes(file_bytes: bytes) -> pd.ExcelFile:
    return pd.ExcelFile(io.BytesIO(file_bytes))

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower().replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _parse_numeric_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace("\u00a0", " ", regex=False)
    # eliminar letras/símbolos excepto separadores
    s = s.str.replace(r"[^0-9\.,\-]+", "", regex=True)
    def _fix(x: str) -> str:
        if x == "" or x.lower() in {"nan","none","-"}: return ""
        if "." in x and "," in x:
            x = x.replace(".", "").replace(",", ".")
        elif "," in x and "." not in x:
            x = x.replace(",", ".")
        # demasiados puntos
        if x.count(".") > 1:
            first = x.find(".")
            x = x[:first+1] + x[first+1:].replace(".", "")
        return x
    s = s.apply(_fix)
    return pd.to_numeric(s, errors="coerce")

def _next_ensayo_col(df_cons: pd.DataFrame) -> str:
    # Busca 'Puntaje Ensayo N' existente y devuelve el siguiente
    nums = []
    pat = re.compile(r"(?i)^puntaje\s+ensayo\s+(\d+)\s*$")
    for c in df_cons.columns:
        m = pat.match(str(c).strip())
        if m:
            try:
                nums.append(int(m.group(1)))
            except:
                pass
    n = (max(nums) + 1) if nums else 2
    name = f"Puntaje Ensayo {n}"
    while name in df_cons.columns:
        n += 1
        name = f"Puntaje Ensayo {n}"
    return name

def _detect_score_col(df: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in df.columns]
    # Prioridades
    preferidas = ["Puntaje Ensayo 1", "FK", "TOTAL"]
    for p in preferidas:
        if p in df.columns:
            return p
    # Heurística por nombre
    for c in df.columns:
        n = str(c).lower()
        if any(k in n for k in ("puntaje","simce","ensayo")) and str(c) != "NOMBRE ESTUDIANTE":
            return c
    # Fallback: primera numérica
    for c in df.columns:
        if c not in ("NOMBRE ESTUDIANTE","__key") and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def _ensure_only_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Si hay columnas duplicadas con el mismo nombre, conservamos la primera
    return df.loc[:, ~df.columns.duplicated()]

def _safe_to_numeric_inplace(df: pd.DataFrame, col: str):
    if col in df.columns:
        # si por alguna razón col es DataFrame (duplicados), toma la primera
        data = df[col]
        if isinstance(data, pd.DataFrame):
            serie = data.iloc[:,0]
        else:
            serie = data
        df[col] = pd.to_numeric(serie, errors="coerce")
        # quitar duplicados de columnas homónimas
        return _ensure_only_unique_columns(df)
    return df

# ===================================================
# 1) EXTRAER PUNTAJES (COMPLEJO → NORMALIZADO)
# ===================================================
def extraer_hoja(df_raw: pd.DataFrame,
                 fila_encabezado: int = 9,   # 10 humano
                 fila_ini: int = 10,         # 11 humano
                 fila_fin: int = 80,         # por seguridad
                 fallback_nombre_col: int = 2) -> pd.DataFrame:
    """
    Devuelve DataFrame con ['NOMBRE ESTUDIANTE', 'Puntaje Ensayo 1'].
    Busca encabezados; si no, usa heurísticas y fallback fila a fila.
    """
    headers = df_raw.iloc[fila_encabezado].astype(str).fillna("").str.strip().str.lower()
    # Col nombres
    idx_name = next((i for i,v in enumerate(headers) if "nombre" in v and "estudiante" in v), None)
    if idx_name is None:
        idx_name = fallback_nombre_col

    # Col puntaje (por header)
    idx_score = next((i for i,v in enumerate(headers) if ("puntaje" in v and ("simce" in v or "ensayo" in v))), None)

    sub = df_raw.iloc[fila_ini:fila_fin].copy()
    nombres = sub.iloc[:, idx_name].astype(str).str.strip()
    invalid = {"", "nan", "nombre estudiante", "curso", "correctas", "a","b","c","d","e"}
    mask_valid = ~nombres.str.lower().isin(invalid)
    sub_valid = sub[mask_valid].copy()

    def _serie_to_num(sr: pd.Series) -> pd.Series:
        sr = sr.astype(str).str.strip().str.replace("\u00a0"," ", regex=False)
        sr = sr.str.replace(r"[^0-9\.,\-]+","", regex=True)
        # normalizar separadores
        def _fix(x: str) -> str:
            if x == "" or x.lower() in {"nan","none","-"}: return ""
            if "." in x and "," in x:
                x = x.replace(".","").replace(",",".")
            elif "," in x and "." not in x:
                x = x.replace(",",".")
            if x.count(".") > 1:
                first = x.find("."); x = x[:first+1] + x[first+1:].replace(".","")
            return x
        sr = sr.apply(_fix)
        return pd.to_numeric(sr, errors="coerce")

    # Si no hay header de puntaje claro, elegir la columna más "numérica"
    if idx_score is None:
        best_idx, best_ratio = None, -1.0
        for j in range(sub_valid.shape[1]):
            nums = _serie_to_num(sub_valid.iloc[:,j])
            ratio = nums.between(0, 1000, inclusive="both").mean()
            if ratio > best_ratio:
                best_ratio, best_idx = ratio, j
        idx_score = best_idx

    puntajes = _serie_to_num(sub.iloc[:, idx_score])
    puntajes_valid = puntajes[mask_valid].copy()
    nombres_valid = nombres[mask_valid].str.replace(r"\s+"," ", regex=True).str.strip()

    # Fallback fila-a-fila: buscar mejor número en la fila si falta
    if puntajes_valid.isna().any():
        headers_full = headers
        def _mejor_puntaje_fila(row: pd.Series) -> float:
            mejor = np.nan
            mejor_desde_header = False
            for j, cell in enumerate(row):
                num = _serie_to_num(pd.Series([cell])).iloc[0]
                if pd.isna(num) or not (50 <= num <= 1000):
                    continue
                h = str(headers_full.iloc[j]).lower()
                es_header = ("puntaje" in h) or ("simce" in h) or ("ensayo" in h) or (h=="fk") or (h=="total")
                if es_header:
                    mejor, mejor_desde_header = num, True
                elif not mejor_desde_header:
                    if pd.isna(mejor) or num > mejor:
                        mejor = num
            return mejor
        fallback_series = sub_valid.apply(_mejor_puntaje_fila, axis=1)
        puntajes_valid = puntajes_valid.combine_first(fallback_series)

    out = pd.DataFrame({
        "NOMBRE ESTUDIANTE": nombres_valid.values,
        "Puntaje Ensayo 1": puntajes_valid.values
    })
    out = out[out["NOMBRE ESTUDIANTE"] != ""].reset_index(drop=True)
    return out

# =============================
# UI: 1) Extraer
# =============================
if vista.startswith("1)"):
    st.title("📥 Extraer puntajes (complejo → normalizado)")

    up_complex = st.file_uploader("Sube el Excel complejo (con múltiples hojas)", type=["xlsx"], key="f1_complex")
    fila_enc = st.number_input("Fila de encabezados (base 0)", min_value=0, value=9, step=1)
    fila_ini = st.number_input("Primera fila de estudiantes (base 0)", min_value=0, value=10, step=1)
    fila_fin = st.number_input("Última fila límite (exclusiva, base 0)", min_value=20, value=80, step=1)

    if up_complex:
        xlsc = load_excel_bytes(up_complex.getvalue())
        hojas = xlsc.sheet_names
        st.info(f"Hojas detectadas: {hojas}")
        df_map = {}
        bar = st.progress(0.0, text="Procesando hojas...")
        for i, hoja in enumerate(hojas, start=1):
            df_raw = pd.read_excel(xlsc, sheet_name=hoja, header=None)
            df_norm = extraer_hoja(df_raw, fila_encabezado=fila_enc, fila_ini=fila_ini, fila_fin=fila_fin)
            df_map[hoja] = df_norm
            bar.progress(i/len(hojas), text=f"Procesando {hoja} ({i}/{len(hojas)})")
        bar.empty()

        # Mostrar preview + descarga
        for hoja, dfh in df_map.items():
            st.subheader(f"📄 {hoja}")
            st.dataframe(dfh.head(10))

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            for hoja, dfh in df_map.items():
                dfh.to_excel(writer, index=False, sheet_name=hoja[:31])
        st.download_button(
            "📦 Descargar Excel normalizado (todas las hojas)",
            data=out.getvalue(),
            file_name="excel_normalizado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.session_state["normalizado_bytes"] = out.getvalue()

# ===================================================
# 2) ANÁLISIS POR CURSO (sobre normalizados)
# ===================================================
if vista.startswith("2)"):
    st.title("📊 Análisis por curso")
    up_norm = st.file_uploader("Sube un Excel normalizado (o usa el generado en 1)", type=["xlsx"], key="f2_norm")
    criterio = st.radio("Criterio de cortes", ["SIMCE", "PAES"], horizontal=True)

    if up_norm or "normalizado_bytes" in st.session_state:
        xlsn = load_excel_bytes(up_norm.getvalue() if up_norm else st.session_state["normalizado_bytes"])
        cols = st.columns(2)
        total = {"Insuficiente":0, "Intermedio":0, "Adecuado":0}
        for i, hoja in enumerate(xlsn.sheet_names):
            df = pd.read_excel(xlsn, sheet_name=hoja)
            score_col = _detect_score_col(df) or "Puntaje Ensayo 1"
            serie = _parse_numeric_series(df[score_col])

            def clasif(x):
                if pd.isna(x): return "Sin datos"
                if criterio=="SIMCE":
                    if 0 <= x <= 250: return "Insuficiente"
                    if 251 <= x <= 285: return "Intermedio"
                    if x > 285: return "Adecuado"
                else:
                    if 0 <= x <= 599: return "Insuficiente"
                    if 600 <= x <= 799: return "Intermedio"
                    if x >= 800: return "Adecuado"
                return "Sin datos"

            cats = serie.apply(clasif).value_counts().reindex(["Insuficiente","Intermedio","Adecuado","Sin datos"], fill_value=0)
            for k in ["Insuficiente","Intermedio","Adecuado"]:
                total[k] += int(cats[k])

            col = cols[i%2]
            with col:
                st.subheader(hoja)
                fig, ax = plt.subplots(figsize=(5.5,4))
                vals = [cats["Insuficiente"], cats["Intermedio"], cats["Adecuado"]]
                labels = ["Insuficiente","Intermedio","Adecuado"]
                ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90, shadow=True)
                ax.set_title(f"Distribución ({criterio})")
                st.pyplot(fig)

        st.subheader("🔵 Distribución global")
        fig, ax = plt.subplots(figsize=(6,4))
        vals = [total["Insuficiente"], total["Intermedio"], total["Adecuado"]]
        labels = ["Insuficiente","Intermedio","Adecuado"]
        ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90, shadow=True)
        ax.set_title(f"Global ({criterio})")
        st.pyplot(fig)
    else:
        st.info("Sube un archivo normalizado o genera uno en el módulo 1.")

# ===================================================
# 3) CONSOLIDAR PUNTAJES (agregar ensayo)
# ===================================================
if vista.startswith("3)"):
    st.title("📂 Consolidar puntajes (agregar nuevo ensayo)")
    up_complex = st.file_uploader("Excel complejo (nuevo ensayo)", type=["xlsx"], key="f3_complex")
    up_consol = st.file_uploader("Consolidado histórico (hojas por curso)", type=["xlsx"], key="f3_consol")

    if up_complex and up_consol:
        xlsc = load_excel_bytes(up_complex.getvalue())
        xlsh = load_excel_bytes(up_consol.getvalue())

        resumen: List[Dict] = []
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            for hoja in xlsh.sheet_names:
                df_cons = pd.read_excel(xlsh, sheet_name=hoja)
                # detectar columna de nombres
                col_nombres = next((c for c in df_cons.columns if "nombre" in str(c).lower() and "estudiante" in str(c).lower()), None)
                if col_nombres is None:
                    resumen.append({"Hoja": hoja, "Coincidencias": 0, "Sin coincidencia": len(df_cons), "Columna nueva": "(no aplica)"})
                    df_cons.to_excel(writer, index=False, sheet_name=hoja[:31])
                    continue

                # extraer de complejo -> normalizado de esta hoja
                try:
                    df_raw = pd.read_excel(xlsc, sheet_name=hoja, header=None)
                    df_new = extraer_hoja(df_raw)  # siempre produce 'Puntaje Ensayo 1'
                except Exception:
                    df_new = pd.DataFrame(columns=["NOMBRE ESTUDIANTE","Puntaje Ensayo 1"])

                # normalizar llaves
                df_cons["__key"] = df_cons[col_nombres].map(_norm)
                df_new["__key"]  = df_new["NOMBRE ESTUDIANTE"].map(_norm)

                # decidir nombre del nuevo ensayo
                nuevo_col = _next_ensayo_col(df_cons)

                # preparar lado derecho sólo con '__key' + nuevo_col
                right = df_new[["__key","Puntaje Ensayo 1"]].rename(columns={"Puntaje Ensayo 1": nuevo_col})

                # merge
                df_merge = df_cons.merge(right, on="__key", how="left")

                # priorizar valores existentes; si no hay, tomar nuevos
                if nuevo_col in df_merge.columns and nuevo_col in df_cons.columns:
                    df_merge[nuevo_col] = pd.to_numeric(df_merge[nuevo_col], errors="coerce")
                if nuevo_col in df_merge.columns and nuevo_col not in df_cons.columns:
                    df_merge = _safe_to_numeric_inplace(df_merge, nuevo_col)

                # limpiar auxiliares/duplicados
                df_merge.drop(columns=["__key"], inplace=True, errors="ignore")
                df_merge = _ensure_only_unique_columns(df_merge)

                # conteo de coincidencias
                coinc = int(df_merge[nuevo_col].notna().sum()) if nuevo_col in df_merge.columns else 0
                resumen.append({"Hoja": hoja, "Coincidencias": coinc, "Sin coincidencia": int(len(df_merge)-coinc), "Columna nueva": nuevo_col})

                # guardar hoja
                df_merge.to_excel(writer, index=False, sheet_name=hoja[:31])

        # descargar
        st.success("Consolidado actualizado generado.")
        st.dataframe(pd.DataFrame(resumen))
        st.download_button(
            "📥 Descargar CONSOLIDADO ACTUALIZADO",
            data=out.getvalue(),
            file_name="consolidado_actualizado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.session_state["consolidado_bytes"] = out.getvalue()
    else:
        st.info("Sube el Excel complejo y tu consolidado histórico para continuar.")

# ===================================================
# 4) ANÁLISIS POR ESTUDIANTE (sobre consolidado)
# ===================================================
if vista.startswith("4)"):
    st.title("🎯 Análisis por estudiante")
    up_consol = st.file_uploader("Sube consolidado (o usa el generado en 3)", type=["xlsx"], key="f4_consol")

    if up_consol or "consolidado_bytes" in st.session_state:
        xls = load_excel_bytes(up_consol.getvalue() if up_consol else st.session_state["consolidado_bytes"])
        hoja = st.selectbox("Elige curso (hoja)", xls.sheet_names, key="f4_hoja")
        df = pd.read_excel(xls, sheet_name=hoja)

        col_nombres = next((c for c in df.columns if "nombre" in str(c).lower() and "estudiante" in str(c).lower()), None)
        if col_nombres is None:
            st.error("No se encontró columna de nombres en la hoja seleccionada.")
        else:
            estudiante = st.selectbox("Elige estudiante", df[col_nombres].dropna().unique())
            row = df[df[col_nombres]==estudiante].copy()
            if row.empty:
                st.info("No se encontró el estudiante en esta hoja.")
            else:
                # columnas de ensayos, en orden
                pat = re.compile(r"(?i)^puntaje\s+ensayo\s+\d+\s*$")
                cols = [c for c in df.columns if pat.match(str(c).strip())]
                if not cols:
                    st.warning("No hay columnas de puntajes tipo 'Puntaje Ensayo n'.")
                else:
                    vals = _parse_numeric_series(row[cols].iloc[0])
                    mask = vals.notna()
                    x_labels = list(pd.Index(cols)[mask])
                    y_vals = list(vals[mask].astype(float))
                    if not y_vals:
                        st.info("Sin puntajes numéricos para graficar.")
                    else:
                        fig, ax = plt.subplots(figsize=(7,4))
                        ax.plot(range(len(x_labels)), y_vals, marker="o", linestyle="-")
                        off = (max(y_vals)-min(y_vals))*0.03 if len(y_vals)>1 else 5
                        for i, y in enumerate(y_vals):
                            ax.text(i, y+off, f"{y:.2f}", ha="center", fontsize=9)
                        ax.set_title(f"Evolución de {estudiante} – {hoja}")
                        ax.set_xlabel("Ensayos")
                        ax.set_ylabel("Puntaje")
                        ax.grid(True)
                        ax.set_xticks(range(len(x_labels)))
                        ax.set_xticklabels(x_labels, rotation=25, fontsize=9)
                        st.pyplot(fig)
                        st.success(f"📊 Promedio: **{float(np.nanmean(np.array(y_vals))):.2f}**")
    else:
        st.info("Sube el consolidado en este módulo o genera uno en el módulo 3.")

# ===================================================
# 5) ESTUDIANTES CON RENDIMIENTO MÁS BAJO
# ===================================================
if vista.startswith("5)"):
    st.title("📉 Estudiantes con rendimiento más bajo")
    up_consol = st.file_uploader("Sube consolidado (o usa el generado en 3)", type=["xlsx"], key="f5_consol")
    top_n = st.slider("¿Cuántos mostrar por curso?", 5, 30, 10)

    if up_consol or "consolidado_bytes" in st.session_state:
        xls = load_excel_bytes(up_consol.getvalue() if up_consol else st.session_state["consolidado_bytes"])
        for hoja in xls.sheet_names:
            st.subheader(f"Curso: {hoja}")
            df = pd.read_excel(xls, sheet_name=hoja)
            col_nombres = next((c for c in df.columns if "nombre" in str(c).lower() and "estudiante" in str(c).lower()), None)
            if col_nombres is None:
                st.warning(f"No se encontró columna de nombres en {hoja}.")
                continue

            # última columna de ensayo
            pat = re.compile(r"(?i)^puntaje\s+ensayo\s+(\d+)\s*$")
            ens_cols = []
            for c in df.columns:
                m = pat.match(str(c).strip())
                if m:
                    ens_cols.append((int(m.group(1)), c))
            if not ens_cols:
                st.warning(f"No hay columnas 'Puntaje Ensayo n' en {hoja}.")
                continue
            ens_cols.sort()
            last_col = ens_cols[-1][1]

            df_show = df[[col_nombres, last_col]].copy()
            df_show[last_col] = _parse_numeric_series(df_show[last_col])
            df_show = df_show.dropna(subset=[last_col]).sort_values(by=last_col, ascending=True).head(top_n)
            st.table(df_show.reset_index(drop=True))
    else:
        st.info("Sube el consolidado o genera uno en el módulo 3.")

# ===================================================
# 6) ANÁLISIS DE PREGUNTAS Y DISTRACTORES (complejo)
# ===================================================
if vista.startswith("6)"):
    st.title("📝 Análisis de preguntas y distractores")
    up_complex = st.file_uploader("Sube el Excel complejo (del ensayo a analizar)", type=["xlsx"], key="f6_complex")

    if up_complex:
        xls = load_excel_bytes(up_complex.getvalue())
        hoja = st.selectbox("Elige curso (hoja)", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=hoja, header=None)

        # Estas posiciones pueden variar por archivo, pero son las más comunes en tus planillas
        # Claves y preguntas
        claves = df.iloc[8, 3:68].tolist()     # D9:BP9
        preguntas = df.iloc[9, 3:68].tolist()  # D10:BP10
        valid_idx = [i for i, c in enumerate(claves) if pd.notna(c) and str(c).strip() != ""]
        claves_fil = [claves[i] for i in valid_idx]
        preguntas_fil = [preguntas[i] for i in valid_idx]

        # Respuestas estudiantes
        respuestas = df.iloc[10:80, 3:68]      # D11:BP80

        resumen = []
        for pos, clave in zip(valid_idx, claves_fil):
            col = respuestas.iloc[:, pos]
            total = col.notna().sum()
            aciertos = (col.astype(str).str.strip().str.lower() == str(clave).strip().lower()).sum()
            pct = aciertos/total*100 if total>0 else 0.0

            # Conteo A..E típico (puede variar en archivos)
            conteos_col = 3 + pos
            try:
                conteos = df.iloc[59:64, conteos_col]
                alternativas = ["A","B","C","D","E"]
                dist = dict(zip(alternativas, conteos))
            except Exception:
                dist = {}

            obs = ""
            if dist:
                tot = float(sum([v for v in dist.values() if pd.notna(v)]))
                if tot > 0:
                    dist_pct = {k: (float(v)/tot*100.0 if pd.notna(v) else 0.0) for k,v in dist.items()}
                    # evitar marcar E si duplica D
                    if "D" in dist_pct and "E" in dist_pct and dist.get("D", None)==dist.get("E", None):
                        dist_pct.pop("E", None)
                    # distraactor fuerte
                    inc = {k:v for k,v in dist_pct.items() if k.lower()!=str(clave).strip().lower()}
                    if inc:
                        kmax = max(inc, key=inc.get)
                        if inc[kmax] > 50: obs = f"Distractor fuerte: {kmax}"
                        vals = list(inc.values())
                        if pct < 50 and len(vals)>1 and (max(vals)-min(vals) < 10):
                            obs = "Alta dispersión"

            resumen.append({"Pregunta": preguntas_fil[valid_idx.index(pos)],
                            "Correcta": clave,
                            "% Aciertos": round(pct,2),
                            "Observación": obs})

        df_res = pd.DataFrame(resumen)
        st.subheader("📊 Resumen de preguntas")
        st.dataframe(df_res)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(df_res["Pregunta"], df_res["% Aciertos"])
        ax.set_title(f"% de aciertos por pregunta – {hoja}")
        ax.set_xlabel("Pregunta"); ax.set_ylabel("% Aciertos")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("Sube el Excel complejo para analizar.")
