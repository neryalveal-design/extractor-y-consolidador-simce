
import pandas as pd
import unicodedata

def normalizar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).strip().lower()
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    return texto

def analyze_data(files):
    all_data = []

    for file in files:
        try:
            df = pd.read_excel(file, header=None)

            # Buscar la fila donde está "NOMBRE ESTUDIANTE"
            header_row = df.apply(lambda row: row.astype(str).str.contains("NOMBRE ESTUDIANTE", case=False, na=False)).any(axis=1)
            header_index = header_row[header_row].index[0]

            df = pd.read_excel(file, header=header_index)

            # Normalizar nombres de columnas
            df.columns = [str(col).strip().lower() for col in df.columns]

            if "nombre estudiante" not in df.columns:
                raise ValueError("No se encontró la columna 'nombre estudiante'")

            df.rename(columns={"nombre estudiante": "nombre"}, inplace=True)

            # Inferir curso desde alguna celda previa o agregar manualmente
            curso = None
            for i in range(header_index):
                fila = df.iloc[i] if i < len(df) else None
                if fila is not None and fila.astype(str).str.contains("medio", case=False).any():
                    curso = fila.dropna().values[-1]
                    break
            if not curso:
                curso = "curso_desconocido"

            df["curso"] = curso
            df["nombre"] = df["nombre"].apply(normalizar_texto)
            all_data.append(df[["nombre", "curso"] + [col for col in df.columns if col not in ["nombre", "curso"]]])

        except Exception as e:
            raise RuntimeError(f"Error al procesar {file.name}: {e}")

    final_df = pd.concat(all_data, ignore_index=True)
    cursos = final_df["curso"].unique()
    estudiantes = final_df["nombre"].nunique()

    return final_df, cursos, estudiantes
