# 📊 Extractor de Puntajes SIMCE y Ensayos

Esta aplicación te permite:

1. Consolidar múltiples archivos de ensayo por curso.
2. Agregar puntajes a estructuras existentes por grupo.
3. Exportar resultados a Excel o PDF.

## 🚀 Uso

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Estructura del proyecto

- `app.py`: Interfaz principal Streamlit
- `utils.py`: Procesamiento de datos
- `excel_exporter.py`: Exportación a Excel
- `pdf_exporter.py`: Exportación a PDF
- `.streamlit/config.toml`: Tema visual
- `assets/`: Logos o recursos visuales
- `sample_data/`: Datos de muestra
