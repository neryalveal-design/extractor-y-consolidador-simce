from fpdf import FPDF
from io import BytesIO

def export_to_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Consolidado de Puntajes", ln=1, align='C')
    pdf.ln(10)

    for index, row in data.iterrows():
        text = ", ".join(f"{col}: {row[col]}" for col in data.columns)
        pdf.multi_cell(0, 10, text)

    output = BytesIO()
    pdf.output(output)
    output.seek(0)
    return output
