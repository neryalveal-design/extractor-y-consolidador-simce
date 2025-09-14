import pandas as pd
from io import BytesIO

def export_to_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Consolidado')
    output.seek(0)
    return output
