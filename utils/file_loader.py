import io, os
import pandas as pd
import arff

def load_dataset(uploaded_file):
    ext=os.path.splitext(uploaded_file.name)[1].lower()
    data=uploaded_file.getvalue()
    if ext==".csv":
        return pd.read_csv(io.BytesIO(data))
    if ext in [".xls",".xlsx"]:
        return pd.read_excel(io.BytesIO(data))
    if ext==".arff":
        obj=arff.load(io.StringIO(data.decode("utf-8",errors="replace")))
        return pd.DataFrame(obj["data"],columns=[x[0] for x in obj["attributes"]])
    raise ValueError("Supported formats: CSV, XLS, XLSX, ARFF")
