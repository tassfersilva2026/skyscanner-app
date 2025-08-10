import streamlit as st, pandas as pd
st.set_page_config(page_title="Skyscanner Behavior", layout="wide")
st.title("Análises Skyscanner")

# caminho do seu parquet
PARQUET = "OFERTAS.parquet"
df = pd.read_parquet(PARQUET)

st.caption(f"Linhas: {len(df):,}")
st.dataframe(df.head(100), use_container_width=True)
