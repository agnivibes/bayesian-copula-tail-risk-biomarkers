import pandas as pd
import pyreadstat

def read_xpt(path):
    df, meta = pyreadstat.read_xport(path)
    print("\n==============================")
    print("FILE:", path)
    print("SHAPE:", df.shape)
    print("COLUMNS (first 30):", list(df.columns)[:30])
    return df

glu = read_xpt("GLU_J.xpt")
ghb = read_xpt("GHB_J.xpt")

# NHANES variable names commonly used:
# GLU: LBXGLU (Plasma fasting glucose, mg/dL)
# GHB: LBXGH (Glycohemoglobin / HbA1c, %)

needed_glu = ["SEQN", "LBXGLU"]
needed_ghb = ["SEQN", "LBXGH"]

print("GLU has:", [c for c in needed_glu if c in glu.columns])
print("GHB has:", [c for c in needed_ghb if c in ghb.columns])

df_glu_ghb = (
    glu[needed_glu]
    .merge(ghb[needed_ghb], on="SEQN", how="inner")
    .dropna()
    .rename(columns={"LBXGLU": "glu", "LBXGH": "ghb"})
)

print("Final merged dataset shape:", df_glu_ghb.shape)
display(df_glu_ghb.describe())
display(df_glu_ghb.head())

outname = "NHANES_2017_2018_glu_ghb.csv"
df_glu_ghb.to_csv(outname, index=False)
print("Saved:", outname)
