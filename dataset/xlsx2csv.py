import pandas as pd
from pandas import DataFrame

df = DataFrame(pd.read_excel("./cph.xlsx", 0))
df.to_csv("./cph.csv", index=False)
