import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from bloque01.datos import preparar_clustering

df, X, _ = preparar_clustering()
print("Positivos totales:", (df["Result"] == "positive").sum())  # debe dar 807

Z = linkage(X, method="ward")
df["cluster"] = fcluster(Z, 3, criterion="maxclust")
print(pd.crosstab(df["cluster"], df["Result"]))