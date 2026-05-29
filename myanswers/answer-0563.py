import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluar_clusters_por_periodo(df, fecha_col, k_min=2, k_max=5):
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df["__trimestre__"] = (
        df[fecha_col].dt.year.astype(str)
        + "-Q"
        + df[fecha_col].dt.quarter.astype(str)
    )

    trimestres = sorted(df["__trimestre__"].unique())

    feature_cols = [
        c for c in df.columns
        if c not in (fecha_col, "__trimestre__")
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    resultados = []
    for trim in trimestres:
        sub = df[df["__trimestre__"] == trim]
        if len(sub) < k_min * 2:
            continue

        X = sub[feature_cols].to_numpy()
        X_imp = SimpleImputer(strategy="mean").fit_transform(X)
        X_sc = StandardScaler().fit_transform(X_imp)

        mejor_k = None
        mejor_sil = -np.inf
        for k in range(k_min, k_max + 1):
            if k >= len(X_sc):
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_sc)
            if len(np.unique(labels)) < 2:
                continue
            sil = silhouette_score(X_sc, labels)
            if sil > mejor_sil:
                mejor_sil = sil
                mejor_k = k

        if mejor_k is not None:
            resultados.append({
                "trimestre": trim,
                "mejor_k": mejor_k,
                "silhouette_score": mejor_sil,
            })

    return pd.DataFrame(
        resultados, columns=["trimestre", "mejor_k", "silhouette_score"]
    ).reset_index(drop=True)
