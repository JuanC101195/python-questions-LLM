import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def generar_caso_de_uso_segmentar_paisajes_sonoros():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(80, 301))
    n_clusters = int(rng.integers(2, 5))

    intensidad_media = rng.normal(loc=60.0, scale=10.0, size=n_filas)
    intensidad_max = intensidad_media + rng.normal(loc=10.0, scale=3.0, size=n_filas)
    frecuencia_media = rng.gamma(shape=3.0, scale=1000.0, size=n_filas)
    frecuencia_dominante = frecuencia_media * rng.uniform(0.8, 1.5, size=n_filas)
    diversidad_espectral = rng.uniform(0.0, 5.0, size=n_filas)
    timestamp = np.arange(n_filas, dtype=float)

    df = pd.DataFrame({
        "intensidad_media": intensidad_media,
        "intensidad_max": intensidad_max,
        "frecuencia_media": frecuencia_media,
        "frecuencia_dominante": frecuencia_dominante,
        "diversidad_espectral": diversidad_espectral,
        "timestamp": timestamp,
    })

    for col in df.columns:
        if col == "timestamp":
            continue
        mask = rng.random(n_filas) < 0.10
        df.loc[mask, col] = np.nan

    input_data = {"df": df.copy(), "n_clusters": n_clusters}

    X = df.drop(columns=["timestamp"])
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    varianza_explicada_pca = [float(v) for v in pca.explained_variance_ratio_]

    modelo_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = modelo_cluster.fit_predict(X_pca)

    if len(np.unique(labels)) < 2:
        silhouette = -1.0
    else:
        silhouette = round(
            float(silhouette_score(X_scaled, labels, metric="euclidean")), 4
        )

    output_data = {
        "labels": labels,
        "silhouette": silhouette,
        "varianza_explicada_pca": varianza_explicada_pca,
    }

    return input_data, output_data


if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_segmentar_paisajes_sonoros()
    print("INPUT:")
    print(f"n_clusters: {input_data['n_clusters']}")
    print(input_data["df"].head())
    print("\nOUTPUT:")
    print(f"silhouette: {output_data['silhouette']}")
    print(f"varianza_explicada_pca: {output_data['varianza_explicada_pca']}")
    print(f"labels (primeras 10): {output_data['labels'][:10]}")
