import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


def extraer_patrones_consumo(df, n_components):
    X = df.to_numpy()

    modelo = NMF(
        n_components=n_components,
        init="random",
        random_state=42,
        max_iter=500
    )

    W = modelo.fit_transform(X)
    H = modelo.components_

    X_reconstruida = np.dot(W, H)
    rmse = float(np.sqrt(np.mean((X - X_reconstruida) ** 2)))

    return W, H, rmse


def generar_caso_de_uso_extraer_patrones_consumo():
    rng = np.random.default_rng()

    n_muestras = int(rng.integers(20, 60))
    n_features = int(rng.integers(5, 10))

    data = rng.uniform(0, 100, size=(n_muestras, n_features))

    columnas = [f"franja_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columnas)

    max_componentes = min(n_muestras, n_features)
    if max_componentes <= 2:
        n_components = 1
    else:
        n_components = int(rng.integers(2, max_componentes))

    input_data = {
        "df": df,
        "n_components": n_components
    }

    output_data = extraer_patrones_consumo(df, n_components)

    return input_data, output_data


if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_extraer_patrones_consumo()

    print("INPUT:")
    print(input_data)

    print("\nOUTPUT:")
    print(output_data)