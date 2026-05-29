import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest


def generar_caso_de_uso_detectar_anomalias_red():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(100, 301))
    n_anomalas = max(1, int(round(n_filas * 0.05)))
    n_normales = n_filas - n_anomalas

    duracion_n = rng.exponential(scale=2.0, size=n_normales)
    bytes_env_n = rng.gamma(shape=2.0, scale=500.0, size=n_normales)
    bytes_rec_n = rng.gamma(shape=2.0, scale=600.0, size=n_normales)
    num_conex_n = rng.exponential(scale=3.0, size=n_normales)
    puerto_n = rng.choice(
        np.array([80, 443, 22, 21, 53, 8080], dtype=float),
        size=n_normales,
    )

    duracion_a = rng.exponential(scale=2.0, size=n_anomalas) * 50.0
    bytes_env_a = rng.gamma(shape=2.0, scale=500.0, size=n_anomalas) * 100.0
    bytes_rec_a = rng.gamma(shape=2.0, scale=600.0, size=n_anomalas) * 100.0
    num_conex_a = rng.exponential(scale=3.0, size=n_anomalas) * 30.0
    puerto_a = rng.integers(1024, 65536, size=n_anomalas).astype(float)

    duracion = np.concatenate([duracion_n, duracion_a])
    bytes_env = np.concatenate([bytes_env_n, bytes_env_a])
    bytes_rec = np.concatenate([bytes_rec_n, bytes_rec_a])
    num_conex = np.concatenate([num_conex_n, num_conex_a])
    puerto = np.concatenate([puerto_n, puerto_a])

    idx = rng.permutation(n_filas)
    df = pd.DataFrame({
        "duracion": duracion[idx],
        "bytes_enviados": bytes_env[idx],
        "bytes_recibidos": bytes_rec[idx],
        "num_conexiones": num_conex[idx],
        "puerto_destino": puerto[idx],
    })

    for col in df.columns:
        mask = rng.random(n_filas) < 0.05
        df.loc[mask, col] = np.nan

    input_data = {"df": df.copy()}

    X = df.select_dtypes(include=[np.number])

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X_imp)

    modelo = IsolationForest(contamination=0.05, random_state=42)
    pred = modelo.fit_predict(X_sc)

    df_resultado = df.copy()
    df_resultado["anomalia"] = np.where(pred == -1, 1, 0)

    porcentaje_anomalias = round(float(df_resultado["anomalia"].mean()), 4)

    output_data = {
        "df_resultado": df_resultado,
        "porcentaje_anomalias": porcentaje_anomalias,
        "modelo": modelo,
    }

    return input_data, output_data


if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_detectar_anomalias_red()
    print("INPUT:")
    print(input_data["df"].head())
    print("\nOUTPUT:")
    print(f"porcentaje_anomalias: {output_data['porcentaje_anomalias']}")
    print(output_data["df_resultado"].head())
