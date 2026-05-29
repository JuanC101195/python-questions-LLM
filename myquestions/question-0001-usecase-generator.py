import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def generar_caso_de_uso_clasificar_pulsos_radio():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(50, 201))

    frecuencia_central = rng.gamma(shape=2.0, scale=200.0, size=n_filas)
    ancho_banda = rng.exponential(scale=15.0, size=n_filas)
    flujo = rng.gamma(shape=1.5, scale=5.0, size=n_filas)
    snr = rng.exponential(scale=8.0, size=n_filas)

    es_neutron = ((flujo > np.median(flujo)) & (snr > np.median(snr))).astype(int)

    df = pd.DataFrame({
        "frecuencia_central": frecuencia_central,
        "ancho_banda": ancho_banda,
        "flujo": flujo,
        "relacion_señal_ruido": snr,
        "es_neutron": es_neutron,
    })

    target_col = "es_neutron"

    feature_cols = [c for c in df.columns if c != target_col]
    for col in feature_cols:
        mask = rng.random(n_filas) < 0.10
        df.loc[mask, col] = np.nan

    input_data = {
        "df": df.copy(),
        "target_col": target_col,
    }

    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    qt = QuantileTransformer(output_distribution="uniform", random_state=42)
    X_qt = qt.fit_transform(X_imp)

    scaler = StandardScaler()
    X_proc = scaler.fit_transform(X_qt)

    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_proc, y)
    accuracy = round(float(modelo.score(X_proc, y)), 4)

    output_data = {"modelo": modelo, "accuracy": accuracy}

    return input_data, output_data


if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_clasificar_pulsos_radio()
    print("INPUT:")
    print(f"target_col: {input_data['target_col']}")
    print(input_data["df"].head())
    print("\nOUTPUT:")
    print(f"modelo: {output_data['modelo']}")
    print(f"accuracy: {output_data['accuracy']}")
