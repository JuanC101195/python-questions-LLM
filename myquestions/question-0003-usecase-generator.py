import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluar_regularizacion_ridge(df, target_col, alphas):
    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Seleccionar solo columnas numéricas
    X = X.select_dtypes(include=[np.number])

    resultados = {}

    for alpha in alphas:
        modelo = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])

        scores = cross_val_score(
            modelo,
            X,
            y,
            cv=5,
            scoring="neg_mean_squared_error"
        )

        rmse_promedio = float(np.mean(np.sqrt(-scores)))
        resultados[alpha] = rmse_promedio

    mejor_alpha = min(resultados, key=resultados.get)
    mejor_rmse = resultados[mejor_alpha]

    return resultados, mejor_alpha, mejor_rmse


def generar_caso_de_uso_evaluar_regularizacion_ridge():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(60, 120))
    n_features = int(rng.integers(4, 8))

    data = {}

    for i in range(n_features):
        col = rng.normal(loc=0, scale=1, size=n_filas)

        # Introducir algunos valores faltantes
        mask = rng.random(n_filas) < 0.1
        col[mask] = np.nan

        data[f"feature_{i}"] = col

    df = pd.DataFrame(data)

    # Crear target continua a partir de combinación lineal de variables
    X_temp = df.fillna(df.mean())
    coeficientes = rng.uniform(-3, 3, size=n_features)
    ruido = rng.normal(0, 0.5, size=n_filas)

    y = X_temp.to_numpy() @ coeficientes + ruido
    df["target"] = y

    alphas_disponibles = [0.01, 0.1, 1.0, 10.0, 100.0]
    cantidad_alphas = int(rng.integers(3, len(alphas_disponibles) + 1))
    alphas = [float(a) for a in rng.choice(alphas_disponibles, size=cantidad_alphas, replace=False)]
    alphas.sort()

    input_data = {
        "df": df,
        "target_col": "target",
        "alphas": alphas
    }

    output_data = evaluar_regularizacion_ridge(df, "target", alphas)

    return input_data, output_data


if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_evaluar_regularizacion_ridge()

    print("INPUT:")
    print(input_data)

    print("\nOUTPUT:")
    print(output_data)