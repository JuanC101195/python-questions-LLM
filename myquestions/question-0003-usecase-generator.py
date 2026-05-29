import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def generar_caso_de_uso_predecir_popularidad_meme():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(100, 501))

    likes = rng.gamma(shape=2.0, scale=500.0, size=n_filas)
    shares = (rng.pareto(a=1.5, size=n_filas) + 1.0) * 50.0
    longitud_texto = rng.integers(20, 281, size=n_filas).astype(float)
    numero_hashtags = rng.integers(0, 16, size=n_filas).astype(float)
    tiempo_publicacion = rng.integers(0, 24, size=n_filas).astype(float)

    popularidad = (
        np.log1p(likes) * 5.0
        + np.log1p(shares) * 8.0
        + np.sqrt(numero_hashtags) * 2.0
        - np.abs(tiempo_publicacion - 18) * 0.5
        + rng.normal(0, 3.0, size=n_filas)
    )
    pmin, pmax = popularidad.min(), popularidad.max()
    if pmax > pmin:
        popularidad = (popularidad - pmin) / (pmax - pmin) * 100.0

    df = pd.DataFrame({
        "likes": likes,
        "shares": shares,
        "longitud_texto": longitud_texto,
        "numero_hashtags": numero_hashtags,
        "tiempo_publicacion": tiempo_publicacion,
        "popularidad": popularidad,
    })

    target_col = "popularidad"

    feature_cols = [c for c in df.columns if c != target_col]
    for col in feature_cols:
        mask = rng.random(n_filas) < 0.10
        df.loc[mask, col] = np.nan

    input_data = {"df": df.copy(), "target_col": target_col}

    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=X.columns)

    hora = X_imp_df["tiempo_publicacion"].to_numpy()

    pt = PowerTransformer(method="yeo-johnson")
    X_pt = pt.fit_transform(X_imp)
    X_pt_df = pd.DataFrame(X_pt, columns=X.columns)

    X_pt_df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    X_pt_df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    X_pt_df = X_pt_df.drop(columns=["tiempo_publicacion"])

    selector = SelectPercentile(score_func=f_regression, percentile=60)
    X_sel = selector.fit_transform(X_pt_df.to_numpy(), y)
    n_features_seleccionadas = int(X_sel.shape[1])

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_sel, y)
    y_pred = modelo.predict(X_sel)
    rmse = round(float(np.sqrt(mean_squared_error(y, y_pred))), 4)

    output_data = {
        "modelo": modelo,
        "rmse": rmse,
        "n_features_seleccionadas": n_features_seleccionadas,
    }

    return input_data, output_data


if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_predecir_popularidad_meme()
    print("INPUT:")
    print(f"target_col: {input_data['target_col']}")
    print(input_data["df"].head())
    print("\nOUTPUT:")
    print(f"rmse: {output_data['rmse']}")
    print(f"n_features_seleccionadas: {output_data['n_features_seleccionadas']}")
