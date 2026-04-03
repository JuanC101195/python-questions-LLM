import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def reducir_perfiles_categoricos(df, cat_cols, n_components):
    # Codificación one-hot de las columnas categóricas indicadas
    X_encoded = pd.get_dummies(df[cat_cols], columns=cat_cols)

    # Reducción de dimensionalidad con TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reducida = svd.fit_transform(X_encoded)

    # Varianza explicada acumulada
    varianza_acumulada = float(np.sum(svd.explained_variance_ratio_))

    return X_reducida, varianza_acumulada


def generar_caso_de_uso_reducir_perfiles_categoricos():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(40, 90))

    ciudades = [
        "Bogota", "Medellin", "Cali", "Barranquilla", "Cartagena",
        "Bucaramanga", "Manizales", "Pereira", "Tunja", "Pasto"
    ]
    planes = [
        "Basico", "Estandar", "Premium", "Empresarial", "Familiar", "Plus"
    ]
    canales = [
        "Web", "App", "Tienda", "CallCenter", "Distribuidor"
    ]
    segmentos = [
        "Joven", "Adulto", "Pyme", "Corporativo", "Estudiante", "Hogar"
    ]

    df = pd.DataFrame({
        "ciudad": rng.choice(ciudades, size=n_filas),
        "tipo_plan": rng.choice(planes, size=n_filas),
        "canal_compra": rng.choice(canales, size=n_filas),
        "segmento": rng.choice(segmentos, size=n_filas)
    })

    # Agregamos una columna extra no categórica del problema para que el input sea más realista
    df["antiguedad_meses"] = rng.integers(1, 60, size=n_filas)

    cat_cols = ["ciudad", "tipo_plan", "canal_compra", "segmento"]

    # Contar cuántas columnas tendrá la codificación one-hot
    X_encoded = pd.get_dummies(df[cat_cols], columns=cat_cols)
    max_componentes = min(X_encoded.shape[0], X_encoded.shape[1]) - 1

    # Asegurar un n_components válido y no trivial
    if max_componentes < 2:
        n_components = 1
    else:
        n_components = int(rng.integers(2, min(6, max_componentes) + 1))

    input_data = {
        "df": df,
        "cat_cols": cat_cols,
        "n_components": n_components
    }

    output_data = reducir_perfiles_categoricos(df, cat_cols, n_components)

    return input_data, output_data


if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_reducir_perfiles_categoricos()

    print("INPUT:")
    print(input_data)

    print("\nOUTPUT:")
    print(output_data)