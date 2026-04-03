import numpy as np
import pandas as pd


def eliminar_multicolinealidad(df, threshold):
    # Seleccionar columnas numéricas
    df_numeric = df.select_dtypes(include=[np.number])

    # Calcular matriz de correlación absoluta
    corr_matrix = df_numeric.corr().abs()

    # Tomar solo la parte superior de la matriz
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Identificar columnas a eliminar
    columnas_eliminadas = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    # Eliminar columnas
    df_filtrado = df_numeric.drop(columns=columnas_eliminadas)

    return df_filtrado, columnas_eliminadas


def generar_caso_de_uso_eliminar_multicolinealidad():
    np.random.seed()

    # Número aleatorio de filas y columnas
    n_filas = np.random.randint(30, 100)
    n_cols = np.random.randint(4, 8)

    data = {}

    # Generar columnas base
    for i in range(n_cols):
        data[f"col_{i}"] = np.random.randn(n_filas)

    df = pd.DataFrame(data)

    # Introducir correlación artificial (clave para que el caso no sea trivial)
    if n_cols >= 2:
        col_base = np.random.choice(df.columns)
        col_nueva = np.random.choice(df.columns)

        if col_base != col_nueva:
            df[col_nueva] = df[col_base] * (0.8 + 0.2 * np.random.rand())

    # Threshold aleatorio
    threshold = np.random.uniform(0.7, 0.95)

    # Crear input
    input_data = {
        "df": df,
        "threshold": threshold
    }

    # Calcular output esperado
    df_filtrado, columnas_eliminadas = eliminar_multicolinealidad(df, threshold)

    output_data = (df_filtrado, columnas_eliminadas)

    return input_data, output_data


if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_eliminar_multicolinealidad()

    print("INPUT:")
    print(input_data)
    print("\nOUTPUT:")
    print(output_data)