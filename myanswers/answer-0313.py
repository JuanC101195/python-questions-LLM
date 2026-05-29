import numpy as np
import pandas as pd


def eliminar_multicolinealidad(df, threshold):
    df_numeric = df.select_dtypes(include=[np.number])

    corr_matrix = df_numeric.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    columnas_eliminadas = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    df_filtrado = df_numeric.drop(columns=columnas_eliminadas)

    return df_filtrado, columnas_eliminadas
