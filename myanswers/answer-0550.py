import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def clasificar_riesgo_binario(df, target_col, umbral):
    df_num = df.select_dtypes(include=[np.number])
    X = df_num.drop(columns=[target_col])
    y = (df_num[target_col].to_numpy() >= umbral).astype(int)

    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X, y)
    return modelo.predict(X)
