import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def linear_regression(X, y):
    rmse_linear = []
    mae_linear = []
    r2_linear = []

    rmse_linear_train = []
    r2_linear_train = []

    for seed in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        rmse_linear.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        mae_linear.append(mean_absolute_error(y_test, y_pred_test))
        r2_linear.append(r2_score(y_test, y_pred_test))

        rmse_linear_train.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        r2_linear_train.append(r2_score(y_train, y_pred_train))

    return pd.DataFrame({
        "RMSE_teste": rmse_linear,
        "MAE_teste": mae_linear,
        "R2_teste": r2_linear,
        "RMSE_treino": rmse_linear_train,
        "R2_treino": r2_linear_train
    })
