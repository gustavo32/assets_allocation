import logging
from typing import Dict, Tuple
from functools import reduce
from operator import add

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pmdarima as pm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.special import softmax

pd.options.display.float_format = '{:,.4f}'.format


def _normalize_sum_to_one(y_preds):
    test_df = y_preds.sort_index().clip(0, 20)
    total_sum = reduce(add, [test_df[l] for l in test_df.columns])
    return test_df.apply(lambda col: col / total_sum.values)


def _get_shifted_data(df, index, period):
    test_df = pd.DataFrame({"d": [1]*len(index)}, index=index)
    test_X_df = pd.DataFrame(test_df, index=df.index).shift(period)
    test_index_prox_day = test_X_df[test_X_df.d == 1].index
    return df.loc[test_index_prox_day.values]


def _print_models_performance(y_preds, y_test, model_name):
    logger = logging.getLogger(__name__)
    mean_score = []
    mean_mse = []

    for k in y_preds.columns:
        score = r2_score(y_test[k], y_preds[k])
        mse = mean_squared_error(y_test[k], y_preds[k])
        logger.info(f"[{model_name}] Model {k} has a coefficient R^2 of {score} on test data.")
        logger.info(f"[{model_name}] Model {k} has a MSE of {mse} on test data.")
        mean_score.append(score)
        mean_mse.append(mse)

    logger.info(f"[{model_name}] Models mean has a coefficient R^2 of {np.asarray(mean_score).mean()} on test data.")
    logger.info(f"[{model_name}] Models mean has a MSE of {np.asarray(mean_mse).mean()} on test data.")


def _print_approach_performance(assets_allocation, approach_name, make_plot):
    total_sum = assets_allocation.sum()
    total_std = assets_allocation.std()
    print(f"\n{approach_name}")
    print("RETURN:", total_sum)
    print("RISK:", total_std)
    print("RETURN_RISK:", total_sum / (total_std + 0.5))
    if make_plot:
        plt.plot(assets_allocation.cumsum(), label=approach_name)


def merge_feat_tables(delete_first_n_rows, **kwargs):
    mapped_dfs = map(lambda key: kwargs[key].add_prefix(key+"_"), kwargs.keys())
    mapped_dfs = list(mapped_dfs)
    return reduce(
        lambda left, right: pd.merge(left,
                                     right,
                                     on=['Date']),
        mapped_dfs)[delete_first_n_rows:]


def separate_labels_and_features(df):
    labels = [c for c in df.columns if c.endswith("return_risk")]

    y = df[labels].shift(-1)
    y = _normalize_sum_to_one(y)

    print(y["ivvb11_return_risk"].sum())
    print(y["bvsp_return_risk"].sum())

    return df.drop(columns=labels), y


def split_data(X: pd.DataFrame, y: pd.DataFrame, training_size: int) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/training.yml.
    Returns:
        Split data.
    """

    size = int(len(X) * training_size)
    X_train, y_train = X[:size], y[:size]
    X_test, y_test = X[size:-1], y[size:-1]

    return X_train, X_test, y_train, y_test


def _forecast_exogenous_variables(df, horizon):
    forecasts = {}
    for c in df.columns:
        model = pm.auto_arima(df[c])
        forecasts[c] = model.predict(horizon)
    return pd.DataFrame(forecasts)


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, indices, experiment) -> Dict[str,LinearRegression]:
    models = {}
    y_preds = {}

    X_train = X_train.clip(-100, 100)
    training_indices = [index for index, v in indices.items() if v.get("training", False)]
    for index in training_indices:
        train_df = X_train[indices[index].training]
        return_risk_col = f"{index}_return_risk"

        if experiment["model"] == "linear_regression":
            model = LinearRegression(**experiment["params"])
            model.fit(train_df, y_train[return_risk_col] * (X_train[f"{index}_close"].sum() / len(X_train)))
            models[index] = model
            y_pred = model.predict(train_df)

        elif experiment["model"] == "arima":
            #  * (X_train[f"{index}_close"].sum() / len(X_train))
            model = pm.auto_arima(y_train[return_risk_col],
                                  X=train_df,
                                  **experiment["params"])
            models[index] = model
            y_pred = model.predict(len(y_train), X=train_df)

        elif experiment["model"] == "fixed":
            print(index)
            print(X_train[f"{index}_close"].sum())
            print(X_train[f"{index}_close"].std())
            print(y_train[return_risk_col].sum())
            y_pred = X_train[f"{index}_close"].sum() / len(X_train)
            models[index] = y_pred

        else:
            raise (KeyError, "There is no such option!")

        y_preds[return_risk_col] = y_pred

    y_preds = pd.DataFrame(y_preds, index=y_train.index)
    y_preds = _normalize_sum_to_one(y_preds)

    _print_models_performance(y_preds, y_train, experiment["model"].upper() + "_OVERFIT")

    return models


def evaluate_model(
    models: Dict, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, indices, experiment
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressors: Trained models.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_preds = {}
    for index, model in models.items():
        test_df = X_test[indices[index].training]
        return_risk_col = f"{index}_return_risk"

        if experiment["model"] == "linear_regression":
            y_pred = model.predict(test_df)
        elif experiment["model"] == "arima":
            print("building forecasts")
            forecasts_df = _forecast_exogenous_variables(X_train, len(y_test))
            print("finishing forecasts")
            y_pred = model.predict(len(y_test), X=forecasts_df)
        elif experiment["model"] == "fixed":
            y_pred = model
        else:
            raise(KeyError, "There is no such option!")
        y_preds[return_risk_col] = y_pred

    y_preds = pd.DataFrame(y_preds, index=y_test.index)
    y_preds = _normalize_sum_to_one(y_preds)

    _print_models_performance(y_preds, y_test, experiment["model"].upper())

    return y_preds


def compare_performance_traditional_scenario(X, y, y_preds, make_plots=False):
    close_columns = [indices.replace("return_risk", "close") for indices in y_preds.columns]
    test_index = y_preds.index

    X_preds = _get_shifted_data(X, test_index, period=1)[close_columns]
    X_preds_perfect = X.loc[test_index][close_columns]
    y_preds_perfect = y.shift(1).loc[test_index]

    # traditional approach
    trad_X = X_preds * (1/len(y_preds.columns))
    sum_columns = reduce(add, [trad_X[l] for l in close_columns])
    _print_approach_performance(sum_columns, approach_name="TRADITIONAL", make_plot=make_plots)


    # ibova approach
    sum_columns = X_preds["bvsp_close"]
    _print_approach_performance(sum_columns, approach_name="IBOVA", make_plot=make_plots)

    # ibova approach
    sum_columns = X_preds["ivvb11_close"]
    _print_approach_performance(sum_columns, approach_name="IVVB11", make_plot=make_plots)

    # ipca approach
    sum_columns = X_preds["posfixado_ipca_close"]
    _print_approach_performance(sum_columns, approach_name="IPCA", make_plot=make_plots)


    # our approach
    test_df = _normalize_sum_to_one(y_preds)
    app_X = X_preds.values * test_df.values
    sum_columns = app_X.sum(axis=1)
    sum_columns = pd.Series(sum_columns, index=X_preds.index)
    _print_approach_performance(sum_columns, approach_name="OUR APPROACH", make_plot=make_plots)

    allocation_df = test_df.copy()


    #perfect approach
    total_sum = reduce(add, [y_preds_perfect[l] for l in y_preds_perfect.columns])
    test_df = y_preds_perfect.apply(lambda col: col / total_sum.values)
    app_X = X_preds_perfect.values * test_df.values
    sum_columns = app_X.sum(axis=1)
    sum_columns = pd.Series(sum_columns, index=X_preds.index)
    _print_approach_performance(sum_columns, approach_name="BEST APPROACH", make_plot=make_plots)

    if make_plots:
        plt.legend()
        plt.show()

    diff = (allocation_df - y_preds_perfect).abs()
    total_diff = reduce(add, [diff[l] for l in diff.columns])
    plt.plot(total_diff)
    plt.show()

    [plt.plot(allocation_df[c], label=c.replace("_return_risk", "")) for c in allocation_df.columns]
    plt.legend()
    plt.show()