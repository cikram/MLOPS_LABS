"""Step 5 (second part) - baseline and model, evaluated with the MAE in minutes."""
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .common import FEATURES, SEED, TARGET


def train_and_evaluate(train_df, test_df, features=FEATURES):
    y_train, y_test = train_df[TARGET], test_df[TARGET]

    baseline = DummyRegressor(strategy="median").fit(train_df[features], y_train)
    mae_base = mean_absolute_error(y_test, baseline.predict(test_df[features]))

    model = HistGradientBoostingRegressor(max_iter=200, random_state=SEED).fit(train_df[features], y_train)
    mae_model = mean_absolute_error(y_test, model.predict(test_df[features]))

    return model, {"baseline_mae_min": round(float(mae_base), 3), "test_mae_min": round(float(mae_model), 3)}