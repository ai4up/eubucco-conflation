import xgboost as xgb
import pandas as pd

XGBOOST_PARAMS = {
    "tree_method": "hist",
    "objective": "binary:logistic"
}
HYPERPARAMETERS = {
    "colsample_bytree": 0.7,
    "subsample": 0.7,
}


def predict_match(model_path: str, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts whether pairs of geometries match using a pre-trained XGBoost model.
    """
    model = xgb.XGBClassifier(random_state=42, **HYPERPARAMETERS, **XGBOOST_PARAMS)
    model.load_model(model_path)

    fts = model.get_booster().feature_names
    X = candidate_pairs[fts]
    candidate_pairs["match"] = model.predict(X).astype(bool)

    return candidate_pairs
