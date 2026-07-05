from src.gradient_boosting import GBDT, GBDT_PARAM_GRID
from src.random_forest import RF, RF_PARAM_GRID
from src.tune import tune_model
from src.utils import compare


def main():
    # Hyperparameter tuning for Gradient Boosting classifier
    BEST_GBDT, BEST_GBDT_PARAMS, BEST_GBDT_SCORE = tune_model(GBDT, GBDT_PARAM_GRID);

    # Hyperparameter tuning for Gradient Boosting classifier
    BEST_RF, BEST_RF_PARAMS, BEST_RF_SCORE = tune_model(RF, RF_PARAM_GRID);

   

if __name__ == "__main__":
    main()