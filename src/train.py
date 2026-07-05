from src.gradient_boosting import GBDT, GBDT_PARAM_GRID
from src.random_forest import RF, RF_PARAM_GRID
from src.tune import tune_model
from src.utils import compare, save_model, save_training_results, save_metrics


def main():
    # Hyperparameter tuning for Gradient Boosting classifier
    BEST_GBDT, BEST_GBDT_PARAMS, BEST_GBDT_SCORE = tune_model(GBDT, GBDT_PARAM_GRID);

    # Hyperparameter tuning for Gradient Boosting classifier
    BEST_RF, BEST_RF_PARAMS, BEST_RF_SCORE = tune_model(RF, RF_PARAM_GRID);

    # Save Tuned Gradient Boosting Classifier
    save_model(BEST_GBDT, "gradient_booster.pkl")

    # Save Tuned Random Forest
    save_model(BEST_RF, "random_forest.pkl")

    # Save tune parameters in JSON format
    save_training_results({
        "Random Forest": {
            "best_params": BEST_RF_PARAMS,
            "best_score": BEST_RF_SCORE,
        },
        "Gradient Boosting": {
            "best_params": BEST_GBDT_PARAMS,
            "best_score": BEST_GBDT_SCORE,
        },
    })

    # Save metrics as a reusable CSV file
    metrics = compare({
        "Random Forest": BEST_RF,
        "Gradient Boosting": BEST_GBDT,
    })

    save_metrics(metrics)

   

if __name__ == "__main__":
    main()