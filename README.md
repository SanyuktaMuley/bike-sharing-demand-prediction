# Bike Sharing Demand Prediction

A machine learning project that predicts hourly bike rental demand from calendar and weather information.

## Project Overview

The dataset contains hourly bike-sharing records. The goal is to predict the total number of rentals (`count`) for each timestamp in the test data.

The notebook walks through:

1. Loading and inspecting the data
2. Checking dimensions, data types, and missing values
3. Exploring demand patterns with visualizations
4. Extracting useful datetime features
5. Creating interaction features for hour and working-day effects
6. Comparing linear, Ridge, and Lasso regression models
7. Selecting model hyperparameters using 5-fold cross-validation
8. Evaluating models with RMSLE
9. Creating and validating the final submission file

## Files

| File | Description |
|---|---|
| `bike_sharing_demand_prediction.ipynb` | Complete analysis, feature engineering, model training, evaluation, and prediction code |
| `bike_train.csv` | Training data containing features and the target columns |
| `bike_test.csv` | Test data containing features without the target |
| `submission.csv` | Final predicted rental counts for the test timestamps |
| `model_report.pdf` | Exported report with the analysis results |

## Dataset

The training data includes:

- `datetime`: Date and hour of the observation
- `season`: Encoded season category
- `holiday`: Whether the day is a holiday
- `workingday`: Whether the day is a working day
- `weather`: Encoded weather category
- `temp`: Temperature
- `atemp`: Feels-like temperature
- `humidity`: Relative humidity
- `windspeed`: Wind speed
- `casual`: Rentals by casual users
- `registered`: Rentals by registered users
- `count`: Total rentals and prediction target

The test data contains the same input features but does not contain the target columns.

## Approach

Datetime values are converted into calendar features including year, month, day, hour, weekday, and day of year. Additional interaction features are created to represent different demand curves for working and non-working days:

- `hour_work`
- `hour_weekday`
- `year_month`

Categorical variables are one-hot encoded. Continuous variables are expanded with polynomial features and standardized. The target is transformed using `log1p`, which is appropriate because rental counts are right-skewed and the evaluation metric is RMSLE.

The final model is a regularized Ridge regression model using polynomial weather features. The Ridge regularization strength is selected using cross-validation and the one-standard-error rule.

## Evaluation

The project uses Root Mean Squared Logarithmic Error (RMSLE):

```text
RMSLE = sqrt(mean((log(1 + prediction) - log(1 + actual))^2))
```

This metric focuses on relative prediction error and reduces the influence of very large rental counts.

## How to Run

1. Open `bike_sharing_demand_prediction.ipynb` in Jupyter Notebook or VS Code.
2. Place the notebook and CSV files in the same folder.
3. Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

4. Run the notebook cells from top to bottom.
5. The final model writes predictions to `submission.csv`.

## Result

The engineered models substantially outperform the raw-feature linear baselines because they capture time-of-day, working-day, seasonal, and weather effects. The final submission contains one non-negative prediction for every timestamp in the test dataset.
