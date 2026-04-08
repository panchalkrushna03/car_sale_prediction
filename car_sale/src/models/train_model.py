# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# Constraint wrapper to ensure positive predictions
class ConstrainedRFRegressor:
    def __init__(self, model, min_value=0):
        self.model = model
        self.min_value = min_value
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = self.model.predict(X)
        # Ensure predictions are not negative (minimum is 0)
        return np.maximum(predictions, self.min_value)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Trains the model and saves it as pickle file along with the preprocessor.
    """
    logger = logging.getLogger(__name__)
    logger.info('training model')

    # Load configuration
    project_dir = Path(__file__).resolve().parents[2]
    config_path = project_dir / 'config' / 'config.yaml'
    params_path = project_dir / 'config' / 'params.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    # Load processed training data
    train_df = pd.read_csv(input_filepath)
    logger.info(f'Loaded processed training data with shape: {train_df.shape}')

    # Separate features and target
    target = 'Units_Sold'  # Back to predicting Units_Sold which may be more predictable
    X_train = train_df.drop(columns=[target])  # Only remove the target column
    y_train = train_df[target]

    # Load validation data for evaluation
    val_path = Path(input_filepath).parent / 'val_processed.csv'
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        X_val = val_df.drop(columns=[target])  # Only remove the target column
        y_val = val_df[target]
        logger.info(f'Loaded validation data with shape: {val_df.shape}')
    else:
        logger.warning('Validation data not found, using train/test split')
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=params['train']['test_size'],
            random_state=config['model']['random_state']
        )

    # Create a wrapper class to ensure positive predictions
    # (class is defined at module level for pickling)

    # Create and train Random Forest model (best and most stable algorithm)
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=config['model']['random_state'],
        n_jobs=-1,
        bootstrap=True
    )
    
    # Wrap model to ensure positive predictions
    model = ConstrainedRFRegressor(rf_model, min_value=0)
    model.fit(X_train, y_train)
    logger.info('Random Forest model trained successfully with positive prediction constraint')

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    logger.info(f'Model evaluation - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}')

    # Save the model
    model_path = Path(output_filepath) / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'Model saved to {model_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
