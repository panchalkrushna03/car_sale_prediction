# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import pickle
from pathlib import Path
import yaml


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, model_filepath, output_filepath):
    """ Makes predictions using the trained model and saves results.
    """
    logger = logging.getLogger(__name__)
    logger.info('making predictions')

    # Load model
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)
    logger.info(f'Loaded model from {model_filepath}')

    # Load data for prediction (use test set)
    test_path = Path(input_filepath).parent / 'test_processed.csv'
    if test_path.exists():
        df = pd.read_csv(test_path)
        logger.info(f'Loaded test data for prediction with shape: {df.shape}')
        # Separate features from target for prediction
        target = 'Units_Sold'  # Back to predicting Units_Sold
        if target in df.columns:
            X_test = df.drop(columns=[target])  # Only remove the target column
            y_true = df[target]
        else:
            X_test = df
            y_true = None
    else:
        df = pd.read_csv(input_filepath)
        logger.info(f'Loaded data for prediction with shape: {df.shape}')
        X_test = df

    # Make predictions
    predictions = model.predict(X_test)
    logger.info('Predictions made successfully')

    # Save predictions
    output_path = Path(output_filepath) / 'predictions.csv'
    result_df = X_test.copy()
    result_df['predicted_units_sold'] = predictions
    if y_true is not None:
        result_df['actual_units_sold'] = y_true
    result_df.to_csv(output_path, index=False)
    logger.info(f'Predictions saved to {output_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
