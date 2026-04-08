# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import yaml


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        Splits data before preprocessing to avoid data leakage.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load configuration
    project_dir = Path(__file__).resolve().parents[2]
    config_path = project_dir / 'config' / 'config.yaml'
    params_path = project_dir / 'config' / 'params.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    # Load data
    df = pd.read_csv(input_filepath)
    logger.info(f'Loaded data with shape: {df.shape}')

    # Basic preprocessing before splitting
    # Convert Year and Month to datetime
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

    # Handle missing values
    df = df.dropna()
    logger.info(f'After dropping NA: {df.shape}')

    # Prepare features and target
    target = 'Units_Sold'  # Back to predicting Units_Sold
    features = ['Year', 'Month', 'Region', 'Model', 'Avg_Price_EUR',
                'BEV_Share', 'Premium_Share', 'GDP_Growth', 'Fuel_Price_Index']

    X = df[features]
    y = df[target]

    # Define output path
    output_path = Path(output_filepath)

    # Split data first to avoid data leakage
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=config['model']['random_state']
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config['model']['random_state']
    )

    logger.info(f'Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}')

    # Create preprocessing pipeline
    categorical_features = params['features']['categorical_features']
    numerical_features = params['features']['numerical_features']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Fit preprocessor on training data only
    preprocessor.fit(X_train)
    logger.info('Preprocessor fitted on training data')

    # Transform all datasets
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to DataFrames for saving
    # Get feature names after preprocessing
    num_features = numerical_features
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features = list(num_features) + list(cat_features)

    X_train_df = pd.DataFrame(X_train_processed, columns=all_features)
    X_val_df = pd.DataFrame(X_val_processed, columns=all_features)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_features)

    # Add target columns
    X_train_df[target] = y_train.values
    X_val_df[target] = y_val.values
    X_test_df[target] = y_test.values

    # Save processed datasets
    output_path = Path(output_filepath)
    X_train_df.to_csv(output_path / 'train_processed.csv', index=False)
    X_val_df.to_csv(output_path / 'val_processed.csv', index=False)
    X_test_df.to_csv(output_path / 'test_processed.csv', index=False)

    # Save preprocessor to models folder
    models_path = Path(output_filepath).parent.parent / 'models'
    models_path.mkdir(exist_ok=True)
    preprocessor_path = models_path / 'preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    logger.info(f'Saved processed datasets to {output_path} and preprocessor to {models_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
