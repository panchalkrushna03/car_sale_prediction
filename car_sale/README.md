Car_Sale
==============================

A short description of the project.

## Running the Web Application

To run the FastAPI web application for car sales prediction:

### Option 1: Using the batch file (Windows)
```bash
run_api.bat
```

### Option 2: Direct command (ensure you're in the car_sale directory)
```bash
cd car_sale
python src/app.py
```

### Option 3: Using full path
```bash
python D:\Car_sale_project\car_sale\src\app.py
```

### Option 4: Using uvicorn directly
```bash
cd car_sale/src
uvicorn app:app --reload
```

Then open your browser and go to: http://127.0.0.1:8000

## Testing the API

To test the API endpoints:

```bash
python test_api.py
```

This will test the health check and prediction endpoints with the example data.

## Example Data

Example input data is available in `example_data.json` for testing the model.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── config             <- Configuration files for the project (config.yaml, params.yaml)
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── environment.yml    <- Conda environment file for reproducing the analysis environment
    │
    ├── Dockerfile         <- Docker configuration for containerizing the project
    │
    ├── example_data.json  <- Example input data for testing the model
    │
    ├── test_api.py        <- Test script for the FastAPI application
    │
    ├── run_api.bat        <- Windows batch file to run the API server
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
