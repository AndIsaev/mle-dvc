# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os


# оценка качества модели
def evaluate_model():
    # прочитайте файл с гиперпараметрами params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # загрузите результат прошлого шага: fitted_model.pkl
    with open('data/initial_data.csv', 'rb') as df:
        data = pd.read_csv(df)

    with open('models/fitted_model.pkl', 'rb') as fd:
        pipeline = joblib.load(fd)

    # реализуйте основную логику шага с использованием прочтённых гиперпараметров
    cv_strategy = StratifiedKFold(n_splits=5)
    cv_res = cross_validate(
        pipeline,
        data,
        data[params['target_col']],
        cv=cv_strategy,
        n_jobs=params['n_jobs'],
        scoring=params['metrics'],
    )
    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)

        # сохраните результата кросс-валидации в cv_res.json
    os.makedirs('cv_results', exist_ok=True)

    with open('cv_results/cv_res.json', 'w') as f:
        json.dump(cv_res, f)


if __name__ == '__main__':
    evaluate_model()
