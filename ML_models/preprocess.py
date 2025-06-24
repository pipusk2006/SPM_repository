import pandas as pd
import kagglehub
from sklearn.impute import IterativeImputer

def preprocess():
    path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
    df = pd.read_csv(f"{path}/healthcare-dataset-stroke-data.csv")

    # Удаляем столбец id (он не нужен для обучения)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    map_bin = {
        'gender': {'Male': 1, 'Female': 0},
        'ever_married': {'Yes': 1, 'No': 0},
        'Residence_type': {'Urban': 1, 'Rural': 0},
        'smoking_status': {'never smoked': 0, 'formerly smoked': 0.5, 'smokes': 1}
    }

    for col, mapping in map_bin.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = pd.get_dummies(df, columns=['work_type'], prefix='work')

    df = df[df['stroke'].notna()]
    df['stroke'] = df['stroke'].astype(int)

    imputer = IterativeImputer(random_state=42)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('stroke')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    return df





