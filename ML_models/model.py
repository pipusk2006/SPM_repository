import pandas as pd
import numpy as np
import os
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import kagglehub

# Загрузка локального датасета
df_train = pd.read_csv("train(43).csv")

# Загрузка дополнительного датасета с Kaggle (если нужно)
path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
df_kaggle = pd.read_csv(f"{path}/healthcare-dataset-stroke-data.csv")

# Используем df_train как основной
df = df_train.copy()

# Удаление столбца id
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Преобразование категориальных признаков
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
df['smoking_status'] = df['smoking_status'].map({
    'never smoked': 0,
    'formerly smoked': 0.5,
    'smokes': 1
})

# One-hot encoding для work_type
df = pd.get_dummies(df, columns=['work_type'], prefix='work')

# Удаление строк без метки stroke
df = df[df['stroke'].notna()]
df['stroke'] = df['stroke'].astype(int)

# Импутация пропущенных значений
mice_imputer = IterativeImputer(random_state=42)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('stroke')
df[numeric_cols] = mice_imputer.fit_transform(df[numeric_cols])

X = df.drop(columns=['stroke'])
y = df['stroke']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Балансировка обучающей выборки
sampler = SMOTETomek(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)

# Обучение модели
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Оценка модели
y_pred = rf_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Сохранение модели
joblib.dump(rf_model, "random_forest_stroke_model.pkl")
joblib.dump(scaler, "scaler.pkl")


import pandas as pd
import joblib

def RandomForestModel(gender, age, hypertension, heart_disease, ever_married,
                      work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    if not hasattr(RandomForestModel, "rf_model"):
        RandomForestModel.rf_model = joblib.load("random_forest_stroke_model.pkl")
        RandomForestModel.scaler = joblib.load("scaler.pkl")

    rf_model = RandomForestModel.rf_model
    scaler = RandomForestModel.scaler

    gender = 1 if gender == 'Male' else 0
    ever_married = 1 if ever_married == 'Yes' else 0
    Residence_type = 1 if Residence_type == 'Urban' else 0

    smoking_status_map = {
        'never smoked': 0,
        'formerly smoked': 0.5,
        'smokes': 1
    }
    smoking_status = smoking_status_map.get(smoking_status, 0)

    work_type_cols = ['work_Govt_job', 'work_Never_worked', 'work_Private', 'work_Self-employed', 'work_children']
    work_type_encoded = dict.fromkeys(work_type_cols, 0)
    work_type_col = f'work_{work_type}'
    if work_type_col in work_type_encoded:
        work_type_encoded[work_type_col] = 1

    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status,
        **work_type_encoded
    }

    input_df = pd.DataFrame([input_data])

    # Масштабирование
    input_scaled = scaler.transform(input_df)

    # Предсказание
    prob = rf_model.predict_proba(input_scaled)[0][1]
    return round(prob * 100, 2)

