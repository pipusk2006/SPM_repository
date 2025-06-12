import pandas as pd
import kagglehub
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load datasets
df_train = pd.read_csv("train(43).csv")
path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
df_kaggle = pd.read_csv(f"{path}/healthcare-dataset-stroke-data.csv")

# 2. Add 'stroke' column to df_test if not present
if 'stroke' not in df_kaggle.columns:
    df_kaggle['stroke'] = None  # (In this case, Kaggle data already has 'stroke')

# 3. Align columns in both dataframes
common_cols = list(set(df_train.columns) | set(df_kaggle.columns))
df_train = df_train.reindex(columns=common_cols)
df_kaggle = df_kaggle.reindex(columns=common_cols)

# 4. Concatenate dataframes
df = pd.concat([df_train, df_kaggle], ignore_index=True)

# 5. Convert categorical features to numeric codes
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
df['smoking_status'] = df['smoking_status'].map({
    'never smoked': 0,
    'formerly smoked': 0.5,
    'smokes': 1
})

# 6. One-hot encode the work_type feature
df = pd.get_dummies(df, columns=['work_type'], prefix='work')

# 7. Remove any rows without a stroke label
df = df[df['stroke'].notna()]
df['stroke'] = df['stroke'].astype(int)

# 8. Impute missing numeric values using IterativeImputer (MICE)
mice_imputer = IterativeImputer(random_state=42)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('stroke')
df[numeric_cols] = mice_imputer.fit_transform(df[numeric_cols])

# 9. Train-test split (80/20)
X = df.drop(columns=['stroke'])
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 11. Balance the training set using SMOTE + Tomek Links
sampler = SMOTETomek(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)

y_pred_rf = rf_model.predict(X_test_scaled)
print("📊 Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred_rf, digits=4))




def RandomForestModel(gender, age, hypertension, heart_disease, ever_married,
                      work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    import numpy as np
    import pandas as pd

    # 1. Преобразование категориальных признаков
    gender = 1 if gender == 'Male' else 0
    ever_married = 1 if ever_married == 'Yes' else 0
    Residence_type = 1 if Residence_type == 'Urban' else 0

    # Курение: 0 = никогда не курил, 0.5 = курил раньше, 1 = курит
    smoking_status_map = {
        'never smoked': 0,
        'formerly smoked': 0.5,
        'smokes': 1
    }
    smoking_status = smoking_status_map.get(smoking_status, 0)  # по умолчанию 0

    # One-hot encoding для work_type
    work_type_cols = ['work_Govt_job', 'work_Never_worked', 'work_Private', 'work_Self-employed', 'work_children']
    work_type_encoded = dict.fromkeys(work_type_cols, 0)
    work_type_col = f'work_{work_type}'
    if work_type_col in work_type_encoded:
        work_type_encoded[work_type_col] = 1

    # 2. Формирование DataFrame с одним наблюдением
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

    # 3. Преобразование признаков (импутация не нужна — пользователь должен передать полные значения)
    input_scaled = scaler.transform(input_df)

    # 4. Предсказание вероятности положительного класса
    prob = rf_model.predict_proba(input_scaled)[0][1]

    # 5. Вернуть вероятность в процентах
    return round(prob * 100, 2)
