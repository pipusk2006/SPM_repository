import pandas as pd
import joblib
import os

def RandomForestModel(
    gender: str,
    age: float,
    hypertension: int,
    heart_disease: int,
    ever_married: str,
    work_type: str,
    Residence_type: str,
    avg_glucose_level: float,
    bmi: float,
    smoking_status: str,
    threshold=0.245,
    weights={
        'main': 0.55,
        'c1': 0.12,
        'fnx': 0.02,
        'c2': 0.15,
        'c3': 0.15,
        'final_corr': [0.0, 0.04, 0.07, 0.09, 0.11]
    },
    model_dir='saved_models'
) -> float:
    # Загрузка моделей
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    main = joblib.load(os.path.join(model_dir, 'main.pkl'))
    model_c1 = joblib.load(os.path.join(model_dir, 'model_c1.pkl'))
    model_fnx = joblib.load(os.path.join(model_dir, 'model_fnx.pkl'))
    model_c2 = joblib.load(os.path.join(model_dir, 'model_c2.pkl'))
    model_c3 = joblib.load(os.path.join(model_dir, 'model_c3.pkl'))
    model_final_corr = joblib.load(os.path.join(model_dir, 'model_final_corr.pkl'))

    # Категориальные преобразования
    map_bin = {
        'gender': {'Male': 1, 'Female': 0},
        'ever_married': {'Yes': 1, 'No': 0},
        'Residence_type': {'Urban': 1, 'Rural': 0},
        'smoking_status': {'never smoked': 0, 'formerly smoked': 0.5, 'smokes': 1}
    }

    # Сбор всех данных
    data = {
        'gender': map_bin['gender'].get(gender, 0),
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': map_bin['ever_married'].get(ever_married, 0),
        'Residence_type': map_bin['Residence_type'].get(Residence_type, 0),
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': map_bin['smoking_status'].get(smoking_status, 0),
        'work_Govt_job': 1 if work_type == 'Govt_job' else 0,
        'work_Never_worked': 1 if work_type == 'Never_worked' else 0,
        'work_Private': 1 if work_type == 'Private' else 0,
        'work_Self-employed': 1 if work_type == 'Self-employed' else 0,
        'work_children': 1 if work_type == 'children' else 0,
    }

    df_input = pd.DataFrame([data])
    x_scaled = scaler.transform(df_input)

    # Предсказания моделей
    p_main = main.predict_proba(x_scaled)[:, 1][0]
    p1 = model_c1.predict_proba(x_scaled)[:, 1][0]
    p_fnx = model_fnx.predict_proba(x_scaled)[:, 1][0]
    p2 = model_c2.predict_proba(x_scaled)[:, 1][0]
    p3 = model_c3.predict_proba(x_scaled)[:, 1][0]
    p_final = model_final_corr.predict_proba(x_scaled)[:, 1][0]

    # Дифференцированный подсчёт риска
    for i, w_final in enumerate(weights['final_corr']):
        ensemble = (
            weights['main'] * p_main +
            weights['c1'] * p1 +
            weights['fnx'] * p_fnx +
            weights['c2'] * p2 +
            weights['c3'] * p3 +
            w_final * p_final
        )
        if ensemble > threshold:
            base = [95, 85, 70, 55, 35]
            step = [10, 15, 15, 20, 10]
            risk = base[i] + (ensemble - threshold) / (1 - threshold) * step[i]
            return min(round(risk, 2), 100.0)

    # Если ни одна ступень не сработала — риск оценивается по последней модели
    return round(p_final * 30, 2)


