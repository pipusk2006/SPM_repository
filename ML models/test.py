import pandas as pd
import kagglehub
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

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

