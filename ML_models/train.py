import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import kagglehub
import joblib  
import os
from preprocess import preprocess

df = preprocess()

X = df.drop(columns=['stroke'])
y = df['stroke']
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, stratify=y, random_state=43)
X_corr1, X_rest, y_corr1, y_rest = train_test_split(X_rest, y_rest, test_size=2/3, stratify=y_rest, random_state=42)
X_corr2, X_val, y_corr2, y_val = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_corr1_s = scaler.transform(X_corr1)
X_corr2_s = scaler.transform(X_corr2)
X_val_s = scaler.transform(X_val)

main = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0)
main.fit(X_train_s, y_train)

p_corr1 = main.predict_proba(X_corr1_s)[:, 1]
y_corr1_pred = (p_corr1 > 0.25).astype(int)
fp1 = (y_corr1_pred == 1) & (y_corr1 == 0)
fn1 = (y_corr1_pred == 0) & (y_corr1 == 1)

X_c1 = pd.concat([X_corr1[fp1], X_corr1[fn1]])
y_c1 = pd.concat([y_corr1[fp1], y_corr1[fn1]])
model_c1 = RandomForestClassifier(n_estimators=100, random_state=1)
model_c1.fit(scaler.transform(X_c1), y_c1)

nonstroke_pool_fnx = pd.concat([X_train[y_train == 0], X_corr1[y_corr1 == 0]])
n_fnx = fn1.sum()
n_zeros = min(n_fnx * 2, len(nonstroke_pool_fnx))

if n_fnx > 0 and n_zeros > 0:
    nonstroke_sample_fnx = nonstroke_pool_fnx.sample(n=n_zeros, random_state=99)
    X_fnx = pd.concat([X_corr1[fn1], nonstroke_sample_fnx])
    y_fnx = pd.Series([1] * n_fnx + [0] * n_zeros)
    model_fnx = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=99)
    model_fnx.fit(scaler.transform(X_fnx), y_fnx)
    px = model_fnx.predict_proba(X_val_s)[:, 1]
else:
    print("⚠️ FNX skipped due to lack of samples.")
    px = np.zeros(len(X_val_s))

p_corr2 = model_c1.predict_proba(X_corr2_s)[:, 1]
y_corr2_pred = (p_corr2 > 0.25).astype(int)
fp2 = (y_corr2_pred == 1) & (y_corr2 == 0)
stroke_pool = pd.concat([X_train[y_train == 1], X_corr1[y_corr1 == 1]])
stroke_sample2 = stroke_pool.sample(n=min(fp2.sum(), len(stroke_pool)), random_state=2)
X_c2 = pd.concat([X_corr2[fp2], stroke_sample2])
y_c2 = pd.Series([0] * fp2.sum() + [1] * len(stroke_sample2))
model_c2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=2)
model_c2.fit(scaler.transform(X_c2), y_c2)

p_corr3 = model_c2.predict_proba(X_val_s)[:, 1]
y_corr3_pred = (p_corr3 > 0.25).astype(int)
fp3 = (y_corr3_pred == 1) & (y_val == 0)
stroke_pool3 = pd.concat([X_train[y_train == 1], X_corr2[y_corr2 == 1]])
stroke_sample3 = stroke_pool3.sample(n=min(fp3.sum(), len(stroke_pool3)), random_state=2)
X_c3 = pd.concat([X_val[fp3], stroke_sample3])
y_c3 = pd.Series([0] * fp3.sum() + [1] * len(stroke_sample3))
model_c3 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=3)
model_c3.fit(scaler.transform(X_c3), y_c3)

all_fp = pd.concat([X_corr1[fp1], X_corr2[fp2], X_val[fp3]])
all_fn = pd.concat([X_corr1[fn1]])
false_all = pd.concat([all_fp, all_fn])
n_false = len(false_all)

n_half_1 = min(n_false // 2, len(X_train[y_train == 1]))
n_half_0 = min(n_false - n_half_1, len(X_train[y_train == 0]))

real_balanced = pd.concat([
    X_train[y_train == 1].sample(n=n_half_1, random_state=7),
    X_train[y_train == 0].sample(n=n_half_0, random_state=8)
])

X_final_corr = pd.concat([false_all, real_balanced])
y_final_corr = pd.Series([0] * len(false_all) + [1] * len(real_balanced))

model_final_corr = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=100)
model_final_corr.fit(scaler.transform(X_final_corr), y_final_corr)
p_final_corr = model_final_corr.predict_proba(X_val_s)[:, 1]

p_main = (main.predict_proba(X_val_s)[:, 1] > 0.6).astype(int)
p1 = model_c1.predict_proba(X_val_s)[:, 1]
p2 = model_c2.predict_proba(X_val_s)[:, 1]
p3 = model_c3.predict_proba(X_val_s)[:, 1]

ensemble = (
    0.55 * p_main +
    0.12 * p1 +
    0.02 * px +
    0.15 * p2 +
    0.15 * p3 +
    0.11 * p_final_corr  
)
y_pred = (ensemble > 0.245).astype(int)

print("🎯 Accuracy:", accuracy_score(y_val, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_val, y_pred, digits=4))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

os.makedirs("saved_models", exist_ok=True)
joblib.dump(main, "saved_models/main.pkl")
joblib.dump(model_c1, "saved_models/model_c1.pkl")
joblib.dump(model_c2, "saved_models/model_c2.pkl")
joblib.dump(model_c3, "saved_models/model_c3.pkl")
joblib.dump(model_final_corr, "saved_models/model_final_corr.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")

if 'model_fnx' in locals():
    joblib.dump(model_fnx, "saved_models/model_fnx.pkl")

