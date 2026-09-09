import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
n = 140000

humidity9am = np.random.uniform(20, 100, n)
humidity3pm = np.random.uniform(20, 100, n)
pressure9am = np.random.uniform(990, 1030, n)
pressure3pm = np.random.uniform(990, 1030, n)
temp9am = np.random.uniform(10, 35, n)
temp3pm = np.random.uniform(10, 35, n)
cloud9am = np.random.uniform(0, 8, n)
cloud3pm = np.random.uniform(0, 8, n)
windspeed9am = np.random.uniform(0, 50, n)
windspeed3pm = np.random.uniform(0, 50, n)

feature_names = ["Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
                  "Temp9am", "Temp3pm", "Cloud9am", "Cloud3pm",
                  "WindSpeed9am", "WindSpeed3pm"]

X = np.column_stack([humidity9am, humidity3pm, pressure9am, pressure3pm,
                      temp9am, temp3pm, cloud9am, cloud3pm,
                      windspeed9am, windspeed3pm])

z_h = (humidity3pm - humidity3pm.mean()) / humidity3pm.std()
z_c = (cloud3pm - cloud3pm.mean()) / cloud3pm.std()
z_p = (pressure3pm - pressure3pm.mean()) / pressure3pm.std()

signal = 1.0 * z_h + 0.8 * z_c - 0.6 * z_p
signal_noisy = signal + np.random.normal(0, 0.95, n)
threshold = np.quantile(signal_noisy, 0.78)
y_class = (signal_noisy > threshold).astype(int)

rainfall_signal = 2 + 1.5 * z_h + 1.0 * z_c - 0.8 * z_p
y_reg = np.clip(rainfall_signal + np.random.normal(0, 0.78, n), 0, None)

X_train, X_test, y_train, y_test, y_train_reg, y_test_reg = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=== Classification: Rain Tomorrow ===")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"{name:22s} Accuracy: {acc:.3f}  F1: {f1:.3f}")

print("\n=== Hyperparameter Tuning (Gradient Boosting) ===")

param_grid = {
    "max_iter": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "l2_regularization": [0.0, 0.1, 1.0]
}

gb = HistGradientBoostingClassifier(random_state=42)

search = RandomizedSearchCV(
    gb, param_distributions=param_grid, n_iter=8,
    scoring="f1_macro", cv=3, random_state=42, n_jobs=-1
)
search.fit(X_train_scaled, y_train)
best_model = search.best_estimator_

tuned_preds = best_model.predict(X_test_scaled)
print("Best Params:", search.best_params_)
print("Tuned Accuracy:", accuracy_score(y_test, tuned_preds))
print("Tuned F1-score:", f1_score(y_test, tuned_preds, average="macro"))

print("\n=== Regression: Rainfall Amount ===")

reg = HistGradientBoostingRegressor(random_state=42)
reg.fit(X_train_scaled, y_train_reg)
reg_preds = reg.predict(X_test_scaled)

mae = mean_absolute_error(y_test_reg, reg_preds)
rmse = np.sqrt(mean_squared_error(y_test_reg, reg_preds))
r2 = r2_score(y_test_reg, reg_preds)

print(f"MAE: {mae:.3f}  RMSE: {rmse:.3f}  R2: {r2:.3f}")

print("\n=== Feature Importance (Tuned Model) ===")

from sklearn.inspection import permutation_importance

result = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=5, random_state=42, n_jobs=-1)
pairs = sorted(zip(feature_names, result.importances_mean), key=lambda x: x[1], reverse=True)
for name, score in pairs:
    print(f"{name:15s} {score:.4f}")
