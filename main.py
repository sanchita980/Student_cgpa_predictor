import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from math import sqrt

# Load Dataset
data = pd.read_excel("pms2.xlsx")

# Mapping categorical features
stress_map = {"Low": 2, "Medium": 1, "High": 0}
motivation_map = {"Low": 0, "Medium": 1, "High": 2}
health_map = {"Low": 0, "Medium": 1, "High": 2}
data["Active Backlog"] = data["Active Backlog"].map({"No": 0, "Yes": 1})
data["Stress Level"] = data["Stress Level"].map(stress_map)
data["Motivation Level"] = data["Motivation Level"].map(motivation_map)
data["Health Condition"] = data["Health Condition"].map(health_map)

# Clean data
data = data.dropna()

# Feature and target split
X = data.drop("sem2", axis=1)
Y = data["sem2"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 1. Linear Regression
Lr_Model = LinearRegression().fit(X_train, y_train)
Lr_pred = Lr_Model.predict(X_test)
Lr_mae = sqrt(mean_absolute_error(y_test, Lr_pred))
Lr_rmse = sqrt(mean_squared_error(y_test, Lr_pred))
Lr_r2 = r2_score(y_test, Lr_pred)
pickle.dump(Lr_Model, open("Lr_model.pkl", "wb"))

# 2. Decision Tree
DT_Model = DecisionTreeRegressor().fit(X_train, y_train)
DT_pred = DT_Model.predict(X_test)
DT_mae = sqrt(mean_absolute_error(y_test, DT_pred))
DT_rmse = sqrt(mean_squared_error(y_test, DT_pred))
DT_r2 = r2_score(y_test, DT_pred)
pickle.dump(DT_Model, open("DT_model.pkl", "wb"))

# 3. Tuned Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2')
rf_grid.fit(X_train, y_train)
Rf_Model = rf_grid.best_estimator_
RF_pred = Rf_Model.predict(X_test)
Rf_mae = sqrt(mean_absolute_error(y_test, RF_pred))
Rf_rmse = sqrt(mean_squared_error(y_test, RF_pred))
Rf_r2 = r2_score(y_test, RF_pred)
pickle.dump(Rf_Model, open("Rf_model.pkl", "wb"))

# 4. Gradient Boosting
gb_model = GradientBoostingRegressor().fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_mae = sqrt(mean_absolute_error(y_test, gb_pred))
gb_rmse = sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)
pickle.dump(gb_model, open("gb_model.pkl", "wb"))

# 5. Tuned XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1]
}
xgb_grid = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=3, scoring='r2')
xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_
xgb_pred = xgb_model.predict(X_test)
xgb_mae = sqrt(mean_absolute_error(y_test, xgb_pred))
xgb_rmse = sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)
pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))

# 6. K-Nearest Neighbors
knn_model = KNeighborsRegressor().fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_mae = sqrt(mean_absolute_error(y_test, knn_pred))
knn_rmse = sqrt(mean_squared_error(y_test, knn_pred))
knn_r2 = r2_score(y_test, knn_pred)
pickle.dump(knn_model, open("knn_model.pkl", "wb"))

# 7. SVR
svr_model = SVR().fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)
svr_mae = sqrt(mean_absolute_error(y_test, svr_pred))
svr_rmse = sqrt(mean_squared_error(y_test, svr_pred))
svr_r2 = r2_score(y_test, svr_pred)
pickle.dump(svr_model, open("svr_model.pkl", "wb"))

# Summary
print("\nModel Performance:\n")
print(f"{'Model': <25} {'MAE': <10} {'RMSE': <10} {'R2_Score'}")
print("-" * 50)
print(f"{'Linear Regression': <25} {Lr_mae:.3f}     {Lr_rmse:.3f}     {Lr_r2:.3f}")
print(f"{'Decision Tree': <25} {DT_mae:.3f}     {DT_rmse:.3f}     {DT_r2:.3f}")
print(f"{'Random Forest (Tuned)': <25} {Rf_mae:.3f}     {Rf_rmse:.3f}     {Rf_r2:.3f}")
print(f"{'Gradient Boosting': <25} {gb_mae:.3f}     {gb_rmse:.3f}     {gb_r2:.3f}")
print(f"{'XGBoost (Tuned)': <25} {xgb_mae:.3f}     {xgb_rmse:.3f}     {xgb_r2:.3f}")
print(f"{'K-Nearest Neighbors': <25} {knn_mae:.3f}     {knn_rmse:.3f}     {knn_r2:.3f}")
print(f"{'Support Vector Regressor': <25} {svr_mae:.3f}     {svr_rmse:.3f}     {svr_r2:.3f}")

# Prediction Input
def get_user_input():
    try:
        sem1 = float(input("Sem 1 CGPA: "))
        study_hours = float(input("Daily Study Hours: "))
        sleep_hours = float(input("Daily Sleeping Hours: "))
        backlog = int(input("Active Backlog (0=No, 1=Yes): "))
        extra = float(input("Extra Curricular Hours per Week: "))
        screen_time = float(input("Screen Time (hrs/day): "))
        health = int(input("Health Condition (0=Low, 1=Medium, 2=High): "))
        stress = int(input("Stress Level (0=High, 1=Medium, 2=Low): "))
        motivation = int(input("Motivation Level (0=Low, 1=Medium, 2=High): "))

        return [[sem1, study_hours, sleep_hours, backlog, extra, screen_time,
                 health, stress, motivation]]
    except ValueError:
        print("Invalid input!")
        return None

input_data = get_user_input()

if input_data:
    models = {
        "Random Forest": Rf_Model,
        "Decision Tree": DT_Model,
        "Linear Regression": Lr_Model,
        "Gradient Boosting": gb_model,
        "XGBoost": xgb_model,
        "K-Nearest Neighbors": knn_model,
        "Support Vector Regressor": svr_model
    }

    for name, model in models.items():
        pred = model.predict(input_data)
        print(f"{name} Prediction: Predicted Semester 2 CGPA: {pred[0]:.2f}")

# Bar Chart
model_names = list(models.keys())
mae_values = [Lr_mae, DT_mae, Rf_mae, gb_mae, xgb_mae, knn_mae, svr_mae]
rmse_values = [Lr_rmse, DT_rmse, Rf_rmse, gb_rmse, xgb_rmse, knn_rmse, svr_rmse]
r2_values = [Lr_r2, DT_r2, Rf_r2, gb_r2, xgb_r2, knn_r2, svr_r2]

X_pos = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(X_pos - width, mae_values, width, label="MAE", color="Seagreen")
bars2 = ax.bar(X_pos, rmse_values, width, label="RMSE", color="Gold")
bars3 = ax.bar(X_pos + width, r2_values, width, label="R^2", color="SteelBlue")

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(X_pos)
ax.set_xticklabels(model_names, rotation=15)
ax.legend()

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("model_Comparison_bar_graph.png", dpi=600)
plt.show()