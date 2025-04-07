import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay
from ucimlrepo import fetch_ucirepo

# Configuracion
st.set_page_config(page_title="Glass Type Classification", layout="wide")
st.title("Clasificación de Tipos de Vidrio con RandomForestClassifier")

# Cargar dataset
glass_data = fetch_ucirepo(id=42)
X = glass_data.data.features
y = glass_data.data.targets['Type_of_glass']

# Mostrar distribucion de clases
st.subheader("Distribución de Clases")
fig, ax = plt.subplots()
sns.countplot(x=y, ax=ax)
st.pyplot(fig)

# Escalado y split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GridSearch para mejores hiperparámetros
st.subheader("Entrenamiento del Modelo RandomForestClassifier")
params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)

st.write("**Mejores hiperparámetros:**", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluación
st.subheader("Métricas de Evaluación")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Matriz de confusión
st.subheader("Matriz de Confusión")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig)

# Curvas ROC
st.subheader("Curvas ROC (One-vs-Rest)")
classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)
y_score = best_model.predict_proba(X_test)

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"Clase {classes[i]} (AUC = {roc_auc:.2f})")

ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - RandomForestClassifier")
ax.legend(loc="lower right")
st.pyplot(fig)

st.success("Modelo RandomForestClassifier entrenado y evaluado correctamente.")
