import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Загружаем данные
df = pd.read_excel("./data.xlsx")

# Преобразование категориальных признаков в числовые
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Разделение данных на обучающую и тестовую выборки
X = df.drop('Что вы предпочитаете?', axis=1)
y = df['Что вы предпочитаете?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Тестирование различных значений k
best_k = 1
best_scores = {
    'Test Homogeneity': 0,
    'Test Completeness': 0,
    'Test V_measure': 0,
    'Test Adjusted_rand': 0,
    'Test Adjusted_mutual_info': 0,
    'Test Silhouette': 0,
    'Test Precision': 0,
    'Test Recall': 0,
    'Test F1': 0,
    'Test Roc_auc': 0
}

for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Рассчет метрик
    scores = {
        'Test Homogeneity': homogeneity_score(y_test, y_pred),
        'Test Completeness': completeness_score(y_test, y_pred),
        'Test V_measure': v_measure_score(y_test, y_pred),
        'Test Adjusted_rand': adjusted_rand_score(y_test, y_pred),
        'Test Adjusted_mutual_info': adjusted_mutual_info_score(y_test, y_pred),
        'Test Silhouette': silhouette_score(X_test, y_pred),
        'Test Precision': precision_score(y_test, y_pred),
        'Test Recall': recall_score(y_test, y_pred),
        'Test F1': f1_score(y_test, y_pred),
        'Test Roc_auc': roc_auc_score(y_test, y_pred)
    }

    print(f'k={k}:')
    for metric_name, score in scores.items():
        print(f'  {metric_name}: {score}')

    # Выбор лучшего k
    if any(score > best_scores[metric_name] for metric_name, score in scores.items()):
        best_k = k
        best_scores = scores

print(f'Лучшее k: {best_k}')
print(f'С лучшими метриками:')
for metric_name, score in best_scores.items():
    print(f'  {metric_name}: {score}')