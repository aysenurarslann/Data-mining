import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

# Veri setini yükleme
data_path = "../../Masaüstü/pythonProject/hitters.csv"
data_df = pd.read_csv(data_path)

# Eksik verileri içermeyen özelliklerin seçilmesi
X = data_df.dropna().drop(['Salary'], axis=1)

# Kategorik değişkenleri One-Hot Encoding ile dönüştürme
X_encoded = pd.get_dummies(X, columns=['League', 'Division', 'NewLeague'], drop_first=True)

# Hedef değişken
y = data_df.dropna()['Salary']

# Veriyi standardize etme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Randomized PCA modeli oluşturma
rpca = PCA(n_components=2, svd_solver='randomized')  # İki bileşene dönüştürme
X_rpca = rpca.fit_transform(X_scaled)

# Yeni veri setinin oluşturulması
rpca_df = pd.DataFrame(data=X_rpca, columns=['PC1', 'PC2'])
rpca_df['Salary'] = y.values

# Randomized PCA sonuçlarının görselleştirilmesi
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', data=rpca_df, hue='Salary', palette='viridis')
plt.title('Randomized PCA Sonuçları')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Kutu grafikleri oluşturma
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_df[['Salary']], color='lightblue')
plt.title('Salary Kutu Grafiği')
plt.xlabel('Salary')
plt.grid(True)
plt.show()

# Karar Ağacı Regresyon Modeli
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.34, random_state=42)

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
dt_y_pred_train = dt_reg.predict(X_train)
dt_y_pred_test = dt_reg.predict(X_test)

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)
rf_y_pred_train = rf_reg.predict(X_train)
rf_y_pred_test = rf_reg.predict(X_test)

# Gerçek ve Tahmin Edilen Değerlerin Görselleştirilmesi
plt.figure(figsize=(10, 6))

# Karar Ağacı Regresyon Modeli
plt.subplot(1, 2, 1)
plt.scatter(y_train, dt_y_pred_train, color='blue')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--k')
plt.title('Karar Ağacı Regresyon Modeli (Eğitim)')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')

plt.subplot(1, 2, 2)
plt.scatter(y_test, dt_y_pred_test, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
plt.title('Karar Ağacı Regresyon Modeli (Test)')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')

plt.tight_layout()
plt.show()

# Rastgele Orman Regresyon Modeli
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, rf_y_pred_train, color='red')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--k')
plt.title('Rastgele Orman Regresyon Modeli (Eğitim)')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_y_pred_test, color='red')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
plt.title('Rastgele Orman Regresyon Modeli (Test)')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')

plt.tight_layout()
plt.show()


# K-means kümeleme modeli oluşturma
kmeans_cluster = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans_cluster.fit_predict(X_scaled)

# Kümeleme sonuçlarını veri çerçevesine ekleme
cluster_df = pd.DataFrame(data=X_rpca, columns=['PC1', 'PC2'])
cluster_df['Cluster'] = cluster_labels

# Kümeleme sonuçlarının görselleştirilmesi
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', data=cluster_df, hue='Cluster', palette='viridis')
plt.title('K-means Kümeleme Sonuçları')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Başarı oranlarının hesaplanması
# Eğitim seti üzerinde RMSE
dt_rmse_train = mean_squared_error(y_train, dt_y_pred_train, squared=False)
rf_rmse_train = mean_squared_error(y_train, rf_y_pred_train, squared=False)

# Test seti üzerinde RMSE
dt_rmse_test = mean_squared_error(y_test, dt_y_pred_test, squared=False)
rf_rmse_test = mean_squared_error(y_test, rf_y_pred_test, squared=False)

# Çapraz doğrulama (CV) üzerinde RMSE
dt_rmse_cv = -np.mean(cross_val_score(dt_reg, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error'))
rf_rmse_cv = -np.mean(cross_val_score(rf_reg, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error'))

# Başarı oranlarını bir sözlükte topla
performance_metrics = {
    "Karar Ağacı Regresyon Modeli": {"Eğitim seti RMSE": dt_rmse_train, "Test seti RMSE": dt_rmse_test, "Çapraz doğrulama (CV) RMSE": dt_rmse_cv},
    "Rastgele Orman Regresyon Modeli": {"Eğitim seti RMSE": rf_rmse_train, "Test seti RMSE": rf_rmse_test, "Çapraz doğrulama (CV) RMSE": rf_rmse_cv}
}

# Başarı oranlarını yazdırma
for model, metrics in performance_metrics.items():
    print(model + ":")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")
