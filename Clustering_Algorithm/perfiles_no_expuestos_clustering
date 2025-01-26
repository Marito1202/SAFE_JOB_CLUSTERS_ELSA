# %%
scaler = StandardScaler()
df_scaled1 = scaler.fit_transform(df_tmp1)

# %%
kpca = KernelPCA(n_components=2, kernel='cosine', gamma=0.1,random_state=0)
X_kpca = kpca.fit_transform(df_scaled1)

plt.figure(figsize=(8, 6))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1])
plt.title("Kernel PCA with Cosine Kernel")
plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.show()

# %%
pca = PCA(n_components=2,random_state=0)
df_pca2 = pca.fit_transform(df_scaled1)
df_pca2 = pd.DataFrame(data=df_pca2, columns=['PC1', 'PC2'])

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=df_pca2)
# plt.title('PCA scatter plot')
plt.xlabel('1st component', fontsize=16)
plt.ylabel('2nd component', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# %%
pca = PCA(random_state=0)
pca.fit(df_scaled1)

plt.figure(figsize=(10,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.grid(True)
plt.show()

# %%
df_pca = pd.read_csv('kernel_pca_acoso_17102024.csv').to_numpy()

# %%
from sklearn.mixture import GaussianMixture

# %%
gmm = GaussianMixture(random_state=0,n_components=2)
cluster_labels = gmm.fit_predict(df_pca)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=cluster_labels, cmap='plasma')

# %%
plt.scatter(df_pca2.to_numpy()[:, 0], df_pca2.to_numpy()[:, 1], c=cluster_labels, cmap='plasma')

# %%
df_rev1 = dfper[dfper['target'].isin([0])].copy()
df_rev1['segmento'] = cluster_labels

# %%
df_rev1['segmento'].value_counts()

# %% [markdown]
# # Graficando resultados

# %%
