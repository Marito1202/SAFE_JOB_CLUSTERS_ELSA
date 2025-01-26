from perfiles_no_expuestos_feat_eng import *

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
def grafica_por_dimension(df,
                          code_dimension,
                          code_question,
                          num_partitions,
                          max_char_per_line=50,
                          figsize=(10,6),
                          title=None,
                          cat_omit=None,x_legend = 0.91,y_legend = 0.5):
    if title == None:
        title = m_per[m_per['code']==f'{code_question}']['pregunta'].values[0]
    if cat_omit==None:
        agg_data = df.groupby([f'{code_dimension}',f'{code_question}']).agg(CTD_PERSONAS = ('ad_001','count')).reset_index()
        total_counts = df.groupby(f'{code_dimension}').agg(TOTAL_PERSONAS=('ad_001', 'count')).reset_index()
    elif cat_omit!=None:
        tmp = df[df[f'{code_question}']!=f'{cat_omit}']
        agg_data = tmp.groupby([f'{code_dimension}',f'{code_question}']).agg(CTD_PERSONAS = ('ad_001','count')).reset_index()
        total_counts = tmp.groupby(f'{code_dimension}').agg(TOTAL_PERSONAS=('ad_001', 'count')).reset_index()

    
    agg_data = agg_data.merge(total_counts, on=f'{code_dimension}',how='left')
    agg_data['PROPORCION'] = agg_data['CTD_PERSONAS'] / agg_data['TOTAL_PERSONAS']
    
    if len(title) > max_char_per_line:
        title = '\n'.join(textwrap.wrap(title, break_long_words=False, max_lines=num_partitions))

    category_mapping = {category: idx + 1 for idx, category in enumerate(agg_data[code_dimension].unique())}
    agg_data['category_num'] = agg_data[code_dimension].map(category_mapping)

    plt.figure(figsize=figsize)
    sns.barplot(data=agg_data, x=f'category_num', y='PROPORCION', hue=f'{code_question}')
    plt.title(f'{title}')
    plt.ylabel('Proporción (%)')
    plt.xlabel(f'{code_dimension}')
    plt.legend(title='Respuestas', bbox_to_anchor=(1.00, 1), loc='upper left')
    plt.xticks(rotation=0, ha='center', fontsize=10)
    
    category_legend = [f'{num}: {category}' for category, num in category_mapping.items()]
    plt.figtext(x_legend, y_legend, '\n'.join(category_legend), fontsize=10, ha='left')
    
    plt.show()

# %%
def grafica_por_dimension_relativa(df,
                          code_dimension,
                          code_question,
                          num_partitions,
                          max_char_per_line=50,
                          figsize=(10,6),
                          title=None,
                          cat_omit=None,x_legend = 0.91,y_legend = 0.5):
    if title == None:
        title = m_per[m_per['code']==f'{code_question}']['pregunta'].values[0]
    if cat_omit==None:
        agg_data = df.groupby([f'{code_dimension}',f'{code_question}']).agg(CTD_PERSONAS = ('ad_001','count')).reset_index()
        total_counts = df.groupby(f'{code_question}').agg(TOTAL_PERSONAS=('ad_001', 'count')).reset_index()
    elif cat_omit!=None:
        tmp = df[df[f'{code_question}']!=f'{cat_omit}']
        agg_data = tmp.groupby([f'{code_dimension}',f'{code_question}']).agg(CTD_PERSONAS = ('ad_001','count')).reset_index()
        total_counts = tmp.groupby(f'{code_question}').agg(TOTAL_PERSONAS=('ad_001', 'count')).reset_index()

    
    agg_data = agg_data.merge(total_counts, on=f'{code_question}',how='left')
    agg_data['PROPORCION'] = agg_data['CTD_PERSONAS'] / agg_data['TOTAL_PERSONAS']
    
    if len(title) > max_char_per_line:
        title = '\n'.join(textwrap.wrap(title, break_long_words=False, max_lines=num_partitions))

    category_mapping = {category: idx + 1 for idx, category in enumerate(agg_data[code_question].unique())}
    agg_data['category_num'] = agg_data[code_question].map(category_mapping)

    plt.figure(figsize=figsize)
    sns.barplot(data=agg_data, x=f'category_num', y='PROPORCION', hue=f'{code_dimension}',palette='tab10')
    plt.title(f'{title}')
    plt.ylabel('Proporción (%)')
    plt.xlabel(f'{code_question}')
    plt.legend(title='Segmentos', bbox_to_anchor=(1.00, 1), loc='upper left')
    plt.xticks(rotation=0, ha='center', fontsize=10)
    
    category_legend = [f'{num}: {category}' for category, num in category_mapping.items()]
    plt.figtext(x_legend, y_legend, '\n'.join(category_legend), fontsize=10, ha='left')
    
    plt.show()

# %%
grafica_por_dimension(df_rev1,'segmento','sp_001',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','sp_002',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','sp_003',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','sp_012',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_001',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_002',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_003',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_008',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_009',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_011',3,cat_omit='Otro',y_legend=0.47)

# %%
tolerancia = [col for col in dfper.columns if 'tol_' in col]
df_rev1['flg_tol_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['No veo nada de malo con esa situación.', 'No son formas de comportarse en el trabajo, pero no es acoso.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_tol_neg'])['measurement_process_id'].count().to_frame().reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','cpt_001',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','cpt_002',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','cpt_003',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','cpt_004',3,cat_omit='Otro',y_legend=0.47)

# %%
# tolerancia = [col for col in dfper.columns if 'cpt_' in col]
# Repetitivo y jerarquia
tolerancia = ['cpt_002','cpt_004']
df_rev1['flg_cpt_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['No.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_cpt_neg'])['measurement_process_id'].count().to_frame().reset_index()

# %%
tolerancia = ['cpt_001','cpt_003']
# Repetitivo y jerarquia
df_rev1['flg_cpt_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['No.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_cpt_neg'])['measurement_process_id'].count().to_frame().reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','con_001',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','con_002',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','con_003',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','con_004',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','con_005',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','con_006',3,cat_omit='Otro',y_legend=0.47)

# %%
tolerancia = ['con_004','con_006']
df_rev1['flg_con_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['De acuerdo.','Totalmente de acuerdo.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_con_neg'])['measurement_process_id'].count().to_frame().reset_index()

# %%
tolerancia = ['con_001','con_002','con_003','con_005']
df_rev1['flg_con_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['De acuerdo.','Totalmente de acuerdo.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_con_neg'])['measurement_process_id'].count().to_frame().reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','name',3,cat_omit='Otro',y_legend=0.47,title='')

# %%
df_rev1['case_001_case_resolution'].value_counts(dropna=False,normalize=True)

# %%
grafica_por_dimension(df_rev1,'segmento','case_001_case_resolution',3,cat_omit='Otro',y_legend=0.47,title='')

# %%
grafica_por_dimension(df_rev1,'segmento','case_002_case_resolution',3,cat_omit='Otro',y_legend=0.47,title='')

# %%
grafica_por_dimension(df_rev1,'segmento','case_003_case_resolution',3,cat_omit='Otro',y_legend=0.47,title='')

# %%
grafica_por_dimension(df_rev1,'segmento','case_004_case_resolution',3,cat_omit='Otro',y_legend=0.47,title='')

# %% [markdown]
# # Calculo de AUC

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

def calculate_auc_with_decision_tree(df, categorical_columns, target_column):
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    
    auc_scores = {}
    
    for col in categorical_columns:
        X = pd.get_dummies(df[col], prefix=col)
        y = df[target_column]
        
        clf = DecisionTreeClassifier(random_state=42)
        y_pred_proba = cross_val_predict(clf, X, y, cv=5, method='predict_proba')
    
        auc_per_class = []
        for i in range(len(le.classes_)):
            try:
                auc = roc_auc_score((y == i).astype(int), y_pred_proba[:, i])
                auc_per_class.append(auc)
            except ValueError:
                print(f"No se pudo calcular AUC para {col}, clase {i}")

        auc_scores[col] = np.mean(auc_per_class)
    
    return auc_scores

# %%
categorical_columns = dfper.drop(columns=['target','measurement_process_id']).columns
target_column = 'target'
auc_scores = calculate_auc_with_decision_tree(dfper, categorical_columns, target_column)
print(auc_scores)

# %%
dfauc = pd.DataFrame(zip(auc_scores.keys(),auc_scores.values()),columns=['variable','AUC'])

# %%
dfauc.sort_values('AUC',ascending=False)


