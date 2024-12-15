df_tmp1 = dfper[dfper['target'].isin([1,2])].copy()

list_n_categorias = []
list_pct_cat_mayoritaria = []
list_pct_cat_minoritaria = []
list_cat_mayoritaria = []
list_cat_minoritaria = []

for col in df_tmp1.columns:
    n_categorias = df_tmp1[col].value_counts().shape[0]
    list_n_categorias.append(n_categorias)
    pct_cat_mayoritaria = df_tmp1[col].value_counts(normalize=True).sort_values(ascending=False).values[0]
    pct_cat_minoritaria = df_tmp1[col].value_counts(normalize=True).sort_values(ascending=False).values[-1]
    list_pct_cat_mayoritaria.append(pct_cat_mayoritaria)
    list_pct_cat_minoritaria.append(pct_cat_minoritaria)
    cat_mayoritaria = df_tmp1[col].value_counts(normalize=True).sort_values(ascending=False).index[0]
    cat_minoritaria = df_tmp1[col].value_counts(normalize=True).sort_values(ascending=False).index[-1]
    list_cat_mayoritaria.append(cat_mayoritaria)
    list_cat_minoritaria.append(cat_minoritaria)

# %%
tmp = pd.DataFrame(zip(df_tmp1.columns,list_n_categorias,list_pct_cat_mayoritaria,list_pct_cat_minoritaria),columns=['variable','n_categorias','pct_max','pct_min'])
tmp = tmp.sort_values('n_categorias')
tmp = tmp.reset_index(drop=True)

# %%
unicat = list(tmp[tmp['n_categorias']==1].variable.values)

# %%
info_per = [columna for columna in tmp['variable'] if columna.startswith('ip')]
info_lab = [columna for columna in tmp['variable'] if columna.startswith('il')]
info_ad = [columna for columna in tmp['variable'] if columna.startswith('ad')]
info_act = [columna for columna in tmp['variable'] if columna.startswith('act')]

# %%
df_tmp1 = df_tmp1.drop(unicat+['measurement_process_id']+info_per+info_ad+info_lab+info_act,axis=1)

# %%
df_tmp1 = df_tmp1.drop(columns=['Acoso_Tecnico','Testigo_Tecnico','Acoso_Declarado','Testigo_Declarado','Acoso_Total','Testigo_Total','target'])

# %% [markdown]
# # Tratamiento de missing

# %% [markdown]
# ## Analisis de missing

# %%
m_per = pd.DataFrame({'n_missing': dfper.isnull().sum(), 'pct_missing': dfper.isnull().mean()*100})
m_per = m_per.sort_values('pct_missing',ascending=True)
m_per = m_per.reset_index()
m_per = m_per.rename(columns={'index':'code'})
m_per = per.merge(m_per,on='code',how='right')

# %% [markdown]
# ## Tratamiento

# %%
var_list = m_per[m_per['categoria']=='Acciones Acoso Declarado']
var_list = var_list['code'].sort_values()
var_list = list(var_list.values)

# %%
df_tmp1[var_list] = df_tmp1[var_list].fillna('No aplica')

# %%
var_list = m_per[(m_per['categoria']=='Testigos Declarados')&(m_per['n_missing']!=1)]
var_list = var_list['code'].sort_values()
var_list = list(var_list.values)

# %%
df_tmp1[var_list] = df_tmp1[var_list].fillna('No aplica')

# %%
var_list = m_per[(m_per['categoria']=='Barreras de denuncia')&(m_per['code']!='bad_010')]
var_list = var_list['code'].sort_values()
var_list = list(var_list.values)

# %%
df_tmp1[var_list] = df_tmp1[var_list].fillna('No aplica')

# %%
var_list = m_per[(m_per['categoria']=='Costos')]
var_list = var_list['code'].sort_values()
var_list = list(var_list.values)

# %%
df_tmp1[var_list] = df_tmp1[var_list].fillna('No aplica')

# %%
df_tmp1['fre_001'] = df_tmp1['fre_001'].fillna('No aplica')

# %%
df_tmp1['pl_001'] = df_tmp1['pl_001'].fillna('No aplica')

# %%
var_list = m_per[(m_per['categoria']=='Perfil Acosador - Declarado')]
var_list = var_list['code'].sort_values()
var_list = list(var_list.values)

# %%
df_tmp1[var_list] = df_tmp1[var_list].fillna('No aplica')

# %%
var_list = m_per[(m_per['categoria']=='Acoso Técnico')]
var_list = var_list['code'].sort_values()
var_list = list(var_list.values)

# %%
df_tmp1[var_list] = df_tmp1[var_list].fillna('No aplica')

# %%
var_list = m_per[(m_per['categoria']=='Testigos Técnicos')]
var_list = var_list['code'].sort_values()
var_list = list(var_list.values)

# %%
df_tmp1[var_list] = df_tmp1[var_list].fillna('No aplica')

# %%
var_list = [col for col in dfper.columns if 'case_' in col]
df_tmp1[var_list] = df_tmp1[var_list].fillna('No aplica')

# %%
df_tmp1 = df_tmp1.drop(['bad_010'],axis=1)

# %%
temporal = pd.DataFrame(df_tmp1.isnull().sum(),columns=['n_missing'])
temporal= temporal.sort_values('n_missing')

# %%
temporal = temporal.reset_index()

# %%
for i in temporal[temporal['n_missing']>0]['index'].values:
    moda = df_tmp1[i].mode().values[0]
#     print(i,moda)
    df_tmp1[i] = df_tmp1[i].fillna(moda)

# %% [markdown]
# # Codificacion de variables

# %%
def cat_to_dummies(df,columna):
    unique_cats = list(df[f'{columna}'].unique())
    unique_dict = dict(zip(unique_cats,[f'{columna}_{i}' for i in range(len(unique_cats))]))
    out = pd.get_dummies(df[f'{columna}']).rename(columns=unique_dict)
    print(unique_dict)
    return out

# %%
tmp = cat_to_dummies(df_tmp1,'name')
df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
df_tmp1 = df_tmp1.drop(columns=['name'])

# %%
tmp = cat_to_dummies(df_tmp1,'sp_001')
df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
df_tmp1 = df_tmp1.drop(columns=['sp_001'])

# %%
tmp = cat_to_dummies(df_tmp1,'sp_002')
df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
df_tmp1 = df_tmp1.drop(columns=['sp_002'])

# %%
tmp = cat_to_dummies(df_tmp1,'sp_003')
df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
df_tmp1 = df_tmp1.drop(columns=['sp_003'])

# %%
df_tmp1['sp_004'] = df_tmp1['sp_004'].replace({True:1,False:0})
df_tmp1['sp_005'] = df_tmp1['sp_005'].replace({True:1,False:0})
df_tmp1['sp_006'] = df_tmp1['sp_006'].replace({True:1,False:0})
df_tmp1['sp_007'] = df_tmp1['sp_007'].replace({True:1,False:0})
df_tmp1['sp_008'] = df_tmp1['sp_008'].replace({True:1,False:0})
# df_tmp1['sp_009'] = df_tmp1['sp_009'].replace({True:1,False:0})
df_tmp1['sp_010'] = df_tmp1['sp_010'].replace({True:1,False:0})
df_tmp1['sp_011'] = df_tmp1['sp_011'].replace({True:1,False:0})

# %%
tmp = cat_to_dummies(df_tmp1,'sp_012')
df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
df_tmp1 = df_tmp1.drop(columns=['sp_012'])

# %%
tmp = cat_to_dummies(df_tmp1,'pl_001')
df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
df_tmp1 = df_tmp1.drop(columns=['pl_001'])

# %%
tolerancia = [col for col in df_tmp1.columns if col.startswith('tol')&(len(col)<8)]

# %%
for columna in tolerancia:
    tmp = cat_to_dummies(df_tmp1,columna)
    df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
    df_tmp1 = df_tmp1.drop(columns=[columna])

# %%
df_tmp1['cpt_001'] = df_tmp1['cpt_001'].replace({'Sí.':1,'No.':0})
df_tmp1['cpt_002'] = df_tmp1['cpt_002'].replace({'Sí.':1,'No.':0})
df_tmp1['cpt_003'] = df_tmp1['cpt_003'].replace({'Sí.':1,'No.':0})
df_tmp1['cpt_004'] = df_tmp1['cpt_004'].replace({'Sí.':1,'No.':0})

# %%
vars_analysis = [col for col in df_tmp1.columns if (col.startswith('pad'))]

# %%
for columna in vars_analysis:
    tmp = cat_to_dummies(df_tmp1,columna)
    df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
    df_tmp1 = df_tmp1.drop(columns=[columna])

# %%
vars_analysis = [col for col in df_tmp1.columns if (col.startswith('aad')&(len(col)==7))]

# %%
for columna in vars_analysis:
    tmp = cat_to_dummies(df_tmp1,columna)
    df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
    df_tmp1 = df_tmp1.drop(columns=[columna])

# %%
vars_analysis = [col for col in df_tmp1.columns if (col.startswith('bad'))]

# %%
for columna in vars_analysis:
    tmp = cat_to_dummies(df_tmp1,columna)
    df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
    df_tmp1 = df_tmp1.drop(columns=[columna])

# %%
vars_analysis = [col for col in df_tmp1.columns if (col.startswith('at'))]

# %%
for columna in vars_analysis:
    tmp = cat_to_dummies(df_tmp1,columna)
    df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
    df_tmp1 = df_tmp1.drop(columns=[columna])

# %%
tmp = cat_to_dummies(df_tmp1,'fre_001')
df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
df_tmp1 = df_tmp1.drop(columns=['fre_001'])

# %%
vars_analysis = [col for col in df_tmp1.columns if (col.startswith('cos')&(len(col)<=8))]

# %%
for columna in vars_analysis:
    tmp = cat_to_dummies(df_tmp1,columna)
    df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
    df_tmp1 = df_tmp1.drop(columns=[columna])

# %%
vars_analysis = [col for col in df_tmp1.columns if (col.startswith('td'))]

# %%
for columna in vars_analysis:
    tmp = cat_to_dummies(df_tmp1,columna)
    df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
    df_tmp1 = df_tmp1.drop(columns=[columna])

# %%
vars_analysis = [col for col in df_tmp1.columns if (col.startswith('tt'))]

# %%
for columna in vars_analysis:
    tmp = cat_to_dummies(df_tmp1,columna)
    df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
    df_tmp1 = df_tmp1.drop(columns=[columna])

# %%
vars_analysis = [col for col in df_tmp1.columns if (col.startswith('con'))]

# %%
for columna in vars_analysis:
    tmp = cat_to_dummies(df_tmp1,columna)
    df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
    df_tmp1 = df_tmp1.drop(columns=[columna])

# %%
vars_analysis = [col for col in df_tmp1.columns if (col.startswith('case'))]

# %%
for columna in vars_analysis:
    tmp = cat_to_dummies(df_tmp1,columna)
    df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
    df_tmp1 = df_tmp1.drop(columns=[columna])

# %%
tmp = cat_to_dummies(df_tmp1,'sat_001')
df_tmp1 = pd.concat([df_tmp1,tmp],axis=1)
df_tmp1 = df_tmp1.drop(columns=['sat_001'])

# %% [markdown]
# # Validacion de missing

# %%
tmp = pd.DataFrame(df_tmp1.isnull().sum()).reset_index()
tmp.columns = ['var','nmiss']
tmp = tmp.sort_values('nmiss',ascending=True)
tmp = tmp.reset_index(drop=True)
# OK

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
df_pca = pca.fit_transform(df_scaled1)
df_pca = pd.DataFrame(data=df_pca, columns=['PC1', 'PC2'])

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=df_pca)
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
# pca = PCA(n_components=100,random_state=0)
# df_pca = pca.fit_transform(df_scaled1)
kpca = KernelPCA(n_components=100, kernel='cosine', gamma=0.1,random_state=0)
df_pca = kpca.fit_transform(df_scaled1)

# %%
from sklearn.mixture import GaussianMixture

# %%
gmm = GaussianMixture(random_state=0,n_components=3)
cluster_labels = gmm.fit_predict(df_pca)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=cluster_labels, cmap='plasma')

# %% [markdown]
# # Calculando clusters

# %%
n_components = np.arange(1, 11)

# %%
aic_values = []
bic_values = []

# Ajustar el modelo y calcular AIC/BIC para cada número de clústeres
for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(df_pca)
    aic_values.append(gmm.aic(df_pca))
    bic_values.append(gmm.bic(df_pca))

# %%
plt.figure(figsize=(8, 4))
plt.plot(n_components, aic_values, label='AIC', marker='o')
plt.plot(n_components, bic_values, label='BIC', marker='o')
plt.xlabel('Número de Clústeres')
plt.ylabel('Puntaje')
plt.legend()
plt.title('AIC y BIC para determinar el número óptimo de clústeres')
plt.show()

# %%
log_likelihoods = []

# Calcular log-likelihood para cada número de clústeres
for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(df_pca)
    log_likelihoods.append(gmm.score(df_pca))  # Promedio de log-likelihood

# Graficar la log-likelihood
plt.figure(figsize=(8, 4))
plt.plot(n_components, log_likelihoods, marker='o')
plt.xlabel('Número de Clústeres')
plt.ylabel('Log-Likelihood')
plt.title('Curva de Log-Likelihood para determinar el número óptimo de clústeres')
plt.show()

# %%
df_rev1 = dfper[dfper['target'].isin([1,2])].copy()

# %%
df_rev1['segmento'] = cluster_labels

# %%
df_rev1['segmento'].value_counts()

# %%
plt.pie(df_rev1['segmento'].value_counts().values,labels=df_rev1['segmento'].value_counts().index)
plt.show()

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
grafica_por_dimension(df_rev1,'segmento','sp_001',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','sp_002',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','sp_003',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','td_001',3,cat_omit='Otro',y_legend=0.46)

# %%
df_rev1.groupby(['segmento','td_001'],dropna=False)['measurement_process_id'].count().to_frame().reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','ad_001',3,cat_omit='Otro',y_legend=0.47)

# %%
df_rev1.groupby(['segmento','ad_001'],dropna=False)['measurement_process_id'].count().to_frame().reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','ad_014',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','Acoso_Tecnico',3,cat_omit='Otro',y_legend=0.47,title='')

# %%
df_rev1.groupby(['segmento','Acoso_Tecnico']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='Acoso_Tecnico',index='segmento').reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','Testigo_Tecnico',3,cat_omit='Otro',y_legend=0.47,title='')

# %%
df_rev1.groupby(['segmento','Testigo_Tecnico']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='Testigo_Tecnico',index='segmento').reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','tol_001',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_002',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_003',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_004',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_005',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_006',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_007',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_008',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_009',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_010',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_011',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','tol_012',3,cat_omit='Otro',y_legend=0.46)

# %%
tolerancia = [col for col in dfper.columns if 'tol_' in col]
df_rev1['flg_tol_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['No veo nada de malo con esa situación.', 'No son formas de comportarse en el trabajo, pero no es acoso.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_tol_neg']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='flg_tol_neg',index='segmento').reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','cpt_001',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','cpt_002',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','cpt_003',3,cat_omit='Otro',y_legend=0.46)

# %%
grafica_por_dimension(df_rev1,'segmento','cpt_004',3,cat_omit='Otro',y_legend=0.46)

# %%
tolerancia = ['cpt_002','cpt_004','cpt_001','cpt_003']
df_rev1['flg_tol_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['No.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_tol_neg']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='flg_tol_neg',index='segmento').reset_index()

# %%
tolerancia = ['cpt_002','cpt_004']
df_rev1['flg_tol_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['No.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_tol_neg']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='flg_tol_neg',index='segmento').reset_index()

# %%
tolerancia = ['cpt_001','cpt_003']
df_rev1['flg_tol_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['No.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_tol_neg']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='flg_tol_neg',index='segmento').reset_index()

# %%
df_rev1.groupby(['segmento','cpt_001']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='cpt_001',index='segmento').reset_index()

# %%
df_rev1.groupby(['segmento','cpt_002']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='cpt_002',index='segmento').reset_index()

# %%
df_rev1.groupby(['segmento','cpt_003']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='cpt_003',index='segmento').reset_index()

# %%
df_rev1.groupby(['segmento','cpt_004']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='cpt_004',index='segmento').reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','con_001',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','con_004',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','con_006',3,cat_omit='Otro',y_legend=0.47)

# %%
tolerancia = ['con_004','con_006']
df_rev1['flg_con_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['De acuerdo.','Totalmente de acuerdo.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_con_neg']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='flg_con_neg',index='segmento').reset_index()

# %%
tolerancia = ['con_001','con_002','con_003','con_005']
df_rev1['flg_con_neg'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['De acuerdo.','Totalmente de acuerdo.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','flg_con_neg']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='flg_con_neg',index='segmento').reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','act_001',3,cat_omit='Otro',y_legend=0.47)

# %%
grafica_por_dimension(df_rev1,'segmento','act_002',3,cat_omit='Otro',y_legend=0.47)

# %%
testigo_activo = ['act_001', 'act_002', 'act_003', 'act_004', 'act_005']
testigo_pasivo = ['act_006', 'act_007', 'act_008']
df_rev1['testigo_activo_flg_pos'] = df_rev1[testigo_activo].apply(lambda row: 1 if any(x in [True] for x in row) else 0, axis=1)
df_rev1['testigo_pasivo_flg_pos'] = df_rev1[testigo_pasivo].apply(lambda row: 1 if any(x in [True] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','testigo_activo_flg_pos'],dropna=False).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='testigo_activo_flg_pos',index='segmento').reset_index()

# %%
df_rev1.groupby(['segmento','testigo_pasivo_flg_pos']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='testigo_pasivo_flg_pos',index='segmento').reset_index()

# %%
grafica_por_dimension(df_rev1,'segmento','tt_001',3,cat_omit='Otro',y_legend=0.47)

# %%
# Generar variable de acoso total (indique quien es declarado y quien es tecnico)

# %%
df_rev1['ad_001'].value_counts()

# %%
df_rev1['Acoso_Tecnico'].value_counts()

# %%
def acoso_total_col(ad_001,acoso_tecnico):
    if ad_001 == 'Sí, me ha pasado.':
        return 'acoso declarado'
    elif acoso_tecnico == 1:
        return 'acoso tecnico'
    else:
        return 'otro'

# %%
df_rev1['Acoso_Total_Col'] = df_rev1.apply(lambda x: acoso_total_col(x['ad_001'],x['Acoso_Tecnico']),axis=1)

# %%
df_rev1.groupby(['segmento','Acoso_Total_Col']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='Acoso_Total_Col',index='segmento').reset_index()

# %%
df_rev1['td_001'].value_counts().index

# %%
def testigo_total_col(ad_001,acoso_tecnico):
    if ad_001 == 1:
        return 'testigo declarado'
    elif acoso_tecnico == 1:
        return 'testigo tecnico'
    else:
        return 'otro'

# %%
df_rev1['Testigo_Total_Col'] = df_rev1.apply(lambda x: testigo_total_col(x['Testigo_Declarado'],x['Testigo_Tecnico']),axis=1)

# %%
df_rev1.groupby(['segmento','Testigo_Total_Col']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='Testigo_Total_Col',index='segmento').reset_index()

# %%
df_rev1.groupby(['segmento','Testigo_Total']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='Testigo_Total',index='segmento').reset_index()

# %%
tolerancia = ['aad_001', 'aad_002', 'aad_003', 'aad_004', 'aad_005', 'aad_008']
df_rev1['aad_flg_pos'] = df_rev1[tolerancia].apply(lambda row: 1 if any(x in ['Sí.'] for x in row) else 0, axis=1)

# %%
df_rev1.groupby(['segmento','aad_flg_pos']).agg(cantidad=('measurement_process_id','count')).reset_index().pivot_table(values='cantidad',columns='aad_flg_pos',index='segmento').reset_index()


