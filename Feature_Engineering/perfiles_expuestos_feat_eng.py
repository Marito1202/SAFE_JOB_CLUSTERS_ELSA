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
