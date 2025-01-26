df_tmp1 = dfper[dfper['target']==0].copy()

list_n_categorias = []
list_pct_cat_mayoritaria = []
list_pct_cat_minoritaria = []
list_cat_mayoritaria = []
list_cat_minoritaria = []

for col in df_tmp1.columns:
    n_categorias = df_tmp1[col].value_counts().shape[0]
    if n_categorias > 0: 
        list_n_categorias.append(n_categorias)
        pct_cat_mayoritaria = df_tmp1[col].value_counts(normalize=True).sort_values(ascending=False).values[0]
        pct_cat_minoritaria = df_tmp1[col].value_counts(normalize=True).sort_values(ascending=False).values[-1]
        list_pct_cat_mayoritaria.append(pct_cat_mayoritaria)
        list_pct_cat_minoritaria.append(pct_cat_minoritaria)
        cat_mayoritaria = df_tmp1[col].value_counts(normalize=True).sort_values(ascending=False).index[0]
        cat_minoritaria = df_tmp1[col].value_counts(normalize=True).sort_values(ascending=False).index[-1]
        list_cat_mayoritaria.append(cat_mayoritaria)
        list_cat_minoritaria.append(cat_minoritaria)
    else:
        list_n_categorias.append(np.nan)
        list_pct_cat_mayoritaria.append(np.nan)
        list_pct_cat_minoritaria.append(np.nan)

# %%
tmp = pd.DataFrame(zip(df_tmp1.columns,list_n_categorias,list_pct_cat_mayoritaria,list_pct_cat_minoritaria),columns=['variable','n_categorias','pct_max','pct_min'])
tmp = tmp.sort_values('n_categorias')
tmp = tmp.reset_index(drop=True)

# %%
unicat = tmp[(tmp['n_categorias'].isnull())|(tmp['n_categorias']==1)]['variable'].tolist()

# %%
info_per = [columna for columna in tmp['variable'] if columna.startswith('ip')]
info_lab = [columna for columna in tmp['variable'] if columna.startswith('il')]
info_ad = [columna for columna in tmp['variable'] if columna.startswith('ad')]
info_td = [columna for columna in tmp['variable'] if columna.startswith('td')]

# %%
df_tmp1 = df_tmp1.drop(unicat+['measurement_process_id']+info_per+info_lab+info_ad+info_td,axis=1)

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
df_tmp1['pl_001'] = df_tmp1['pl_001'].fillna('No aplica')

df_tmp1 = df_tmp1.drop(columns=['pl_001'])

# %%
var_list = [col for col in dfper.columns if 'case_' in col]
df_tmp1[var_list] = df_tmp1[var_list].fillna('No aplica')

df_tmp1 = df_tmp1.drop(columns=var_list)

# %% [markdown]
# ### Analizando correlacion de casos con otras variables

# %%
from scipy.stats import chi2_contingency

# %%
def cramerV(label,x):
    confusion_matrix = pd.crosstab(label, x)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r,k = confusion_matrix.shape
    phi2 = chi2/n
    phi2corr = max(0,phi2-((k-1)*(r-1))/(n-1))
    rcorr = r - ((r - 1) ** 2) / ( n - 1 )
    kcorr = k - ((k - 1) ** 2) / ( n - 1 )
    try:
        if min((kcorr - 1),(rcorr - 1)) == 0:
            warnings.warn(
            "Unable to calculate Cramer's V using bias correction. Consider not using bias correction",RuntimeWarning)
            v = 0
            print("If condition Met: ",v)
        else:
            v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
            print("Else condition Met: ",v)
    except:
        print("inside error")
        v = 0
    return v

# %%
def plot_cramer(df):
    cramer = pd.DataFrame(index=df.columns,columns=df.columns)
    for column_of_interest in df.columns:
        try:
            temp = {}

            columns = df.columns
            for j in range(0,len(columns)):
                v = cramerV(df[column_of_interest],df[columns[j]])
                cramer.loc[column_of_interest,columns[j]] = v
                if (column_of_interest==columns[j]):
                    pass
                else:
                    temp[columns[j]] = v
            cramer.fillna(value=np.nan,inplace=True)
        except:
            print('Dropping row:',column_of_interest)
            pass
    return cramer

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
vars_analysis = [col for col in df_tmp1.columns if (col.startswith('con'))]

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
