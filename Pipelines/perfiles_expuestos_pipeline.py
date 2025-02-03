exec(open(f"{work_dir}\Downloads\SAFE_JOB_CLUSTERS_ELSA-main\SAFE_JOB_CLUSTERS_ELSA-main\Feature_Engineering\perfiles_expuestos_feat_eng.py").read())

exec(open(f"{work_dir}\Downloads\SAFE_JOB_CLUSTERS_ELSA-main\SAFE_JOB_CLUSTERS_ELSA-main\Clustering_Algorithm\perfiles_expuestos_clustering.py").read())

exec(open(f"{work_dir}\Downloads\SAFE_JOB_CLUSTERS_ELSA-main\SAFE_JOB_CLUSTERS_ELSA-main\Reporting\perfiles_expuestos_reporting.py").read())

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
