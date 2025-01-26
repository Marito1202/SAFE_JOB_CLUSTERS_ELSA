from perfiles_expuestos_feat_eng import *
from perfiles_expuestos_clustering import *

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


