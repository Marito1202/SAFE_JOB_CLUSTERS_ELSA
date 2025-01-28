from perfiles_no_expuestos_feat_eng import *
from perfiles_no_expuestos_clustering import *

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


