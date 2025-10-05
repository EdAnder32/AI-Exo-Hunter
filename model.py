import os
import numpy
import pandas
import joblib
import optuna
import tsfresh
import sklearn
import seaborn
import lightgbm
import matplotlib
from google.colab import drive


def EDA(file):
  pandas.set_option('display.max_columns', 200)
  target_col = 'target'
  print("Shape from file: ", file.shape)
  # display(file.head())
  # print(file.dtypes)

def PipelineCreation(file, target_col):
  from sklearn.model_selection import train_test_split
  from sklearn.pipeline import Pipeline
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.compose import ColumnTransformer

  id_cols = [c for c in ['id', 'time', 'index'] if c in file.columns]
  x = file.drop(columns=[target_col] + id_cols, errors='ignore')
  y = file[target_col]

  numeric_features = x.select_dtypes(include=['number']).columns.tolist()
  categorical_features = x.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
  # print("Numeric: ", len(numeric_features))
  # print("Categorical: ", len(categorical_features))
  # print("X: ", x)
  # print("Y: ", y)

  numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),])

  #categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
  #                                    ('ohe', OneHotEncoder(handle_unknown='ignore')),])
  categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

  print(f"Num:{len(numeric_features)}")
  preprocessor = ColumnTransformer([
      ('num', numeric_transformer, numeric_features),
      ('cat', categorical_transformer, categorical_features),
  ], remainder='drop')
  return x, y, preprocessor

def Training(x_train_clean, y_train, x_val_clean, y_val):
  # from lightgbm import LGBMClassifier
  # scale_pos_weight = (y_train==0).sum() / max(1, (y_train==1).sum())
  # model = LGBMClassifier(
  #     objective='binary',
  #     n_estimators=10000,
  #     learning_rate=0.05,
  #     num_leaves=31,
  #     random_state=42,
  #     scale_pos_weight=scale_pos_weight
  # )
  # model.fit(
  #     x_train_tr, y_train,
  #     eval_set=[(x_val_tr, y_val)],
  #     eval_metric='auc',
  #     #my version does not support that method. I need to use callbacks
  #     #early_stopping_rounds=100,
  #     #callbacks=[early_stopping(100), log_evaluation(100)]
  # )
  # print("Best iteration:", model.best_iteration_)
  # print("Train AUC:", model.best_score_['training']['auc'])
  # print("Valid AUC:", model.best_score_['valid_0']['auc'])
  from lightgbm import LGBMClassifier

  # model = LGBMClassifier(
  #     objective='binary',
  #     n_estimators=2000,
  #     learning_rate=0.05,
  #     num_leaves=31,
  #     min_child_samples=1,
  #     random_state=42
  # )

  model = LGBMClassifier(
    objective='binary',
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=1,
    min_split_gain=0.0,  # permite splits mesmo sem ganho positivo aparente
    min_data_in_leaf=1,  # remove restrição mínima
    random_state=42
)

  model.fit(
      x_train_clean, y_train,
      eval_set=[(x_val_clean, y_val)],
      eval_metric='auc',
  )

  print("Best iteration:", model.best_iteration_)


  return model



from sklearn.model_selection import train_test_split
if os.path.exists('/content/drive') == 0:
  drive.mount('/content/drive')

labels = pandas.read_csv('/content/drive/MyDrive/AI_assets/labels.csv')
light_curves  = pandas.read_csv('/content/drive/MyDrive/AI_assets/light_curves.csv')
metadata = pandas.read_csv('/content/drive/MyDrive/AI_assets/metadata.csv')
#data = pandas.read_csv('/content/drive/MyDrive/exoplanets_normalized_data.csv')
data = pandas.read_csv('/content/drive/MyDrive/data.csv')

# im gonna change this under cause i nedd more classes. Using binary interpretation insted of 'CONFIRMED' will help a lot
#x, y, preprocessor = PipelineCreation(data, target_col='kepoi_name')

data['target'] = data['koi_disposition'].map(
    lambda v: 1 if v == "CONFIRMED" else 0
)

x, y, preprocessor = PipelineCreation(data, target_col='target')
print("First step done. 1 -> PIPELINE CREATION")
# print("X: ", x)
# print("Y: ", y)
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.20, stratify=y, random_state=42
 )
preprocessor.fit(x_train)
x_train_tr = preprocessor.transform(x_train)
x_val_tr = preprocessor.transform(x_val)
EDA(labels)

#debug purposes
#print("Second step done. 2 -> EDA DONE")
#print("X_train shape:", x_train_tr.shape)
#print("X_val shape:", x_val_tr.shape)
#print("y_train distribution:\n", y_train.value_counts())
#print("y_val distribution:\n", y_val.value_counts())
#print("Check for NaNs:", x_train_tr.isna().sum().sum(), "in train,", x_val_tr.isna().sum().sum(), "in val")
import pandas as pd

x_train_df = pd.DataFrame(x_train_tr)
x_val_df = pd.DataFrame(x_val_tr)

# Remover colunas sem variância
valid_cols = x_train_df.columns[x_train_df.var() > 0]
x_train_clean = x_train_df[valid_cols]
x_val_clean = x_val_df[valid_cols]

import numpy as np
import pandas as pd

# X_train é seu conjunto de treino (DataFrame)
variancias = x_train_df.var()
sem_variancia = (variancias == 0).sum()

# print(f"Número de colunas com variância zero: {sem_variancia}/{len(variancias)}")
# print("Exemplo de variâncias não nulas:")
# print(variancias[variancias > 0].head())

# print("Número de colunas após filtragem:", len(valid_cols))
# print("Shape final de treino:", x_train_clean.shape)
# print("Distribuição de classes:")
# print(y_train.value_counts(normalize=True))
# print("Exemplo de valores:")
# print(x_train_clean.head())


# print("Tipo de dados:", x_train_clean.dtypes.unique())
# print("Faixa de valores nas primeiras colunas:")
# print(x_train_clean.iloc[:, :5].describe())
# import numpy as np

# print("Min:", np.min(x_train_clean.values))
# print("Max:", np.max(x_train_clean.values))
# print("Mean:", np.mean(x_train_clean.values))
# print("Std:", np.std(x_train_clean.values))
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMClassifier

# === 1. Remover colunas quase constantes ===
var_sel = VarianceThreshold(threshold=1e-4)
x_train_filtered = var_sel.fit_transform(x_train_clean)
x_val_filtered = var_sel.transform(x_val_clean)

#print("Shape após VarianceThreshold:", x_train_filtered.shape)

# === 2. Padronizar ===
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_filtered)
x_val_scaled = scaler.transform(x_val_filtered)

# === 3. Treinar modelo ===
# model = LGBMClassifier(
#     objective='binary',
#     n_estimators=2000,
#     learning_rate=0.05,
#     num_leaves=31,
#     random_state=42
# )

# model.fit(
#     x_train_scaled, y_train,
#     eval_set=[(x_val_scaled, y_val)],
#     eval_metric='auc'
# )

# print("Best iteration:", model.best_iteration_)

#model = Training(x_train_clean, y_train, x_val_clean, y_val)
#print("Finished Training. 3!!")
# import numpy as np
# import pandas as pd

# # Converter se ainda estiver em numpy array
# x_train_df = pd.DataFrame(x_train_clean)

# # 1️⃣ Correlação média entre cada feature e o target
# corrs = []
# for col in x_train_df.columns:
#     try:
#         corrs.append(abs(np.corrcoef(x_train_df[col], y_train)[0,1]))
#     except:
#         corrs.append(0)
# mean_corr = np.nanmean(corrs)
# print("Correlação média com o target:", mean_corr)

# # 2️⃣ Contar quantas colunas têm correlação > 0.05
# useful = np.sum(np.array(corrs) > 0.05)
# print("Features com correlação > 0.05:", useful, "/", len(corrs))


# A litle degub made by GPT
# print("Best iteration:", model.best_iteration_)
# print("Train AUC:", model.best_score_['training']['auc'])
# print("Valid AUC:", model.best_score_['valid_0']['auc'])
import numpy as np

if hasattr(x_train_clean, "todense"):
    x_train_clean = np.array(x_train_clean.todense(), dtype=np.float32)
    x_val_clean = np.array(x_val_clean.todense(), dtype=np.float32)
else:
    x_train_clean = np.array(x_train_clean, dtype=np.float32)
    x_val_clean = np.array(x_val_clean, dtype=np.float32)
print("Treino - tipo:", type(x_train_clean))
print("Treino - shape:", x_train_clean.shape)
print("Treino - dtype:", x_train_clean.dtype)
print("Valores únicos de y:", np.unique(y_train, return_counts=True))

#model = Training(x_train_clean, y_train, x_val_clean, y_val)
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    objective='binary',
    learning_rate=0.01,
    n_estimators=2000,
    random_state=42
)
model.fit(x_train_clean, y_train)
train_score = model.score(x_train_clean, y_train)
print("Treino score:", train_score)

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_val_clean)
acc = accuracy_score(y_val, y_pred)
print(f"Acurácia: {acc:.4f}")



