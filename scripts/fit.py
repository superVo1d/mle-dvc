import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import yaml
import os
import joblib

# обучение модели
def fit_model():
	with open('params.yaml', 'r') as fd:
		params = yaml.safe_load(fd)

	data = pd.read_csv('data/initial_data.csv')

	cat_features = data.select_dtypes(include='object')
	potential_binary_features = cat_features.nunique() == 2

	binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
	other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
	num_features = data.select_dtypes(['float'])

	preprocessor = ColumnTransformer(
		[
		('binary', OneHotEncoder(drop=params['one_hot_drop']), binary_cat_features.columns.tolist()),
		('cat', OneHotEncoder(drop=params['one_hot_drop']), other_cat_features.columns.tolist()),
		('num', StandardScaler(), num_features.columns.tolist())
		],
		remainder='drop',
		verbose_feature_names_out=False
	)

	model = LogisticRegression(
		C=params['lr_c'],
		penalty=params['lr_penalty']
	)

	pipeline = Pipeline(
		[
			('preprocessor', preprocessor),
			('model', model)
		]
	)
	pipeline.fit(data, data[params['target_col']]) 

	os.makedirs('models', exist_ok=True)

	joblib.dump(pipeline, 'models/fitted_model.pkl')


if __name__ == '__main__':
	fit_model()