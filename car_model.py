from sklearn.ensemble import RandomForestRegressor


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def cross_val_mse_r2(X_train, y_train, func):

	mse = -sum(cross_val_score(func, X_train, y_train, scoring='neg_mean_squared_error'))/3
	r2 = sum(cross_val_score(func, X_train, y_train, scoring='r2'))/3

	func_name = str(func.__class__.__name__)

	print("{0:27} Train CV | MSE: {1:5.4} | R2: {2:5.4}".format(func_name, mse, r2))
	return mse, r2



def rf_grid(X_train, y_train, X_test, y_test):
	random_forest_grid = {'max_depth': [3, None],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [2, 4],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False],
                      'n_estimators': [10, 20, 40, 80],
                      'random_state': [1]}

	rf_gridsearch = GridSearchCV(RandomForestRegressor(),
	                             random_forest_grid,
	                             n_jobs=-1,
	                             verbose=True,
	                             scoring='neg_mean_squared_error')
	rf_gridsearch.fit(X_train, y_train)

	print( "best parameters:", rf_gridsearch.best_params_ )

	best_rf_model = rf_gridsearch.best_estimator_
	preds = best_rf_model.predict(X_test)

	print("MSE of Best RF Model:", mean_squared_error(y_test, preds))



def score_avg(car_means, true_means, names):
	
	preds = []
	for name in names:
		preds.append(car_means.loc[car_means.index == name, :].mpg)

	return np.sum((preds - true_means)**2)/len(names)



def create_model(params):

	cars = pd.read_csv('cars.csv')
	params = ['cylinders',
			'horsepower',
			'weight',
			'origin',
			'displacement',
			'acceleration',
			'model']

	# ## Mean of car_names
	# name_mpg = cars[['car_name', 'mpg']]
	# car_means = name_mpg.groupby('car_name').mean()
	# true_mpgs = cars.mpg.values
	# names = cars.car_name.values
	# print(score_avg(car_means, true_mpgs, names), '\n')
	
	## Setting target
	y = cars.mpg.values

	## Dropping bad cols for data
	cars.drop(['mpg'], axis=1, inplace=True)

	## Only taking the features entered
	X = cars[params].values

	## Splitting data
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)


	######### RANDOM FORREST #########
	rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=1)
	rf.fit(X, y)
	return rf
	pickle.dump(rf, open('randomf.p', 'wb'))

	#rf_mse = cross_val_mse_r2(X_train, y_train, rf)[0]
	

	## Grid Search on RF for best params
	#rf_grid(X_train, y_train, X_test, y_test)

