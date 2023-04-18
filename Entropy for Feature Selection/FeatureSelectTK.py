import numpy as np 
import pandas as pd
import scipy as sp
import statsmodels.api as sm
import cupy as cp
import pdb # For debugging || pdb.set_trace()


path = "../data/"
DB2 = pd.read_csv(path+"DB2P8.csv")

def get_regression(_R, withDB2=False):
	"""
	Computes OLS. It ineeds sto be specified if data already
	contains DB2 shots or not. 

	ASSUMING DATA IS ***NOT*** GIVEN IN LOG-SCALE

	Returns:
		data (DataFrame): the data used to compute OLS
		regression: statsmodels.api for ODL;
			use: regression.summary() to see OLS output.
		n_,p_ (int, int): number of observations and columns in data.
	"""

	coeffs = ['IP', 'BT', 'NEL', 'PLTH', 'RGEO', 'KAREA', 'EPS', 'MEFF']

	if withDB2:
		data = _R.copy()
	else:     
		data = pd.concat([DB2, _R],
						 axis=0, 
						 ignore_index=True
						)
	Y_ = data[["TAUTH"]].apply(np.log).to_numpy()
	# Adding a column for the intercept
	_df = data[coeffs].apply(np.abs).apply(np.log)
	_df.insert(
		loc = 0, 
		column = "intercept", 
		value = np.ones(len(_df))
	)
	X_ = _df.to_numpy()
	n_, p_ = X_.shape
	model = sm.OLS(Y_,X_)
	regression = model.fit()
	return data, regression, (n_,p_)



def get_entropy_of_dataset(data, alpha = 0.5):
	"""
	data: DataFrame, expected with scaled data.
	alpha: parameter used to compute the numerical similarity. Default: 0.5.

	Returns: Associated entropy to dataset
	"""
	N, M = data.shape
	
	num_features = data.select_dtypes(include=['int', 'float']).columns.tolist()
	cat_features = data.select_dtypes(include=['object']).columns.tolist()

	cat_data = data[cat_features]; cat_E = np.zeros( len(cat_features) )
	num_data = data[num_features]; num_E = np.zeros( len(num_features) )

	# COMPUTING FOR NUMERICAL DATA

	# Precompute max and min values for each numerical feature
	num_data_gpu = cp.asarray(num_data)
	num_max = num_data_gpu[:, range(len(num_features))].max(axis=0)
	num_min = num_data_gpu[:, range(len(num_features))].min(axis=0)
	D_ij = cp.zeros((N, N))
	for k in range(len(num_features)):
		F_k = num_data_gpu[:, k]
		Fk_norm = cp.divide(F_k, num_max[k] - num_min[k])
		Fk_norm_col = Fk_norm[:, cp.newaxis]
		D_ij += cp.square( Fk_norm_col - Fk_norm )
	D_ij = cp.sqrt(D_ij)  # euclidean distance
	S_ij_num = cp.exp(-alpha * D_ij)  # similarity


	# COMPUTING FOR CATEGORICAL DATA
	S_ij_cat = np.zeros((N, N))
	for k in range(len(cat_features)):
		F_k = cat_data[cat_features[k]].values
		d_ij = np.array(F_k)[:, None] == np.array(F_k)[None, :]
		S_ij_cat += d_ij.astype(int)

	S_ij_cat = S_ij_cat / M


	# COMPUTING ENTROPY OF DATASET

	E_ij_cat = sp.special.xlogy(S_ij_cat, S_ij_cat) + sp.special.xlogy(1-S_ij_cat, 1-S_ij_cat)
	E_ij_cat = cp.nan_to_num( cp.asarray(E_ij_cat), nan=0.0)
	E_cat = - E_ij_cat.sum() 

	E_ij_num = sp.special.xlogy(S_ij_num, S_ij_num) + sp.special.xlogy(1-S_ij_num, 1-S_ij_num)
	E_ij_num = cp.nan_to_num( cp.asarray(E_ij_num), nan=0.0)
	E_num = - E_ij_num.sum()

	E = E_cat + E_num

	return E


def get_ranked_features(data, alpha=0.5):
	"""
	Returns pandas's Series containing the sorted entropy
	associated to the removal of the feature.  

	pd.Series is sorted as: 
	from the most important to the least important features.

	Returns Ranked Features 
	"""	

	E = np.zeros(data.shape[-1])
	for i in range(data.shape[-1]):
		eliminate_col = [data.columns[i]]
		df = data.drop(eliminate_col, axis="columns")
		E[i] = get_entropy_of_dataset(df, alpha)	
	ranked_entropy = pd.Series(E).sort_values(ascending=False)
	return ranked_entropy