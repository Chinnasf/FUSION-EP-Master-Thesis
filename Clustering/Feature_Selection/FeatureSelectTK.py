import numpy as np 
import pandas as pd
import scipy as sp
import pdb # For debugging || pdb.set_trace()



def get_entropy_of_categorical_dataset(data):
	"""
	Only categorical data is expected
	"""
	N, M = data.shape
	S_ij = np.zeros((N,N))

	for k in range(M):
		F_k  = data[ data.columns[k] ].values
		d_ij = np.frompyfunc( lambda x,y: x==y, 2, 1).reduce( np.array( np.meshgrid(F_k,F_k) ) ).astype(int)
		S_ij = S_ij + d_ij

	S_ij = S_ij/M
	E_ij = sp.special.xlogy(S_ij, S_ij) + sp.special.xlogy(1-S_ij, 1-S_ij) 
	E = - E_ij.sum()
	return E


def get_entropy_of_dataset(data, alpha = 0.5):
	np.seterr(divide='ignore', invalid='ignore')
	"""
	data: DataFrame, expected with scaled data.
	alpha: parameter used to compute the similarity. Default: 0.5.
	"""
	N, M = data.shape
	D = [0]*M
	D_ij = np.zeros((N,N))

	for k in range(M):
		F_k  = data[ data.columns[k] ].values
		ΔF_k = (max(F_k) - min(F_k))
		# D[k] = np.frompyfunc( lambda x,y: x-y, 2, 1).reduce( np.array( np.meshgrid(F_k,F_k) ) )
		D[k] = np.subtract.outer(F_k,F_k)
		D_k  = np.square( D[k]/ΔF_k ) 
		D_ij = D_ij + D_k

	D_ij = np.sqrt(D_ij) # euclidean distance
	S_ij = np.exp( - alpha * D_ij ) # similarity
	E_ij = sp.special.xlogy(S_ij, S_ij) + sp.special.xlogy(1-S_ij, 1-S_ij)
	E = - E_ij.sum()    
	return E



def get_ranked_features(data, alpha=0.5, ddtype="numerical"):
	"""
	Returns pandas's Series containing the sorted entropy
	associated to the removal of the feature.  

	pd.Series is sorted as: 
	from the most important to the least important features. 
	"""	
	N,M = data.shape
	E = np.zeros(M)
	data_cols = pd.Series(data.columns)

	if ddtype == "numerical":
		for i in range(M):
			eliminate_col = [data.columns[i]]
			df = data[ data_cols[~data_cols.isin(eliminate_col)].values ]
			E[i] = get_entropy_of_dataset(df)	
		ranked_entropy = pd.Series(E).sort_values(ascending=False)
		return ranked_entropy
	else:
		for i in range(M):
			eliminate_col = [data.columns[i]]
			df = data[ data_cols[~data_cols.isin(eliminate_col)].values ]
			E[i] = get_entropy_of_categorical_dataset(df)	
		ranked_entropy = pd.Series(E).sort_values(ascending=False)
		return ranked_entropy