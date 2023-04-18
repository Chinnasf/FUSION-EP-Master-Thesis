import numpy as np 
import pandas as pd
import scipy as sp
import pdb # For debugging || pdb.set_trace()



def get_entropy_of_dataset(data, alpha = 0.5):
	"""
	data: DataFrame, expected with scaled data.
	alpha: parameter used to compute the numerical similarity. Default: 0.5.
	"""
	N, M = data.shape
	

	num_features = data.select_dtypes(include=['int', 'float']).columns.tolist()
	cat_features = data.select_dtypes(include=['object']).columns.tolist()

	cat_data = data[cat_features]; cat_E = np.zeros( len(cat_features) )
	num_data = data[num_features]; num_E = np.zeros( len(num_features) )

	# COMPUTING FOR NUMERICAL DATA

	D = [0]*M
	D_ij = np.zeros((N,N))
	for k in range( len(num_features) ):
		F_k  = num_data[ num_features[k] ].values
		ΔF_k = (max(F_k) - min(F_k))
		# D[k] = np.frompyfunc( lambda x,y: x-y, 2, 1).reduce( np.array( np.meshgrid(F_k,F_k) ) )
		D[k] = np.subtract.outer(F_k,F_k)
		D_k  = np.square( D[k]/ΔF_k ) 
		D_ij = D_ij + D_k

	D_ij = np.sqrt(D_ij) # euclidean distance
	S_ij_num = np.exp( - alpha * D_ij ) # similarity

	# COMPUTING FOR CATEGORICAL DATA

	S_ij_cat = np.zeros((N,N))
	for k in range(len(cat_features)):
		F_k  = cat_data[ cat_features[k] ].values
		d_ij = np.frompyfunc( lambda x,y: x==y, 2, 1).reduce( np.array( np.meshgrid(F_k,F_k) ) ).astype(int)
		S_ij_cat = S_ij_cat + d_ij
	S_ij_cat = S_ij_cat/M


	S_ij = S_ij_num + S_ij_num

	E_ij = sp.special.xlogy(S_ij, S_ij) + sp.special.xlogy(1-S_ij, 1-S_ij)
	E = - E_ij.sum()    
	return E



def get_ranked_features(data, alpha=0.5):
	"""
	Returns pandas's Series containing the sorted entropy
	associated to the removal of the feature.  

	pd.Series is sorted as: 
	from the most important to the least important features. 
	"""	

	E = np.zeros(data.shape[-1])
	for i in range(data.shape[-1]):
		eliminate_col = [data.columns[i]]
		df = data.drop(eliminate_col, axis="columns")
		E[i] = get_entropy_of_dataset(df, alpha)	
	ranked_entropy = pd.Series(E).sort_values(ascending=False)
	return ranked_entropy