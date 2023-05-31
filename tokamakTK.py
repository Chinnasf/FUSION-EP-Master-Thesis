
import os
import numpy as np 
import pandas as pd
import scipy as sp
import statsmodels.api as sm
import cupy as cp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import pdb # For debugging || pdb.set_trace()


##################### CLASSES ####################################################


class MyCounter(Counter):
	"""
	Uses collections library tou output sorted Counter
	"""
	def __str__(self):
		return "\n".join('{}: {}'.format(k, v) for k, v in sorted(self.items()))
	
class HUEOrder:
	"""
	Used to extract specific keys from an existing dictionary to return another 
	dictionary.
	"""
	def __init__(self, hue_order):
		self.hue_order = hue_order

	def get_values(self, key, subkeys):
		result = {}
		subdict = self.hue_order.get(key, {})
		for subkey in subkeys:
			result[subkey] = subdict[subkey]
		return result


##################### FUNCTIONS ###################################################

def get_colors_per_category(DB5):
	TD_colors = plt.cm.get_cmap('flag', 20) #Paired, flag
	colors_ = sns.color_palette('viridis', 20)
	CSS_colors = [mcolors.to_hex(TD_colors(i)) for i in range(20)]

	# Needed for Improved Visualization in 2D Plots
	HUE_ORDER = {}

	HUE_ORDER["ICSCHEME"] = dict(zip(sorted(DB5["ICSCHEME"].unique()), colors_[:len(DB5["ICSCHEME"].unique())]))
	HUE_ORDER["ELMTYPE"] = dict(zip(sorted(DB5["ELMTYPE"].unique()), colors_[:len(DB5["ELMTYPE"].unique())]))
	HUE_ORDER["HYBRID"] = dict(zip(sorted(DB5["HYBRID"].unique()), colors_[:len(DB5["HYBRID"].unique())]))
	HUE_ORDER["AUXHEAT"] = dict(zip(sorted(DB5["AUXHEAT"].unique()), colors_[:len(DB5["AUXHEAT"].unique())]))
	HUE_ORDER["DIVMAT"] = dict(zip(sorted(DB5["DIVMAT"].unique()), colors_[:len(DB5["DIVMAT"].unique())]))
	HUE_ORDER["WALMAT"] = dict(zip(sorted(DB5["WALMAT"].unique()), colors_[:len(DB5["WALMAT"].unique())]))
	HUE_ORDER["EVAP"] = dict(zip(sorted(DB5["EVAP"].unique()), colors_[:len(DB5["EVAP"].unique())]))
	HUE_ORDER["ECHMODE"] = dict(zip(sorted(DB5["ECHMODE"].unique()), colors_[:len(DB5["ECHMODE"].unique())]))
	HUE_ORDER["PELLET"] = dict(zip(sorted(DB5["PELLET"].unique()), colors_[:len(DB5["PELLET"].unique())]))
	HUE_ORDER["CONFIG"] = dict(zip(sorted(DB5["CONFIG"].unique()), colors_[:len(DB5["CONFIG"].unique())]))
	HUE_ORDER["LIMMAT"] = dict(zip(sorted(DB5["LIMMAT"].unique()), colors_[:len(DB5["LIMMAT"].unique())]))
	HUE_ORDER["TOK"] = dict(zip(sorted(DB5["TOK"].unique()), colors_[:len(DB5["TOK"].unique())]))

	# Needed for 3D plots
	PX_ORDER = {}

	PX_ORDER["ICSCHEME"] = dict(zip(sorted(DB5["ICSCHEME"].unique()), CSS_colors[:len(DB5["ICSCHEME"].unique())]))
	PX_ORDER["ELMTYPE"] = dict(zip(sorted(DB5["ELMTYPE"].unique()), CSS_colors[:len(DB5["ELMTYPE"].unique())]))
	PX_ORDER["HYBRID"] = dict(zip(sorted(DB5["HYBRID"].unique()), CSS_colors[:len(DB5["HYBRID"].unique())]))
	PX_ORDER["AUXHEAT"] = dict(zip(sorted(DB5["AUXHEAT"].unique()), CSS_colors[:len(DB5["AUXHEAT"].unique())]))
	PX_ORDER["DIVMAT"] = dict(zip(sorted(DB5["DIVMAT"].unique()), CSS_colors[:len(DB5["DIVMAT"].unique())]))
	PX_ORDER["WALMAT"] = dict(zip(sorted(DB5["WALMAT"].unique()), CSS_colors[:len(DB5["WALMAT"].unique())]))
	PX_ORDER["EVAP"] = dict(zip(sorted(DB5["EVAP"].unique()), CSS_colors[:len(DB5["EVAP"].unique())]))
	PX_ORDER["ECHMODE"] = dict(zip(sorted(DB5["ECHMODE"].unique()), CSS_colors[:len(DB5["ECHMODE"].unique())]))
	PX_ORDER["PELLET"] = dict(zip(sorted(DB5["PELLET"].unique()), CSS_colors[:len(DB5["PELLET"].unique())]))
	PX_ORDER["CONFIG"] = dict(zip(sorted(DB5["CONFIG"].unique()), CSS_colors[:len(DB5["CONFIG"].unique())]))
	PX_ORDER["LIMMAT"] = dict(zip(sorted(DB5["LIMMAT"].unique()), CSS_colors[:len(DB5["LIMMAT"].unique())]))
	PX_ORDER["TOK"] = dict(zip(sorted(DB5["TOK"].unique()), CSS_colors[:len(DB5["TOK"].unique())]))

	return (HUE_ORDER, PX_ORDER)





def clean_categorical_data(db5):
	"""
	Clean categorical data in a pandas DataFrame by filling missing values and standardizing categories.

	Args:
	db5 (pandas.DataFrame): A pandas DataFrame with categorical columns to be cleaned.

	Returns:
	pandas.DataFrame: A cleaned pandas DataFrame with standardized categories and filled missing values.
	"""	
	DB5 = db5.copy()
	
	specific_categorical = ["PREMAG","HYBRID","CONFIG","ELMTYPE","ECHMODE",
				   "ICSCHEME","AUXHEAT","EVAP","PELLET", "DIVNAME","WALMAT","DIVMAT","LIMMAT"] 

	DB5[specific_categorical] = DB5[specific_categorical].fillna('UNKNOWN')
	DB5["DIVNAME"]   = DB5["DIVNAME"].str.replace("NONAME","UNKNOWN",regex=False)

	DB5["DIVMAT"] = DB5["DIVMAT"].str.replace("CC","C",regex=False)
	DB5["DIVMAT"] = DB5["DIVMAT"].str.replace("TI1","TI12",regex=False)
	DB5["DIVMAT"] = DB5["DIVMAT"].str.replace("TI2","TI12",regex=False)

	DB5["DIVNAME"] = DB5["DIVNAME"].str.replace("(DIV-I)|(DV-IPRE)|(DV-IPOST)",
												"DV-I",regex=True)
	DB5["DIVNAME"] = DB5["DIVNAME"].str.replace("(DIV-II)|(DV-IIc)|(DV-II-C)|(DV-IIb)|(DV-IIc)|(DV-IId)|(DV-IId)",
												"DV-II",regex=True)
	DB5["DIVNAME"] = DB5["DIVNAME"].str.replace("(MARK0)|(MARKI)|(MARKIIA)|(MARKGB)|(MARKGBSR)|"+
												"(MARKIA)|(MARKIAP)|(MARKSR)|(MARKA)|(MARKP)",
												"MARK",regex=True)

	DB5["ICSCHEME"]   = DB5["ICSCHEME"].str.replace("OFF","NONE",regex=False)
	DB5["HYBRID"]   = DB5["HYBRID"].str.replace("HYBRID","YES",regex=False)


	DB5["EVAP"] = DB5["EVAP"].str.replace("CARBH","C-H",regex=True)
	DB5["EVAP"] = DB5["EVAP"].str.replace("CARB","C",regex=True)
	DB5["EVAP"] = DB5["EVAP"].str.replace("BOROC","C-BO",regex=True)
	DB5["EVAP"] = DB5["EVAP"].str.replace("(BOROA)|(BOROB)|(BOROX)|(BOR)","BO",regex=True)

	DB5["PELLET"] = DB5["PELLET"].str.replace("GP_D","D",regex=False)
	DB5["PELLET"] = DB5["PELLET"].str.replace("GP_H","H",regex=False)
	
	return DB5


def encode_categorical_ohe(data):
	"""
	Encodes categorical data using OneHotEncoder.

	Parameters:
	- data: pandas.DataFrame, the data to be encoded

	Returns:
	- ohe_data: pandas.DataFrame, the encoded data
	"""
	encoder = OneHotEncoder() # create a OneHotEncoder object
	transformed = encoder.fit_transform(data) # fit and transform the data
	ohe_data = pd.DataFrame(transformed.toarray(),
						 columns=encoder.get_feature_names_out( 
							 data.columns
						 )
	) # create a DataFrame with the encoded data and column names

	return ohe_data




def impute_with_mean(series):
	return series.fillna(series.mean())


def clean_numerical_data(data, scaling=True):
	"""
	Takes a DataFrame `data` with numerical data and returns a cleaned DataFrame with missing 
	values filled using the mean value of that column for each year and month, followed by the mean 
	value for each `tokamak`. The function then fills any remaining missing values with zeros 
	and standardizes the numerical data. 

	Args:
		data (pandas.DataFrame): A DataFrame with numerical data.

	Returns:
		pandas.DataFrame: A cleaned DataFrame with numerical data only.
	"""
	df = data.copy()

	# Passing DATE to datetime
	df["DATE"] = df["DATE"].astype(str).replace(r"(\d{6})00", r"\g<1>01", regex=True)
	# Data type per feature
	num_features = df.select_dtypes(include=['int', 'float']).columns.tolist()    
	df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
	df['year'] = df['DATE'].dt.year
	df['month'] = df['DATE'].dt.month


	for tokamak in df.TOK.unique():
		for col in df.columns[df.isnull().sum() > 0]:
			# Fill NA with mean per month and year:
			tem = df.groupby(['year', 'month'])[[col]].mean().reset_index()
			tem.rename(columns={col: f'{col}_mean'}, inplace=True)
			# Merge and fill NA:
			df = pd.merge(df, tem, how='left', on=['year', 'month'])
			df.loc[df[col].isna(),col] = df[f'{col}_mean']
			df.drop(f'{col}_mean', axis=1, inplace=True)
	# Fill NA per tokamak
	df.fillna(df.groupby("TOK").mean(numeric_only=True), inplace=True)
	df = df[num_features]
	# Fill NA with general table / zeros
	#df = df.apply(lambda x: x.fillna(x.mean()))
	df.fillna(0, inplace=True)
	if scaling:
		df = StandardScaler().fit_transform(df)
		df = pd.DataFrame(df, columns=num_features)
	return df

def get_ECT_regression(data_, features=['IP', 'BT', 'NEL', 'PLTH', 'RGEO', 'KAREA', 'EPS', 'MEFF']):
	"""
	Computes OLS. for energy confinement time (ECT), given data_ (pd.DataFrame). 
	ASSUMING DATA IS ***NOT*** GIVEN IN LOG-SCALE

	Returns:
		regression: statsmodels.api for ODL
		(use: regression.summary() to see OLS output.)
	"""

	data = data_.copy()
	Y_ = np.log(data["TAUTH"])
	# Adding a column for the intercept
	X_ = sm.add_constant(np.log(data[features].apply(np.abs)))

	# MODEL RECEIVES LOG DATA WITH INTERCEPT

	model = sm.OLS(Y_,X_)
	regression = model.fit()
	
	return regression



def get_entropy_of_dataset(data, alpha = 0.5):
	"""
	Computes the entropy of a given dataset using a combination of numerical and categorical data.
	
	Args:
	data: A pandas.DataFrame with the dataset to compute the entropy on.
	alpha: A float used to compute the numerical similarity. Default: 0.5.
	
	Returns:
	A float representing the entropy of the given dataset.
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
	S_ij_cat = cp.asarray(S_ij_cat)


	# COMPUTING ENTROPY OF DATASET

	warnings.filterwarnings("ignore")

	# Log_2 used for shannon entropy interpretation as bits.
	# If log() is used, then shannon entropy is in nats.

	E_ij_cat = cp.log2(S_ij_cat, S_ij_cat) + cp.log2(1-S_ij_cat, 1-S_ij_cat)
	E_ij_cat = cp.nan_to_num( cp.asarray(E_ij_cat), nan=0.0)
	E_cat = - E_ij_cat.sum() 

	E_ij_num = cp.log2(S_ij_num, S_ij_num) + cp.log2(1-S_ij_num, 1-S_ij_num)
	E_ij_num = cp.nan_to_num( cp.asarray(E_ij_num), nan=0.0)
	E_num = - E_ij_num.sum()

	warnings.simplefilter("default")


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



def scatter_data_comparison(data, params):
	import warnings
	warnings.filterwarnings("ignore")
	warnings.simplefilter("ignore")
	"""
	Plots a group comparison using scatter plots with multiple subplots.

	Args:
		data (list of pandas.DataFrame): containing two pandas.DataFrames to be compared. 
		params (dict) containing:
			cat_params (list): List of categorical parameters for comparison.
			xy_params (list): List of the columns names for x and y: scatter plot.
			x_label (str): Label for the x-axis.
			y_label (str): Label for the y-axis.
			plot_size(tuple): containing (int, int) for figsize.
			fig_format (str, optional): Format of the output figure file.
			x_minmax (tuple, optional): Tuple of minimum and maximum values for the x-axis. Defaults to None
			y_minmax (tuple, optional): Tuple of minimum and maximum values for the y-axis. Defaults to None            
			save_fig (bool, optional): Whether to save the figure. Defaults to False.

	Returns:
		sns Plot
	"""
	
	default_params = {
		"x__minmax": (None, None),
		"y__minmax": (None, None),
		"fig_name" : "sns_plot_comparing_data",
		"fig_format": "pdf",
		"save_fig": False
	}
	
	# Update default params with provided params
	default_params.update(params)
	
	# Extract params from dictionary
	HUE_ORDER  = params["HUE_ORDER"]
	cat_params = params["cat_params"]
	xy__params = params["xy__params"]
	x___label  = params["x___label"]
	y___label  = params["y___label"]
	plot_size  = params["plot_size"]
	legend_pos = params["legend_pos"]
	legend_in_decreasing = params["legend_in_decreasing"]

	if legend_in_decreasing:
		locs = [legend_pos]*3 + ["upper right"]*3
	else:
		locs = ["upper right"]*3 + [legend_pos]*3

	# Access updated default params
	x__minmax  = default_params["x__minmax"]
	y__minmax  = default_params["y__minmax"]
	fig_format = default_params["fig_format"]
	save_fig   = default_params["save_fig"]  

	fig_name__ = f"scatter_plot__{xy__params}__{cat_params}.pdf"  
	
	data1, data2 = data

	if len(cat_params) > 1:
		fig, axs = plt.subplots(2, len(cat_params), figsize=plot_size)
		fig.subplots_adjust(hspace=0.4)

		data_ = data1.copy()
		for i in range(len(cat_params)):
			sns.scatterplot(data=data_, x=xy__params[0], y=xy__params[1], 
							hue=cat_params[i], hue_order=HUE_ORDER[cat_params[i]], 
							ax=axs[0,i], legend=legend_in_decreasing
							)
		data_ = data2.copy()
		for j in range(len(cat_params)):
			sns.scatterplot(data=data_, x=xy__params[0], y=xy__params[1], 
							hue=cat_params[j], hue_order=HUE_ORDER[cat_params[j]], 
							ax=axs[1,j], legend=bool(1 - legend_in_decreasing)
							)
		
		for i, ax in enumerate(axs.flatten()):
			ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
			ax.tick_params(axis='x', labelsize=13)
			ax.tick_params(axis='y', labelsize=13)
			ax.set_ylim(y__minmax[0],y__minmax[1])
			ax.set_xlim(x__minmax[0],x__minmax[1])  
			ax.set_xlabel("")
			ax.set_ylabel("")
			ax.legend(fontsize=11, title=(cat_params*2)[i], loc=locs[i])
		# Add text label for x-labels at the bottom of the plot
		fig.text(0.5, 0.03, x___label, 
				 ha='center', va='bottom', fontsize=15
				)
		# Add text label for y-labels at the left of the plot
		fig.text(0.07, 0.5, y___label, 
				 ha='left', va='center', rotation='vertical', fontsize=15
				)
		axs[0, 1].set_title("Decreasing Dataset\n", fontsize=18)
		axs[1, 1].set_title("Unaffected Dataset\n", fontsize=18);

		if save_fig:
			plt.savefig(fig_path+fig_name__+"."+fig_format, format=fig_format, dpi=800, bbox_inches='tight');
			
		return plt.show()

	else:
		fig, axs = plt.subplots(1, 2, figsize=plot_size)
		data_ = data1.copy()
		sns.scatterplot(data=data_, x=xy__params[0], y=xy__params[1], s=90, hue=cat_params[0], hue_order=HUE_ORDER[cat_params[0]], ax=axs[0], legend=False)
		data_ = data2.copy()
		sns.scatterplot(data=data_, x=xy__params[0], y=xy__params[1], s=90, hue=cat_params[0], hue_order=HUE_ORDER[cat_params[0]], ax=axs[1])
		axs[0].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
		axs[1].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
		axs[0].tick_params(axis='x', labelsize=13); axs[1].tick_params(axis='x', labelsize=13)
		axs[0].tick_params(axis='y', labelsize=13); axs[1].tick_params(axis='y', labelsize=13)
		axs[0].set_ylim(y__minmax[0],y__minmax[1]); axs[1].set_ylim(y__minmax[0],y__minmax[1])
		axs[0].set_xlim(x__minmax[0],x__minmax[1]); axs[1].set_xlim(x__minmax[0],x__minmax[1])
		axs[0].set_xlabel(""); axs[1].set_xlabel("")
		axs[0].set_ylabel(""); axs[1].set_ylabel("")
		axs[0].legend(fontsize=11, title=(cat_params*2)[0], loc='upper right')
		axs[1].legend(fontsize=11, title=(cat_params*2)[0], loc=legend_pos)
		# Add text label for x-labels at the bottom of the plot
		fig.text(0.5, -0.05, x___label, 
				 ha='center', va='bottom', fontsize=15
				)
		# Add text label for y-labels at the left of the plot
		fig.text(0.07, 0.5, y___label, 
				 ha='left', va='center', rotation='vertical', fontsize=15
				)
		axs[0].set_title("Decreasing Dataset\n", fontsize=20)
		axs[1].set_title("Unaffected Dataset\n", fontsize=20);

		if save_fig:
			plt.savefig(fig_path+fig_name__+"."+fig_format, format=fig_format, dpi=800, bbox_inches='tight');

		return plt.show()

# A way to obtain info about multicollinearity

def get_multicollinearity_info(X, features):
	"""
	Input: X (np.array) (#observations x #regressors)
	Return DataFrame with infor
	"""
	# rowvar=False indicates that each column represents a variable
	cov_mat = np.cov(X.T)
	eigenvals = np.linalg.eigvals(cov_mat)
	_, sing_vals, _ = np.linalg.svd(X)

	# Computing the condition index
	condition_index = [ (sing_vals.max()/n) for n in sing_vals]

	# Computing the VIF | variance inflation factors
	from statsmodels.stats.outliers_influence import variance_inflation_factor
	VIF = [variance_inflation_factor(X, col_indx) for col_indx in range(X.shape[-1])]

	# Variance Decomposition | values above 0.5 or 0.6 are often considered relatively high
	variance_decomposition = eigenvals / eigenvals.sum()

	df = pd.DataFrame([eigenvals, condition_index]).T
	df.index = features
	df.columns = ["eigenvalues", "CI"]
	df["VIF"] = VIF
	df["VD"] = variance_decomposition

	return df

# Another way to obtain info about multicollinearity

def prepare_data(data, 
				 columns=['IP', 'BT', 'NEL', 'PLTH', 'RGEO', 'KAREA', 'EPS', 'MEFF'], 
				 add_intercept=True):
	"""
	Prepare data such tha the ECT regression can be assessed
	"""
	data = data[columns]
	data = data.apply(np.abs).apply(np.log)
	if add_intercept:
		data = sm.add_constant(data)
	return data

def scale_data(df, centered=False, add_intercept=True):
	"""
	df (pd.DataFrame) containing the data numerical to be prepared for ECT analysis
	"""
	if centered:
		df = df - df.describe().loc["mean"]
	X  = (df / df.apply(lambda x: np.linalg.norm(x)))
	if add_intercept:
		X = sm.add_constant(X)
	return X

def get_condition_number(X, scale=True, add_intercept=False):
	if scale:
		X = scale_data(X, centered=False, add_intercept=add_intercept)
	# Get Singular Values
	_,S,_ = np.linalg.svd(X)
	# Return condition index
	return max(S)/min(S)

def get_condition_index(X, scale=True):
	if scale:
		X = scale_data(X)
	# Get Singular Values
	_,S,_ = np.linalg.svd( X )
	# Return condition indexes
	return max(S)/S

def get_pi_matrix(X, features, scale=True, intercept=False):
	if scale:
		X = scale_data(X, centered=False, add_intercept=intercept)
	cols = features
	if intercept:
		cols = ["intrcp"] + features
	m = len(cols)
		
	U,D,VT = np.linalg.svd( X ) # Transpose of V ----> ask Joe.
	V_sq   = np.square(VT.T); mu_sq = np.square(D)

	φ_kj   = np.zeros((m,m))
	φ_k    = np.zeros(m)
	pi_jk  = np.zeros((m,m))

	for k in range(m):
		φ_kj[k,:] = V_sq[k,:] / mu_sq
		φ_k[k] = φ_kj[k,:].sum()
		pi_jk[:,k]  = φ_kj[k,:] / φ_k[k]

	# Pi-Matrix (Variance Decomposition)
	Π = pd.DataFrame(pi_jk, 
					 index=[f"μ{i}" for i in range(m)], 
					 columns=cols) 
	print(f"Condition Number: {max(D)/min(D)}")
	return Π