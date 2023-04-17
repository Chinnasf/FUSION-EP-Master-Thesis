import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import matplotlib.patches as mpatches
import seaborn as sns

from collections import Counter


plt.rc('font',family = 'serif')

colors_ = sns.color_palette('viridis', 20)

coeffs = ['IP', 'BT', 'NEL', 'PLTH', 'RGEO', 'KAREA', 'EPS', 'MEFF']

path = "../data/"
fig_path = "../../../LATEX/Latex Images/"



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



def process_DB5_cat_data(db5):
    """
    Customized function to process categorical data for DB5.
    """
    DB5 = db5.copy()
    TOK_characteristics = ["DIVNAME","WALMAT","DIVMAT","LIMMAT"]
    categorical = ["PREMAG","HYBRID","CONFIG","ELMTYPE","ECHMODE",
                   "ICSCHEME","AUXHEAT","EVAP","PELLET"] + TOK_characteristics 

    DB5[categorical] = DB5[categorical].fillna('UNKNOWN')
    DB5["DIVNAME"]   = DB5["DIVNAME"].str.replace("NONAME","UNKNOWN",regex=False)


    DB5["PELLET"] = DB5["PELLET"].str.replace("GP_D","D",regex=False)
    DB5["PELLET"] = DB5["PELLET"].str.replace("GP_H","H",regex=False)

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

    DB5["EVAP"] = DB5["EVAP"].str.replace("CARBH","C-H",regex=True)
    DB5["EVAP"] = DB5["EVAP"].str.replace("CARB","C",regex=True)
    DB5["EVAP"] = DB5["EVAP"].str.replace("BOROC","C-BO",regex=True)
    DB5["EVAP"] = DB5["EVAP"].str.replace("(BOROA)|(BOROB)|(BOROX)|(BOR)","BO",regex=True)
    #DB5["EVAP"] = DB5["EVAP"].str.replace("DECABOA","C",regex=True)
    
    return DB5



def scatter_data_comparison(data, params):
	"""
	Plots a group comparison using scatter plots with multiple subplots.

	Args:
		data (list): containing two DataFrames to be compared. 
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

	# Access updated default params
	x__minmax  = default_params["x__minmax"]
	y__minmax  = default_params["y__minmax"]
	fig_format = default_params["fig_format"]
	save_fig   = default_params["save_fig"]  

	fig_name__ = f"scatter_plot__{xy__params}__{cat_params}.pdf"  
	
	data1, data2 = data
	
	fig, axs = plt.subplots(2, len(cat_params), figsize=plot_size)
	fig.subplots_adjust(hspace=0.4)

	data_ = data1.copy()
	for i in range(len(cat_params)):
		sns.scatterplot(data=data_, x=xy__params[0], y=xy__params[1], 
						hue=cat_params[i], hue_order=HUE_ORDER[cat_params[i]], ax=axs[0,i])
	data_ = data2.copy()
	for j in range(len(cat_params)):
		sns.scatterplot(data=data_, x=xy__params[0], y=xy__params[1], 
						hue=cat_params[j], hue_order=HUE_ORDER[cat_params[j]], ax=axs[1,j])

	for i, ax in enumerate(axs.flatten()):
		ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
		ax.tick_params(axis='x', labelsize=13)
		ax.tick_params(axis='y', labelsize=13)
		ax.set_ylim(y__minmax[0],y__minmax[1])
		ax.set_xlim(x__minmax[0],x__minmax[1])  
		ax.set_xlabel("")
		ax.set_ylabel("")
		ax.legend(fontsize=11, title=(cat_params*2)[i])

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

