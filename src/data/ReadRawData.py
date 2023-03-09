# import packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.compat import lzip
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from arch.unitroot import ADF, PhillipsPerron, DFGLS
from arch import arch_model
from pmdarima.arima import auto_arima
import warnings
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

# import data and create variables

gss = pd.read_csv("trends-gss.csv")
variables = ["year", "progunlaw", "age", "hapmar", "degree", "realinc"]
sub = gss[variables].copy()

# recode variables

sub['nprogunlaw'] = np.where(sub['progunlaw'] == 1, 1, 0)
sub['baplus'] = np.where(sub['degree'] >= 3, 1, 0)
sub['happinessmar'] = np.where(sub['hapmar'] == 1, 1, 0)
sub['income'] = sub['realinc']

# group variables by year

by_year = sub.groupby('year', as_index = False).agg('mean').replace({0.000000: np.nan})

# add years to dataframe

add_years_df = pd.DataFrame(data = pd.Series([1979, 1981, 1992, 1995] + list(np.arange(1997, 2009, 2))),
                            columns = ['year'])
                            
by_year = pd.concat([by_year, add_years_df], sort = False, ignore_index = True)

# sort by year and make index

by_year = by_year.sort_values('year')
by_year = by_year.set_index("year", drop = False)
by_year_ts = by_year.interpolate(method = 'linear')

# multiply percentages by 100

by_year_ts['progunlaw_pct'] = by_year_ts['progunlaw']*100
by_year_ts['ba_pct'] = by_year_ts['baplus']*100
by_year_ts['hapmar_pct'] = by_year_ts['happinessmar']*100