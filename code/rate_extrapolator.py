#/* -------------------------------------------------------------------------------------
#   Egorov Group
#   University of Virginia
#   Mohan Shankar
#
#   rate_extrapolator.py
# "This code extrapolates rate constants as wall thickness tends to zero"
#---------------------------------------------------------------------------------------- */

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Trange = np.array([100, 200, 400, 600, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000])

gaps = np.array([2, 4, 6, 8, 10])

df = pd.read_excel("/Users/mohan/Desktop/Research/pib_review/data/extrap_rate_plot.xlsx").dropna()


extrap_rates = np.zeros((len(Trange), len(gaps))) # matrix where constant T across row, constant gap size down a column
extrap_errors = np.zeros((len(Trange), len(gaps))) # matrix where constant T across row, constant gap size down a column
ratio_rates_error = np.zeros((len(Trange), len(gaps))) # matrix where constant T across row, constant gap size down a column

for i, T in enumerate(Trange):
    for j, gap_size in enumerate(gaps):
        filtered_df = df[(df['Temperature [K]'] == T) & (df['Gap Size'] == gap_size)]
        result = filtered_df.head(5) # Take the first five rows that meet the criteria
        
        wall_thickness = result.iloc[:, 1]
        rates = result.iloc[:, -1]
        
        order = 2

        fit, cov_matrix = np.polyfit(wall_thickness, rates, order, cov = True) # fit 3rd order polynomial to data and return covariance matrix

        errors = np.sqrt(np.diag(cov_matrix))

        extrap_rates[i][j] = fit[-1] # grab y-int

        extrap_errors[i][j] = errors[-1] # grab y-int error

        ratio_rates_error[i][j] = np.abs( fit[-1] / errors[-1] )


gaps = np.array([0, 2, 4, 6, 8, 10])

final_extrap_rates = np.row_stack((gaps, np.column_stack((Trange, extrap_rates)))) # add leftmost column vector denoting temp and topmost row vector denoting gap size

final_extrap_errors = np.row_stack((gaps, np.column_stack((Trange, extrap_errors)))) # add leftmost column vector denoting temp and topmost row vector denoting gap size

final_ratio_errors = np.row_stack((gaps, np.column_stack((Trange, ratio_rates_error)))) # add leftmost column vector denoting temp and topmost row vector denoting gap size


# np.savetxt('/Users/mohan/Desktop/extrap_rates.csv', final_extrap_rates, delimiter=',') # CHANGE PATH
# np.savetxt('/Users/mohan/Desktop/extrap_error.csv', final_extrap_errors, delimiter=',') # CHANGE PATH
# np.savetxt('/Users/mohan/Desktop/extrap_ratio.csv', final_ratio_errors, delimiter=',') # CHANGE PATH

