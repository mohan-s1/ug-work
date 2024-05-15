#/* -------------------------------------------------------------------------------------
#   Egorov Group
#   University of Virginia
#   Mohan Shankar
#
#   rate_calculator.py
# "This code finds plateaus for f-s correlation fncs in order to calculate rate constants"
#---------------------------------------------------------------------------------------- */

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


Trange = np.array([100, 200, 400, 600, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000]) 

hbar = 1.054571817e-34 # hbar in SI Units [J*s]

m = 1.67262192e-27 # mass of proton in kg

L = 1e-10 # 1 angstrom in meters

kb = 1.380649e-23 # Boltzmann constant in SI units

#-------------------------------------------------------------------------------- */
# Calculate Partition Functions
n = np.arange(1, 21, 1)
Q = np.empty(shape = len(Trange)) # initialize empty array of dimensions equal to Trange
E = (np.pi ** 2 * hbar ** 2) / (2 * m * L ** 2)

for i, temp in enumerate(Trange):
    T = temp
    beta = 1 / (kb * T)
    zz = np.exp(-beta * E * n** 2)
    final = np.sum(zz)
    Q[i] = final**2

print("Partition Functions: \n", Q)

#-------------------------------------------------------------------------------- */
# FUNCTION DEFINITIONS

T = 1500 # change this to grab flux-side correlation function at different temperature

chunk_size = 50 # LEAVE THIS ALONE 

gap_size = 6 # change this to grab flux-side correlation function at different gap size

wall_size = 2 # change this to grab flux-side correlation function at different wall thickness 

temp_index = np.where(Trange == T) # get index of Trange array where the temperature match 
#-------------------------------------------------------------------------------- */
# READ IN DATA
df = pd.read_csv("/Users/mohan/Desktop/Research/pib_review/data/"+str(wall_size)+"pt_wall/"+str(gap_size)+"pt_gap/side-flux"+str(Trange[temp_index][0])+"K.txt") # Trange[temp_index] returns a one-element array, so we grab the first (only) element in there

# df = pd.read_csv("/Users/mohan/Desktop/Refile/2pt_wall/6pt_gap/side-flux"+str(Trange[temp_index][0])+".txt", sep = ' ', header = None).T # if Alec style; my 2 pt wall, 6pt gap are Alec style

t = df.iloc[:, 0].to_numpy() # first column is time array
c = df.iloc[:, 1].to_numpy() # second column is correlation function array

#-------------------------------------------------------------------------------- */
# DEFINE FUNCTIONS
def zero_slope(data, chunksize = chunk_size, max_slope = .04): # function to find where the numerical derivative is ~ 0 based on chunk size (looking at points forwards and backwards) and max_slope
    midindex = chunksize / 2
    is_plateau = np.zeros((data.shape[0]))
    for index in range(int(midindex), len(data) - int(midindex)):
        front = index - midindex
        back = index + midindex
        chunk = data[int(front) : int(back)]
        dy_dx = abs(chunk[0] - chunk[-1])/chunksize
        if (0 <= dy_dx < max_slope):
            is_plateau[index] = 1.0
    return is_plateau

def ranges(nums): # return tuples of consecutive integers (i.e. find "widths" of plateaus in step function)
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))
#-------------------------------------------------------------------------------- */
# ANNOYING STUFF
small = np.nanmin(c/t) # find min of c/t wile dropping NaN values, used as a scaling factor for step function 

step_function = zero_slope(data = c, chunksize = chunk_size, max_slope = small*10) # array of all 0's and 1's based on where the plateaus are

indices = [idx for idx,val in enumerate(step_function) if val == 1] # grab indices where value is 1

new = ranges(indices) # create tuples of consecutive numbers (i.e. find indices for each plateau)

lengths = np.zeros(len(new)) # make empty array with same dimensions as "new" array 

for i, val in enumerate(new): # print out tuples
    print("Tuples", val)
    lengths[i] = val[1] - val[0] 

#-------------------------------------------------------------------------------- */
# NUMERICALLY FIND LARGEST PLATEAU AND SET INDEX (`use_this`) TO THE MIDDLE INDEX OF LONGEST PLATEAU

max_value = max(lengths) # find longest plateau 

max_index = list(lengths).index(max_value) # find index of longest plateau 

largest_tuple = new[max_index] # grab largest tuple 

use_this = (largest_tuple[0] + (largest_tuple[1] - largest_tuple[0])/2) # automatically find widest plateau numerically; index for y value of csf we want to use for calculating the rate 

# use_this = (40 + (59 - 40)/2) # manually set index for widest plateau; index for y value of csf we want to use for calculating the rate 

print("Number of index used", use_this)

csf = c[int(use_this)]

print("Value of csf", csf)

print("Time", t[int(use_this)])

rate = (csf/Q[temp_index]) * 6.57966e15 # rate; multiply hartree value by conversion factor to get Hz
print("Rate [Hz]", rate)
short_rate = np.around(rate, decimals = -6)
print("Truncated Rate [Hz]", short_rate)
#-------------------------------------------------------------------------------- */
# PLOT
plt.plot(t, c, color = 'RoyalBlue')
plt.plot(t, step_function*np.nanmax(c), color = 'darkorange') # scale the step function by the maximum of the correlation function so that plotting makes sense
plt.axvline(x = t[int(use_this)], color = 'k', linestyle = '--', label = 'plateau value')
plt.text(t[int(use_this)], c[int(use_this)] + np.nanmin(c), str(short_rate)+"Hz")
plt.xlabel("t")
plt.ylabel(r"$C_{fs}$")
# plt.savefig("/Users/mohan/Desktop/well-behaved_ff_1000k_2wall_6gap.png", dpi = 500)
plt.show()