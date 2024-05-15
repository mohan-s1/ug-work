#/* --------------------------------------------------------------------------------
#   Egorov Group
#   University of Virginia
#   Mohan Shankar
#
#   correlation.py
#   "This code calculates correlation functions for a given system in parallel"
#-------------------------------------------------------------------------------- */
# DEPENDENCIES
import numpy as np
from multiprocessing import Pool # relevant package for parallel processing temperatures
from datetime import datetime
#-------------------------------------------------------------------------------- */
# INPUTS & LOAD IN DATA (PART 1)

start_time = datetime.now()
print("Starting!", start_time)
kb = 3.166830e-6 # Boltzmann constant in hartree
har = 2.194746e5 # hartree in cm^{-1}

Trange = np.array([100, 200, 400, 600, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000])

tmin = 0
tmax = 10000
nt = 1000
dt = (tmax - tmin)/nt

hbar = 1.0

with np.load('Energy.npz') as data:
    E = np.array([data['energies']])
    fx2 = data['fx2']

nn = np.shape(fx2)[0]
# print("Energies", E)
epsilon= np.log(1e-16)
zero = E[0]
dE = E - E.transpose()
sE = E + E.transpose()

dEm = dE.copy()

np.fill_diagonal(dEm, 1)

ttot = np.arange(tmin, tmax+1, dt)
CF = np.zeros((2, nt+1))
CS = np.zeros((2, nt+1))
CF[0] = ttot
CS[0] = ttot

end_time =  datetime.now()
print('Duration of Part One: {}'.format(end_time - start_time))
#-------------------------------------------------------------------------------- */
# Correlation Functions (PART 2)

start_time = datetime.now()

def f(T):
    beta = 1/(kb*T)
    cut = 2 * epsilon/beta 
    over = np.where(E > zero - cut)
    if np.shape(over) == 0:
        print(f'Not converged at {T}')
        Q = nn
    else:
        print(f'At {T}K, {over[0]} functions are significant')
        if over[0] <= nn:
            Q = over[0]
        else:
            Q = nn

    for time in range(nt + 1):
        t = tmin + time * dt
        zz = np.exp(-beta * 0.5 * sE[0:Q, 0:Q]) * fx2[0:Q, 0:Q]
        CF1 = zz * np.cos(dE[0:Q, 0:Q] * t)
        zzd = np.diagonal(zz)
        CS1 = zz * np.sin(dE[0:Q, 0:Q] * t) / dEm[0:Q, 0:Q]
        CSd = zzd * t
        np.fill_diagonal(CS1, CSd)

        CF[1, time] = np.sum(CF1)
        CS[1, time] = np.sum(CS1)
        
    return CS, CF

    # with open(f'Side-Flux{T}.txt', 'w') as f:
    #     np.savetxt(f, CS, fmt = '%1.4e')
    # with open(f'Flux-Flux{T}.txt', 'w') as f:s
    #     np.savetxt(f, CF, fmt = '%1.4e')


if __name__ == '__main__':
    #num_cores=os.getenv('SLURM_CPUS_PER_TASK') # get number of cores based on slurm script
    pool = Pool(processes=10) # use as many cores as allocated by slurm script
    fargs=zip(Trange)

    result = pool.starmap(f, fargs) # result is a list of all returned values for CS, CF at a given T in order (i.e. at Trange[i]: CS = results[i][0], CF = results[i][1])

    # result is a 4-D object; result[a][b][c][d] --> a denotes which temp;
    # b = 0 --> CS; b = 1 --> CF
    # [c][d] are then indices for a 2D matrix where the first column
    # corresponds to time and the second the relevant correlation function

    for i, val in enumerate(Trange):
        np.savetxt('side-flux'+str(val)+'K.txt', np.transpose((result[i][0][0], result[i][0][1])), delimiter = ',', header="Time , Side-Flux", fmt='%1.4e') 

        # result[i][a][b] means Trange[i]; [a] = 0 means Flux-Side (CS);
        # [b] = 0 means time column; b = 1 means correlation function;
        # free index [d] corresponds to a particular element in column so it's unused
    
    for i, val in enumerate(Trange):
        np.savetxt('flux-flux'+str(val)+'K.txt', np.transpose((result[i][1][0], result[i][1][1])), delimiter = ',', header="Time , Flux-Flux", fmt='%1.4e') 
    
        # result[i][a][b] means Trange[i]; [a] = 1 means Flux-Flux (CF) [b] = 0 means
        # time column while b = 1 means correlation function;
        # [c][d] are then indices for a 2D matrix where the first column
        # corresponds to time and the second the relevant correlation function

    pool.close()
    pool.join()


end_time =  datetime.now()
print('Duration of Part Two: {}'.format(end_time - start_time))
#-------------------------------------------------------------------------------- */