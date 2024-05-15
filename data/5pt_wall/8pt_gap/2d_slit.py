#/* --------------------------------------------------------------------------------
#   Egorov Group
#   University of Virginia
#   Mohan Shankar
#
#   2d_slit.py
#   "This file calculates eigenfunctions of particle in a box in the presence of a slit"
#-------------------------------------------------------------------------------- */
# DEPENDENCIES
import numpy as np
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------- */
# INPUTS
me = 5.485799e-4 # Electron mass in daltons
m_au = 1.0 / me

lx = 1.0 # Angstroms
ly = 8.0 # Angstroms

a0 = 0.529177258 # Bohr radius

lx_au = lx/a0
ly_au = ly/a0 

nnx = 60 
nny = 480 

xmin = 0
xmax = lx_au

ymin = 0
ymax = ly_au

dx = (xmax-xmin)/(nnx+1)
dy = (ymax-ymin)/(nny+1)

pi = np.pi

nd = 2 # 
wall_thickness = 5
cd = np.zeros(nd) 
nn = 10000 

gap_size = 8.0
#-------------------------------------------------------------------------------- */
# FUNCTION DEFINITIONS

def kron_sum(A1, A2):
    '''
    Assumes A1, A2 are nxn, mxm matrices where n can be equal to m
    '''
    i1 = np.identity(len(A1[0]))
    i2 = np.identity(len(A2[0]))
    return np.kron(A1,i2) + np.kron(i1, A2)


def PIB_one(points, lmax, lmin, mass, hbar=1):  
    '''
    function to return eigenvalues and eigenvectors of a Hamiltonian matrix 
    eigvals returned in array where eigvals[0] gives a float corresponding to n = 1; units of Joules since I defined my analytic formula with SI units
    eigvecs returned in array of arrays where eigvecs[:, i] (column vectors) gives an array corresponding to n = i at each point in the box defined by grid spacing dx
    '''
    dn = (lmax-lmin)/(points+1)
    dn2 = dn**2  # second derivative
    H = np.zeros((points, points)) # initialize 
    z = -pi**2/3.0  # weight for diagonal
    for i in range(points):
        for j in range(points):
            if i == j:
                H[i][j] = z # weight for diagonals of matrix
            else:
                H[i][j] = (2/(i-j)**2)*((-1)**(i-j+1))  # weights for non-diagonals
    H *= (-1/(2*mass*dn2)) # hbar = 1 hence 1/(2 * mass * dn2)
    return H
#-------------------------------------------------------------------------------- */
# CREATE MATRICES 
h1_x = PIB_one(nnx, xmax, xmin, m_au) # 1-D Hamiltonian from x points
h1_y = PIB_one(nny, ymax, ymin, m_au) # 1-D Hamiltonian from y points

H = kron_sum(h1_x, h1_y) # 2-D Hamiltonian
# STENCIL FOR FLUX ALONG DIVIDING SURFACE
for k in range(nd):
    zz = 1.0
    for j in range(nd):
        if j != k:
            zz *= ((j+1)**2) / ((j+1)**2 - (k+1)**2)
        else:
            continue
    cd[k] = zz/(2*(k+1))

print("Matrices created!")
#-------------------------------------------------------------------------------- */
# MAKE EFFUSION SLIT
i = -1

for k in range(nnx):
    for j in range(nny):
        i = i+1
        x = dx * k + xmin
        y = dy * j + ymin
        if (1.0 / a0) < y < (1.0 / a0 + wall_thickness * dy):
            if x <(0.5 * xmax - (gap_size/2.0) * dx) or x >(0.5 * xmax + (gap_size/2.0) * dx):
                H[i] = np.zeros(nnx * nny)
                H[:, i] = np.zeros(nnx * nny)
                H[i, i] = -1
print("Slit created!")
#-------------------------------------------------------------------------------- */
# CALCULATE EIGENVECTORS AND VALUES
eigvals, eigvecs = np.linalg.eigh(H) # find eigenvalues and eigenvectors
print("Eigs found!")
#-------------------------------------------------------------------------------- */
# CLEANUP OF DATA
psi = np.transpose(eigvecs) # vectors returned in column form so take transpose for easier indexing

psi = psi[np.argsort(eigvals)]
energies = eigvals[np.argsort(eigvals).real]

cut = np.where(energies > 0)
print(cut)
energies = energies[cut]
psi = psi[cut]

for i in range(len(psi)):
    normalization = np.sqrt(np.sum(psi[i]**2 * dy * dx))
    psi[i] = psi[i]/normalization

# np.savez("eigs_file", psi = psi, eigvals = eigvals)
print("Eigs saved!")
#-------------------------------------------------------------------------------- */
# FIRST DERIVATIVE OVER DIVIDING SURFACE

dpsi = np.zeros((nn, nnx))

y = (1/a0) + 1.0 * dy
i0 = int((y-ymin)/dy)

for j in range(nn):
    for i2 in range(nnx):
        ix = nny * i2 + i0
        zz = 0.0
        for k in range(nd):
            kk = k + 1
            zz += cd[k] * (psi[j][ix + kk] - psi[j][ix-kk])
        zz = zz/dy
        dpsi[j, i2] = zz

print("First derivative over dividing surface found!")
#-------------------------------------------------------------------------------- */
# CALCULATION OF FLUX SQUARED
        
fx2 = np.zeros((nn, nn))

for j1 in range(nn):
    for j2 in range(nn):
        zz = 0.0
        for i2 in range(nnx):
            ix = nny * i2 + i0
            zz0 = dpsi[j1, i2] * psi[j2, ix] - psi[j1, ix] * dpsi[j2, i2]
            zz0 = zz0 * dy / (2.0 * m_au)
            zz += zz0
        zz = zz**2
        fx2[j1, j2] = zz

with open("Energy.npz", "wb") as f:
    np.savez(f, energies = energies, fx2 = fx2, )

print("Job was successfully completed!")
#-------------------------------------------------------------------------------- */