import random

from FoKL import FoKLRoutines
from FoKL import getKernels
from Discretization import discretize
from Discretization import localCoverage
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd

phis = getKernels.sp500()
# ======================================================================================================================
npoints = 20
# Inputs:
X_line = np.linspace(0,1,npoints)
Y_line = np.linspace(0,1,npoints)
# grid
X_grid = X_line
Y_grid = Y_line
for i in range(npoints-1):
    X_grid = np.vstack([X_grid,X_line])
    Y_grid = np.hstack([Y_grid, Y_line])
Z_data = []
dk = 1/(npoints*npoints)
k = 1
for i in range(npoints):
    for j in range(npoints):
        Z_data.append(k*np.sin(np.sqrt(np.power(X_line[i], 2) + np.power(Y_line[j], 2))) + np.random.normal(0,0.03))


# Data:
# Z_grid = np.loadtxt('DATA_nois.csv',dtype=float,delimiter=',')
inputs = pd.read_csv("armIn.csv")
data = pd.read_csv("armD.csv")
#
inF = inputs.to_numpy()[0:1000, 1:7]
dF = data.to_numpy()[0:1000, 1:4]
dFT = np.zeros([np.shape(dF)[0],1])
for i in range(np.shape(dF)[0]):
    if dF[i][2] == 0:
        dFT[i] = np.pi/2
    else:
        dFT[i] = np.arctan(dF[i][0]/dF[i][2])



# # Reshaping grid matrices into vectors via fortran index order:
# m, n = np.shape(X_grid) # = np.shape(Y_grid) = np.shape(Z_grid) = dimensions of grid
# X = np.reshape(X_grid, (m*n,1), order='F')
# Y = np.reshape(Y_grid, (m*n,1), order='F')
# Z = np.reshape(Z_grid, (m*n,1), order='F')

# Initializing FoKL model with some user-defined hyperparameters (leaving others blank for default values):
kmax = 5
power = 2
dmin = 0.01
dmax = 5
csummax = 1/16

centers, label, Dins, cin, cout, cov, icov = discretize(np.hstack([inF[:,1:3],inF[:,4:6]]), dFT, dmin, dmax, csummax, kmax)

inputs_max = np.max(cin)

for ii in range(len(inputs_max)):
    inputs_min = np.min(cin[:, ii])
    if inputs_max[ii] != 1 or inputs_min != 0:
        if inputs_min == inputs_max[ii]:
            inputs[:, ii] = np.ones(len(inputs[:, ii]))
            warnings.warn("'inputs' contains a column of constants which will not improve the model's fit.",
                          category=UserWarning)
        else:  # normalize
            inputs[:, ii] = (inputs[:, ii] - inputs_min) / (inputs_max[ii] - inputs_min)

model = FoKLRoutines.FoKL()

betas_all = []
mtx_all = []



for i in range(np.shape(centers)[0]):

    # Running emulator routine to fit model to training data as a function of the corresponding training inputs:
    betas, mtx, evs = model.fit(cin[i], cout[i], train=0.75)

    # Provide feedback to user before the figure from coverage3() pops up and pauses the code:
    print("\nDone! Please close the figure to continue.\n")

    # Evaluating and visualizing predicted values of data as a function of all inputs (train set plus test set):


    # Store any values from iteration if performing additional post-processing or analysis:
    betas_all.append(betas)
    mtx_all.append(mtx)



    # Reset the model so that all attributes of the FoKL class are removed except for the hyperparameters:
    model.clear()


for ii in range(len(inputs_max)):
    inputs_min = np.min(cin[:, ii])
    if inputs_max[ii] != 1 or inputs_min != 0:
        if inputs_min == inputs_max[ii]:
            inputs[:, ii] = np.ones(len(inputs[:, ii]))
            warnings.warn("'inputs' contains a column of constants which will not improve the model's fit.",
                          category=UserWarning)
        else:  # normalize
            inputs[:, ii] = (inputs[:, ii] - inputs_min) / (inputs_max[ii] - inputs_min)

localCoverage(betas_all, mtx_all, phis, centers, np.hstack([inF[:,1:3],inF[:,4:6]]), dFT, 1000, 1, icov)
# ======================================================================================================================

# Post-processing:
print("\nThe results are as follows:")


