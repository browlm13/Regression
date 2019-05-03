#!/usr/bin/env python

"""


The data in the accompanying ../data/climate_data.txt file comes from
https://data.giss.nasa.gov/gistemp/tabledata_v3/GLB.Ts.txt

Column 1 -- Year offset from 1880.  To convert to the calendar year,
            just add 1880.

Column 2 -- Average January-December global temperature in degrees
            Farenheit.  This data was computed by taking the "J-D"
            data column from the online data set, scaling by 1.8/100
            to convert to deviation in degrees Farenheit, and adding
            57.2 to add back in the "mean" surface air temperature.



"""

__author__ = "l.j. Brown"
__version__ = "1.0.1"

# imports

# internal
import logging

# external
import numpy as np
import matplotlib.pyplot as plt

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#
# CONSTANTS
#
FILE_PATH = "../data/climate_data.txt"
DELIM = '   '

#
# a.) Load Climate Data
# 

# cols: [year, temperature]
data = np.loadtxt( FILE_PATH, usecols = (1,2), delimiter=DELIM, 
				   dtype=np.float32, unpack=True ) 

#
# Helper Methods
#

# [TODO]: build more general phi(y,y), b(t,y) least sqaures constructor

def poly_M(t, degree=1):
	# cols : [t^degree, ..., t^1, t^0]

	# create empty matrix
	M = np.empty(shape=(t.shape[0],degree+1), dtype=np.float32)

	# set last column to 1 (t^0)
	M[:,-1] = np.ones(shape=t.shape)

	# set each preceding column as the previous * ts : 
	for col in range(degree-1,-1,-1): 
		M[:,col] = M[:,col+1]*t

	return M


#
# b.) Construct the over-determined linear system Ax = b corresponding to fitting a polynomial of the form:
#         
#                             p(t) = x1*t^3 +x2*t^2 +x3*t +x4
#
#     Through the supplied data, i.e., the ith row of the matrix A will contain the entries:
# 	
#							      [ ti^3, ti^2, ti, 1 ]
#
#     And bi = tempi, where ti is the ith ‘year’ and tempi is the ith ‘temperature’ from the data set, 
#     and the solution vector will be:
#
#                                 x = [􏰀 x1 x2 x3 x4 ].􏰁T

# [year, temperature]
t, b = data 

# Construct Matrix A
A = poly_M(t, degree=3)

#
# c.) Use the Python command numpy.linalg.qr to construct the factorization A = QR
#

Q, R = np.linalg.qr(A)

#
# d.) Solve the least-squares problem for 
#

# 1) Compute c = Q^T b
c = Q.T @ b

# 2) Solve Rx = c for x (backwards substitution).
x1 = np.linalg.solve(R,c)

# build model 1
p = np.poly1d(x1)

#
# e.) Set an array dates to equal 􏰀0 1 ... 145􏰁, corresponding to the date range 1880 through 2025.
#

dates = np.linspace(0, 145, num=146)

#
# f.) Set an array projections1 to equal your model p(t) from equation (1), evaluated at each date in the dates array.
#

projections1 = p(dates)

#
# g) Construct the over-determined linear system By = d 
#    corresponding to fitting a model of the form:
#
#        q(t) = exp(x1*t^2 + x2*t + x3)
#

# modify b: log(q(t)) 
d = np.log(b)

# construct B
B = poly_M(t, degree=2)

#
# h) Use the Python command numpy.linalg.svd 
#    to construct the factorization B = U ΣV.T 
#    Check the documentation for this function to understand the structures that are returned!
#

U, S, Vh = np.linalg.svd(B, full_matrices=False)

#*** if rank r < min{m, n} we additonally seek the x̄ such that ∥x̄∥2 is minimal.

# numerical rank : σ1 ≥···≥σk > ε ≥ σk+1 ≥ ···
numerical_rank = np.linalg.matrix_rank(B, tol=1e-16)

# 1. Compute c = U.(∗)b, and partition this into c = [c_hat, w]
c = U.T @ d
c_hat = c[:numerical_rank]

# 2. Compute y_hat = S_pinv @ c_hat.
S_pinv = 1/S[:numerical_rank]
y_hat = S_pinv * c_hat 			# Note: y = [y_hat, 0]

# 4. Compute x = V y
x2 = Vh.T[:,:numerical_rank] @ y_hat

# model 2
p2 = np.poly1d(x2)
q = lambda t: np.exp(p2(t))

# model 2 projections
projections2 = q(dates)

#
# Plot
#

plt.plot(t, b, '-k', marker='o', linestyle='None', markersize=1, label='true')
plt.plot(dates, projections1, '-b', label='projections1')
plt.plot(dates, projections2, '--r', label='projections2')

plt.title("avg temperature v. year")
plt.xlabel("year")
plt.ylabel("temp")
plt.legend(loc='upper left')

plt.show()
