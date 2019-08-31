import argparse
import json
from functools import partial
from math import cos, pi, sin

import matplotlib.pyplot as plt
import numba
import numpy as np

from jacobian import jacobian
from residual import residual
from makethetafunction import maketheta

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfile',
                    help='A JSON file with all simulation params.', required=True)
args = parser.parse_args()

with open(args.inputfile, 'r') as inp:
    params = json.loads(inp.read())

# Simulation parameters
R = params['R']  # Vehicle radius
L1 = params['L1']  # Link1 length
L2 = params['L2']  # Link2 length
H = params['H']

Ct = params['Ct']  # Torsional drag coeff
Cdv = params['Cdv']  # Vehicle drag coeff
Cd = params['Cd']  # Link drag coeff

St = eval(params['St'])  # Projected area for torsional drag
Sv = eval(params['Sv'])  # Projected area for vehicle drag

rho = params['rho']  # density of water
Ma = eval(params['Ma'])  # Added mass per unit link length
Mv = eval(params['Mv'])  # Mass of vehicle
Iv = eval(params['Iv'])  # Moment of Inertia of vehicle

theta1 = params['theta1']  # Angle of link 1
theta2 = params['theta2']  # Angle of link 2

# Create a function to evaluate angular inputs
theta = maketheta(theta1, theta2)

res = partial(residual, R=R, L1=L1, L2=L2, H=H, Ct=Ct, Cdv=Cdv,
              Cd=Cd, St=St, Sv=Sv, rho=rho, Ma=Ma, Mv=Mv, Iv=Iv,
              theta=theta)

jac = partial(jacobian, R=R, L1=L1, L2=L2, H=H, Ct=Ct, Cdv=Cdv,
              Cd=Cd, St=St, Sv=Sv, rho=rho, Ma=Ma, Mv=Mv, Iv=Iv,
              theta=theta)

t = 10*np.random.rand(1)[0]
y = 10*np.random.rand(6)
yd = 10*np.random.rand(6)
yd[0:3] = y[3:6]

hlist = np.logspace(-8,-3,101)
error = np.zeros(hlist.shape[0])

c = 1.0
jac_direct = jac(c, t, y, yd)
for index, hi in enumerate(hlist):
    jac_approx = np.zeros_like(jac_direct)
    for i in range(6):
        for j in range(6):
            yplus = np.copy(y)
            yminus = np.copy(y)
            yplus[j] += hi
            yminus[j] -= hi
            fplus = res(t, yplus, yd)
            fminus = res(t,yminus, yd)

            ydplus = np.copy(yd)
            ydminus = np.copy(yd)
            ydplus[j] += hi
            ydminus[j] -= hi

            if i > 2:
                yn = np.copy(y)
                yn[3:6] = ydplus[0:3]
                fdplus = res(t, yn, ydplus)
                yn = np.copy(y)
                yn[3:6] = ydminus[0:3]
                fdminus = res(t, yn, ydminus)
            else:
                fdplus = res(t, y, ydplus)
                fdminus = res(t, y, ydminus)

            jac_approx[i,j] += (fplus[i]-fminus[i])/(2.0*hi) +\
                    c*(fdplus[i]-fdminus[i])/(2.0*hi) 

    tempErr = np.linalg.norm(jac_direct - jac_approx)
    error[index] = tempErr

plt.loglog(hlist, error)
plt.show()
