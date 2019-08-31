import argparse
import json
from functools import partial

import matplotlib.pyplot as plt
import numba
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
from numpy import array, concatenate, cos, isnan, pi, savetxt, sin, zeros

from jacobian import jacobian
from makethetafunction import maketheta
from residual import residual

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfile',
                    help='A JSON file with all simulation params.', required=True)
args = parser.parse_args()

with open(args.inputfile, 'r') as inp:
    params = json.loads(inp.read())

t = 0.0
y = zeros(6)
yp = zeros(6)

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

# -------------------------------------------------------------------------------
# Set up the initial conditions
# -------------------------------------------------------------------------------
# The initial velocity should NOT be 0 at t=0
y[2] = params['phi']
y[3] = params['xvelocity']
y[4] = params['yvelocity']

# Set the y3==yp1,y4==yp2,y5==yp3
yp[0:3] = y[3:6]
f = res(t, y, yp)
yp[3] = -f[3]/Mv
yp[4] = -f[4]/Mv
yp[5] = -f[5]/Iv

# Check Jacobian
g = jac(0.1, t, y, yp)
isNotValid = isnan(g).any()

if isNotValid == True:
    raise ValueError('Please check initial conditions. Jacobian has NaN.\n')

# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(res, y, yp)
imp_mod.jac = jac  # Sets the jacobian

# Sets the options to the problem
# Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod)  # Create a IDA solver
imp_sim.algvar = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
imp_sim.suppress_alg = True

# Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
imp_sim.make_consistent('IDA_YA_YDP_INIT')

# Simulate 4*pi seconds with 1000 communication points
endtime = eval(params['endtime'])
N = params['timepoints']
t_sol, y_sol, yd_sol = imp_sim.simulate(endtime, N)

t = array(t_sol)
t = t.reshape((N + 1, 1))

# Save results
yd = yd_sol[:, 3:6]
outputfile = params['Output']  # Output file name
outArray = concatenate((t, y_sol, yd_sol[:, 3:6]), axis=1)
savetxt(outputfile, outArray, fmt='%20.16f', delimiter=',')

# Extract position from solution
x_p = y_sol[:, 0]
y_p = y_sol[:, 1]
p_p = y_sol[:, 2]

x_v = y_sol[:, 3]
y_v = y_sol[:, 4]
p_v = y_sol[:, 5]

x_a = yd_sol[:, 3]
y_a = yd_sol[:, 4]
p_a = yd_sol[:, 5]

# Plot trajectory

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
#ax.set_aspect('equal')
ax.plot(x_p, y_p)
fig.tight_layout()
plt.savefig('Trajectory.pdf')