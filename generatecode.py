from functools import partial

from sympy import cos, sin, pi, atan2, sqrt, acos
import sympy as sp
import sympy.physics.vector as v

v.printing.init_printing()
sp.init_printing()
pycode = partial(sp.pycode, fully_qualified_modules=False)

E = v.ReferenceFrame('E')
t = sp.Symbol('t')

t1 = sp.Function('theta1')(t)
t2 = sp.Function('theta2')(t)
p = sp.Function('phi')(t)
x = sp.Function('x')(t)
y = sp.Function('y')(t)

r1, r2, L1, L2, R = sp.symbols('r1, r2, L1, L2, R',real=True)

##### Kinematics #####
X = x*E.x + y*E.y
eR = cos(p)*E.x + sin(p)*E.y

# Link 1
er1 = cos(t1+p)*E.x + sin(t1+p)*E.y

Xp = X + R*eR + r1*er1
Vp = Xp.diff(t,E)
n1 = sin(t1+p)*E.x - cos(t1+p)*E.y
Vn1 = v.dot(Vp,n1)*n1

# Link2
er2 = cos(t2+t1+p)*E.x + sin(t2+t1+p)*E.y
Xq = X + R*eR + L1*er1 + r2*er2 
Vq = Xq.diff(t,E)
n2 = sin(t2+t1+p)*E.x - cos(t2+t1+p)*E.y
Vn2 = v.dot(Vq,n2)*n2

##### Dynamics #####
Ma,Mv,Iv = sp.symbols('Ma,Mv,Iv', real=True)

s1,s2,s3 = sp.symbols('s1,s2,s3',real=True)

# Z1 = sp.Rational(1,2)*rho*Cd*H
# Z2 = sp.Rational(1,2)*rho*Cdv*Sv
# Z3 = sp.Rational(1,2)*rho*Ct*St

Z1, Z2, Z3 = sp.symbols('Z1, Z2, Z3')
rt1 = sqrt(R**2 + sp.Rational(9,16)*L1**2 + sp.Rational(3,2)*L1*R*cos(t1))
cos_t3 = ((3*L1 + 4*R*cos(t1))/4/rt1)
L3 = R*cos(p) + L1*cos(p + t1) + sp.Rational(3,4)*L2*cos(p + t1 + t2)
L4 = R*sin(p) + L1*sin(p + t1) + sp.Rational(3,4)*L2*sin(p + t1 + t2)
rt2 = sqrt(L3**2 + L4**2)
t6 = acos(L4/rt2) - pi/2 + (p + t1 + t2)

# When Vn1 < 0 s1 = -1 and when Vn2 < 0 s2 = -1
# Otherwise, s1 = 1 and s2 = 1
dF1m = -Ma*Vn1.diff(t,E)
dF1s = -s1*Z1*v.dot(Vp,n1)**2*n1
dF2m = -Ma*Vn2.diff(t,E)
dF2s = -s2*Z1*v.dot(Vq,n2)**2*n2

# Total force due to links
F1xm = sp.integrate(v.dot(dF1m,E.x),(r1,0,L1))
F1ym = sp.integrate(v.dot(dF1m,E.y),(r1,0,L1))
F1xs = sp.integrate(v.dot(dF1s,E.x),(r1,0,L1))
F1ys = sp.integrate(v.dot(dF1s,E.y),(r1,0,L1))
F2xm = sp.integrate(v.dot(dF2m,E.x),(r2,0,L2))
F2ym = sp.integrate(v.dot(dF2m,E.y),(r2,0,L2))
F2xs = sp.integrate(v.dot(dF2s,E.x),(r2,0,L2))
F2ys = sp.integrate(v.dot(dF2s,E.y),(r2,0,L2))

F1x = F1xm + F1xs;
F1y = F1ym + F1ys;
F2x = F2xm + F2xs;
F2y = F2ym + F2ys;

Fd1 = sqrt(F1x**2 + F1y**2);
Fd2 = sqrt(F2x**2 + F2y**2);

# The derivatives of x and y
x1 = x.diff(t)
y1 = y.diff(t)
p1 = p.diff(t)
x2 = x.diff(t, 2)
y2 = y.diff(t, 2)
p2 = p.diff(t, 2)

dt1 = t1.diff(t)
dt2 = t2.diff(t)
ddt1 = t1.diff(t, 2)
ddt2 = t2.diff(t, 2)

# Resistance force due to drag
Fvx = -Z2*sqrt(x1**2 + y1**2)*x1;
Fvy = -Z2*sqrt(x1**2 + y1**2)*y1;

T1 = s1*rt1*Fd1*cos_t3;
T2 = s2*rt2*Fd2*cos(t6);
# When p1 > 0, s3 = 1, when p1 < 0, s3 = -1
Tv = -s3*Z3*p1**2;

f3 = Mv*x2 - (Fvx + F1x + F2x)
f4 = Mv*y2 - (Fvy + F1y + F2y)
f5 = Iv*p2 - (T1 + T2 + Tv)

df3dx = f3.diff(x)
df3dy = f3.diff(y)
df3dp = f3.diff(p)
df3dx1 = f3.diff(x1)
df3dy1 = f3.diff(y1)
df3dp1 = f3.diff(p1)
df3dx2 = f3.diff(x2)
df3dy2 = f3.diff(y2)
df3dp2 = f3.diff(p2)

df4dx = f4.diff(x)
df4dy = f4.diff(y)
df4dp = f4.diff(p)
df4dx1 = f4.diff(x1)
df4dy1 = f4.diff(y1)
df4dp1 = f4.diff(p1)
df4dx2 = f4.diff(x2)
df4dy2 = f4.diff(y2)
df4dp2 = f4.diff(p2)

df5dx = f5.diff(x)
df5dy = f5.diff(y)
df5dp = f5.diff(p)
df5dx1 = f5.diff(x1)
df5dy1 = f5.diff(y1)
df5dp1 = f5.diff(p1)
df5dx2 = f5.diff(x2)
df5dy2 = f5.diff(y2)
df5dp2 = f5.diff(p2)

# Make substitutions to prepare for code generation
subs = [(p2, sp.Symbol('p2')),
        (y2, sp.Symbol('y2')),
        (x2, sp.Symbol('x2')),
        (p1, sp.Symbol('p1')),
        (y1, sp.Symbol('y1')),
        (x1, sp.Symbol('x1')),
        (p, sp.Symbol('p')),
        (y, sp.Symbol('y')),
        (x, sp.Symbol('x')),
        (ddt1, sp.Symbol('ddt1')),
        (dt1, sp.Symbol('dt1')),
        (t1, sp.Symbol('t1')),
        (ddt2, sp.Symbol('ddt2')),
        (dt2, sp.Symbol('dt2')),
        (t2, sp.Symbol('t2'))]

f3 = f3.subs(subs)
f4 = f4.subs(subs)
f5 = f5.subs(subs)

df3dx = df3dx.subs(subs)
df3dy = df3dy.subs(subs)
df3dp = df3dp.subs(subs)
df3dx1 = df3dx1.subs(subs)
df3dy1 = df3dy1.subs(subs)
df3dp1 = df3dp1.subs(subs)
df3dx2 = df3dx2.subs(subs)
df3dy2 = df3dy2.subs(subs)
df3dp2 = df3dp2.subs(subs)

df4dx = df4dx.subs(subs)
df4dy = df4dy.subs(subs)
df4dp = df4dp.subs(subs)
df4dx1 = df4dx1.subs(subs)
df4dy1 = df4dy1.subs(subs)
df4dp1 = df4dp1.subs(subs)
df4dx2 = df4dx2.subs(subs)
df4dy2 = df4dy2.subs(subs)
df4dp2 = df4dp2.subs(subs)

df5dx = df5dx.subs(subs)
df5dy = df5dy.subs(subs)
df5dp = df5dp.subs(subs)
df5dx1 = df5dx1.subs(subs) 
df5dy1 = df5dy1.subs(subs) 
df5dp1 = df5dp1.subs(subs) 
df5dx2 = df5dx2.subs(subs) 
df5dy2 = df5dy2.subs(subs) 
df5dp2 = df5dp2.subs(subs)

def sym():
    """ Infinite iterator for subexpressions
    """
    i = 0
    while True:
        yield sp.Symbol('term{}'.format(i))
        i += 1

# Eliminate common sub-expressions from residues 
terms = sym()
funcrep, funcexpr = sp.cse([f3,f4,f5], symbols=terms)

# Eliminate common sub-expressions from jacobians 
terms = sym()
exprs = [df3dx, df3dy, df3dp, df3dx1, df3dy1, df3dp1, df3dx2, df3dy2, df3dp2,
         df4dx, df4dy, df4dp, df4dx1, df4dy1, df4dp1, df4dx2, df4dy2, df4dp2,
         df5dx, df5dy, df5dp, df5dx1, df5dy1, df5dp1, df5dx2, df5dy2, df5dp2]

jacrep, jacexpr = sp.cse(exprs, symbols=terms)

# Create code for the residual and write it to file
with open('residual.py', 'w') as funcfile:
    funcfile.write(
            'import numba\n'
            'import numpy as np\n'
            'from math import sqrt, sin, cos, acos, atan2\n\n'
            '@numba.njit\n'
            'def residual(t, yin, yp, R, L1, L2, H, Ct, Cdv, Cd, St, Sv, rho, Ma, Mv, Iv, theta):\n'
            '    """Calculates the residual of the system of equations G(t, y, yp) == 0."""\n'
            '    x,y,p,x1,y1,p1 = yin\n'
            '    x2,y2,p2 = yp[3:6]\n'
                 
            '    t1,dt1,ddt1,t2,dt2,ddt2 = theta(t)\n'
                 
            '    # Test direction of Vn1\n'
            '    r1 = 0.5*L1\n'
            '    Vn1 = (-R*sin(p)*p1 - r1*(p1 + dt1)*sin(p + t1) + x1)*sin(p + t1) - (R*cos(p)*p1 + r1*(p1 + dt1)*cos(p + t1) + y1)*cos(p + t1)\n'
                 
            '    s1 = 1.0 if Vn1 > 0 else -1.0\n'
            '    # Test direction of Vn2\n'
            '    r2 = 0.5*L2    \n'
            '    Vn2 = (-L1*(p1 + dt1)*sin(p + t1) - R*sin(p)*p1 - r2*(p1 + dt1 + dt2)*sin(p + t1 + t2) + x1)*sin(p + t1 + t2) - (L1*(p1 + dt1)*cos(p + t1) + R*cos(p)*p1 + r2*(p1 + dt1 + dt2)*cos(p + t1 + t2) + y1)*cos(p + t1 + t2)\n'
            '    s2 = 1.0 if Vn2 > 0 else -1.0\n'
            '    s3 = 1.0 if p1 > 0 else -1.0\n'
            '    Z1 = 0.5*rho*Cd*H\n'
            '    Z2 = 0.5*rho*Cdv*Sv\n'
            '    Z3 = 0.5*rho*Ct*St\n\n'
            '    # Now calculate the residuals\n'
            '    f = np.empty(6)\n'
            '    f[0] = yp[0] - x1;\n'
            '    f[1] = yp[1] - y1;\n'
            '    f[2] = yp[2] - p1;\n\n'
            '    # First calculate the common sub-expressions\n')
    for lhs, rhs in funcrep:
        funcfile.write('    {} = {}\n'.format(lhs, pycode(rhs)))

    funcfile.write('\n    # Now the remaining elements of `f`\n')
    for lhs, rhs in zip(['f[3]', 'f[4]', 'f[5]'], funcexpr):
        funcfile.write('    {} = {}\n'.format(lhs, pycode(rhs)))

    funcfile.write('\n    return f\n')

# Now create code for the jacobian and write it to file
with open('jacobian.py', 'w') as jacfile:
    jacfile.write(
            'import numba\n'
            'import numpy as np\n'
            'from math import sqrt, sin, cos, acos, atan2\n\n'
            '@numba.njit\n'
            'def jacobian(c, t, yin, yp, R, L1, L2, H, Ct, Cdv, Cd, St, Sv, rho, Ma, Mv, Iv, theta):\n'
            '    """Calculates the jacobian of the system of equations G(t, y, yp) == 0."""\n'
            '    x,y,p,x1,y1,p1 = yin\n'
            '    x2,y2,p2 = yp[3:6]\n'
                 
            '    t1,dt1,ddt1,t2,dt2,ddt2 = theta(t)\n'
                 
            '    # Test direction of Vn1\n'
            '    r1 = 0.5*L1\n'
            '    Vn1 = (-R*sin(p)*p1 - r1*(p1 + dt1)*sin(p + t1) + x1)*sin(p + t1) - (R*cos(p)*p1 + r1*(p1 + dt1)*cos(p + t1) + y1)*cos(p + t1)\n'
                 
            '    s1 = 1.0 if Vn1 > 0 else -1.0\n'
            '    # Test direction of Vn2\n'
            '    r2 = 0.5*L2    \n'
            '    Vn2 = (-L1*(p1 + dt1)*sin(p + t1) - R*sin(p)*p1 - r2*(p1 + dt1 + dt2)*sin(p + t1 + t2) + x1)*sin(p + t1 + t2) - (L1*(p1 + dt1)*cos(p + t1) + R*cos(p)*p1 + r2*(p1 + dt1 + dt2)*cos(p + t1 + t2) + y1)*cos(p + t1 + t2)\n'
            '    s2 = 1.0 if Vn2 > 0 else -1.0\n'
            '    s3 = 1.0 if p1 > 0 else -1.0\n'
            '    Z1 = 0.5*rho*Cd*H\n'
            '    Z2 = 0.5*rho*Cdv*Sv\n'
            '    Z3 = 0.5*rho*Ct*St\n\n'
            '    # Now calculate the jacobian\n'
            '    dfdy = np.zeros((6, 6))\n'
            '    dfdyp = np.zeros((6, 6))\n'
            '    dfdy[0,3] = -1.0\n'
            '    dfdy[1,4] = -1.0\n'
            '    dfdy[2,5] = -1.0\n\n'
            '    dfdyp[0,0] = 1.0\n'
            '    dfdyp[1,1] = 1.0\n'
            '    dfdyp[2,2] = 1.0\n\n'
            '    # First calculate the common sub-expressions\n')
    for lhs, rhs in jacrep:
        jacfile.write('    {} = {}\n'.format(lhs, pycode(rhs)))

    jacfile.write('\n    # Now the remaining elements of `dfdy`\n')
    exprstr = ['df3dx', 'df3dy', 'df3dp',
               'df3dx1', 'df3dy1', 'df3dp1',
               'df3dx2', 'df3dy2', 'df3dp2',
               'df4dx', 'df4dy', 'df4dp',
               'df4dx1', 'df4dy1', 'df4dp1',
               'df4dx2', 'df4dy2', 'df4dp2',
               'df5dx', 'df5dy', 'df5dp',
               'df5dx1', 'df5dy1', 'df5dp1',
               'df5dx2', 'df5dy2', 'df5dp2']
    for lhs, rhs in zip(exprstr, jacexpr):
        jacfile.write('    {} = {}\n'.format(lhs, pycode(rhs)))

    jacfile.write('\n    # Fill the jacobian matrices\n')
    for i in range(3, 6):
        for j, v in enumerate(['x', 'y', 'p', 'x1', 'y1', 'p1']):
            jacfile.write('    dfdy[{0},{1}] = df{0}d{2}\n'.format(i, j, v))

    jacfile.write('\n')
    for i in range(3, 6):
        for j, v in enumerate(['x1', 'y1', 'p1', 'x2', 'y2', 'p2']):
            jacfile.write('    dfdyp[{0},{1}] = df{0}d{2}\n'.format(i, j, v))

    jacfile.write('\n    jac = dfdy + c*dfdyp\n\n')
    jacfile.write('    return jac\n')

