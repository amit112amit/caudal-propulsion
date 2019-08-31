from functools import partial
import sympy as sp

def maketheta(theta1, theta2):
    """
    Create a numba jitted function to evaluate theta and its derivatives.
    """
    t = sp.Symbol('t')
    t1 = sp.sympify(theta1)
    t2 = sp.sympify(theta2)
    dt1 = t1.diff(t)
    dt2 = t2.diff(t)
    ddt1 = t1.diff(t, 2)
    ddt2 = t2.diff(t, 2)
    code = [
        'import math',
        'from numba import njit',
        '@njit',
        'def theta(t):',
        '    """ Calculate hinge angles input."""',
    ]

    for term in ['t1', 'dt1', 'ddt1', 't2', 'dt2', 'ddt2']:
        code.append('    {0} = {1}'.format(term, sp.pycode(eval(term))))
    
    code.append('    return t1, dt1, ddt1, t2, dt2, ddt2')

    funccode = '\n'.join(code)
    exec(funccode, globals())
    return globals()['theta']


if __name__ == "__main__":
    theta1 = 'sin(t)'
    theta2 = 'cos(t)'
    func = maketheta(theta1, theta2)
    print(func(3.141))
