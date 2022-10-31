"""

This program solves Initial Value Problems (IVP).
We support three numerical meothds: Euler, Rk2, and Rk4

Author: Kuo-Chuan Pan, NTHU 2022.10.06
For the course, computational physics lab

"""

import numpy as np

def solve_ivp(derive_func, y0, t, dt, N, method, args):
    """
    Solve Initial Value Problems. 

    :param derive_func: a function to describe the derivative of the desired function
    :param y0: an array. The initial state
    :param dt: the step time
    :param N: the number of steps.
    :param method: string. Numerical method to compute. 
                   We support "Euler", "RK2" and "RK4".
    :param *args: extra arguments for the derive func.

    :return: array_like. solutions. 
    """
    sol_pos, sol_vel = np.array([]), np.array([])
    t = 0
    for _ in range(N):
        t += dt
        sol_pos = np.append(sol_pos, y0[0])
        sol_vel = np.append(sol_vel, y0[1]) 
        y0 = _update(derive_func, y0, t, dt, method, *args)

    return [sol_pos, sol_vel]

def _update(derive_func, y0, t, dt, method, *args):
    """
    Update the IVP with different numerical method

    :param derive_func: the derivative of the function y'
    :param y0: the initial conditions at time t
    :param dt: the time step dt
    :param method: the numerical method
    :param *args: extral parameters for the derive_func

    :return: the next step condition y0

    """

    if method=="Euler":
        ynext = _update_euler(derive_func,y0, t, dt,*args)
    elif method=="RK2":
        ynext = _update_rk2(derive_func,y0, t, dt,*args)
    elif method=="RK4":
        ynext = _update_rk4(derive_func,y0,t, dt,*args)
    else:
        print("Error: mysolve doesn't supput the method",method)
        quit()
    return ynext

def _update_euler(derive_func,y0, t, dt,*args):
    """
    Update the IVP with the Euler's method

    :return: the next step solution y

    """
    y0 = np.add(y0, derive_func(y0, t, *args) * dt)

    return y0 # <- change here. just a placeholder

def _update_rk2(derive_func, y0, t, dt,*args):
    """
    Update the IVP with the RK2 method

    :return: the next step solution y
    """

    k1 = derive_func(y0, t, *args)
    y_temp = y0 + dt * k1 
    k2 = derive_func(y_temp, t, *args) 
    
    y0 = np.add(y0, (dt / 2) * (k1 + k2))
    # note: if use: y0 += (dt / 2) * (k1 + k2) ==> error
    # Yet use y0 = y0 + (dt / 2) * (k1 + k2) ==> it can work, and np.add() can work too.

    return y0 # <- change here. just a placeholder

def _update_rk4(derive_func,y0, t, dt,*args):
    """
    Update the IVP with the RK4 method

    :return: the next step solution y
    """

    k1 = derive_func(y0, t, *args)
    y_temp = y0 + (dt/2) * k1 # temp for virtual step y*
    k2 = derive_func(y_temp, t, *args)
    y_temp = y0 + (dt/2) * k2
    k3 = derive_func(y_temp, t, *args)
    y_temp = y0 + dt * k3
    k4 = derive_func(y_temp, t, *args)
    
    y0 = np.add(y0, (1/6) * dt * (k1 + 2*k2 + 2*k3 + k4))

    return y0 # <- change here. just a placeholder


if __name__=='__main__':


    """
    
    Testing mysolver.solve_ivp()

    Kuo-Chuan Pan 2022.10.07

    """


    def oscillator(y,t,K,M):
        '''
        This is the function (osci) defined in the [position, velocity] 
        and [derivative(position), derivative(velocity)]
        :param y: [position, velocity]
        :param k: spring constants
        :param m: mass constant
        '''
        
        yder =  np.zeros(2)
        yder[0] = y[1]
        yder[1] = -y[0] * K/M # the difinition of the acceleration, which is depend on the position.
        print(t) # check to update time
        return yder

    K, M  = 1, 1
    N, t = 100, 20
    dt = t/N
    y0 = np.array([1, 0]) # [pos_0, vel_0]

    sol = solve_ivp(oscillator, y0, t, dt, N, method="RK2", args=(K,M))

    # print("sol=",sol)
    print("Done!")