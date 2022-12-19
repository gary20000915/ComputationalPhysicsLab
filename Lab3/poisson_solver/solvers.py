import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numba import njit, prange
from .mesh import Mesh2D

"""
Solver to solve for Laplace/Poisson's equation

"""

def set_boundary(nx, ny, buff_size, boundary, mesh: Mesh2D):
    mesh[0, :]  = boundary[0]
    mesh[ny+buff_size, :] = boundary[1]
    mesh[:, 0]  = boundary[2]
    mesh[:, nx+buff_size] = boundary[3]

@njit(parallel = True)
def j_kernel(u, u_temp, nx, ny, buff_size):
    for i in prange(1, nx + 2*buff_size - 1, 1):
        for j in prange(1, ny + 2*buff_size - 1, 1):
            u[i, j] = 0.25 * (u_temp[i+1, j] + u_temp[i, j+1] + u_temp[i-1, j] + u_temp[i, j-1])
    return u
    
@njit(parallel = True)
def gs_kernel(u, nx, ny, buff_size):
    for i in prange(1, nx + 2*buff_size - 1, 1):
        for j in prange(1, ny + 2*buff_size - 1, 1):
            u[i, j] = 0.25 * (u[i+1, j] + u[i, j+1] + u[i-1, j] + u[i, j-1])
    return u

def solve(name, tor, boundary, mesh: Mesh2D):
    u         = mesh.get_mesh()
    nx        = mesh.get_nx()
    ny        = mesh.get_ny()
    buff_size = mesh.get_buff_size()
    
    set_boundary(nx, ny, buff_size, boundary, u)
    
    err     = 10
    err_arr = np.array([])
    n       = 0
    
    while err > tor:
        u_temp = np.copy(u)
        
        if name == "Jacobi":
            u = j_kernel(u, u_temp, nx, ny, buff_size)
        elif name == "Gauss":
            u = gs_kernel(u, nx, ny, buff_size)
        else:
            print("Error: unknown kernel!")
            break

        err = np.sqrt(np.sum(np.square(u - u_temp))) / (nx * ny)
        err_arr = np.append(err_arr, err)
        n += 1
        # check
        # if n % 100 == 0:
        #     print(err, tor)
            # print(u)
            # plt.imshow(u.reshape(nx+2*buff_size, ny+2*buff_size), origin = 'lower', extent=[-1, 1, -1, 1])
            # plt.colorbar()
            # plt.contour(u, colors = 'white', extent=[-1, 1, -1, 1])
        if n == 1e5:
            break
        
    return u.reshape(nx+2*buff_size, ny+2*buff_size), err_arr, n



if __name__=='__main__':

    nx, ny = 4, 4
    buff_size=1
    tor = 1e-10
    boundary = np.zeros((4, nx + 2*buff_size))
    boundary[0] =np.ones(nx + 2*buff_size)
    mesh = Mesh2D(nx = nx, ny = ny, buff_size=buff_size)

    u = solve("Gauss", tor, boundary, mesh)[1]
    print(u)
    print("TEST")