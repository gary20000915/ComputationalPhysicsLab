import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numba import jit, njit, prange
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
def kernel(u, u_temp, nx, ny, buff_size):
    for i in prange(1, nx + 2*buff_size - 1, 1):
        for j in prange(1, ny + 2*buff_size - 1, 1):
            u[i, j] = 0.25 * (u_temp[i+1, j] + u_temp[i, j+1] + u_temp[i-1, j] + u_temp[i, j-1])
    
def solve(tor, boundary, mesh: Mesh2D):
    u         = mesh.get_mesh()
    nx        = mesh.get_nx()
    ny        = mesh.get_ny()
    buff_size = mesh.get_buff_size()
    
    set_boundary(nx, ny, buff_size, boundary, u)
    
    err = 0
    n   = 0
    # print(u)
    # plt.imshow(u, origin = 'lower', extent=[-1, 1, -1, 1])
    # plt.colorbar()
    
    while err < tor:
        u_temp    = np.copy(u)
        
        kernel(u, u_temp, nx, ny, buff_size)
        # for i in range(1, nx + 2*buff_size - 1, 1):
        #     for j in range(1, ny + 2*buff_size - 1, 1):
        #         u[i, j] = 0.25 * (u_temp[i+1, j] + u_temp[i, j+1] + u_temp[i-1, j] + u_temp[i, j-1])
                
        err = np.sqrt(np.sum(np.square(u - u_temp))) / (nx * ny)
        n += 1
        # if n % 10 == 0:
            # print(err, tor)
            # print(u)
            # plt.imshow(u.reshape(nx+2*buff_size, ny+2*buff_size), origin = 'lower', extent=[-1, 1, -1, 1])
            # plt.colorbar()
            # plt.contour(u, colors = 'white', extent=[-1, 1, -1, 1])
        if n == 1e5:
            break
        
    return u.reshape(nx+2*buff_size, ny+2*buff_size)



if __name__=='__main__':

    nx, ny = 4, 4
    buff_size=1
    tor = 1
    boundary = np.zeros((4, nx + 2*buff_size))
    boundary[0] =np.ones(nx + 2*buff_size)
    mesh = Mesh2D(nx = nx, ny = ny, buff_size=buff_size)

    u = solve(tor, boundary, mesh)
    print("TEST")