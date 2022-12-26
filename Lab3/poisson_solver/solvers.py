import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numba import njit, prange
from .mesh import Mesh2D

"""
Solver to solve for Laplace/Poisson's equation

"""

def set_boundary(nx, ny, buff_size, mesh: Mesh2D):
    boundary    = np.zeros((4, nx + 2*buff_size))
    boundary[1] = np.ones(nx + 2*buff_size)
    mesh[0, :]  = boundary[0]
    mesh[ny+buff_size, :] = boundary[1]
    mesh[:, 0]  = boundary[2]
    mesh[:, nx+buff_size] = boundary[3]

@njit(parallel = True)
def j_kernel(u, u_temp, nx, ny, buff_size):
    for i in prange(1, int(nx + 2*buff_size - 1), 1):
        for j in prange(1, int(ny + 2*buff_size - 1), 1):
            u[i, j] = 0.25 * (u_temp[i+1, j] + u_temp[i, j+1] + u_temp[i-1, j] + u_temp[i, j-1])
    return u
    
@njit(parallel = True)
def gs_kernel(u, nx, ny, buff_size):
    for i in prange(1, int(nx + 2*buff_size) - 1, 1):
        for j in prange(1, int(ny + 2*buff_size - 1), 1):
            u[i, j] = 0.25 * (u[i+1, j] + u[i, j+1] + u[i-1, j] + u[i, j-1])
    return u

@njit(parallel = True)
def SOR_kernel(u, u_temp, nx, ny, buff_size, w):
    for i in prange(1, int(nx + 2*buff_size - 1), 1):
        for j in prange(1, int(ny + 2*buff_size - 1), 1):
            u[i, j] = 0.25 * (u[i+1, j] + u[i, j+1] + u[i-1, j] + u[i, j-1])
            u[i, j] = (1-w) * u_temp[i,j] + w * u[i, j]
    return u

def res(u, nx, ny, buff_size, Mesh2D):
    nx = int(nx * 0.5)
    ny = int(ny * 0.5)
    r = Mesh2D(nx = nx, ny = ny, buff_size=buff_size).get_mesh()
    for i in range(1, nx-1, 2):
        ii = int((i - 1) * 0.5)
        for j in range(1, ny-1, 2):
            jj = int((j - 1) * 0.5)
            r[ii, jj] = 0.25 * (u[i,j] + u[i+1,j] + u[i,j+1] + u[i+1,j+1])
            
    set_boundary(nx, ny, buff_size, r)
    # r = r.reshape(int(0.5 * (nx - 2*buff_size)), int(0.5 * (ny - 2*buff_size)))
    return [r, nx, ny]

def pro(u, nx, ny, buff_size, Mesh2D):
    # p = np.zeros(int(2 * (nx + ny))).reshape(int(2*nx), int(2*ny))
    nx = int(nx * 2)
    ny = int(ny * 2)
    p = Mesh2D(nx = nx, ny = ny, buff_size=buff_size).get_mesh()
    for i in range(1, nx, 2):
        ii = int((i - 1) * 0.5)
        for j in range(1, ny, 2):
            jj = int((j - 1) * 0.5)
            p[i  , j ] = u[ii,jj]
            p[i+1,j  ] = u[ii,jj]
            p[i+1,j+1] = u[ii,jj]
            p[i  ,j+1] = u[ii,jj]
            
    set_boundary(nx, ny, buff_size, p)
    return [p, nx, ny]

def muti(name, u_temp, u, nx, ny, w, buff_size):
    
    if name == "Jacobi":
        u = j_kernel(u, u_temp, nx, ny, buff_size)
    elif name == "Gauss":
        u = gs_kernel(u, nx, ny, buff_size)
    elif name == "SOR":
        u = SOR_kernel(u, u_temp, nx, ny, buff_size, w)
    else:
        print("Error: unknown kernel!")
    
    return u

def solve(name, tor, mesh: Mesh2D, w):
    u         = mesh.get_mesh()
    nx        = mesh.get_nx()
    ny        = mesh.get_ny()
    buff_size = mesh.get_buff_size()
    
    set_boundary(nx, ny, buff_size, u)
    
    err     = 10
    err_arr = np.array([])
    n       = 0
    
    while err > tor:
        
        u_temp = np.copy(u)
        
        if name == "Jacobi":
            u = j_kernel(u, u_temp, nx, ny, buff_size)
        elif name == "Gauss":
            u = gs_kernel(u, nx, ny, buff_size)
        elif name == "SOR":
            u = SOR_kernel(u, u_temp, nx, ny, buff_size, w)
        elif name == "multi":
            if n == 0:
                u = res(u, nx, ny, buff_size, Mesh2D)
                u_temp = np.copy(u[0])
                nx, ny = u[1], u[2]
                ans = u[0]
                print("start for MultiGrid")
            if n == 10:
                u = res(u, nx, ny, buff_size, Mesh2D)
                u_temp = np.copy(u[0])
                nx, ny = u[1], u[2]
                ans = u[0]
            if n == 20:
                u = res(u, nx, ny, buff_size, Mesh2D)
                u_temp = np.copy(u[0])
                nx, ny = u[1], u[2]
                ans = u[0]
            if n == 30:
                u = pro(u, nx, ny, buff_size, Mesh2D)
                u_temp = np.copy(u[0])
                nx, ny = u[1], u[2]
                ans = u[0]
            if n == 40:
                u = pro(u, nx, ny, buff_size, Mesh2D)
                u_temp = np.copy(u[0])
                nx, ny = u[1], u[2]
                ans = u[0]
            if n == 50:
                u = pro(u, nx, ny, buff_size, Mesh2D)
                u_temp = np.copy(u[0])
                nx, ny = u[1], u[2]
                ans = u[0]
                print("end for MultiGrid")
                
            u = muti("SOR", u_temp, ans, nx, ny, w, buff_size)
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
        if n == 1e4:
            print("Warning too many steps!")
            break
        
    # return u.reshape(nx+2*buff_size, ny+2*buff_size), err_arr, n
    return u.reshape(nx + 2*buff_size, ny + 2*buff_size), err_arr, n



if __name__=='__main__':

    nx, ny = 100, 100
    buff_size=1
    tor = 0
    mesh = Mesh2D(nx = nx, ny = ny, buff_size=buff_size)

    u = solve("multi", tor, mesh, 1.2)[1]
    print(u)
    print("TEST")