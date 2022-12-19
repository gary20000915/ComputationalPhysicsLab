"""
This file define classes for generating 2D meshes.

"""
import numpy as np
from numba import jit, int32, float64
from numba.experimental import jitclass

class Mesh2D:
    def __init__(self, nx:int = 10, ny:int = 10, buff_size = 1, xmin = 0, xmax = 1, ymin = 0, ymax = 1):
        self._nx = nx
        self._ny = ny
        self._nbuff = buff_size
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax   
        
        self._setup()
        
        return

    def _setup(self):
        self._dx = (self._xmax - self._xmin) / (self._nx + 1)
        self._dy = (self._ymax - self._ymin) / (self._ny + 1)
        
        self._istart = self._nbuff
        self._istartGC = 0
        self._iend = self._nbuff + self._nx - 1
        self._iendGC = 2 * self._nbuff + self._nx - 1
        self._nxGC = 2 * self._nbuff + self._nx
        
        self._jstart = self._nbuff
        self._jstartGC = 0
        self._jend = self._nbuff + self._ny - 1
        self._jendGC = 2 * self._nbuff + self._ny - 1
        self._nyGC = 2 * self._nbuff + self._ny
        
        x = np.linspace(self._xmin, self._xmax, self._nxGC)
        y = np.linspace(self._ymin, self._ymax, self._nyGC)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self._mesh = xx * 0
        self._xx = xx
        self._yy = yy
        self._x = x
        self._y = y
        
        return
        
    def get_nx(self):
        return self._nx
    def set_nx(self, nx):
        self._nx = nx
        self._setup()
        return

    def get_ny(self):
        return self._ny
    def set_ny(self, ny):
        self._ny = ny
        self._setup()
        return

    def get_buff_size(self):
        return self._nbuff
    def set_buff_size(self, buff_size):
        self._nbuff = buff_size
        self._setup()
        return

    def get_xmin(self):
        return self._xmin
    def set_xmin(self, xmin):
        self._xmin = xmin
        self._setup()
        return 

    def get_xmax(self):
        return self._xmax
    def set_xmax(self, xmax):
        self._xmax = xmax
        self._setup()
        return 

    def get_ymin(self):
        return self._ymin
    def set_ymin(self, ymin):
        self._ymin = ymin
        self._setup()
        return

    def get_ymax(self):
        return self._ymax
    def set_ymax(self, ymax):
        self._ymax = ymax
        self._setup()
        return

    def get_istart(self):
        return self._istart
    def set_istart(self, istart):
        self._istart = istart
        self._setup()
        return

    def get_istartGC(self):
        return self._istartGC
    def set_istartGC(self, istartGC):
        self._istartGC = istartGC
        self._setup()
        return

    def get_iend(self):
        return self._iend
    def set_iend(self, iend):
        self._iend = iend
        self._setup()
        return
    
    def get_iendGC(self):
        return self._iendGC
    def set_iendGC(self, iendGC):
        self._iendGC = iendGC
        self._setup()
        return

    def get_jstart(self):
        return self._jstart
    def set_jstart(self, jstart):
        self._jstart = jstart
        self._setup()
        return

    def get_jstartGC(self):
        return self._jstartGC
    def set_jstartGC(self, jstartGC):
        self._jstartGC = jstartGC
        self._setup()
        return

    def get_jend(self):
        return self._jend
    def set_jend(self, jend):
        self._jend = jend
        self._setup()
        return
    
    def get_jendGC(self):
        return self._jendGC
    def set_jendGC(self, jendGC):
        self._jendGC = jendGC
        self._setup()
        return

    def get_mesh(self):
        return self._mesh
    def set_mesh(self, mesh):
        self._mesh = mesh
        if (mesh.size != self._mesh.size): 
            print("error! size conflict!")
        return
    
    def get_xx(self):
        return self._xx
    def set_xx(self, xx):
        self._xx = xx
        self._setup()
        return

    def get_yy(self):
        return self._yy
    def set_yy(self, yy):
        self._yy = yy
        self._setup()
        return
    
    def get_x(self):
        return self._x
    def set_x(self, x):
        self._x = x
        self._setup()
        return

    def get_y(self):
        return self._y
    def set_y(self, y):
        self._y = y
        self._setup()
        return

    def get_nx(self):
        return self._nx
    def set_nx(self, nx):
        self._nx = nx
        self._setup()
        return
    
    def get_ny(self):
        return self._ny
    def set_ny(self, ny):
        self._ny = ny
        self._setup()
        return


if __name__=='__main__':
    mesh = Mesh2D(nx = 3, ny = 3, buff_size=1)
    # mesh.set_nx(32)
    # mesh.set_ny(32)
    
    u = mesh.get_mesh()
    nx = mesh.get_nx()
    ny = mesh.get_ny()
    buff = mesh.get_buff_size()

    print(u)
    print(f"Testing ... nx={nx}, ny={ny}, buff ={buff}")
    print('Done')