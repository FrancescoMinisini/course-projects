import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny):
        self.px = Poisson(Lx, Nx)
        self.py = Poisson(Ly, Ny)

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        x = np.linspace(0, self.px.L, self.px.N+1)
        y = np.linspace(0, self.py.L, self.py.N+1)
        smesh = np.meshgrid(x, y, indexing='ij')
        return smesh

    def laplace(self):
        """Return a vectorized Laplace operator"""
        D2X = self.px.D2()
        D2Y = self.py.D2()
        return (sparse.kron(D2X, sparse.eye(self.py.N+1)) + sparse.kron(sparse.eye(self.px.N+1), D2Y))

    def assemble(self, f=None, g=None):
        """Return assembled coefficient matrix A and right hand side vector b"""
        xij, yij = self.create_mesh()
        F = sp.lambdify((x, y), f)(xij, yij)
        
        if g is None:
            G = np.zeros_like(F)
        else:
            G = sp.lambdify((x, y), g)(xij, yij)

        A = self.laplace()
        B = np.ones((self.px.N+1, self.py.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]

        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        
        b = F.ravel()
        b[bnds] = G.ravel()[bnds]

        return A , b 

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        xij, yij = self.create_mesh()
        return np.sqrt(self.px.dx*self.py.dx*np.sum((u - sp.lambdify((x, y), ue)(xij, yij))**2))


    def __call__(self, f=implemented_function('f', lambda x, y: 2)(x, y), g=None):
        """Solve Poisson's equation with a given right hand side function

        Parameters
        ----------
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(f=f, g=g)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

def test_poisson2d():
    um = x*(1-x)*y*(1-y) + x  
    f  = um.diff(x, 2) + um.diff(y, 2)
    g  = um  
    solver = Poisson2D(1, 1, 100, 100)
    u = solver(f, g=g)
    err = solver.l2_error(u, um)
    assert err < 5e-4, f"Error too large: {err}"


