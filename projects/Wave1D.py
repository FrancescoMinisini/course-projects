import numpy as np
import sympy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from numbers import Number

x, t, c, L = sp.symbols("x,t,c,L")


class Wave1D:
    """Class for solving the wave equation

    Parameters
    ----------
    N : int
        Number of uniform spatial intervals
    L0 : number
        The extent of the domain, which is [0, L]
    c0 : number, optional
        The wavespeed
    cfl : number, optional
        CFL number
    u0 : Sympy function of x, t, c and L
        Used for specifying initial condition
    """

    def __init__(
        self,
        N: int,
        L0: Number = 1.0,
        c0: Number = 1,
        cfl: Number = 1,
        u0: sp.Expr = sp.exp(-200 * (x - L / 2 + c * t) ** 2),
    ):
        self.N = N
        self.L = L0
        self.c = c0
        self.cfl = cfl
        self.x = np.linspace(0, L0, N + 1)
        self.dx = L0 / N
        self.u0 = u0
        self.unp1 = np.zeros(N + 1)
        self.un = np.zeros(N + 1)
        self.unm1 = np.zeros(N + 1)

    def D2(self, bc: dict[str, int]) -> sparse.spmatrix:
        """Return second order differentiation matrix

        Paramters
        ---------
        bc : int
            Boundary condition
            - 0 Dirichlet
            - 1 Neumann
            - 2 Open boundary
            - 3 periodic
        Note
        ----
        The returned matrix is not divided by dx**2
        """
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), "lil")
        
        coeff = [-2 , 2]
        
        left =bc["left"]
        right = bc["right"]
        if left == 3 and right == 3:
            D[0, :]  = 0.0
            D[0, 0]  = -2.0
            D[0, 1]  =  1.0
            D[0, -2] =  1.0

            D[-1, :]  = 0.0
            D[-1, -1] = -2.0
            D[-1, -2] =  1.0

        elif left == 3 or right == 3:
            raise RuntimeError(f"Periodic bound cond cannot be mixed bc = [left:{left}, right:{right}]")
        elif left == 0:
            D[0, :] = 0.0
        elif left == 1:
            D[0, :] = 0.0
            D[0 , 0:2] = coeff
        elif left == 2:
            pass
        else: raise RuntimeError(f"Wrong bc =  [left : {left} , right : {right}]")
        
        if right == 3:
            pass
        elif right == 0:
            D[-1, :] = 0.0
        elif right == 1:
            D[-1, : ] = 0.0
            D[-1 , -2:] = coeff[::-1]
        elif right == 2:
            pass
        else: raise RuntimeError(f"Wrong bc =  [left : {left} , right : {right}]")
        
        return D

    def apply_bcs(self, bc: dict[str, int], u: np.ndarray | None = None):
        """Apply boundary conditions to solution vector
        
        Parameters
        ----------
        bc : int
            Boundary condition in space
            - 0 Dirichlet
            - 1 Neumann
            - 2 Open boundary
            - 3 periodic
        u : array, optional
            The solution array to fix at boundaries
            If not provided, use self.unp1

        """
        u = u if u is not None else self.unp1
        left = bc["left"] 
        right = bc["right"]
        
        C = float(self.cfl)
        un   = self.un      # u^n
        unm1 = self.unm1    # u^{n-1}

        if left == 3 or right ==3:
            if left == 3 and right == 3:
                u[-1] = u[0] 
            else: raise  RuntimeError(f"Periodic bound cond cannot be mixed bc = [left : {left} , right : {right}]")
        elif  left == 0:
            u[0] = 0.0
        elif left == 1:
            pass
        elif left == 2:
            u[0] = 2.0*(1.0 - C)*un[0] - ((1.0 - C)/(1.0 + C))*unm1[0] + (2.0*C**2/(1.0 + C))*un[1]
        else: raise RuntimeError(f"Wrong bc =  [left : {left} , right : {right}]")

        if right == 3:
            pass
        elif  right == 0:
            u[-1] = 0.0
        elif right == 1:
            pass
        elif right == 2:
            u[-1] = 2.0*(1.0 - C)*un[-1] - ((1.0 - C)/(1.0 + C))*unm1[-1] + (2.0*C**2/(1.0 + C))*un[-2]
        else:
            raise RuntimeError(f"Wrong bc =  [left : {left} , right : {right}]")

    @property
    def dt(self) -> float:
        return self.cfl * self.dx / self.c

    def __call__(
        self,
        Nt: int,
        cfl: Number | None = None,
        bc: dict[str , int] = {"left" : 0, "right" : 0},
        ic: int = 0,
        save_step: int = 100,
    ) -> dict[int, np.ndarray]:
        """Solve wave equation

        Parameters
        ----------
        Nt : int
            Number of time steps
        cfl : number
            CFL number
        bc : int, optional
            Boundary condition in space
            - 0 Dirichlet
            - 1 Neumann
            - 2 Open boundary
            - 3 periodic
        ic : int, optional
            Initial conditions
            - 0 Specify un = u(x, t=0) and unm1 = u(x, t=-dt)
            - 1 Specify un = u(x, t=0) and u_t(x, t=0) = 0
        save_step : int, optional
            Save solution every save_step time step

        Returns
        -------
        Dictionary with key, values as timestep, array of solution
        The number of items in the dictionary is Nt/save_step, and
        each value is an array of length N+1

        """
        D = self.D2(bc)
        self.cfl = C = self.cfl if cfl is None else cfl
        dt = self.dt
        u0 = sp.lambdify(x, self.u0.subs({L: self.L, c: self.c, t: 0}))

        # First step. Set un and unm1
        self.unm1[:] = u0(self.x)  # unm1 = u(x, 0)
        plotdata = {0: self.unm1.copy()}
        if ic == 0:  # use sympy function for un = u(x, dt)
            u0 = sp.lambdify(x, self.u0.subs({L: self.L, c: self.c, t: dt}))
            self.un[:] = u0(self.x)

        else:  # use u_t = 0 for un = u(x, dt)
            self.un[:] = self.unm1 + 0.5 * C**2 * (D @ self.unm1)
            self.apply_bcs(bc, self.un)
        if save_step == 1:
            plotdata[1] = self.un.copy()

        for n in range(2, Nt + 1):
            self.unp1[:] = 2 * self.un - self.unm1 + C**2 * (D @ self.un)
            self.apply_bcs(bc)
            self.unm1[:] = self.un
            self.un[:] = self.unp1
            if n % save_step == 0:  # save every save_step timestep
                plotdata[n] = self.unp1.copy()

        return plotdata

    def plot_with_offset(self, data: dict[int, np.ndarray]) -> None:
        v = np.array(list(data.values()))
        t = np.array(list(data.keys()))
        dt = t[1] - t[0]
        v0 = abs(v).max()
        fig = plt.figure(facecolor="k")
        ax = fig.add_subplot(111, facecolor="k")
        for i, u in data.items():
            ax.plot(self.x, u + i * v0 / dt, "w", lw=2, zorder=i)
            ax.fill_between(
                self.x, u + i * v0 / dt, i * v0 / dt, facecolor="k", lw=0, zorder=i - 1
            )
        plt.show()

    def animation(self, data: dict[int, np.ndarray]) -> None:
        from matplotlib import animation

        fig, ax = plt.subplots()
        v = np.array(list(data.values()))
        t = np.array(list(data.keys()))
        save_step = t[1] - t[0]
        (line,) = ax.plot(self.x, data[0])
        ax.set_ylim(v.min(), v.max())

        def update(frame):
            line.set_ydata(data[frame * save_step])
            return (line,)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data), blit=True)
        ani.save(
            "wavemovie.apng", writer="pillow", fps=5
        )  # This animated png opens in a browser
        ani.to_jshtml()
        plt.show()


def test_pulse_bcs():
    sol = Wave1D(100, cfl=1, L0=2, c0=1)
    data = sol(100, bc= {"left" : 0 , "right" : 0}, ic=0, save_step=100)
    assert np.linalg.norm(data[0] + data[100]) < 1e-12
    data = sol(100, bc= {"left" : 0 , "right" : 0}, ic=1, save_step=100)
    assert np.linalg.norm(data[0] + data[100]) < 1e-12
    data = sol(100, bc= {"left" : 1 , "right" : 1}, ic=0, save_step=100)
    assert np.linalg.norm(data[0] - data[100]) < 1e-12
    data = sol(100, bc= {"left" : 1 , "right" : 1}, ic=1, save_step=100)
    assert np.linalg.norm(data[0] - data[100]) < 1e-12
    data = sol(100, bc= {"left" : 2 , "right" : 2}, ic=0, save_step=100)
    assert np.linalg.norm(data[100]) < 1e-12
    data = sol(100, bc={"left" : 2 , "right" : 2}, ic= 1, save_step=100)
    assert np.linalg.norm(data[100]) < 1e-12
    data = sol(100, bc= {"left" : 3, "right" : 3}, ic=0, save_step=100)
    assert np.linalg.norm(data[0] - data[100]) < 1e-12
    data = sol(100, bc= {"left" : 3, "right" : 3}, ic=1, save_step=100)
    assert np.linalg.norm(data[0] - data[100]) < 1e-12


if __name__ == "__main__":
    bc = {"left" : 3 , "right" : 3}
    sol = Wave1D(100, cfl=1, L0=2, c0=1)
    data = sol(Nt = 100, bc = bc, save_step=3, ic=0)
    sol.animation(data)
    # test_pulse_bcs()
    # data = sol(200, bc=2, ic=0, save_step=100)
