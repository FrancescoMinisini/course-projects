import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    size = len(u)
    u_prime = np.zeros(size)
    u_prime[0] = (u[1] - u[0]) / dt
    u_prime[-1] = (u[-1] - u[-2]) / dt
    for i in range(1, size - 1):
        u_prime[i] = (u[i + 1] - u[i - 1]) / (2 * dt)
    return u_prime


def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    size = len(u)
    u_prime = np.zeros(size)
    u_prime[0] = (u[1] - u[0]) / dt
    u_prime[-1] = (u[-1] - u[-2]) / dt
    u_prime[1:-1] = (u[2:]-u[:-2])/(2*dt)
    return u_prime



def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
    