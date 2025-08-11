import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm.notebook import tqdm
from pyfftw.interfaces import numpy_fft as fft
import pyfftw
pyfftw.config.NUM_THREADS = 8
pyfftw.interfaces.cache.enable()

class Wave2D:
    def __init__(self, nx=200, ny=200, Lx=50.0, Ly=50.0, dt=0.02, damping=0.99):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dt = dt
        self.damping = damping

        # Grid
        self.x = np.linspace(0, Lx, nx, endpoint=False)
        self.y = np.linspace(0, Ly, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Wavenumbers
        self.kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
        self.ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.K = np.sqrt(self.KX ** 2 + self.KY ** 2)
        self.K[0, 0] = 1e-9  # avoid division by zero

        # Initial condition: Zero everywhere
        self.eta = np.zeros((nx, ny))
        self.eta_t = np.zeros_like(self.eta)
        self.eta_prev = self.eta - dt * self.eta_t  # back step for leapfrog

        self.omega = self.compute_omega()


    def fft2(self, f):
        return fft.fft2(f)

    def ifft2(self, F):
        return fft.ifft2(F).real

    def step(self, eta, eta_prev):
        eta_hat = self.fft2(eta)
        accel = self.ifft2(- (self.omega ** 2) * eta_hat)
        eta_next = 2 * eta - eta_prev + self.dt ** 2 * accel
        eta_next *= self.damping  # Apply uniform damping
        return eta_next

    def iterate(self):
        eta_next = self.step(self.eta, self.eta_prev)
        self.eta_prev, self.eta = self.eta, eta_next

    def add_impulse(self, x, y, amplitude, sigma):
        amp = amplitude / (2 * np.pi * sigma ** 2)
        s = int(3 * sigma)
        for i in range(-s, s+1):
            for j in range(-s, s+1):
                a = amp * np.exp(-(i**2 + j**2) / (2 * sigma**2))
                self.eta[x + i, y + j] += a


class WaterWave2D(Wave2D):
    def __init__(self, nx=200, ny=200, Lx=50.0, Ly=50.0, g=9.81, h=100.0, dt=0.02, damping=0.99):
        self.g = g
        self.h = h
        super().__init__(nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, damping=damping)

    def compute_omega(self):
        return np.sqrt(self.g * self.K * np.tanh(self.K * self.h))


class LinearWave2D(Wave2D):
    def __init__(self, nx=200, ny=200, Lx=50.0, Ly=50.0, dt=0.02, damping=0.99, speed=10.0):
        self.speed = speed
        super().__init__(nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, damping=damping)

    def compute_omega(self):
        return self.speed * self.K
