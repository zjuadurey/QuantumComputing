# compute_utils.py -------------------------------------------------
import numpy as np
from numpy.fft import fft2, ifft2

def compute_fluid_quantities(psi1, psi2):
    """Return (rho, ux, uy, vorticity) for two-component wave-function."""
    N = psi1.shape[0]
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing='ij')

    psi1_k = fft2(psi1);  psi2_k = fft2(psi2)

    dpsi1_x = ifft2(1j*KX*psi1_k);  dpsi1_y = ifft2(1j*KY*psi1_k)
    dpsi2_x = ifft2(1j*KX*psi2_k);  dpsi2_y = ifft2(1j*KY*psi2_k)

    rho = np.abs(psi1)**2 + np.abs(psi2)**2
    ux  = (np.real(psi1)*np.imag(dpsi1_x) - np.imag(psi1)*np.real(dpsi1_x)
          +np.real(psi2)*np.imag(dpsi2_x) - np.imag(psi2)*np.real(dpsi2_x)) / rho
    uy  = (np.real(psi1)*np.imag(dpsi1_y) - np.imag(psi1)*np.real(dpsi1_y)
          +np.real(psi2)*np.imag(dpsi2_y) - np.imag(psi2)*np.real(dpsi2_y)) / rho
    vor = np.real(ifft2(1j*KX*fft2(uy) - 1j*KY*fft2(ux)))
    return rho, ux, uy, vor
