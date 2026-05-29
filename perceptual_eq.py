import numpy as np
from scipy import signal, optimize

class PerceptualEQ:
    def __init__(self, num_bands=10, min_freq=30, max_freq=18000, q=1.4):
        self.num_bands = num_bands
        self.q = q
        self.center_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands)
        self.gain_limits = (-15.0, 15.0)

    def _peaking_biquad(self, f0, gain_db, fs):
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2 * self.q)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])
        return b, a

    def compute_magnitude_response(self, gains, freqs, fs):
        H = np.ones(len(freqs))
        for i, f0 in enumerate(self.center_freqs):
            if abs(gains[i]) < 0.05:
                continue
            b, a = self._peaking_biquad(f0, gains[i], fs)
            _, h = signal.freqz(b, a, worN=freqs, fs=fs)
            H *= np.abs(h)
        return H

    def apply_eq_to_spectrum_db(self, spectrum_db, gains, freqs, fs):
        H = self.compute_magnitude_response(gains, freqs, fs)
        return spectrum_db + 20 * np.log10(np.maximum(H, 1e-12))

    def perceptual_error(self, gains, model, target_perceptual, current_spectrum_db, freqs, fs):
        new_spec = self.apply_eq_to_spectrum_db(current_spectrum_db, gains, freqs, fs)
        new_perceptual = model.compute_perceptual_spectrum(new_spec, freqs)
        mask = (freqs > 20) & (freqs < 20000)
        return np.mean((new_perceptual[mask] - target_perceptual[mask]) ** 2)

    def solve(self, model, target_perceptual, current_spectrum_db, freqs, fs, initial_gains=None):
        if initial_gains is None:
            initial_gains = np.zeros(self.num_bands)
        bounds = [self.gain_limits] * self.num_bands
        result = optimize.minimize(
            lambda g: self.perceptual_error(g, model, target_perceptual, current_spectrum_db, freqs, fs),
            initial_gains, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 60, 'ftol': 1e-4}
        )
        return result.x, result.fun