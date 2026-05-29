import numpy as np

class PsychoacousticModel:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.min_db = -100.0

    def _masking_contribution(self, f_mask, level_db, f_target):
        if f_mask < 20 or f_target < 20:
            return self.min_db
        ratio = f_target / f_mask
        if ratio <= 0:
            return self.min_db
        octave_diff = np.log2(ratio)
        if octave_diff >= 0:          # upward (higher freqs)
            slope = -12.0
        else:                         # downward
            slope = -27.0
        return level_db + slope * abs(octave_diff)

    def compute_perceptual_spectrum(self, spectrum_db, freqs):
        n = len(freqs)
        perceptual = np.full(n, self.min_db)
        for i in range(n):
            level = spectrum_db[i]
            if level < -70:
                continue
            f = freqs[i]
            for j in range(n):
                contrib = self._masking_contribution(f, level, freqs[j])
                if contrib > perceptual[j]:
                    perceptual[j] = contrib
        # rough hearing threshold floor
        threshold = -15 - 0.008 * (freqs - 2000)**2 / 2000
        return np.maximum(perceptual, threshold)