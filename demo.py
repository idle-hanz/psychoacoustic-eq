import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

from psychoacoustic_model import PsychoacousticModel
from perceptual_eq import PerceptualEQ


def generate_test_signal(duration=4.0, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = (0.55 * np.sin(2*np.pi*55*t) +
           0.45 * np.sin(2*np.pi*180*t) +
           0.35 * np.sin(2*np.pi*450*t) +
           0.30 * np.sin(2*np.pi*1200*t) +
           0.25 * np.sin(2*np.pi*3500*t) +
           0.20 * np.sin(2*np.pi*8500*t) +
           0.08 * np.random.randn(len(t)))
    return (sig / np.max(np.abs(sig)) * 0.75).astype(np.float32), fs


def main():
    print("=== Psychoacoustic EQ Demo ===")
    fs = 44100
    audio, fs = generate_test_signal()

    N = 4096
    chunk = audio[:N]
    fft = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(N, 1/fs)
    mags_db = 20 * np.log10(np.abs(fft) + 1e-12)

    model = PsychoacousticModel(fs)
    current = model.compute_perceptual_spectrum(mags_db, freqs)

    # Plot current perceptual spectrum
    plt.figure(figsize=(14, 7))
    plt.semilogx(freqs, current, label="Current Perceptual Spectrum", color="#1f77b4", lw=2)
    plt.title("Psychoacoustic Spectrum — What it actually sounds like in your head")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Perceived Level (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlim(20, 20000)
    plt.ylim(-90, 5)
    plt.tight_layout()
    plt.show(block=False)

    print("\n>>> Click points on the plot to draw your TARGET curve, then close the window.")

    target_points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close('all')

    if len(target_points) < 2:
        target_points = [(25, -25), (80, -8), (300, -18), (1200, -22), (4500, -10), (12000, -15), (18000, -28)]

    tf = np.array([p[0] for p in target_points])
    tl = np.array([p[1] for p in target_points])
    target = np.interp(freqs, tf, tl, left=tl[0], right=tl[-1])

    eq = PerceptualEQ()
    print("\nOptimizing EQ gains...")
    best_gains, err = eq.solve(model, target, mags_db, freqs, fs)
    print(f"✓ Done. Error = {err:.2f} dB²")
    for f, g in zip(eq.center_freqs, best_gains):
        print(f"  {f:7.0f} Hz : {g:+6.1f} dB")

    # Apply to full audio
    sos_list = []
    for i, f0 in enumerate(eq.center_freqs):
        if abs(best_gains[i]) > 0.1:
            b, a = eq._peaking_biquad(f0, best_gains[i], fs)
            sos_list.append(signal.tf2sos(b, a))
    processed = signal.sosfilt(np.vstack(sos_list), audio) if sos_list else audio.copy()
    processed = np.clip(processed, -0.99, 0.99)
    wavfile.write("output.wav", fs, (processed * 32767).astype(np.int16))
    print("Saved output.wav")

    # Comparison plots
    new_mags = 20 * np.log10(np.abs(np.fft.rfft(processed[:N])) + 1e-12)
    new_per = model.compute_perceptual_spectrum(new_mags, freqs)

    plt.figure(figsize=(14, 9))
    plt.subplot(2,1,1)
    plt.semilogx(freqs, current, label="Original", alpha=0.7)
    plt.semilogx(freqs, target, label="Your Target", color="green", ls="--", lw=2)
    plt.semilogx(freqs, new_per, label="After EQ", color="red", lw=2)
    plt.legend()
    plt.title("Perceptual Spectrum")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlim(20, 20000)

    plt.subplot(2,1,2)
    plt.semilogx(freqs, mags_db, label="Original FFT", alpha=0.5)
    plt.semilogx(freqs, new_mags, label="After EQ FFT", alpha=0.85)
    plt.legend()
    plt.title("Technical Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlim(20, 20000)
    plt.tight_layout()
    plt.show()

    print("\n✓ Done! Listen to output.wav — the red line should hug your green target.")


if __name__ == "__main__":
    main()