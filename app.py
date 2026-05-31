import gradio as gr
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# ==================== PSYCHOACOUSTIC MODEL ====================
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
        slope = -12.0 if octave_diff >= 0 else -27.0
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
        threshold = -15 - 0.008 * (freqs - 2000)**2 / 2000
        return np.maximum(perceptual, threshold)

# ==================== PERCEPTUAL EQ ====================
class PerceptualEQ:
    def __init__(self, num_bands=10):
        self.num_bands = num_bands
        self.q = 1.4
        self.center_freqs = np.logspace(np.log10(30), np.log10(18000), num_bands)

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

    def apply_eq(self, audio, gains, fs):
        y = audio.copy()
        for i, f0 in enumerate(self.center_freqs):
            if abs(gains[i]) < 0.1:
                continue
            b, a = self._peaking_biquad(f0, gains[i], fs)
            y = signal.lfilter(b, a, y)
        return np.clip(y, -0.99, 0.99)

# ==================== GRADIO APP ====================
model = PsychoacousticModel()
eq = PerceptualEQ()
fs = 44100
N = 8192

def load_and_analyze(file):
    if file is None:
        # Built-in test signal
        t = np.linspace(0, 5, int(fs * 5), endpoint=False)
        audio = (0.5 * np.sin(2*np.pi*60*t) + 
                 0.4 * np.sin(2*np.pi*180*t) + 
                 0.3 * np.sin(2*np.pi*450*t) + 
                 0.25 * np.sin(2*np.pi*1200*t) + 
                 0.2 * np.sin(2*np.pi*3500*t) + 
                 0.15 * np.sin(2*np.pi*8000*t))
        audio = audio / np.max(np.abs(audio)) * 0.8
    else:
        import soundfile as sf
        audio, rate = sf.read(file)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != fs:
            audio = signal.resample(audio, int(len(audio) * fs / rate))
        audio = audio.astype(np.float32)[:int(30*fs)]
    
    chunk = audio[:N]
    fft = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(N, 1/fs)
    mags_db = 20 * np.log10(np.abs(fft) + 1e-12)
    perceptual = model.compute_perceptual_spectrum(mags_db, freqs)
    
    return audio, freqs, perceptual, "✅ Audio loaded. Move the sliders below to adjust EQ in real time."

def update_perceptual_spectrum(*gains_and_audio):
    gains = np.array(gains_and_audio[:10])
    audio = gains_and_audio[10]
    freqs = gains_and_audio[11]
    
    if audio is None:
        return None
    
    chunk = audio[:N]
    fft = np.fft.rfft(chunk)
    mags_db = 20 * np.log10(np.abs(fft) + 1e-12)
    
    # Apply current EQ to chunk
    processed_chunk = eq.apply_eq(chunk, gains, fs)
    new_fft = np.fft.rfft(processed_chunk)
    new_mags = 20 * np.log10(np.abs(new_fft) + 1e-12)
    new_perceptual = model.compute_perceptual_spectrum(new_mags, freqs)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogx(freqs, new_perceptual, color="#d62728", linewidth=2.5, label="Perceptual Spectrum (what you hear)")
    ax.set_xlim(20, 20000)
    ax.set_ylim(-95, 8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Perceived Level (dB)")
    ax.set_title("Real-Time Psychoacoustic Spectrum + EQ")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def process_full_audio(audio, *gains):
    gains = np.array(gains)
    if audio is None:
        return None, None
    processed = eq.apply_eq(audio, gains, fs)
    return processed, "✅ EQ applied. Play the processed audio below."

# ==================== BUILD INTERFACE ====================
with gr.Blocks(title="Psychoacoustic EQ - Real Time", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎧 Psychoacoustic EQ (Real-Time)")
    gr.Markdown("**See what you actually hear** (with ear masking) • Move sliders → spectrum updates instantly • Works like normal EQ software")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Upload WAV (or use test signal)", type="filepath")
            load_btn = gr.Button("🔄 Load & Analyze", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### EQ Sliders (move to hear & see changes)")
            
            sliders = []
            band_names = ["60 Hz", "170 Hz", "450 Hz", "1.2 kHz", "3.5 kHz", 
                          "8 kHz", "12 kHz", "16 kHz", "18 kHz", "20 kHz"]
            for i, name in enumerate(band_names):
                s = gr.Slider(minimum=-15, maximum=15, value=0, step=0.5, 
                              label=name, interactive=True)
                sliders.append(s)
        
        with gr.Column(scale=2):
            spectrum_plot = gr.Plot(label="Live Perceptual Spectrum")
            
            with gr.Row():
                play_original = gr.Audio(label="Original Audio", interactive=False)
                play_processed = gr.Audio(label="EQ'd Audio (after sliders)", interactive=False)
            
            apply_btn = gr.Button("🚀 Apply EQ to Full File & Play", variant="secondary")

    # State variables
    audio_state = gr.State()
    freqs_state = gr.State()

    # Load button
    load_btn.click(
        fn=load_and_analyze,
        inputs=[audio_input],
        outputs=[audio_state, freqs_state, spectrum_plot, status]
    ).then(
        fn=lambda a: a,
        inputs=[audio_state],
        outputs=[play_original]
    )

    # Live update when any slider moves
    for s in sliders:
        s.change(
            fn=update_perceptual_spectrum,
            inputs=sliders + [audio_state, freqs_state],
            outputs=[spectrum_plot]
        )

    # Apply full EQ
    apply_btn.click(
        fn=process_full_audio,
        inputs=[audio_state] + sliders,
        outputs=[play_processed, status]
    )

    gr.Markdown("**Tip:** Move the sliders — the red spectrum line updates in real time showing exactly what your ears will hear after masking.")

demo.launch(share=False)