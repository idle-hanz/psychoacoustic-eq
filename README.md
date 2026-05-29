# Psychoacoustic EQ

Exactly what you described — a spectrum that shows what you *actually hear* (with masking) and an EQ where you draw the curve you want and it figures out the filters.

Run `python demo.py`, click the points on the plot to draw your target curve, and watch it solve.

Full details in the code comments. This is the clean, working prototype.

## Installation

```bash
git clone https://github.com/idle-hanz/psychoacoustic-eq.git
cd psychoacoustic-eq
pip install numpy scipy matplotlib
python demo.py
```

## How it works

1. Generates a test signal with several tones
2. Computes the **perceptual spectrum** (FFT + masking model)
3. Lets you click to draw your desired perceived curve
4. Solves for the 10-band parametric EQ gains that achieve it
5. Applies the EQ and saves `output.wav`

This is Stage 1 + 2 exactly as you asked. We can add real-time audio input, better GUI, etc. later.