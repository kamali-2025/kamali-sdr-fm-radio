# Kamali SDR FM Radio Receiver, Live Spectrum Waterfall & Auto-Scanner

A real-time **Software-Defined Radio (SDR) FM Broadcast Receiver**, **FFT Spectrum & Waterfall Analyzer**, and **Automatic Band Scanner** built in Python.

Designed with a **Dual-Mode RF Engine**, the application works immediately out of the box using a built-in **Multi-Station Complex I/Q RF Simulator** (complete with carrier peaks, frequency modulation, 19 kHz stereo pilot tones, and thermal AWGN noise), while seamlessly switching to a physical **RTL-SDR USB dongle** whenever hardware is connected.

---

## Key Features

- **Real-Time DSP FM Demodulation Chain:**
  - Complex baseband frequency shifting and Butterworth low-pass channel filtering (`scipy.signal`)
  - Multi-stage decimation from high-rate RF I/Q samples (`1.024 MSPS`) down to `48 kHz` audio
  - Phase-difference FM discriminator: `np.angle(z[1:] * np.conj(z[:-1]))`
  - Standard `75 us` FM de-emphasis filtering and live audio output via `sounddevice`
- **Live Spectrum & Waterfall GUI (`matplotlib` + `tkinter`):**
  - Real-time Power Spectral Density (PSD) RF graph across a `2.048 MHz` span
  - **Click-to-Tune:** Click directly on any frequency peak in the spectrum plot to tune immediately
  - Scrolling RF Waterfall spectrogram showing signal strength over time
- **Automatic FM Band Scanner (`87.5 - 108.0 MHz`):**
  - Sweeps the FM broadcast band, detects active carriers above the noise floor, and populates clickable station presets
- **Hardware + Simulation Dual Mode:**
  - Automatically detects a connected **RTL-SDR** device (`pyrtlsdr`) or runs in **Demo RF Simulation Mode** with zero physical hardware required

---

## Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kamali-2025/kamali-sdr-fm-radio.git
   cd kamali-sdr-fm-radio
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy scipy matplotlib sounddevice
   ```
   *(Optional, if using physical RTL-SDR USB hardware: `pip install pyrtlsdr`)*

3. **Run the application:**
   ```bash
   python kamali_sdr_fm_radio.py
   ```

---

## Tech Stack

- **Language:** Python 3
- **DSP & Math:** NumPy, SciPy (`scipy.signal`)
- **GUI & Visualization:** Tkinter, Matplotlib (`TkAgg`)
- **Audio Streaming:** SoundDevice (`PortAudio`)
- **SDR Hardware Interface:** `pyrtlsdr` (with built-in Complex I/Q RF Simulator fallback)

---

## Author

Built by **Kamali** ([github.com/kamali-2025](https://github.com/kamali-2025))
