#!/usr/bin/env python3
"""
Kamali SDR FM Radio Receiver, Live Spectrum Waterfall & Auto-Scanner
--------------------------------------------------------------------
Portfolio Project: Real-Time Software-Defined Radio (DSP + GUI)

Features:
1. Dual-Mode RF Engine:
   - Hardware Mode: Connects to a USB RTL-SDR dongle via `pyrtlsdr`.
   - Built-in RF Simulator Mode: Synthesizes realistic complex I/Q radio waves
     (multiple FM broadcast stations, 19 kHz stereo pilot tones, frequency
     modulation, and AWGN RF noise floor) so the full DSP pipeline and audio
     demodulator run with zero physical hardware required.
2. Real-Time DSP Demodulation Chain:
   - Complex frequency shifting & FIR/Butterworth low-pass channel filtering
   - Decimation to intermediate baseband
   - Phase-difference FM discriminator: np.angle(z[1:] * np.conj(z[:-1]))
   - 75 us FM de-emphasis filter & audio resampling to 48 kHz
3. Interactive Matplotlib Dark-Theme RF GUI:
   - Live FFT Power Spectral Density (PSD) plot with click-to-tune
   - Scrolling RF Waterfall spectrogram
   - Automatic FM Band Scanner (87.5 - 108.0 MHz) with peak detection
   - Station preset selector, frequency step buttons, squelch, and volume/mute
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from scipy import signal

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

try:
    from rtlsdr import RtlSdr
    HAS_RTLSDR = True
except Exception:
    HAS_RTLSDR = False


# Simulated local FM broadcast stations (MHz, Station Name, Base Audio Hz, Melody Pattern)
SIM_STATIONS = [
    (88.5, "WUCF Jazz & News", 220.0, [1.0, 1.25, 1.5, 1.875]),
    (92.3, "WWKA Country Hits", 261.63, [1.0, 1.125, 1.25, 1.5]),
    (95.3, "WPYO Orlando Hip-Hop", 196.0, [1.0, 1.2, 1.5, 1.333]),
    (98.9, "WMMO Classic Rock", 293.66, [1.0, 1.5, 1.333, 1.25]),
    (101.1, "WJRR Rock & Alt", 164.81, [1.0, 1.189, 1.498, 1.782]),
    (104.1, "WTKS Real Radio", 246.94, [1.0, 1.333, 1.5, 1.667]),
    (106.7, "WXXL Pop Top-40", 329.63, [1.0, 1.25, 1.5, 2.0]),
]


class SimulatedSDRSource:
    """Generates realistic complex I/Q FM radio samples across a 2.048 MHz span."""

    def __init__(self, sample_rate=1_024_000, center_freq=101_100_000):
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self.t_offset = 0.0
        self.rng = np.random.default_rng(42)

    def set_center_freq(self, freq_hz):
        self.center_freq = float(freq_hz)

    def read_samples(self, num_samples):
        n = int(num_samples)
        t = (np.arange(n, dtype=np.float64) + self.t_offset) / self.sample_rate
        self.t_offset += n

        # Thermal RF noise floor
        noise_power = 0.045
        iq = (
            self.rng.normal(0.0, noise_power, n)
            + 1j * self.rng.normal(0.0, noise_power, n)
        ).astype(np.complex64)

        half_bw = self.sample_rate * 0.48
        beat_idx = int((self.t_offset / self.sample_rate) * 2.2)

        for freq_mhz, _name, base_hz, pattern in SIM_STATIONS:
            st_freq = freq_mhz * 1e6
            offset_hz = st_freq - self.center_freq
            if abs(offset_hz) > half_bw:
                continue

            # Synthesize musical chord + cadence + 19 kHz FM stereo pilot
            note_mult = pattern[beat_idx % len(pattern)]
            f1 = base_hz * note_mult
            f2 = f1 * 1.5
            env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 2.2 * t)
            audio_sig = env * (
                0.55 * np.sin(2.0 * np.pi * f1 * t)
                + 0.35 * np.sin(2.0 * np.pi * f2 * t)
            )
            stereo_pilot = 0.10 * np.sin(2.0 * np.pi * 19_000.0 * t)
            mpx = audio_sig + stereo_pilot

            # FM phase integration (75 kHz max deviation)
            kf = 2.0 * np.pi * 75_000.0 / self.sample_rate
            phase = np.cumsum(mpx) * kf
            carrier = 0.85 * np.exp(1j * (2.0 * np.pi * offset_hz * t + phase))
            iq += carrier.astype(np.complex64)

        return iq

    def close(self):
        pass


class FMDemodulator:
    """Two-stage decimation + phase-difference FM discriminator + de-emphasis."""

    def __init__(self, iq_rate=1_024_000, audio_rate=48_000):
        self.iq_rate = int(iq_rate)
        self.audio_rate = int(audio_rate)
        # Stage 1: 1,024,000 Hz -> 256,000 Hz (decimate by 4)
        self.decim1 = 4
        self.mid_rate = self.iq_rate // self.decim1
        self.b_rf, self.a_rf = signal.butter(5, 95_000 / (0.5 * self.iq_rate))
        self.zi_rf = signal.lfilter_zi(self.b_rf, self.a_rf) * 0j

        # Stage 2: 256,000 Hz -> 48,000 Hz (resample 3 / 16)
        # 75 us FM de-emphasis low-pass filter (~2122 Hz corner)
        tau = 75e-6
        d = np.exp(-1.0 / (self.audio_rate * tau))
        self.b_de = np.array([1.0 - d], dtype=np.float32)
        self.a_de = np.array([1.0, -d], dtype=np.float32)
        self.zi_de = signal.lfilter_zi(self.b_de, self.a_de) * 0.0
        self.last_sample = np.complex64(1.0 + 0j)

    def demodulate(self, iq, volume=0.5, squelch_db=-45.0):
        # Measure channel power inside +/- 95 kHz
        filtered, self.zi_rf = signal.lfilter(self.b_rf, self.a_rf, iq, zi=self.zi_rf)
        iq_mid = filtered[::self.decim1]

        power_db = 10.0 * np.log10(np.mean(np.abs(iq_mid) ** 2) + 1e-12)
        if power_db < squelch_db:
            n_out = int(len(iq_mid) * self.audio_rate / self.mid_rate)
            return np.zeros(n_out, dtype=np.float32), power_db

        # Polar discriminator (phase difference)
        z = np.concatenate(([self.last_sample], iq_mid))
        self.last_sample = iq_mid[-1]
        demod = np.angle(z[1:] * np.conj(z[:-1])).astype(np.float32)

        # Resample 256 kHz -> 48 kHz (up=3, down=16)
        audio = signal.resample_poly(demod, up=3, down=16).astype(np.float32)

        # Apply 75 us de-emphasis filter
        audio, self.zi_de = signal.lfilter(self.b_de, self.a_de, audio, zi=self.zi_de)

        # Normalize & scale by volume
        peak = np.max(np.abs(audio)) + 1e-5
        audio = (audio / max(peak, 0.35)) * float(volume) * 0.65
        return np.clip(audio, -0.95, 0.95).astype(np.float32), power_db


class KamaliSDRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kamali SDR FM Radio • Live Spectrum Waterfall & Scanner")
        self.root.geometry("1220x780")
        self.root.configure(bg="#0f172a")

        self.sample_rate = 1_024_000
        self.audio_rate = 48_000
        self.fft_size = 1024
        self.waterfall_rows = 90

        self.center_freq_mhz = tk.DoubleVar(value=101.1)
        self.volume_var = tk.DoubleVar(value=0.55)
        self.squelch_var = tk.DoubleVar(value=-42.0)
        self.muted = False
        self.running = True
        self.scanning = False

        self.spectrum_queue = queue.Queue(maxsize=4)
        self.audio_queue = queue.Queue(maxsize=12)
        self.waterfall_data = np.full((self.waterfall_rows, self.fft_size), -65.0, dtype=np.float32)

        self.source, self.mode_label = self._init_rf_source()
        self.demod = FMDemodulator(self.sample_rate, self.audio_rate)

        self._build_ui()
        self._start_workers()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_rf_source(self):
        if HAS_RTLSDR:
            try:
                sdr = RtlSdr()
                sdr.sample_rate = self.sample_rate
                sdr.center_freq = int(self.center_freq_mhz.get() * 1e6)
                sdr.gain = "auto"
                return sdr, "HARDWARE: RTL-SDR USB Dongle"
            except Exception:
                pass
        return (
            SimulatedSDRSource(self.sample_rate, self.center_freq_mhz.get() * 1e6),
            "SIMULATION MODE: Multi-Station RF Wave Generator",
        )

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0f172a")
        style.configure("Panel.TFrame", background="#1e293b")
        style.configure("TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#1e293b", foreground="#38bdf8", font=("Segoe UI", 12, "bold"))
        style.configure("TButton", background="#334155", foreground="#f8fafc", font=("Segoe UI", 9, "bold"), padding=6)

        # Top Control Bar
        top = ttk.Frame(self.root, style="Panel.TFrame", padding=10)
        top.pack(fill=tk.X, padx=10, pady=(10, 6))

        ttk.Label(top, text="KAMALI SDR FM RECEIVER", style="Header.TLabel").pack(side=tk.LEFT, padx=(4, 14))

        mode_color = "#22c55e" if "HARDWARE" in self.mode_label else "#facc15"
        self.mode_lbl = tk.Label(
            top, text=self.mode_label, bg="#0f172a", fg=mode_color,
            font=("Consolas", 9, "bold"), padx=10, pady=4
        )
        self.mode_lbl.pack(side=tk.LEFT, padx=6)

        self.signal_lbl = tk.Label(
            top, text="Signal: --.- dBFS", bg="#0f172a", fg="#38bdf8",
            font=("Consolas", 10, "bold"), padx=10, pady=4
        )
        self.signal_lbl.pack(side=tk.RIGHT, padx=6)

        # Tuning & Audio Bar
        ctrl = ttk.Frame(self.root, style="Panel.TFrame", padding=10)
        ctrl.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Label(ctrl, text="Frequency (MHz):", bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(4, 6))

        ttk.Button(ctrl, text="- 0.2", command=lambda: self.step_freq(-0.2)).pack(side=tk.LEFT, padx=2)
        self.freq_entry = tk.Entry(
            ctrl, textvariable=self.center_freq_mhz, width=8,
            bg="#0f172a", fg="#38bdf8", insertbackground="#38bdf8",
            font=("Consolas", 13, "bold"), justify="center", relief=tk.FLAT
        )
        self.freq_entry.pack(side=tk.LEFT, padx=4)
        self.freq_entry.bind("<Return>", lambda _e: self.tune_to(self.center_freq_mhz.get()))
        ttk.Button(ctrl, text="+ 0.2", command=lambda: self.step_freq(0.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Tune", command=lambda: self.tune_to(self.center_freq_mhz.get())).pack(side=tk.LEFT, padx=(4, 14))

        self.scan_btn = ttk.Button(ctrl, text="Auto-Scan Band (87.5-108 MHz)", command=self.start_band_scan)
        self.scan_btn.pack(side=tk.LEFT, padx=6)

        # Volume & Mute on right
        self.mute_btn = ttk.Button(ctrl, text="Mute", command=self.toggle_mute)
        self.mute_btn.pack(side=tk.RIGHT, padx=4)

        vol_scale = tk.Scale(
            ctrl, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.volume_var, bg="#1e293b", fg="#f8fafc",
            highlightthickness=0, troughcolor="#0f172a", length=110, showvalue=False
        )
        vol_scale.pack(side=tk.RIGHT, padx=4)
        tk.Label(ctrl, text="Volume:", bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 9)).pack(side=tk.RIGHT)

        sq_scale = tk.Scale(
            ctrl, from_=-65.0, to=-10.0, resolution=1.0, orient=tk.HORIZONTAL,
            variable=self.squelch_var, bg="#1e293b", fg="#f8fafc",
            highlightthickness=0, troughcolor="#0f172a", length=100, showvalue=False
        )
        sq_scale.pack(side=tk.RIGHT, padx=(4, 12))
        tk.Label(ctrl, text="Squelch:", bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 9)).pack(side=tk.RIGHT)

        # Main Split Area: Left = Spectrum + Waterfall, Right = Detected Stations
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Right Sidebar: Station Scanner List
        sidebar = ttk.Frame(body, style="Panel.TFrame", padding=10, width=270)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="DETECTED FM STATIONS", bg="#1e293b", fg="#38bdf8",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor=tk.W, pady=(0, 6))

        tk.Label(
            sidebar, text="Click a station to tune immediately
(or click directly on the spectrum):",
            bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 8), justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(0, 8))

        self.station_list = tk.Listbox(
            sidebar, bg="#0f172a", fg="#f8fafc", selectbackground="#0284c7",
            font=("Consolas", 10), relief=tk.FLAT, highlightthickness=1, highlightcolor="#334155"
        )
        self.station_list.pack(fill=tk.BOTH, expand=True)
        self.station_list.bind("<<ListboxSelect>>", self._on_station_select)

        for freq_mhz, name, _, _ in SIM_STATIONS:
            self.station_list.insert(tk.END, f"{freq_mhz:5.1f} MHz | {name}")

        audio_status = "Audio Output: ACTIVE (48 kHz)" if HAS_AUDIO else "Audio Output: Visual Only (install sounddevice)"
        tk.Label(
            sidebar, text=audio_status, bg="#1e293b", fg="#94a3b8",
            font=("Segoe UI", 8), wraplength=245, justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(8, 0))

        # Left Spectrum & Waterfall Plots
        plot_frame = ttk.Frame(body, style="Panel.TFrame", padding=6)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8.6, 5.8), dpi=100, facecolor="#0f172a")
        self.ax_psd = self.fig.add_subplot(211, facecolor="#090d16")
        self.ax_wf = self.fig.add_subplot(212, facecolor="#090d16")
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.09, hspace=0.22)

        freqs = self._freq_axis_mhz()
        (self.psd_line,) = self.ax_psd.plot(freqs, np.full(self.fft_size, -60.0), color="#38bdf8", lw=1.2)
        self.center_marker = self.ax_psd.axvline(self.center_freq_mhz.get(), color="#ef4444", ls="--", lw=1.2, label="Tuned Center")

        self.ax_psd.set_ylim(-75, 5)
        self.ax_psd.set_xlim(freqs[0], freqs[-1])
        self.ax_psd.set_ylabel("Power (dBFS)", color="#cbd5e1", fontsize=9)
        self.ax_psd.set_title("Live RF Power Spectrum (Click Anywhere to Tune)", color="#f8fafc", fontsize=10, fontweight="bold")
        self.ax_psd.grid(True, color="#1e293b", linestyle=":", linewidth=0.8)
        self.ax_psd.tick_params(colors="#94a3b8", labelsize=8)

        self.wf_img = self.ax_wf.imshow(
            self.waterfall_data,
            aspect="auto",
            origin="upper",
            cmap="inferno",
            vmin=-65,
            vmax=-5,
            extent=[freqs[0], freqs[-1], self.waterfall_rows, 0],
        )
        self.ax_wf.set_xlabel("Frequency (MHz)", color="#cbd5e1", fontsize=9)
        self.ax_wf.set_ylabel("Time (History)", color="#cbd5e1", fontsize=9)
        self.ax_wf.set_title("Scrolling RF Waterfall Spectrogram", color="#f8fafc", fontsize=10, fontweight="bold")
        self.ax_wf.tick_params(colors="#94a3b8", labelsize=8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        self.root.after(45, self._update_plots)

    def _freq_axis_mhz(self):
        fc = self.center_freq_mhz.get()
        span_mhz = (self.sample_rate / 1e6) * 0.5
        return np.linspace(fc - span_mhz, fc + span_mhz, self.fft_size)

    def tune_to(self, freq_mhz):
        try:
            freq_mhz = round(float(freq_mhz), 1)
        except ValueError:
            return
        freq_mhz = max(87.5, min(108.0, freq_mhz))
        self.center_freq_mhz.set(freq_mhz)
        try:
            self.source.set_center_freq(int(freq_mhz * 1e6))
        except Exception:
            pass

    def step_freq(self, delta):
        self.tune_to(self.center_freq_mhz.get() + delta)

    def toggle_mute(self):
        self.muted = not self.muted
        self.mute_btn.configure(text="Unmute" if self.muted else "Mute")

    def _on_plot_click(self, event):
        if event.inaxes in (self.ax_psd, self.ax_wf) and event.xdata is not None:
            self.tune_to(event.xdata)

    def _on_station_select(self, _event):
        sel = self.station_list.curselection()
        if not sel:
            return
        text = self.station_list.get(sel[0])
        freq_str = text.split("MHz")[0].strip()
        self.tune_to(float(freq_str))

    def start_band_scan(self):
        if self.scanning:
            return
        self.scanning = True
        self.scan_btn.configure(text="Scanning FM Band...")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        found = []
        orig_freq = self.center_freq_mhz.get()
        for f in np.arange(88.1, 108.0, 0.4):
            if not self.running:
                break
            freq_mhz = round(float(f), 1)
            try:
                self.source.set_center_freq(int(freq_mhz * 1e6))
                iq = self.source.read_samples(32 * 1024)
                # Measure narrowband power near center
                b, a = signal.butter(4, 80_000 / (0.5 * self.sample_rate))
                ch = signal.lfilter(b, a, iq)
                pwr = 10.0 * np.log10(np.mean(np.abs(ch) ** 2) + 1e-12)
                if pwr > -28.0:
                    label = next((n for fm, n, _, _ in SIM_STATIONS if abs(fm - freq_mhz) <= 0.2), "Active FM Carrier")
                    found.append((freq_mhz, label, pwr))
            except Exception:
                pass

        self.tune_to(orig_freq)

        def finish_scan():
            self.station_list.delete(0, tk.END)
            for fm, lbl, pwr in found:
                self.station_list.insert(tk.END, f"{fm:5.1f} MHz | {lbl} ({pwr:.0f} dB)")
            self.scan_btn.configure(text="Auto-Scan Band (87.5-108 MHz)")
            self.scanning = False

        self.root.after(0, finish_scan)

    def _start_workers(self):
        threading.Thread(target=self._rf_worker, daemon=True).start()
        if HAS_AUDIO:
            threading.Thread(target=self._audio_worker, daemon=True).start()

    def _rf_worker(self):
        win = np.hanning(self.fft_size).astype(np.float32)
        block_size = 64 * 1024
        while self.running:
            if self.scanning:
                time.sleep(0.05)
                continue
            try:
                iq = self.source.read_samples(block_size)
                # Compute PSD for GUI
                seg = iq[:self.fft_size] * win
                fft_vals = np.fft.fftshift(np.fft.fft(seg))
                psd_db = 20.0 * np.log10(np.abs(fft_vals) / self.fft_size + 1e-7)

                # Demodulate audio
                vol = 0.0 if self.muted else self.volume_var.get()
                audio, pwr_db = self.demod.demodulate(iq, volume=vol, squelch_db=self.squelch_var.get())

                if not self.spectrum_queue.full():
                    self.spectrum_queue.put_nowait((psd_db, pwr_db))
                if HAS_AUDIO and not self.audio_queue.full():
                    self.audio_queue.put_nowait(audio)
            except Exception:
                time.sleep(0.05)

    def _audio_worker(self):
        try:
            with sd.OutputStream(samplerate=self.audio_rate, channels=1, dtype="float32") as stream:
                while self.running:
                    try:
                        chunk = self.audio_queue.get(timeout=0.2)
                        stream.write(chunk)
                    except queue.Empty:
                        pass
        except Exception:
            pass

    def _update_plots(self):
        if not self.running:
            return
        try:
            psd_db, pwr_db = self.spectrum_queue.get_nowait()
            freqs = self._freq_axis_mhz()
            self.psd_line.set_data(freqs, psd_db)
            self.ax_psd.set_xlim(freqs[0], freqs[-1])
            self.center_marker.set_xdata([self.center_freq_mhz.get()])

            self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
            self.waterfall_data[0, :] = psd_db
            self.wf_img.set_data(self.waterfall_data)
            self.wf_img.set_extent([freqs[0], freqs[-1], self.waterfall_rows, 0])

            self.signal_lbl.configure(text=f"Signal: {pwr_db:5.1f} dBFS")
            self.canvas.draw_idle()
        except queue.Empty:
            pass

        self.root.after(45, self._update_plots)

    def _on_close(self):
        self.running = False
        try:
            self.source.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = KamaliSDRApp(root)
    root.mainloop()
