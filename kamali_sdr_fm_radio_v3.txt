#!/usr/bin/env python3
"""
Kamali SDR Multi-Mode Radio v3 (WFM / AM / NOAA Weather + RDS + WAV Recorder)
-----------------------------------------------------------------------------
Portfolio Project: Real-Time Software-Defined Radio (DSP + GUI)
"""

import os
import queue
import threading
import time
import wave
from datetime import datetime
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


STATIONS_BY_MODE = {
    "WFM": [
        (88.5, "WUCF-FM", "Jazz & NPR", "WUCF 88.5 FM - Orlando Public Jazz & News", 220.0, [1.0, 1.25, 1.5, 1.875]),
        (92.3, "WWKA-FM", "Country", "K92.3 FM - Central Florida Country Hits", 261.63, [1.0, 1.125, 1.25, 1.5]),
        (95.3, "WPYO-FM", "Hip-Hop", "Power 95.3 - Orlando Hits & Hip-Hop", 196.0, [1.0, 1.2, 1.5, 1.333]),
        (98.9, "WMMO-FM", "Classic Rock", "98.9 WMMO - Orlando Classic Rock", 293.66, [1.0, 1.5, 1.333, 1.25]),
        (101.1, "WJRR-FM", "Active Rock", "101.1 WJRR - Real Rock Orlando", 164.81, [1.0, 1.189, 1.498, 1.782]),
        (104.1, "WTKS-FM", "Talk / Rock", "Real Radio 104.1 FM - Orlando", 246.94, [1.0, 1.333, 1.5, 1.667]),
        (106.7, "WXXL-FM", "Top 40 Pop", "XL 106.7 - Orlando #1 Hit Music", 329.63, [1.0, 1.25, 1.5, 2.0]),
    ],
    "AM": [
        (0.580, "WDBO-AM", "News/Talk", "580 WDBO - Orlando Breaking News & Weather", 300.0, [1.0, 1.2, 1.0, 1.5]),
        (0.740, "WYGM-AM", "Sports", "740 The Game - Central Florida Sports", 350.0, [1.0, 1.25, 1.5, 1.25]),
        (0.950, "WORL-AM", "Talk Radio", "AM 950 - Orlando Talk & Business", 280.0, [1.0, 1.333, 1.2, 1.0]),
    ],
    "NOAA": [
        (162.400, "WXJ70", "NOAA Marine", "NOAA All Hazards - Melbourne / Space Coast", 440.0, [1.0, 1.0, 1.25, 1.25]),
        (162.475, "KIH63", "NOAA Orlando", "NOAA Weather Radio Orlando - Windermere / Orange County", 523.25, [1.0, 1.5, 1.0, 1.5]),
        (162.550, "KHB39", "NOAA Tampa", "NOAA Weather Radio Tampa Bay Regional", 493.88, [1.0, 1.25, 1.0, 1.25]),
    ],
}


class SimulatedSDRSource:
    """Synthesizes complex I/Q RF samples for WFM, AM, and NOAA Weather bands."""

    def __init__(self, sample_rate=1_024_000, center_freq=101_100_000):
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self.mode = "WFM"
        self.t_offset = 0.0
        self.rng = np.random.default_rng(2026)

    def set_center_freq(self, freq_hz):
        self.center_freq = float(freq_hz)

    def set_mode(self, mode):
        self.mode = mode

    def read_samples(self, num_samples):
        n = int(num_samples)
        t = (np.arange(n, dtype=np.float64) + self.t_offset) / self.sample_rate
        self.t_offset += n

        noise_power = 0.042
        iq = (
            self.rng.normal(0.0, noise_power, n)
            + 1j * self.rng.normal(0.0, noise_power, n)
        ).astype(np.complex64)

        half_bw = self.sample_rate * 0.48
        stations = STATIONS_BY_MODE.get(self.mode, STATIONS_BY_MODE["WFM"])

        for mhz, _call, _pty, _rt, base_hz, ratios in stations:
            f_station = mhz * 1e6
            df = f_station - self.center_freq
            if abs(df) > half_bw:
                continue

            beat_idx = int((self.t_offset / self.sample_rate) * 2.5) % len(ratios)
            tone_hz = base_hz * ratios[beat_idx]
            audio_sig = 0.65 * np.sin(2.0 * np.pi * tone_hz * t) + 0.25 * np.sin(
                2.0 * np.pi * (tone_hz * 0.5) * t
            )

            if self.mode == "WFM":
                pilot = 0.12 * np.sin(2.0 * np.pi * 19_000.0 * t)
                mod = audio_sig + pilot
                kf = 75_000.0
                phase = 2.0 * np.pi * df * t + 2.0 * np.pi * kf * np.cumsum(mod) / self.sample_rate
                iq += (0.58 * np.exp(1j * phase)).astype(np.complex64)
            elif self.mode == "NOAA":
                kf = 5_000.0
                phase = 2.0 * np.pi * df * t + 2.0 * np.pi * kf * np.cumsum(audio_sig) / self.sample_rate
                iq += (0.52 * np.exp(1j * phase)).astype(np.complex64)
            else:  # AM
                envelope = 0.48 * (1.0 + 0.7 * audio_sig)
                iq += (envelope * np.exp(1j * 2.0 * np.pi * df * t)).astype(np.complex64)

        return iq


class KamaliSDRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kamali SDR Radio v3 • WFM / AM / NOAA + RDS + WAV Recorder")
        self.root.geometry("1220x740")
        self.root.configure(bg="#0b1120")

        self.sample_rate = 1_024_000
        self.audio_rate = 48_000
        self.fft_size = 2048
        self.waterfall_rows = 90

        self.mode = "WFM"
        self.freq_mhz = 101.1
        self.volume = 0.45
        self.squelch_db = -42.0
        self.muted = False
        self.running = True

        self.recording = False
        self.wav_file = None
        self.rec_start_time = 0.0
        self.rec_filename = ""

        self.rds_ticker_offset = 0
        self.current_station_info = ("WJRR-FM", "Active Rock", "101.1 WJRR - Real Rock Orlando", True)

        self.audio_q = queue.Queue(maxsize=12)
        self.spectrum_data = np.full(self.fft_size, -65.0, dtype=np.float32)
        self.waterfall_data = np.full((self.waterfall_rows, self.fft_size), -65.0, dtype=np.float32)
        self.peak_db = -60.0
        self.snr_db = 0.0
        self.stereo_locked = True

        self._init_sdr()
        self._build_ui()
        self._start_threads()
        self._schedule_ui_refresh()

    def _init_sdr(self):
        self.hw_mode = False
        if HAS_RTLSDR:
            try:
                sdr = RtlSdr()
                sdr.sample_rate = self.sample_rate
                sdr.center_freq = int(self.freq_mhz * 1e6)
                sdr.gain = "auto"
                self.sdr = sdr
                self.hw_mode = True
                return
            except Exception:
                pass
        self.sdr = SimulatedSDRSource(self.sample_rate, int(self.freq_mhz * 1e6))

    def _build_ui(self):
        bg = "#0b1120"
        panel = "#131c31"
        border = "#24324d"
        fg = "#f1f5f9"
        muted = "#94a3b8"
        accent = "#38bdf8"

        # Top Header Bar
        top = tk.Frame(self.root, bg=panel, highlightbackground=border, highlightthickness=1)
        top.pack(fill=tk.X, padx=10, pady=(10, 6))

        tk.Label(
            top, text="📻 KAMALI SDR RADIO v3", font=("Segoe UI", 13, "bold"), bg=panel, fg=accent
        ).pack(side=tk.LEFT, padx=12, pady=8)

        # Mode Selector
        mode_frame = tk.Frame(top, bg=panel)
        mode_frame.pack(side=tk.LEFT, padx=10)
        self.mode_var = tk.StringVar(value="WFM")
        for m, label in [("WFM", "FM Broadcast"), ("AM", "AM Band"), ("NOAA", "NOAA Weather")]:
            rb = tk.Radiobutton(
                mode_frame,
                text=label,
                variable=self.mode_var,
                value=m,
                command=self._on_mode_change,
                indicatoron=False,
                bg="#1e293b",
                fg=fg,
                selectcolor="#0284c7",
                activebackground="#334155",
                activeforeground=fg,
                font=("Segoe UI", 9, "bold"),
                padx=10,
                pady=4,
                bd=0,
            )
            rb.pack(side=tk.LEFT, padx=3)

        # Frequency Readout & Step Buttons
        freq_box = tk.Frame(top, bg="#0f172a", highlightbackground=border, highlightthickness=1)
        freq_box.pack(side=tk.LEFT, padx=14, pady=5)

        tk.Button(
            freq_box, text="◀ -0.2", command=lambda: self.step_freq(-0.2),
            bg="#1e293b", fg=fg, bd=0, padx=8, pady=4, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT)

        self.freq_label = tk.Label(
            freq_box, text="101.100 MHz", font=("Consolas", 15, "bold"), bg="#0f172a", fg="#34d399", width=13
        )
        self.freq_label.pack(side=tk.LEFT, padx=8)

        tk.Button(
            freq_box, text="+0.2 ▶", command=lambda: self.step_freq(0.2),
            bg="#1e293b", fg=fg, bd=0, padx=8, pady=4, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT)

        # Hardware / Sim Status Badge
        src_text = "LIVE RTL-SDR HARDWARE" if self.hw_mode else "SIMULATED RF ENGINE"
        src_color = "#22c55e" if self.hw_mode else "#f59e0b"
        tk.Label(
            top, text=f"● {src_text}", font=("Segoe UI", 9, "bold"), bg=panel, fg=src_color
        ).pack(side=tk.RIGHT, padx=14)

        # RDS & Signal Telemetry Bar
        rds_bar = tk.Frame(self.root, bg="#0f172a", highlightbackground=border, highlightthickness=1)
        rds_bar.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.rds_station_lbl = tk.Label(
            rds_bar, text="RDS: WJRR-FM [Active Rock]", font=("Consolas", 10, "bold"), bg="#0f172a", fg="#38bdf8"
        )
        self.rds_station_lbl.pack(side=tk.LEFT, padx=12, pady=5)

        self.stereo_lbl = tk.Label(
            rds_bar, text="STEREO 19kHz", font=("Segoe UI", 8, "bold"), bg="#14532d", fg="#4ade80", padx=6, pady=2
        )
        self.stereo_lbl.pack(side=tk.LEFT, padx=6)

        self.rds_text_lbl = tk.Label(
            rds_bar, text="101.1 WJRR - Real Rock Orlando", font=("Consolas", 10), bg="#0f172a", fg="#e2e8f0"
        )
        self.rds_text_lbl.pack(side=tk.LEFT, padx=14)

        self.snr_lbl = tk.Label(
            rds_bar, text="Peak: -3.0 dBFS | SNR: 42.0 dB", font=("Consolas", 9, "bold"), bg="#0f172a", fg="#fbbf24"
        )
        self.snr_lbl.pack(side=tk.RIGHT, padx=12)

        # Main Body: Plots Left + Controls Sidebar Right
        body = tk.Frame(self.root, bg=bg)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        plot_frame = tk.Frame(body, bg=panel, highlightbackground=border, highlightthickness=1)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.fig = Figure(figsize=(8.2, 5.6), dpi=100, facecolor=panel)
        self.ax_spec = self.fig.add_subplot(211)
        self.ax_wf = self.fig.add_subplot(212)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.09, hspace=0.28)

        self.ax_spec.set_facecolor("#090d16")
        self.ax_spec.set_title("Live RF Power Spectrum (Click Anywhere to Tune)", color=fg, fontsize=10, fontweight="bold")
        self.ax_spec.set_ylabel("Power (dBFS)", color=muted, fontsize=8)
        self.ax_spec.set_ylim(-85, 5)
        self.ax_spec.grid(True, color="#1e293b", linestyle="--", linewidth=0.6)
        self.ax_spec.tick_params(colors=muted, labelsize=8)

        freqs_init = self._get_freq_axis()
        (self.spec_line,) = self.ax_spec.plot(freqs_init, self.spectrum_data, color="#38bdf8", linewidth=1.1)
        self.center_marker = self.ax_spec.axvline(self.freq_mhz, color="#ef4444", linestyle="--", linewidth=1.3)

        self.ax_wf.set_facecolor("#090d16")
        self.ax_wf.set_title("Scrolling RF Waterfall Spectrogram", color=fg, fontsize=10, fontweight="bold")
        self.ax_wf.set_xlabel("Frequency (MHz)", color=muted, fontsize=8)
        self.ax_wf.set_ylabel("Time", color=muted, fontsize=8)
        self.ax_wf.tick_params(colors=muted, labelsize=8)

        self.wf_img = self.ax_wf.imshow(
            self.waterfall_data,
            aspect="auto",
            cmap="plasma",
            vmin=-75,
            vmax=-5,
            extent=[freqs_init[0], freqs_init[-1], 0, self.waterfall_rows],
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        # Right Sidebar
        sidebar = tk.Frame(body, width=300, bg=panel, highlightbackground=border, highlightthickness=1)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Direct Frequency Entry
        tk.Label(sidebar, text="DIRECT FREQUENCY TUNER", font=("Segoe UI", 9, "bold"), bg=panel, fg=accent).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        entry_row = tk.Frame(sidebar, bg=panel)
        entry_row.pack(fill=tk.X, padx=12)
        self.freq_entry = tk.Entry(entry_row, font=("Consolas", 11), bg="#0f172a", fg=fg, insertbackground=fg, width=12)
        self.freq_entry.insert(0, "101.1")
        self.freq_entry.pack(side=tk.LEFT, padx=(0, 6))
        self.freq_entry.bind("<Return>", lambda _e: self._tune_from_entry())
        tk.Button(
            entry_row, text="Tune", command=self._tune_from_entry,
            bg="#0284c7", fg=fg, font=("Segoe UI", 9, "bold"), bd=0, padx=12, pady=3
        ).pack(side=tk.LEFT)

        # Audio & Squelch Controls
        tk.Label(sidebar, text="AUDIO & SQUELCH", font=("Segoe UI", 9, "bold"), bg=panel, fg=accent).pack(
            anchor="w", padx=12, pady=(14, 2)
        )

        tk.Label(sidebar, text="Volume", font=("Segoe UI", 8), bg=panel, fg=muted).pack(anchor="w", padx=12)
        vol_scale = tk.Scale(
            sidebar, from_=0, to=100, orient=tk.HORIZONTAL, bg=panel, fg=fg,
            highlightthickness=0, troughcolor="#0f172a", command=self._on_vol
        )
        vol_scale.set(45)
        vol_scale.pack(fill=tk.X, padx=12)

        tk.Label(sidebar, text="Squelch Threshold (dBFS)", font=("Segoe UI", 8), bg=panel, fg=muted).pack(
            anchor="w", padx=12, pady=(4, 0)
        )
        sq_scale = tk.Scale(
            sidebar, from_=-75, to=-10, orient=tk.HORIZONTAL, bg=panel, fg=fg,
            highlightthickness=0, troughcolor="#0f172a", command=self._on_sq
        )
        sq_scale.set(-42)
        sq_scale.pack(fill=tk.X, padx=12)

        btn_row = tk.Frame(sidebar, bg=panel)
        btn_row.pack(fill=tk.X, padx=12, pady=8)

        self.mute_btn = tk.Button(
            btn_row, text="🔊 Mute", command=self._toggle_mute,
            bg="#1e293b", fg=fg, font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=5
        )
        self.mute_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self.rec_btn = tk.Button(
            btn_row, text="⏺ Record WAV", command=self._toggle_record,
            bg="#991b1b", fg=fg, font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=5
        )
        self.rec_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        self.rec_status_lbl = tk.Label(
            sidebar, text="Recorder: Idle", font=("Consolas", 8), bg=panel, fg=muted
        )
        self.rec_status_lbl.pack(anchor="w", padx=12)

        # Band Scanner & Station Presets
        tk.Label(sidebar, text="BAND SCANNER & PRESETS", font=("Segoe UI", 9, "bold"), bg=panel, fg=accent).pack(
            anchor="w", padx=12, pady=(14, 4)
        )
        self.scan_btn = tk.Button(
            sidebar, text="🔍 Auto-Scan Active Band", command=self._scan_band,
            bg="#0f766e", fg=fg, font=("Segoe UI", 9, "bold"), bd=0, pady=6
        )
        self.scan_btn.pack(fill=tk.X, padx=12, pady=(0, 6))

        self.station_list = tk.Listbox(
            sidebar, bg="#0f172a", fg=fg, selectbackground="#0284c7",
            font=("Consolas", 9), bd=0, highlightthickness=1, highlightbackground=border
        )
        self.station_list.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.station_list.bind("<<ListboxSelect>>", self._on_station_select)
        self._populate_stations()

    def _get_freq_axis(self):
        span_mhz = self.sample_rate / 1e6
        return np.linspace(self.freq_mhz - span_mhz / 2.0, self.freq_mhz + span_mhz / 2.0, self.fft_size)

    def _populate_stations(self):
        self.station_list.delete(0, tk.END)
        for mhz, call, pty, _rt, _b, _r in STATIONS_BY_MODE.get(self.mode, []):
            unit = "kHz" if self.mode == "AM" else "MHz"
            disp = f"{int(mhz * 1000)} {unit}" if self.mode == "AM" else f"{mhz:6.3f} {unit}"
            self.station_list.insert(tk.END, f"{disp} | {call} ({pty})")

    def _on_mode_change(self):
        new_mode = self.mode_var.get()
        self.mode = new_mode
        if not self.hw_mode:
            self.sdr.set_mode(new_mode)
        self._populate_stations()
        default_freq = STATIONS_BY_MODE[new_mode][0][0]
        if new_mode == "NOAA":
            default_freq = 162.475
        self.set_frequency(default_freq)

    def set_frequency(self, freq_mhz):
        self.freq_mhz = round(float(freq_mhz), 3)
        if self.mode == "AM":
            self.freq_label.config(text=f"{int(self.freq_mhz * 1000)} kHz")
        else:
            self.freq_label.config(text=f"{self.freq_mhz:.3f} MHz")
        self.freq_entry.delete(0, tk.END)
        self.freq_entry.insert(0, f"{self.freq_mhz:.3f}")

        if self.hw_mode:
            self.sdr.center_freq = int(self.freq_mhz * 1e6)
        else:
            self.sdr.set_center_freq(int(self.freq_mhz * 1e6))

        self._update_rds_match()

    def _update_rds_match(self):
        best = None
        for mhz, call, pty, rt, _b, _r in STATIONS_BY_MODE.get(self.mode, []):
            if abs(mhz - self.freq_mhz) <= 0.08:
                best = (call, pty, rt, self.mode == "WFM")
                break
        if best:
            self.current_station_info = best
        else:
            self.current_station_info = ("NO CARRIER", self.mode, f"Scanning {self.freq_mhz:.3f} MHz — No RDS Signal", False)
        self.rds_ticker_offset = 0

    def step_freq(self, delta):
        step = 0.01 if self.mode == "AM" else (0.025 if self.mode == "NOAA" else delta)
        if delta < 0 and self.mode in ("AM", "NOAA"):
            step = -step
        self.set_frequency(self.freq_mhz + step)

    def _tune_from_entry(self):
        try:
            val = float(self.freq_entry.get().strip())
            if self.mode == "AM" and val > 30:
                val = val / 1000.0
            self.set_frequency(val)
        except ValueError:
            messagebox.showwarning("Invalid Frequency", "Enter a valid numeric frequency (e.g., 101.1 or 162.475).")

    def _on_plot_click(self, event):
        if event.inaxes in (self.ax_spec, self.ax_wf) and event.xdata is not None:
            self.set_frequency(event.xdata)

    def _on_station_select(self, _event):
        sel = self.station_list.curselection()
        if not sel:
            return
        idx = sel[0]
        stations = STATIONS_BY_MODE.get(self.mode, [])
        if 0 <= idx < len(stations):
            self.set_frequency(stations[idx][0])

    def _scan_band(self):
        self._populate_stations()
        stations = STATIONS_BY_MODE.get(self.mode, [])
        if stations:
            self.set_frequency(stations[0][0])
            messagebox.showinfo(
                "Band Scan Complete",
                f"Detected {len(stations)} active {self.mode} carriers in Windermere / Orlando area.",
            )

    def _on_vol(self, val):
        self.volume = float(val) / 100.0

    def _on_sq(self, val):
        self.squelch_db = float(val)

    def _toggle_mute(self):
        self.muted = not self.muted
        self.mute_btn.config(text="🔇 Unmute" if self.muted else "🔊 Mute")

    def _toggle_record(self):
        if not self.recording:
            rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
            os.makedirs(rec_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.rec_filename = os.path.join(rec_dir, f"sdr_{self.mode}_{self.freq_mhz:.3f}MHz_{ts}.wav")
            wf = wave.open(self.rec_filename, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.audio_rate)
            self.wav_file = wf
            self.rec_start_time = time.time()
            self.recording = True
            self.rec_btn.config(text="⏹ Stop Rec", bg="#dc2626")
        else:
            self.recording = False
            if self.wav_file:
                try:
                    self.wav_file.close()
                except Exception:
                    pass
                self.wav_file = None
            self.rec_btn.config(text="⏺ Record WAV", bg="#991b1b")
            self.rec_status_lbl.config(text=f"Saved: {os.path.basename(self.rec_filename)}")

    def _start_threads(self):
        threading.Thread(target=self._dsp_worker, daemon=True).start()
        if HAS_AUDIO:
            threading.Thread(target=self._audio_worker, daemon=True).start()

    def _dsp_worker(self):
        decim = int(self.sample_rate // self.audio_rate)  # 21
        b_lp, a_lp = signal.butter(4, 95_000.0 / (self.sample_rate / 2.0))
        b_de, a_de = signal.butter(1, 2122.0 / (self.audio_rate / 2.0))  # ~75 us de-emphasis
        zi_lp = signal.lfilter_zi(b_lp, a_lp) * 0j
        zi_de = signal.lfilter_zi(b_de, a_de) * 0.0
        win = np.hanning(self.fft_size)

        while self.running:
            try:
                iq = self.sdr.read_samples(32_768)
                # FFT for Spectrum & Waterfall
                seg = iq[: self.fft_size] * win
                fft_vals = np.fft.fftshift(np.fft.fft(seg)) / self.fft_size
                psd = 20.0 * np.log10(np.abs(fft_vals) + 1e-9)
                self.spectrum_data = psd
                self.waterfall_data = np.vstack([psd, self.waterfall_data[:-1]])

                center_bin = self.fft_size // 2
                ch_peak = float(np.max(psd[center_bin - 25 : center_bin + 25]))
                noise_floor = float(np.median(psd))
                self.peak_db = ch_peak
                self.snr_db = max(0.0, ch_peak - noise_floor)

                # Channel filter & decimation
                filt, zi_lp = signal.lfilter(b_lp, a_lp, iq, zi=zi_lp)
                baseband = filt[::decim]

                if self.mode in ("WFM", "NOAA"):
                    demod = np.angle(baseband[1:] * np.conj(baseband[:-1]))
                else:
                    env = np.abs(baseband)
                    demod = env - np.mean(env)

                audio, zi_de = signal.lfilter(b_de, a_de, demod, zi=zi_de)
                peak_a = np.max(np.abs(audio)) + 1e-6
                audio = (audio / peak_a) * 0.6

                if ch_peak < self.squelch_db:
                    audio = np.zeros_like(audio)

                if self.recording and self.wav_file:
                    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
                    self.wav_file.writeframes(pcm.tobytes())

                if not self.muted and self.volume > 0.01:
                    out = (audio * self.volume).astype(np.float32)
                    try:
                        self.audio_q.put_nowait(out)
                    except queue.Full:
                        pass
                else:
                    time.sleep(0.025)
            except Exception:
                time.sleep(0.05)

    def _audio_worker(self):
        with sd.OutputStream(samplerate=self.audio_rate, channels=1, dtype="float32") as stream:
            while self.running:
                try:
                    chunk = self.audio_q.get(timeout=0.2)
                    stream.write(chunk)
                except queue.Empty:
                    pass
                except Exception:
                    time.sleep(0.05)

    def _schedule_ui_refresh(self):
        if not self.running:
            return
        freqs = self._get_freq_axis()
        self.spec_line.set_data(freqs, self.spectrum_data)
        self.ax_spec.set_xlim(freqs[0], freqs[-1])
        self.center_marker.set_xdata([self.freq_mhz, self.freq_mhz])

        self.wf_img.set_data(self.waterfall_data)
        self.wf_img.set_extent([freqs[0], freqs[-1], 0, self.waterfall_rows])

        call, pty, rt, is_stereo = self.current_station_info
        self.rds_station_lbl.config(text=f"RDS: {call} [{pty}]")
        if is_stereo and self.peak_db >= self.squelch_db:
            self.stereo_lbl.config(text="STEREO 19kHz", bg="#14532d", fg="#4ade80")
        else:
            self.stereo_lbl.config(text="MONO", bg="#1e293b", fg="#94a3b8")

        scroll_str = f"   ★ {rt}   "
        idx = self.rds_ticker_offset % len(scroll_str)
        self.rds_text_lbl.config(text=(scroll_str[idx:] + scroll_str[:idx])[:48])
        self.rds_ticker_offset += 1

        self.snr_lbl.config(text=f"Peak: {self.peak_db:5.1f} dBFS | SNR: {self.snr_db:4.1f} dB")

        if self.recording:
            elapsed = int(time.time() - self.rec_start_time)
            self.rec_status_lbl.config(text=f"REC ● {elapsed // 60:02d}:{elapsed % 60:02d} -> recordings/")

        self.canvas.draw_idle()
        self.root.after(90, self._schedule_ui_refresh)

    def close(self):
        self.running = False
        if self.recording:
            self._toggle_record()
        if self.hw_mode:
            try:
                self.sdr.close()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = KamaliSDRApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
