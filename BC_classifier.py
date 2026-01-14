# -*- coding: utf-8 -*-
"""
Breast Cancer Classifier (Enhanced Version, Smooth RT)
Optimized UI & DSP:
- DSP moved to background thread (no blocking Tk mainloop)
- Reuse Matplotlib artists (no cla()/legend()/tight_layout() per frame)
- Throttle PSD/HRV updates (~1.5 s)
- Cached bandpass filter coefficients
- Lighter UI tick (~10 Hz) that only sets line data

Dependencies:
  pip install numpy pandas scipy matplotlib scikit-learn joblib
  pip install pyserial
  (optional, for real-time) pip install pyshimmer
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import time
import logging
import traceback
from functools import wraps
import threading
import random
import os

import numpy as np
import pandas as pd
import joblib

from scipy.signal import butter, filtfilt, find_peaks, welch, detrend
from scipy.interpolate import PchipInterpolator

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ========= CONFIG =========
APP_TITLE = "Breast Cancer Classifier"
LABEL_MAPPING_INFO = "Model expects label mapping: 'Breast cancer'→1, 'Healthy'→0"
DEFAULT_MODEL_PATH = "best_xgb_pipeline.pkl"

LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

# Performance settings
MAX_BUFFER_SIZE = 128 * 300  # 300 seconds max for real-time
UPDATE_INTERVAL_MS = 200  # (kept for non-RT UI tasks)

# Connection settings
CONNECTION_TIMEOUT = 10  # seconds
PORT_SCAN_TIMEOUT = 5    # seconds

# ========= ENHANCED ERROR HANDLING =========

def handle_errors(logger=None):
    """Decorator for comprehensive error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Running {func.__name__}...")
                if hasattr(self, 'progress'):
                    self.progress.start()

                result = func(self, *args, **kwargs)

                if hasattr(self, 'status_var'):
                    self.status_var.set("Ready")
                return result

            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                if logger:
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())

                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Error: {str(e)}")
                if hasattr(self, 'progress'):
                    self.progress.stop()

                user_friendly_msg = self._make_error_user_friendly(e, func.__name__)
                messagebox.showerror("Operation Failed", user_friendly_msg)

                return None
            finally:
                if hasattr(self, 'progress'):
                    self.progress.stop()
        return wrapper
    return decorator

# Import SerialException for COM error handling
try:
    from serial import Serial, SerialException
except ImportError:
    SerialException = Exception

def handle_com_errors(logger=None):
    """Specialized decorator for COM port operations with better false alarm prevention."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Starting {func.__name__}...")

                result = [None]
                exception = [None]

                def target():
                    try:
                        result[0] = func(self, *args, **kwargs)
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                thread.join(timeout=CONNECTION_TIMEOUT)

                if thread.is_alive():
                    # Don't treat timeout as immediate error - might be normal device behavior
                    self.logger.warning(f"{func.__name__} timed out, but this might be normal")
                    return None  # Return gracefully instead of raising error

                if exception[0]:
                    # Only raise exceptions that are truly critical
                    if isinstance(exception[0], (SerialException, ConnectionError)):
                        raise exception[0]
                    else:
                        # Log non-critical exceptions but don't show user errors
                        self.logger.warning(f"Non-critical COM error in {func.__name__}: {exception[0]}")
                        return None

                return result[0]

            except Exception as e:
                error_msg = f"COM error in {func.__name__}: {str(e)}"
                if logger:
                    logger.error(error_msg)

                if hasattr(self, 'status_var'):
                    self.status_var.set(f"COM Error: {str(e)}")

                # Only show user-friendly message for genuine connection issues
                user_friendly_msg = self._make_com_error_user_friendly(e, func.__name__)
                messagebox.showerror("COM Port Error", user_friendly_msg)
                return None

        return wrapper
    return decorator

# ========= ENHANCED REAL-TIME READER =========

def adc_to_mV(raw_value, vref=2.42, gain=6):
    """Convert ADS1292R 24-bit raw to millivolts with validation."""
    try:
        if not isinstance(raw_value, (int, float, np.number)):
            raise ValueError("Invalid raw value type")
        return (float(raw_value) * vref / ((2**23 - 1) * gain)) * 1000.0
    except (ValueError, TypeError, ZeroDivisionError) as e:
        logging.warning(f"ADC conversion error: {e}, using 0.0")
        return 0.0

class EnhancedShimmerRealtimeReader:
    """
    Enhanced real-time reader with better error handling and performance monitoring.
    """
    def __init__(self, buffer_size=128 * 120, sampling_rate=128):
        self.buffer_size = int(buffer_size)
        self.sampling_rate = float(sampling_rate)
        self.ecg_buffer = deque(maxlen=self.buffer_size)
        self.timestamp_buffer = deque(maxlen=self.buffer_size)
        self.running = False
        self._ser = None
        self._shim = None
        self.connection_stats = {
            'packets_received': 0,
            'conversion_errors': 0,
            'last_error': None
        }
        self._previous_quality = 100.0  # For quality smoothing

        # Connection state
        self._connection_lock = threading.Lock()
        self._is_connecting = False

        # Lazy imports with better error handling
        self._pyshimmer_ok = True
        try:
            from serial import Serial  # noqa
            from pyshimmer import ShimmerBluetooth, DataPacket, DEFAULT_BAUDRATE, EChannelType  # noqa
            self._imports_available = True
        except ImportError as e:
            self._pyshimmer_ok = False
            self._imports_available = False
            logging.warning(f"PyShimmer imports not available: {e}")

    @property
    def serial_port(self):
        return self._ser

    @property
    def is_open(self):
        return (self._ser is not None) and getattr(self._ser, "is_open", False)

    @property
    def is_connecting(self):
        return self._is_connecting

    @property
    def data_quality(self):
        """Calculate data quality metric (0-100%) with less sensitivity to transient errors."""
        total_operations = self.connection_stats['packets_received'] + self.connection_stats['conversion_errors']
        if total_operations == 0:
            return 100.0
        
        # Use a weighted average that's less sensitive to recent errors
        success_rate = (self.connection_stats['packets_received'] / total_operations) * 100
        
        # Apply smoothing - don't drop quality too quickly for transient errors
        smoothed_quality = 0.7 * self._previous_quality + 0.3 * success_rate
        self._previous_quality = smoothed_quality
        return max(0, min(100, smoothed_quality))

    def list_available_ports(self):
        """Enhanced port listing with better error handling and timeout"""
        try:
            import serial.tools.list_ports

            def scan_ports():
                try:
                    ports = serial.tools.list_ports.comports()
                    return [(p.device, p.description) for p in ports]
                except Exception as e:
                    logging.error(f"Error scanning ports: {e}")
                    return []

            result = []
            exception = []

            def target():
                try:
                    result.extend(scan_ports())
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=PORT_SCAN_TIMEOUT)

            if thread.is_alive():
                logging.warning("Port scanning timed out")
                return []

            if exception:
                raise exception[0]

            return result

        except Exception as e:
            logging.error(f"Error listing ports: {e}")
            return []

    def connect(self, port_name: str) -> bool:
        """Enhanced connection with validation and timeout handling."""
        if not self._imports_available:
            logging.error("Required imports not available")
            return False

        if not port_name:
            logging.error("No port name provided")
            return False

        with self._connection_lock:
            if self._is_connecting:
                logging.warning("Already connecting to device")
                return False
            self._is_connecting = True

        try:
            from serial import Serial
            from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, EChannelType  # noqa

            self._cleanup()
            logging.info(f"Attempting to connect to {port_name}")

            try:
                self._ser = Serial(port_name, DEFAULT_BAUDRATE, timeout=5)
            except Exception as e:
                raise ConnectionError(f"Failed to open serial port {port_name}: {e}")

            try:
                self._shim = ShimmerBluetooth(self._ser)
                self._shim.initialize()
                self._shim.add_stream_callback(self._stream_cb)
            except Exception as e:
                raise ConnectionError(f"Failed to initialize Shimmer device: {e}")

            self.connection_stats = {'packets_received': 0, 'conversion_errors': 0, 'last_error': None}
            logging.info(f"Successfully connected to {port_name}")
            return True

        except Exception as e:
            error_msg = f"Failed to connect to {port_name}: {e}"
            logging.error(error_msg)
            self.connection_stats['last_error'] = error_msg
            self._cleanup()
            return False
        finally:
            self._is_connecting = False

    def disconnect(self):
        """Enhanced disconnection with error handling."""
        self.stop_streaming()
        self._cleanup()
        logging.info("Disconnected from Shimmer device")

    def _cleanup(self):
        """Enhanced cleanup with better resource management."""
        try:
            if self._shim:
                try:
                    self._shim.shutdown()
                except Exception as e:
                    logging.warning(f"Error during shutdown: {e}")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
        finally:
            self._shim = None
            if self._ser:
                try:
                    if getattr(self._ser, "is_open", False):
                        self._ser.close()
                except Exception as e:
                    logging.error(f"Error closing serial port: {e}")
                finally:
                    self._ser = None
            self._is_connecting = False

    def start_streaming(self):
        """Enhanced streaming start with error handling."""
        if not self._shim:
            raise RuntimeError("Not connected to Shimmer device")
        try:
            self.clear_buffer()
            self.running = True
            self._shim.start_streaming()
            logging.info("Started streaming")
        except Exception as e:
            self.running = False
            raise RuntimeError(f"Failed to start streaming: {e}")

    def stop_streaming(self):
        """Enhanced streaming stop with error handling."""
        self.running = False
        if self._shim:
            try:
                self._shim.stop_streaming()
                logging.info("Stopped streaming")
            except Exception as e:
                logging.error(f"Error stopping streaming: {e}")

    def _stream_cb(self, pkt):
        """Enhanced stream callback with better error resilience and fewer false alarms."""
        if not self.running:
            return
        
        try:
            if not self._imports_available:
                return

            from pyshimmer import EChannelType

            # Choose a typical ECG channel; adjust to your sensor config if needed
            chan = EChannelType.EXG1_CH1_24BIT
            chans = getattr(pkt, "channels", [])

            if chan in chans:
                raw = pkt[chan]
                
                # More robust ADC conversion with better validation
                try:
                    mv = adc_to_mV(raw)
                    
                    # More permissive sanity checks to reduce false alarms
                    if not np.isfinite(mv):
                        self.connection_stats['conversion_errors'] += 1
                        return
                        
                    # Wider acceptable range to accommodate real ECG signals
                    if abs(mv) > 500:  # Increased from 100 to 500 mV
                        self.connection_stats['conversion_errors'] += 1
                        return

                except (ValueError, TypeError) as e:
                    # Don't log every conversion error, just count them
                    self.connection_stats['conversion_errors'] += 1
                    return

                if len(self.timestamp_buffer) == 0:
                    t = 0.0
                else:
                    t = self.timestamp_buffer[-1] + 1.0 / self.sampling_rate

                self.ecg_buffer.append(mv)
                self.timestamp_buffer.append(t)
                self.connection_stats['packets_received'] += 1

        except Exception as e:
            # Don't log every stream error - many are transient
            self.connection_stats['conversion_errors'] += 1
            # Only log occasional errors to avoid spam
            if random.random() < 0.01:  # Log only 1% of errors
                logging.debug(f"Stream callback transient error: {e}")

    def get_buffered_data(self):
        """Get buffered data with validation."""
        try:
            ts_array = np.asarray(self.timestamp_buffer, float)
            ecg_array = np.asarray(self.ecg_buffer, float)

            valid_mask = np.isfinite(ts_array) & np.isfinite(ecg_array)
            return ts_array[valid_mask], ecg_array[valid_mask]

        except Exception as e:
            logging.error(f"Error getting buffered data: {e}")
            return np.array([]), np.array([])

    def clear_buffer(self):
        """Clear buffers."""
        self.ecg_buffer.clear()
        self.timestamp_buffer.clear()
        logging.info("Cleared data buffers")

# ========= SIGNAL/HRV HELPERS (OPTIMIZED) =========

# Cached bandpass coefficients
_BPF_ORDER = 3
_BPF_COEF = None  # (key, (b, a)), where key=(fs, low, high, order)

def _get_bpf(fs, low=8.0, high=20.0):
    global _BPF_COEF
    key = (float(fs), float(low), float(high), _BPF_ORDER)
    if _BPF_COEF and _BPF_COEF[0] == key:
        return _BPF_COEF[1]
    nyq = 0.5 * fs
    hi = min(high, nyq * 0.95)
    b, a = butter(_BPF_ORDER, [low/nyq, hi/nyq], btype="band")
    _BPF_COEF = (key, (b, a))
    return b, a

def bandpass(x: np.ndarray, fs: float, low=8.0, high=20.0, order=3):
    """Optimized bandpass filter with cached coefficients."""
    if fs is None or fs <= 0 or low >= high or len(x) < order * 3:
        return x.copy()
    try:
        b, a = _get_bpf(fs, low, high)
        y = filtfilt(b, a, x)
        return y - np.nanmedian(y)
    except Exception as e:
        logging.warning(f"Bandpass filter failed: {e}, returning original signal")
        return x.copy()

def rr_from_ecg(ecg: np.ndarray, fs: float):
    """
    Optimized R-peak detection with better performance.
    """
    if fs is None or fs <= 0 or len(ecg) < 10:
        return np.array([]), np.array([]), 0, np.array([], dtype=int), ecg

    x = np.asarray(ecg, float)
    x[np.isnan(x)] = 0.0

    # Filter (chunked for very long signals)
    if len(x) > 10000:
        chunk_size = 5000
        ecg_f_chunks = []
        for i in range(0, len(x), chunk_size):
            chunk = x[i:i+chunk_size]
            ecg_f_chunks.append(bandpass(chunk, fs))
        ecg_f = np.concatenate(ecg_f_chunks)
    else:
        ecg_f = bandpass(x, fs)

    if np.nanmax(ecg_f) < np.nanmax(-ecg_f):
        ecg_f = -ecg_f

    dx = np.diff(ecg_f, prepend=ecg_f[0])
    y = dx ** 2

    win = max(1, int(0.15 * fs))

    def fast_moving_integral(signal, window):
        cumsum = np.cumsum(np.insert(signal, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window

    if len(y) > win:
        mwi = fast_moving_integral(y, win)
        pad_left = win // 2
        pad_right = win - pad_left - 1
        mwi = np.pad(mwi, (pad_left, pad_right), mode='edge')
    else:
        mwi = y

    if np.isfinite(mwi).any():
        hi = np.percentile(mwi[np.isfinite(mwi)], 98)
        thr = 0.30 * hi
    else:
        return np.array([]), np.array([]), 0, np.array([], dtype=int), ecg_f

    distance = int(max(1, 0.25 * fs))
    peaks, _ = find_peaks(mwi, height=thr, distance=distance)

    if peaks.size < 6:
        for scale in (0.25, 0.20, 0.15):
            peaks, _ = find_peaks(mwi, height=scale * hi, distance=distance)
            if peaks.size >= 6:
                break

    def _refine_on_ecg(peaks_idx, ecg_f, fs, radius_s=0.05):
        if peaks_idx.size == 0:
            return peaks_idx
        r = max(1, int(radius_s * fs))
        n = len(ecg_f)
        refined = []
        for p in peaks_idx:
            i0 = max(0, p - r)
            i1 = min(n, p + r + 1)
            if i1 - i0 <= 1:
                refined.append(p)
                continue
            local = ecg_f[i0:i1]
            off = int(np.nanargmax(local))
            refined.append(i0 + off)
        return np.asarray(refined, dtype=int)

    peaks = _refine_on_ecg(peaks, ecg_f, fs, radius_s=0.05)

    if peaks.size < 2:
        return np.array([]), np.array([]), int(peaks.size), peaks, ecg_f

    t = np.arange(len(ecg_f)) / fs
    r_times = t[peaks]
    rr = np.diff(r_times)
    rr_t = r_times[1:]

    good = (rr >= 0.3) & (rr <= 2.0)
    return rr_t[good], rr[good], int(peaks.size), peaks, ecg_f

def compute_psd_from_rr(rr, rr_times, interp_fs=4.0, use_highpass=True):
    """Optimized PSD computation."""
    if len(rr_times) < 2:
        return np.array([]), np.array([])

    start, stop = rr_times[0], rr_times[-1]
    if stop <= start:
        return np.array([]), np.array([])

    t_even = np.arange(start, stop, 1.0 / interp_fs)

    if len(rr_times) > 1000:
        step = max(1, len(rr_times) // 1000)
        rr_times_sub = rr_times[::step]
        rr_sub = rr[::step]
        f_rr = PchipInterpolator(rr_times_sub, rr_sub * 1000.0)
    else:
        f_rr = PchipInterpolator(rr_times, rr * 1000.0)

    rr_interp_ms = f_rr(t_even)
    rr_interp_ms = detrend(rr_interp_ms, type="linear")

    if use_highpass:
        nyq_rr = 0.5 * interp_fs
        hp = 0.04 / nyq_rr
        if 0 < hp < 1:
            b_hp, a_hp = butter(2, hp, btype="highpass")
            rr_interp_ms = filtfilt(b_hp, a_hp, rr_interp_ms)

    nperseg = min(256, len(rr_interp_ms))
    if nperseg < 64:
        nperseg = len(rr_interp_ms)

    noverlap = nperseg // 2
    freqs, psd = welch(rr_interp_ms, fs=interp_fs, nperseg=nperseg,
                       noverlap=noverlap, window="hann",
                       detrend=False, scaling="density")
    return freqs, psd

def band_power(freqs, psd, fmin, fmax):
    """Optimized band power calculation."""
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return np.nan
    return float(np.trapz(psd[idx], freqs[idx]))

def features_from_rr(rr, rr_times, interp_fs=4.0, use_highpass=True):
    """Enhanced feature extraction with validation."""
    if len(rr) < 10:
        return None

    freqs, psd = compute_psd_from_rr(rr, rr_times, interp_fs, use_highpass)
    if len(freqs) == 0:
        return None

    lf = band_power(freqs, psd, *LF_BAND)
    hf = band_power(freqs, psd, *HF_BAND)
    
    # Calculate Total Power (TP) - power in the full frequency range
    tp = band_power(freqs, psd, 0.0, 0.4)  # Standard HRV total power range

    if not np.isfinite(lf) or lf < 0 or lf > 1e6:
        lf = np.nan
    if not np.isfinite(hf) or hf < 0 or hf > 1e6:
        hf = np.nan
    if not np.isfinite(tp) or tp < 0 or tp > 1e6:
        tp = np.nan

    ratio = lf / hf if (isinstance(hf, (int, float, np.floating)) and hf > 0 and np.isfinite(lf)) else np.nan

    return {
        "freqs": freqs,
        "psd": psd,
        "LF_ms2": lf,
        "HF_ms2": hf,
        "Total_Power_ms2": tp,  # Changed from LF_HF to Total_Power
        "LF_HF": ratio  # Keep for internal use if needed
    }

# ========= ENHANCED CSV HELPERS =========

def _is_number(s):
    """Enhanced number validation."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def load_shimmer_csv(path: str) -> pd.DataFrame:
    """Enhanced CSV loading with better error handling and performance."""
    try:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(path, skiprows=1, encoding=encoding, low_memory=False)
                if len(df.columns) > 1:
                    break
            except UnicodeDecodeError:
                continue

        if df is None or len(df.columns) == 1:
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue

        if df is None:
            raise ValueError("Could not read CSV file with any encoding")

        first_col = df.columns[0]
        if first_col in df.columns:
            df = df[df[first_col].astype(str).apply(_is_number)].copy()

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

        return df

    except Exception as e:
        logging.error(f"Error loading CSV {path}: {e}")
        raise

def detect_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    """Enhanced timestamp detection with better pattern matching."""
    # First, check for exact matches of common timestamp column names
    exact_matches = ['time_s', 'timestamp_s', 't_s', 'time', 'timestamp', 't']
    
    for exact_col in exact_matches:
        if exact_col in df.columns:
            return exact_col
    
    # Check for case-insensitive matches
    df_columns_lower = [col.lower().strip() for col in df.columns]
    
    for i, col_lower in enumerate(df_columns_lower):
        # Remove common prefixes/suffixes and check for time patterns
        cleaned_col = col_lower.replace('_', '').replace('-', '').replace(' ', '')
        
        # Check for various time patterns
        time_patterns = ['time', 'timestamp', 't', 'dt', 'date', 'sec', 'second']
        
        for pattern in time_patterns:
            if pattern in col_lower or pattern in cleaned_col:
                # Also verify it's a numeric column
                original_col = df.columns[i]
                if pd.api.types.is_numeric_dtype(df[original_col]):
                    return original_col
    
    # If no time pattern found, look for numeric columns that could be timestamps
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    if not numeric_cols:
        return None
    
    # Check each numeric column to see if it looks like a timestamp
    for col in numeric_cols:
        try:
            series = pd.to_numeric(df[col], errors='coerce')
            
            # A good timestamp column should:
            # 1. Have mostly increasing values
            # 2. Have relatively small, positive differences
            # 3. Not have huge jumps
            
            if len(series) > 10:
                diffs = series.diff().dropna()
                
                # Check if values are increasing (timestamps should increase)
                positive_diffs = diffs[diffs > 0]
                negative_diffs = diffs[diffs < 0]
                
                # More than 80% of differences should be positive for a timestamp
                if len(positive_diffs) > 0 and len(positive_diffs) / len(diffs) > 0.8:
                    # Check typical timestamp ranges
                    # If values are in seconds, they're likely in 0-300 range for short recordings
                    # If values are in milliseconds, they're larger
                    mean_val = series.mean()
                    std_val = series.std()
                    
                    # Heuristic: if values look like they could be seconds or milliseconds
                    if (mean_val < 10000 and std_val > 0):  # Likely seconds (0-10000s = ~2.7 hours)
                        return col
                        
        except Exception:
            continue
    
    # Last resort: return first numeric column
    return numeric_cols[0]

def estimate_fs_from_times(t_vals: np.ndarray) -> Optional[float]:
    """Enhanced sampling rate estimation."""
    t = np.asarray(t_vals, float)
    t = t[np.isfinite(t)]

    if t.size < 3:
        return None

    d = np.diff(t)
    d = d[(d > 0) & np.isfinite(d)]

    if d.size == 0:
        return None

    if len(d) > 10:
        mean_d = np.mean(d)
        std_d = np.std(d)
        d = d[(d > mean_d - 3*std_d) & (d < mean_d + 3*std_d)]

    med = float(np.median(d))

    cand_fs = [1.0/med, 1000.0/med, 1_000_000.0/med]
    valid_fs = [fs for fs in cand_fs if 10 <= fs <= 1000]

    return float(min(valid_fs, key=lambda x: abs(x - 128.0))) if valid_fs else 128.0

def pick_ecg_column(df: pd.DataFrame, ts_col: Optional[str]) -> Optional[str]:
    """
    Optimized ECG column detection with caching.
    """
    if hasattr(df, '_ecg_column_cache') and df._ecg_column_cache:
        return df._ecg_column_cache

    cols = list(df.columns)

    def is_la_ra(name: str) -> bool:
        n = name.lower().replace("_", "-").replace(" ", "-")
        return ("ecg" in n) and (("la-ra" in n) or ("la" in n and "ra" in n))

    la_ra = [c for c in cols if is_la_ra(c)]
    if la_ra:
        la_ra.sort(key=lambda c: ("cal" not in c.lower(), "24" not in c.lower(), len(c)))
        result = la_ra[0]
        df._ecg_column_cache = result
        return result

    ecg_any = [c for c in cols if "ecg" in c.lower()]
    if ecg_any:
        ecg_any.sort(key=lambda c: ("cal" not in c.lower(), "24" not in c.lower(), len(c)))
        result = ecg_any[0]
        df._ecg_column_cache = result
        return result

    blacklist = {
        "status", "marker", "event", "button", "sync", "label", "counter",
        "timestamp", "time", "unix", "ms", "battery", "batt", "temp",
        "acc", "accel", "gyro", "mag", "ppg", "resp", "emg", "eda", "gsr"
    }

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != ts_col and all(b not in c.lower() for b in blacklist)
    ]

    if not num_cols:
        return None

    var = df[num_cols].var(numeric_only=True)
    result = var.idxmax()
    df._ecg_column_cache = result
    return result

# ========= ENHANCED APPLICATION =========
class ScrollableFrame(ttk.Frame):
    """Vertical scrollable frame for the right-hand column."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        # Allow mousewheel scrolling
        self.inner.bind_all("<MouseWheel>", self._on_mousewheel)
        self.inner.bind_all("<Button-4>", self._on_mousewheel)   # Linux
        self.inner.bind_all("<Button-5>", self._on_mousewheel)   # Linux

    @property
    def content(self):
        return self.inner

    def _on_mousewheel(self, event):
        delta = -1*(event.delta//120) if event.delta else (1 if event.num == 5 else -1)
        self.canvas.yview_scroll(delta, "units")

class EnhancedApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Setup logging first
        self.setup_logging()

        self.title(APP_TITLE)
        self.geometry("1200x800")
        self.minsize(1200, 700)

        # Initialize state
        self.model = None
        self.scaler = None
        self.df_raw = None
        self.ts_col = None
        self.ecg_col = None
        self.cached_feats = None
        self.realtime_active = False
        self.update_interval_ms = UPDATE_INTERVAL_MS

        # Real-time data collection tracking
        self._collection_start_time = None
        self._collection_duration = 300  # 5 minutes in seconds
        self._min_data_required = True  # Enable 5-minute requirement

        # DSP/RT threading state
        self._rt_lock = threading.Lock()
        self._last_psd_ts = 0.0
        self._last_process_time = 0  # Track last processing time
        self._dsp_queue = deque(maxlen=1)  # last DSP result for UI
        self._worker_running = False
        self._psd_period_s = 1.5  # compute PSD at most every 1.5s
        self._ui_fps_ms = 100     # ECG refresh ~10 Hz

        # Processing time tracking
        self.last_processing_time = 0  # Simple variable for processing time

        # Final analysis state
        self._final_analysis_computed = False
        self._final_features = None

        # R-peak tracking
        self._full_data_ecg = None
        self._full_data_r_peaks = 0
        self._full_data_computed = False

        # Initialize real-time reader
        try:
            self.shimmer_reader = EnhancedShimmerRealtimeReader()
        except Exception as e:
            self.logger.error(f"Failed to initialize real-time reader: {e}")
            self.shimmer_reader = None

        self._init_enhanced_style()
        self._build_enhanced_ui()
        self._autoload_model()
        self._refresh_ports()

    def setup_logging(self):
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('app_enhanced.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _make_error_user_friendly(self, error: Exception, operation: str) -> str:
        """Convert technical errors to user-friendly messages."""
        error_messages = {
            'FileNotFoundError': f"File not found for {operation}. Please check the file path.",
            'PermissionError': f"Permission denied for {operation}. Please check file permissions.",
            'pd.errors.EmptyDataError': f"The selected file is empty for {operation}.",
            'pd.errors.ParserError': f"Could not parse the file for {operation}. Please check the file format.",
            'ValueError': f"Invalid data in file for {operation}. Please check the file content.",
            'MemoryError': f"File too large for {operation}. Try a smaller file or enable chunked processing.",
        }
        for error_type, friendly_msg in error_messages.items():
            if error_type in str(type(error)):
                return friendly_msg
        return f"An unexpected error occurred during {operation}: {str(error)}"

    def _make_com_error_user_friendly(self, error: Exception, operation: str) -> str:
        """Convert COM port errors to user-friendly messages."""
        error_str = str(error).lower()
        if "timeout" in error_str:
            return (
                f"Device connection timed out during {operation}. Please:\n\n"
                "1. Check if the Shimmer device is powered ON\n"
                "2. Verify the correct COM port is selected\n"
                "3. Try reconnecting the USB cable\n"
                "4. Restart the Shimmer device\n"
                "5. Ensure no other program is using the port"
            )
        elif "access denied" in error_str or "permission" in error_str:
            return (
                f"Access denied to COM port during {operation}. Please:\n\n"
                "1. Close any other programs using the COM port\n"
                "2. Run this application as Administrator\n"
                "3. Try a different USB port\n"
                "4. Restart the application"
            )
        elif "could not open port" in error_str or "port not found" in error_str:
            return (
                f"COM port not found during {operation}. Please:\n\n"
                "1. Verify the Shimmer device is connected\n"
                "2. Check Device Manager for the correct COM port\n"
                "3. Try reconnecting the USB cable\n"
                "4. Select a different COM port from the list"
            )
        elif "device not found" in error_str:
            return (
                f"Shimmer device not found during {operation}. Please:\n\n"
                "1. Ensure the Shimmer device is powered ON\n"
                "2. Check the battery level\n"
                "3. Verify the device is in pairing mode\n"
                "4. Try resetting the device"
            )
        elif "pyshimmer" in error_str or "import" in error_str:
            return (
                f"Shimmer library error during {operation}. Please:\n\n"
                "1. Install required packages: pip install pyserial pyshimmer\n"
                "2. Restart the application\n"
                "3. Check if the Shimmer SDK is properly installed"
            )
        else:
            return (
                f"Device communication error during {operation}:\n\n{str(error)}\n\n"
                "Please check:\n1. Device power and connections\n2. COM port selection\n"
                "3. Driver installation\n4. Try reconnecting the device"
            )

    def _init_enhanced_style(self):
        """Enhanced styling with modern look."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background="#f8f9fa")
        style.configure("TLabel", background="#f8f9fa", font=("Segoe UI", 10))
        style.configure("TFrame", background="#f8f9fa")
        style.configure("Primary.TButton",
                        font=("Segoe UI", 10, "bold"),
                        padding=(14, 8),
                        background="#007acc",
                        foreground="white")

        style.configure("Secondary.TButton",
                        font=("Segoe UI", 10),
                        padding=(14, 8),
                        background="#6c757d",
                        foreground="white")

        style.configure("Success.TButton",
                        background="#28a745",
                        foreground="white")

        style.configure("Danger.TButton",
                        background="#dc3545",
                        foreground="white")

        style.configure("Card.TLabelframe",
                        background="white",
                        borderwidth=2,
                        relief="solid",
                        bordercolor="#dee2e6")

        style.configure("Card.TLabelframe.Label",
                        background="white",
                        font=("Segoe UI", 12, "bold"),
                        foreground="#495057")

        style.configure("Good.TLabel", foreground="#28a745", font=("Segoe UI", 10, "bold"))
        style.configure("Warn.TLabel", foreground="#ffc107", font=("Segoe UI", 10, "bold"))
        style.configure("Bad.TLabel",  foreground="#dc3545", font=("Segoe UI", 10, "bold"))
        style.configure("Info.TLabel", foreground="#17a2b8", font=("Segoe UI", 10, "bold"))

    def _build_enhanced_ui(self):
        """Build enhanced UI with modern features."""
        main_container = ttk.Frame(self, padding=(15, 15, 15, 15))
        main_container.pack(fill="both", expand=True)

        header = ttk.Frame(main_container)
        header.pack(fill="x", pady=(0, 3))

        title_frame = ttk.Frame(header)
        title_frame.pack(side="left", fill="x", expand=True)

        ttk.Label(title_frame, text=APP_TITLE,
                  font=("Segoe UI", 18, "bold"),
                  foreground="#2c3e50").pack(side="left")

        self.model_status = ttk.Label(title_frame, text="Model: not loaded",
                                      style="Bad.TLabel")
        self.model_status.pack(side="left", padx=(15, 0))

        control_frame = ttk.Frame(header)
        control_frame.pack(side="right")

        ttk.Button(control_frame, text="Load Model", style="Secondary.TButton",
                   command=self.load_model).pack(side="left", padx=3)

        ttk.Button(control_frame, text="Settings", style="Secondary.TButton",
                   command=self._show_settings).pack(side="left", padx=3)

        self.progress = ttk.Progressbar(main_container, mode='indeterminate')
        self.progress.pack(fill='x', pady=(0, 3))

        content_pane = ttk.Panedwindow(main_container, orient="horizontal")
        content_pane.pack(fill="both", expand=True)

        left_panel = ttk.Frame(content_pane)
        content_pane.add(left_panel, weight=3)
        self._create_enhanced_plots(left_panel)

        right_panel = ttk.Frame(content_pane)
        content_pane.add(right_panel, weight=2)
        self._create_control_panel(right_panel)

        status_bar = ttk.Frame(main_container, relief='sunken')
        status_bar.pack(side='bottom', fill='x', pady=(10, 0))

        self.status_var = tk.StringVar(value="Ready - Select an operation to begin")
        status_label = ttk.Label(status_bar, textvariable=self.status_var,
                                 relief='sunken', anchor='w')
        status_label.pack(fill='x', padx=2, pady=2)

    def _create_enhanced_plots(self, parent):
        """Create enhanced visualization plots with reusable artists."""
        self.fig = plt.Figure(figsize=(10, 8), dpi=100, facecolor='#f8f9fa')
        gs = self.fig.add_gridspec(2, 1, hspace=0.4)

        self.ax_ecg = self.fig.add_subplot(gs[0, 0])
        self.ax_psd = self.fig.add_subplot(gs[1, 0])

        for ax in (self.ax_ecg, self.ax_psd):
            ax.set_facecolor('#ffffff')
            ax.grid(True, alpha=0.3)
            for s in ax.spines.values():
                s.set_color('#dee2e6')

        # ECG artists (created once)
        self.ecg_t_window = 30.0
        self.line_ecg, = self.ax_ecg.plot([], [], '-', linewidth=1.0, alpha=0.9)
        self.rpk_line, = self.ax_ecg.plot([], [], 'o', markersize=3, linestyle='None')

        self.ax_ecg.set_title("Real-time ECG (Last 30s)", fontsize=12, fontweight='bold')
        self.ax_ecg.set_xlabel("Time (s)")
        self.ax_ecg.set_ylabel("ECG (mV)")

        # PSD artists (created once)
        self.line_psd, = self.ax_psd.semilogy([], [], linewidth=1.3)
        self._lf_span = self.ax_psd.axvspan(LF_BAND[0], LF_BAND[1], alpha=0.2, color='orange')
        self._hf_span = self.ax_psd.axvspan(HF_BAND[0], HF_BAND[1], alpha=0.2, color='green')
        self.ax_psd.set_title("Power Spectral Density", fontsize=12, fontweight='bold')
        self.ax_psd.set_xlabel("Frequency (Hz)")
        self.ax_psd.set_ylabel("PSD")
        self.ax_psd.set_xlim(0, 2.0)  # Extended to 2.0 Hz as requested

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def _create_control_panel(self, parent):
        """Create enhanced control panel."""
        results_card = ttk.Labelframe(parent, text="ANALYSIS RESULTS",
                                      style="Card.TLabelframe", padding=15)
        results_card.pack(fill="x", pady=(0, 3))

        self.pred_var = tk.StringVar(value="Prediction: Awaiting data...")
        pred_label = ttk.Label(results_card, textvariable=self.pred_var,
                               font=("Segoe UI", 16, "bold"),
                               foreground="#2c3e50")
        pred_label.pack(anchor="w", pady=(0, 3))

        metrics_grid = ttk.Frame(results_card)
        metrics_grid.pack(fill="x", pady=5)

        self.time_var = tk.StringVar(value="Processing Time: –")
        self.rp_var = tk.StringVar(value="R-Peaks Detected: –")
        self.fs_var = tk.StringVar(value="Sampling Rate: –")
        self.lf_var = tk.StringVar(value="LF Power: –")
        self.hf_var = tk.StringVar(value="HF Power: –")
        self.total_power_var = tk.StringVar(value="Total Power: –")  # Changed from ratio_var
        self.confidence_var = tk.StringVar(value="Confidence: –")

        metrics = [
            (self.time_var, self.rp_var),
            (self.fs_var, self.confidence_var),
            (self.lf_var, self.hf_var),
            (self.total_power_var,)  # Changed from ratio_var
        ]

        for i, row in enumerate(metrics):
            row_frame = ttk.Frame(metrics_grid)
            row_frame.pack(fill="x", pady=2)
            for j, var in enumerate(row):
                ttk.Label(row_frame, textvariable=var, font=("Segoe UI", 10)).pack(
                    side="left", padx=(0, 20))

        action_frame = ttk.Frame(results_card)
        action_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(action_frame, text="Export Results", style="Success.TButton",
                   command=self.export_results).pack(side="left", padx=(0, 10))

        ttk.Button(action_frame, text="Clear Analysis", style="Danger.TButton",
                   command=self.clear_analysis).pack(side="left")

        # REAL-TIME STREAMING
        rt_card = ttk.Labelframe(parent, text="REAL-TIME STREAMING",
                                 style="Card.TLabelframe", padding=15)
        rt_card.pack(fill="x", pady=(0, 3))

        port_frame = ttk.Frame(rt_card)
        port_frame.pack(fill="x", pady=(0, 3))

        ttk.Label(port_frame, text="COM Port:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.port_combo = ttk.Combobox(port_frame, width=25, state="readonly")
        self.port_combo.pack(side="left", padx=10)

        ttk.Button(port_frame, text="Refresh", style="Secondary.TButton",
                   command=self._refresh_ports).pack(side="left", padx=5)

        # One-line, equal-size buttons
        BTN_W = 16  # same width in text-units; tweak as you like

        btn_row = ttk.Frame(rt_card)
        btn_row.pack(fill="x", pady=6)

        self.connect_btn = ttk.Button(btn_row, text="Connect", width=BTN_W,
                                    style="Primary.TButton", command=self._connect_shimmer)
        self.connect_btn.pack(side="left", padx=(0, 8))

        self.disconnect_btn = ttk.Button(btn_row, text="Disconnect", width=BTN_W,
                                        style="Danger.TButton", command=self._disconnect_shimmer)
        self.disconnect_btn.pack(side="left", padx=8)

        self.start_stream_btn = ttk.Button(btn_row, text="Start Streaming", width=BTN_W,
                                        style="Success.TButton", command=self._start_realtime)
        self.start_stream_btn.pack(side="left", padx=8)

        self.stop_stream_btn = ttk.Button(btn_row, text="Stop Streaming", width=BTN_W,
                                        style="Danger.TButton", command=self._stop_realtime)
        self.stop_stream_btn.pack(side="left", padx=8)

        status_frame = ttk.Frame(rt_card)
        status_frame.pack(fill="x", pady=(10, 0))

        self.conn_status = ttk.Label(status_frame, text="● Disconnected", style="Bad.TLabel")
        self.conn_status.pack(side="left", padx=(0, 20))

        self.stream_status = ttk.Label(status_frame, text="● Inactive", style="Warn.TLabel")
        self.stream_status.pack(side="left", padx=(0, 20))

        self.data_quality_status = ttk.Label(status_frame, text="Data Quality: –", style="Info.TLabel")
        self.data_quality_status.pack(side="left")

        # FILE ANALYSIS
        file_card = ttk.Labelframe(parent, text="FILE ANALYSIS",
                                   style="Card.TLabelframe", padding=15)
        file_card.pack(fill="x", pady=(0, 3))

        ttk.Button(file_card, text="Load & Analyze CSV File", style="Primary.TButton",
                   command=self._load_and_analyze_csv).pack(fill="x", pady=5)

    def _update_collection_progress(self):
        """Update progress display for data collection."""
        if self._collection_start_time is None:
            return 1.0  # Collection complete or not required
        
        elapsed = time.time() - self._collection_start_time
        progress = min(elapsed / self._collection_duration, 1.0)
        remaining = max(0, self._collection_duration - elapsed)
        
        # Update progress in UI
        if progress < 1.0:
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            self.pred_var.set(f"Prediction: Collecting data... ({minutes}:{seconds:02d} remaining)")
            self.progress["value"] = progress * 100
        else:
            self.pred_var.set("Prediction: Analyzing data...")
            self.progress["value"] = 100
        
        return progress

    def _get_remaining_time(self):
        """Get formatted remaining collection time."""
        if self._collection_start_time is None:
            return "0:00"
        elapsed = time.time() - self._collection_start_time
        remaining = max(0, self._collection_duration - elapsed)
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        return f"{minutes}:{seconds:02d}"

    @handle_errors(logger=None)
    def _autoload_model(self):
        """Enhanced model autoloading."""
        p = Path(DEFAULT_MODEL_PATH)
        if p.exists():
            try:
                self.model = joblib.load(p)
                self.scaler = getattr(self.model, "scaler_", None)
                self.model_status.config(
                    text=f"Model: {p.name} (v{getattr(self.model, 'version', '1.0')})",
                    style="Good.TLabel"
                )
                self.logger.info(f"Successfully autoloaded model from {p}")
            except Exception as e:
                self.model = None
                self.scaler = None
                self.model_status.config(text="Model: Autoload failed", style="Bad.TLabel")
                self.logger.error(f"Failed to autoload model: {e}")

    @handle_errors(logger=None)
    def load_model(self):
        """Enhanced model loading with validation."""
        p = filedialog.askopenfilename(
            title="Select Trained Model",
            filetypes=[("Pickle Model", "*.pkl"), ("All Files", "*.*")]
        )
        if not p:
            return

        try:
            self.model = joblib.load(p)
            self.scaler = getattr(self.model, "scaler_", None)

            if not hasattr(self.model, 'predict'):
                raise AttributeError("Model does not have predict method")

            model_name = Path(p).name
            self.model_status.config(
                text=f"Model: {model_name} (Loaded)",
                style="Good.TLabel"
            )

            self.status_var.set(f"Model {model_name} loaded successfully")
            self.logger.info(f"Model loaded successfully from {p}")

        except Exception:
            self.model = None
            self.scaler = None
            self.model_status.config(text="Model: Load failed", style="Bad.TLabel")
            raise

    # ========= ENHANCED COM PORT HANDLING =========

    @handle_com_errors(logger=None)
    def _refresh_ports(self):
        """Enhanced port refresh with status update and timeout handling."""
        self.status_var.set("Scanning for available ports...")
        self.progress.start()
        try:
            self._set_com_controls_state(False)
            ports = self.shimmer_reader.list_available_ports() if self.shimmer_reader else []
            items = [f"{dev} — {desc}" for dev, desc in ports]
            self.port_combo["values"] = items

            if items:
                self.port_combo.current(0)
                self.status_var.set(f"Found {len(items)} available port(s)")
                self.logger.info(f"Found {len(items)} COM ports")
            else:
                self.status_var.set("No COM ports found")
                self.logger.warning("No COM ports detected")

        except Exception as e:
            self.status_var.set("Error scanning ports")
            self.logger.error(f"Port scanning failed: {e}")
            raise
        finally:
            self.progress.stop()
            self._set_com_controls_state(True)

    @handle_com_errors(logger=None)
    def _connect_shimmer(self):
        """Enhanced device connection with comprehensive error handling."""
        if not self.shimmer_reader:
            raise RuntimeError("Real-time reader not available")

        sel = self.port_combo.get()
        if not sel:
            raise RuntimeError("Please select a COM port first")

        port = sel.split(" — ")[0].strip()
        self.status_var.set(f"Connecting to {port}...")
        self.progress.start()

        try:
            self._set_com_controls_state(False)
            self.connect_btn.config(state="disabled", text="Connecting...")

            if self.shimmer_reader.is_open:
                self.logger.warning(f"Already connected to {port}")
                self.status_var.set(f"Already connected to {port}")
                return

            ok = self.shimmer_reader.connect(port)
            if ok:
                self.conn_status.config(text="● Connected", style="Good.TLabel")
                self.status_var.set(f"Successfully connected to {port}")
                self.logger.info(f"Connected to Shimmer device on {port}")
                self.start_stream_btn.config(state="normal")
                self.stop_stream_btn.config(state="normal")
            else:
                self.conn_status.config(text="● Disconnected", style="Bad.TLabel")
                raise RuntimeError(f"Failed to connect to {port}")

        except Exception as e:
            self.conn_status.config(text="● Disconnected", style="Bad.TLabel")
            self.status_var.set(f"Connection failed: {str(e)}")
            self.logger.error(f"Connection to {port} failed: {e}")
            raise
        finally:
            self.progress.stop()
            self._set_com_controls_state(True)
            self.connect_btn.config(state="normal", text="Connect")

    @handle_com_errors(logger=None)
    def _disconnect_shimmer(self):
        """Enhanced device disconnection with error handling."""
        if self.shimmer_reader:
            try:
                self.status_var.set("Disconnecting from device...")
                self._stop_realtime()
                self.shimmer_reader.disconnect()
            except Exception as e:
                self.logger.error(f"Error during disconnection: {e}")
            finally:
                self.conn_status.config(text="● Disconnected", style="Bad.TLabel")
                self.stream_status.config(text="● Inactive", style="Warn.TLabel")
                self.data_quality_status.config(text="Data Quality: –")
                self.status_var.set("Disconnected from device")
                self.logger.info("Disconnected from Shimmer device")
                self.start_stream_btn.config(state="disabled")
                self.stop_stream_btn.config(state="disabled")

    def _set_com_controls_state(self, enabled: bool):
        """Enable or disable COM port controls."""
        state = "normal" if enabled else "disabled"
        self.port_combo.config(state=state)
        self.connect_btn.config(state=state)
        self.disconnect_btn.config(state=state)

        if enabled and self.shimmer_reader and self.shimmer_reader.is_open:
            self.start_stream_btn.config(state="normal")
            self.stop_stream_btn.config(state="normal")
        else:
            self.start_stream_btn.config(state="disabled")
            self.stop_stream_btn.config(state="disabled")

    # ========= REAL-TIME LOOP (SMOOTH) =========

    @handle_errors(logger=None)
    def _start_realtime(self):
        """Start real-time streaming with 5-minute collection requirement."""
        if not self.shimmer_reader or not self.shimmer_reader.is_open:
            messagebox.showerror("Connection Error", "Please connect to a device first")
            return

        try:
            self.shimmer_reader.clear_buffer()
            self.shimmer_reader.start_streaming()
            self.realtime_active = True
            self._worker_running = True
            
            # Reset analysis state
            self._final_analysis_computed = False
            self._final_features = None
            self._full_data_ecg = None
            self._full_data_r_peaks = 0
            
            # Start the 5-minute collection timer
            self._collection_start_time = time.time()
            self.stream_status.config(text="● Collecting data...", style="Warn.TLabel")
            self.status_var.set("Collecting data... (5 minutes required before analysis)")
            
            # Disable prediction display initially
            self.pred_var.set("Prediction: Collecting data... (5:00 remaining)")
            self.confidence_var.set("Please wait for sufficient data collection")

            # Start worker threads
            threading.Thread(target=self._rt_worker, daemon=True).start()
            self.after(self._ui_fps_ms, self._rt_ui_tick)

            self.logger.info("Started real-time streaming with 5-minute collection requirement")

        except Exception as e:
            self.stream_status.config(text="● Inactive", style="Bad.TLabel")
            error_msg = f"Failed to start streaming: {str(e)}"
            self.status_var.set(error_msg)
            self.logger.error(error_msg)
            messagebox.showerror("Streaming Error", error_msg)

    @handle_errors(logger=None)
    def _stop_realtime(self):
        """Stop real-time streaming and reset analysis state."""
        self.realtime_active = False
        self._worker_running = False
        self._collection_start_time = None  # Reset collection timer
        self._final_analysis_computed = False  # Reset final analysis flag
        self._final_features = None  # Clear final features
        self._full_data_ecg = None  # Clear full data buffer
        self._full_data_r_peaks = 0  # Reset full data R-peaks
        self.progress["value"] = 0  # Reset progress bar
        self.stream_status.config(text="● Inactive", style="Warn.TLabel")
        self.status_var.set("Real-time streaming stopped")

        if self.shimmer_reader:
            try:
                self.shimmer_reader.stop_streaming()
                self.logger.info("Stopped real-time streaming")
            except Exception as e:
                self.logger.error(f"Error stopping streaming: {e}")

    def _rt_worker(self):
        """Background DSP thread - stops analysis after 5 minutes."""
        while self._worker_running and self.shimmer_reader:
            try:
                # Check collection progress
                collection_progress = self._update_collection_progress()
                
                ts, ecg = self.shimmer_reader.get_buffered_data()
                fs = float(getattr(self.shimmer_reader, 'sampling_rate', 128.0))
                
                # More permissive data check
                if ecg.size < max(int(2*fs), 256):
                    time.sleep(0.1)
                    continue
                            
                # Use timestamp to detect new data instead of buffer size
                current_time = time.time()
                if hasattr(self, '_last_process_time') and (current_time - self._last_process_time) < 0.1:
                    time.sleep(0.05)
                    continue
                            
                start_time = time.time()
                self._last_process_time = current_time

                # Store full data for final analysis - FIXED ACCUMULATION
                if collection_progress < 1.0:
                    if self._full_data_ecg is None:
                        self._full_data_ecg = ecg.copy()
                    else:
                        # Only append NEW data that we haven't stored yet
                        current_buffer_size = len(ecg)
                        stored_size = len(self._full_data_ecg)
                        
                        if current_buffer_size > stored_size:
                            new_data = ecg[stored_size:]
                            self._full_data_ecg = np.concatenate([self._full_data_ecg, new_data])
                            self.logger.debug(f"Accumulated data: {len(self._full_data_ecg)} samples")

                # Process data with error resilience
                try:
                    # Use 30-second window for real-time display
                    display_window_s = 30
                    display_samples = int(display_window_s * fs)
                    seg = ecg[-display_samples:].astype(np.float32, copy=False) if ecg.size > display_samples else ecg.astype(np.float32, copy=False)

                    rr_t, rr, n_peaks, peaks, ecg_f = rr_from_ecg(seg, fs)

                    now = time.time()
                    feats = None
                    
                    # Only compute features during collection OR if we haven't computed final features yet
                    if rr.size >= 8:
                        should_compute_features = False
                        
                        if collection_progress < 1.0:
                            # During collection: compute periodically to show progress
                            if (now - self._last_psd_ts) >= self._psd_period_s:
                                should_compute_features = True
                        else:
                            # After collection: compute final analysis ONCE on full data
                            if not self._final_analysis_computed:
                                should_compute_features = True
                                self._final_analysis_computed = True
                                
                                # Compute final R-peaks on full 5-minute data - FIXED
                                if self._full_data_ecg is not None and len(self._full_data_ecg) > int(10 * fs):  # Reduced minimum
                                    try:
                                        self.logger.info(f"Starting final analysis on {len(self._full_data_ecg)} samples ({len(self._full_data_ecg)/fs:.1f} seconds)")
                                        
                                        # Process the ENTIRE accumulated data
                                        full_rr_t, full_rr, full_n_peaks, full_peaks, full_ecg_f = rr_from_ecg(self._full_data_ecg, fs)
                                        
                                        self._full_data_r_peaks = full_n_peaks
                                        self.logger.info(f"Final R-peak count on full data: {full_n_peaks} (expected: ~{int(len(self._full_data_ecg)/fs * 1.2)})")
                                        
                                        # Compute final features on full data
                                        if full_rr.size >= 8:  # Need sufficient RR intervals
                                            final_feats = features_from_rr(full_rr, full_rr_t, interp_fs=4.0, use_highpass=True)
                                            if final_feats:
                                                self._final_features = final_feats
                                                self.logger.info("Final analysis computed on full accumulated data")
                                            else:
                                                self.logger.warning("Final feature computation failed")
                                        else:
                                            self.logger.warning(f"Insufficient RR intervals for final analysis: {full_rr.size}")
                                            # Fall back to using the last good features
                                            if feats:
                                                self._final_features = feats
                                            
                                    except Exception as e:
                                        self.logger.error(f"Error computing final analysis on full data: {e}")
                                        # Fall back to current features if available
                                        if feats:
                                            self._final_features = feats
                                else:
                                    self.logger.warning(f"Full data insufficient for final analysis: {len(self._full_data_ecg) if self._full_data_ecg is not None else 0} samples")
                                    # Use current features if full data analysis fails
                                    if feats:
                                        self._final_features = feats
                            
                        # Only compute if we should, and skip if we already have final features
                        if should_compute_features and (collection_progress < 1.0 or not self._final_analysis_computed):
                            try:
                                feats = features_from_rr(rr, rr_t, interp_fs=4.0, use_highpass=True)
                                self._last_psd_ts = now
                                
                                if collection_progress < 1.0:
                                    self.stream_status.config(text="● Analyzing", style="Good.TLabel")
                                    self.status_var.set("Real-time analysis active")
                                
                            except Exception as e:
                                # Don't fail entire processing if feature extraction fails
                                self.logger.debug(f"Feature extraction failed: {e}")
                                feats = None

                except Exception as e:
                    # Log but don't crash on signal processing errors
                    if random.random() < 0.05:
                        self.logger.debug(f"Signal processing error: {e}")
                    continue

                processing_time = (time.time() - start_time) * 1000

                with self._rt_lock:
                    self._dsp_queue.clear()
                    self._dsp_queue.append({
                        "fs": fs,
                        "n_peaks": int(n_peaks),  # Real-time 30-second window peaks
                        "peaks": peaks,  # All peaks in the 30-second window
                        "ecg_tail": seg,  # Full 30-second window for display
                        "feats": feats,
                        "processing_time": processing_time,
                        "collection_complete": collection_progress >= 1.0,
                        "collection_progress": collection_progress,
                        "final_analysis_computed": self._final_analysis_computed,
                        "full_data_r_peaks": self._full_data_r_peaks,  # Full accumulated peaks
                        "full_data_samples": len(self._full_data_ecg) if self._full_data_ecg is not None else 0  # Debug info
                    })

                time.sleep(0.05)
                
            except Exception as e:
                self.logger.warning(f"Worker loop error: {e}")
                time.sleep(0.1)

    def _rt_ui_tick(self):
        """Lightweight UI tick - shows final results after 5 minutes."""
        if not self.realtime_active or not self.shimmer_reader:
            return

        try:
            # Update collection progress
            collection_progress = self._update_collection_progress()
            
            # Update data quality
            quality = self.shimmer_reader.data_quality
            if quality > 80:
                quality_text = f"Data Quality: Good"
                quality_style = "Good.TLabel"
            elif quality > 60:
                quality_text = f"Data Quality: Fair"
                quality_style = "Warn.TLabel"
            else:
                quality_text = f"Data Quality: Poor"
                quality_style = "Bad.TLabel"
                
            self.data_quality_status.config(text=quality_text, style=quality_style)

            # Get latest DSP payload
            payload = None
            with self._rt_lock:
                if self._dsp_queue:
                    payload = self._dsp_queue[-1]

            if payload:
                # Display processing time
                if "processing_time" in payload:
                    self.time_var.set(f"Processing Time: {payload['processing_time']:.1f} ms")

                # Update status - FINAL STATE AFTER COLLECTION
                if payload.get("collection_complete", False):
                    if payload.get("final_analysis_computed", False):
                        # FINAL STATE: Show completed analysis
                        self.stream_status.config(text="● Analysis Complete", style="Good.TLabel")
                        self.status_var.set("5-minute data collection complete - Final results ready")
                        
                        # STOP further real-time analysis updates
                        self._worker_running = False
                    else:
                        self.stream_status.config(text="● Finalizing Analysis", style="Warn.TLabel")
                        self.status_var.set("Computing final analysis...")
                else:
                    progress_percent = payload.get("collection_progress", 0) * 100
                    self.stream_status.config(text=f"● Collecting ({progress_percent:.0f}%)", style="Warn.TLabel")

                # Use final features after collection, otherwise use current features
                if payload.get("collection_complete", False) and self._final_analysis_computed and self._final_features:
                    display_feats = self._final_features
                else:
                    display_feats = payload.get("feats")

                # Show correct R-peak count
                if payload.get("collection_complete", False) and payload.get("final_analysis_computed", False):
                    # After collection: show total R-peaks from full 5-minute data
                    r_peaks_display = payload.get("full_data_r_peaks", 0)
                    total_samples = payload.get("full_data_samples", 0)
                    duration = total_samples / payload['fs'] if payload['fs'] > 0 else 0
                    expected_r_peaks = int(duration * 1.2)  # Rough estimate: 72 BPM
                    self.rp_var.set(f"R-Peaks: {r_peaks_display} (full {duration:.0f}s)")
                else:
                    # During collection: show real-time R-peaks from 30-second window
                    r_peaks_display = payload.get("n_peaks", 0)
                    self.rp_var.set(f"R-Peaks: {r_peaks_display} (30s window)")

                # Show prediction when we have features
                if display_feats and payload.get("collection_complete", False) and self._final_analysis_computed:
                    self.fs_var.set(f"Sampling Rate: {payload['fs']:.1f} Hz")
                    self.lf_var.set(f"LF Power: {display_feats['LF_ms2']:.2f}" if np.isfinite(display_feats['LF_ms2']) else "LF Power: N/A")
                    self.hf_var.set(f"HF Power: {display_feats['HF_ms2']:.2f}" if np.isfinite(display_feats['HF_ms2']) else "HF Power: N/A")
                    self.total_power_var.set(f"Total Power: {display_feats['Total_Power_ms2']:.2f}" if np.isfinite(display_feats['Total_Power_ms2']) else "Total Power: N/A")

                    self.cached_feats = (display_feats["LF_ms2"], display_feats["HF_ms2"], display_feats["Total_Power_ms2"])
                    self.predict_cached()
                    
                elif not payload.get("collection_complete", False):
                    # During collection
                    self.pred_var.set(f"Prediction: Collecting data... ({self._get_remaining_time()})")
                    self.confidence_var.set("Minimum 5 minutes data required")
                    self.lf_var.set("LF Power: –")
                    self.hf_var.set("HF Power: –")
                    self.total_power_var.set("Total Power: –")
                elif payload.get("collection_complete", False) and not self._final_analysis_computed:
                    # Collection complete but analysis not done yet
                    self.pred_var.set("Prediction: Finalizing analysis...")
                    self.confidence_var.set("Please wait")

                # Update ECG plot (30-second window) - always show current data for visualization
                fs = payload["fs"]
                seg = payload["ecg_tail"]
                if seg.size:
                    t = np.arange(seg.size, dtype=np.float32) / fs
                    self.line_ecg.set_data(t, seg)
                    # Set x-axis to show 0-30 seconds
                    self.ax_ecg.set_xlim(0, 30)
                    ymin, ymax = float(seg.min()), float(seg.max())
                    pad = max(0.2, 0.1*(ymax - ymin + 1e-6))
                    self.ax_ecg.set_ylim(ymin - pad, ymax + pad)

                # R-peak markers for the 30-second segment
                if payload["peaks"].size and seg.size:
                    # All peaks are already within the 30-second window
                    rp_t = payload["peaks"] / fs
                    rp_y = seg[payload["peaks"]]
                    self.rpk_line.set_data(rp_t, rp_y)
                else:
                    self.rpk_line.set_data([], [])

                # Update PSD plot - show final PSD after collection, otherwise current PSD
                current_feats = payload.get("feats")
                if current_feats and "freqs" in current_feats and "psd" in current_feats:
                    # Show current PSD during collection, final PSD after collection
                    if payload.get("collection_complete", False) and self._final_analysis_computed and self._final_features:
                        # Use final PSD
                        f, p = self._final_features["freqs"], self._final_features["psd"]
                    else:
                        # Use current PSD
                        f, p = current_feats["freqs"], current_feats["psd"]
                    
                    self.line_psd.set_data(f, p)
                    self.ax_psd.set_xlim(0, 2.0)  # Extended to 2.0 Hz as requested
                    if np.isfinite(p).any():
                        self.ax_psd.set_ylim(max(np.nanmin(p)*0.8, 1e-6), np.nanmax(p)*1.2)

                self.canvas.draw_idle()

        except Exception as e:
            self.logger.debug(f"UI update error: {e}")
            
        self.after(self._ui_fps_ms, self._rt_ui_tick)

    # ========= ENHANCED CSV WORKFLOW =========

    @handle_errors(logger=None)
    def _load_and_analyze_csv(self):
        """Enhanced CSV analysis with progress tracking."""
        p = filedialog.askopenfilename(
            title="Open Shimmer CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not p:
            return

        start_time = time.time()

        try:
            self.status_var.set("Loading CSV file...")
            df = load_shimmer_csv(p)

            self.df_raw = df

            self.status_var.set("Detecting data columns...")
            self.ts_col = detect_timestamp_col(df)
            if self.ts_col is None:
                raise RuntimeError("No timestamp column found in the CSV file")

            self.ecg_col = pick_ecg_column(df, self.ts_col)
            if self.ecg_col is None:
                raise RuntimeError("No suitable ECG column found in the CSV file")

            self.status_var.set("Processing ECG signal...")
            ecg, fs = self._get_current_signal()
            if len(ecg) < 10:
                raise RuntimeError("ECG signal too short for analysis")

            self.status_var.set("Detecting R-peaks...")
            rr_t, rr, n_peaks, peaks, ecg_f = rr_from_ecg(ecg, fs)
            if len(rr) < 10:
                raise RuntimeError("Insufficient R-peaks detected for reliable analysis")

            self.status_var.set("Computing HRV features...")
            feats = features_from_rr(rr, rr_t, interp_fs=4.0, use_highpass=True)
            if feats is None:
                raise RuntimeError("Failed to compute HRV features")

            self._update_results(fs, n_peaks, feats)
            self.cached_feats = (feats["LF_ms2"], feats["HF_ms2"], feats["Total_Power_ms2"])

            self.status_var.set("Making prediction...")
            self.predict_cached()

            self.status_var.set("Updating visualizations...")
            self._plot_enhanced_analysis(ecg_f=ecg_f, fs=fs, peaks_idx=peaks, feats=feats)

            # Calculate and display processing time
            processing_time = (time.time() - start_time) * 1000
            self.last_processing_time = processing_time
            self.time_var.set(f"Processing Time: {processing_time:.0f} ms")

            self.status_var.set("Analysis completed successfully")
            self.logger.info(f"CSV analysis completed in {processing_time:.0f} ms")

        except Exception:
            self.status_var.set("Analysis failed")
            raise

    def _get_current_signal(self) -> Tuple[np.ndarray, float]:
        """Enhanced signal extraction with validation."""
        if self.df_raw is None:
            raise RuntimeError("No data loaded")
        if self.ts_col is None or self.ecg_col is None:
            raise RuntimeError("Timestamp or ECG column not detected")

        t_vals = pd.to_numeric(self.df_raw[self.ts_col], errors="coerce").values
        if np.all(np.isnan(t_vals)):
            raise RuntimeError("Timestamp column contains invalid data")

        fs_est = estimate_fs_from_times(t_vals)
        fs = fs_est if fs_est and np.isfinite(fs_est) else 128.0

        ecg = pd.to_numeric(self.df_raw[self.ecg_col], errors="coerce").astype(float).values
        valid_mask = np.isfinite(ecg)
        if np.sum(valid_mask) < len(ecg) * 0.8:
            self.logger.warning("ECG signal contains significant missing data")

        ecg = ecg[valid_mask]
        if len(ecg) < 10:
            raise RuntimeError("ECG signal too short after cleaning")

        return ecg, float(fs)

    def _plot_enhanced_analysis(self, ecg_f: np.ndarray, fs: float,
                                peaks_idx: Optional[np.ndarray], feats: dict):
        """Static plotting for file analysis (not per-frame)."""
        for ax in [self.ax_ecg, self.ax_psd]:
            ax.cla()
            ax.set_facecolor('#ffffff')
            ax.grid(True, alpha=0.3)
            for s in ax.spines.values():
                s.set_color('#dee2e6')

        if ecg_f is not None and len(ecg_f) > 0:
            view_window_s = min(60, len(ecg_f) / fs)
            samples_to_show = int(view_window_s * fs)
            sig = ecg_f[-samples_to_show:] if len(ecg_f) > samples_to_show else ecg_f
            t_ecg = np.arange(len(sig)) / fs

            self.ax_ecg.plot(t_ecg, sig, 'b-', linewidth=1.0, alpha=0.8, label='ECG')

            if peaks_idx is not None and len(peaks_idx) > 0:
                mask = (peaks_idx >= max(0, len(ecg_f) - samples_to_show)) & (peaks_idx < len(ecg_f))
                pk_t = (peaks_idx[mask] - max(0, len(ecg_f) - samples_to_show)) / fs
                pk_y = ecg_f[peaks_idx[mask]]
                self.ax_ecg.scatter(pk_t, pk_y, color='red', s=20, label='R-peaks')

            self.ax_ecg.set_title("ECG Signal with R-Peak Detection", fontsize=12, fontweight='bold')
            self.ax_ecg.set_xlabel("Time (s)")
            self.ax_ecg.set_ylabel("Amplitude")
            self.ax_ecg.legend()

        if feats is not None and "freqs" in feats and "psd" in feats:
            f, p = feats["freqs"], feats["psd"]
            self.ax_psd.semilogy(f, p, 'purple', linewidth=1.5)
            self.ax_psd.axvspan(LF_BAND[0], LF_BAND[1], alpha=0.2, color='orange', label='LF Band')
            self.ax_psd.axvspan(HF_BAND[0], HF_BAND[1], alpha=0.2, color='green', label='HF Band')
            self.ax_psd.set_title("Power Spectral Density", fontsize=12, fontweight='bold')
            self.ax_psd.set_xlabel("Frequency (Hz)")
            self.ax_psd.set_ylabel("PSD (ms²/Hz)")
            self.ax_psd.set_xlim(0, 2.0)  # Extended to 2.0 Hz as requested
            self.ax_psd.legend()

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ========= ENHANCED RESULTS & PREDICTION =========

    def _update_results(self, fs, n_peaks, feats):
        """Enhanced results update with validation."""
        self.fs_var.set(f"Sampling Rate: {fs:.2f} Hz" if fs else "Sampling Rate: –")
        self.rp_var.set(f"R-Peaks Detected: {n_peaks}")

        lf_val = feats['LF_ms2']
        hf_val = feats['HF_ms2']
        total_power_val = feats['Total_Power_ms2']

        self.lf_var.set(f"LF Power: {lf_val:.2f}" if np.isfinite(lf_val) else "LF Power: N/A")
        self.hf_var.set(f"HF Power: {hf_val:.2f}" if np.isfinite(hf_val) else "HF Power: N/A")
        self.total_power_var.set(f"Total Power: {total_power_val:.2f}" if np.isfinite(total_power_val) else "Total Power: N/A")

    @handle_errors(logger=None)
    def predict_cached(self):
        """Enhanced prediction with confidence scoring."""
        if self.model is None:
            self.pred_var.set("Prediction: Load model first")
            self.confidence_var.set("Confidence: –")
            return

        if self.cached_feats is None:
            self.pred_var.set("Prediction: Analyze data first")
            self.confidence_var.set("Confidence: –")
            return

        lf, hf, total_power = self.cached_feats

        if not np.isfinite([lf, hf, total_power]).all():
            self.pred_var.set("Prediction: Invalid features")
            self.confidence_var.set("Confidence: –")
            return

        if lf > 1e6 or hf > 1e6 or total_power > 1e6:
            self.pred_var.set("Prediction: Features out of range")
            self.confidence_var.set("Confidence: –")
            return

        start_time = time.time()
        X = np.array([[lf, hf, total_power]], float)

        try:
            if self.scaler is not None:
                X = self.scaler.transform(X)

            if hasattr(self.model, "predict_proba"):
                prob = float(self.model.predict_proba(X)[0, 1])
                pred = "Breast cancer" if prob >= 0.5 else "Healthy"
                confidence = max(prob, 1 - prob)

                # include p in label
                if "cancer" in pred.lower():
                    self.pred_var.set(f"Prediction: {pred}")
                else:
                    self.pred_var.set(f"Prediction: {pred}")

                self.confidence_var.set(f"Confidence: {confidence:.1%}")

            else:
                y = float(self.model.predict(X)[0])
                pred = "Breast cancer" if y >= 0.5 else "Healthy"
                self.pred_var.set(f"Prediction: {pred} (score={y:.3f})")
                self.confidence_var.set("Confidence: –")

            self.last_processing_time = (time.time() - start_time) * 1000

        except Exception:
            self.pred_var.set("Prediction: Error")
            self.confidence_var.set("Confidence: –")
            raise

    # ========= ENHANCED EXPORT FUNCTIONALITY =========

    def _get_export_duration(self):
        """Get the duration of collected data in seconds."""
        if hasattr(self, '_full_data_ecg') and self._full_data_ecg is not None:
            fs = float(getattr(self.shimmer_reader, 'sampling_rate', 128.0))
            return len(self._full_data_ecg) / fs
        return 0

    def _calculate_30s_rr_intervals(self):
        """Calculate RR intervals for every 30-second segment."""
        if not hasattr(self, '_full_data_ecg') or self._full_data_ecg is None:
            return None
        
        fs = float(getattr(self.shimmer_reader, 'sampling_rate', 128.0))
        ecg_data = self._full_data_ecg
        total_duration = len(ecg_data) / fs
        segment_duration = 30  # seconds
        
        segments_data = []
        
        for segment_start in np.arange(0, total_duration, segment_duration):
            segment_end = min(segment_start + segment_duration, total_duration)
            start_sample = int(segment_start * fs)
            end_sample = int(segment_end * fs)
            
            if end_sample - start_sample < int(10 * fs):  # Skip segments with less than 10s data
                continue
                
            segment_ecg = ecg_data[start_sample:end_sample]
            
            # Detect R-peaks in this segment
            rr_t, rr, n_peaks, peaks, ecg_f = rr_from_ecg(segment_ecg, fs)
            
            if len(rr) > 0:
                segment_stats = {
                    'segment_start_time_s': segment_start,
                    'segment_end_time_s': segment_end,
                    'r_peaks_detected': n_peaks,
                    'mean_rr_interval_ms': np.mean(rr) * 1000 if len(rr) > 0 else 0,
                    'std_rr_interval_ms': np.std(rr) * 1000 if len(rr) > 0 else 0,
                    'min_rr_interval_ms': np.min(rr) * 1000 if len(rr) > 0 else 0,
                    'max_rr_interval_ms': np.max(rr) * 1000 if len(rr) > 0 else 0,
                    'mean_heart_rate_bpm': 60 / np.mean(rr) if len(rr) > 0 and np.mean(rr) > 0 else 0
                }
                
                # Add individual RR intervals
                for i, (rr_time, rr_interval) in enumerate(zip(rr_t, rr)):
                    segment_stats[f'rr_interval_{i+1}_ms'] = rr_interval * 1000
                    
                segments_data.append(segment_stats)
        
        return pd.DataFrame(segments_data) if segments_data else None

    def _save_plot_images(self, export_dir):
        """Save plot images for the analysis."""
        try:
            # Save current PSD plot
            psd_path = os.path.join(export_dir, "power_spectral_density.png")
            self.fig.savefig(psd_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            # Create and save 30-second segment plots
            if hasattr(self, '_full_data_ecg') and self._full_data_ecg is not None:
                self._save_30s_segment_plots(export_dir)
                
        except Exception as e:
            self.logger.warning(f"Could not save plot images: {e}")

    def _save_30s_segment_plots(self, export_dir):
        """Save ECG plots for every 30-second segment."""
        fs = float(getattr(self.shimmer_reader, 'sampling_rate', 128.0))
        ecg_data = self._full_data_ecg
        total_duration = len(ecg_data) / fs
        segment_duration = 30  # seconds
        
        plots_dir = os.path.join(export_dir, "segment_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        for segment_idx, segment_start in enumerate(np.arange(0, total_duration, segment_duration)):
            segment_end = min(segment_start + segment_duration, total_duration)
            start_sample = int(segment_start * fs)
            end_sample = int(segment_end * fs)
            
            if end_sample - start_sample < int(10 * fs):  # Skip segments with less than 10s data
                continue
                
            segment_ecg = ecg_data[start_sample:end_sample]
            
            # Create plot for this segment
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Plot ECG
            t_segment = np.arange(len(segment_ecg)) / fs
            ax.plot(t_segment, segment_ecg, 'b-', linewidth=1, alpha=0.8, label='ECG')
            
            # Detect and plot R-peaks
            rr_t, rr, n_peaks, peaks, ecg_f = rr_from_ecg(segment_ecg, fs)
            
            if peaks.size > 0:
                r_peak_times = peaks / fs
                r_peak_values = segment_ecg[peaks]
                ax.scatter(r_peak_times, r_peak_values, color='red', s=30, 
                          label=f'R-peaks ({n_peaks} detected)', zorder=5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('ECG Amplitude (mV)')
            ax.set_title(f'ECG Segment {segment_idx+1}: {segment_start:.1f}s - {segment_end:.1f}s\n'
                        f'{n_peaks} R-peaks detected, Mean HR: {60/np.mean(rr) if len(rr) > 0 else 0:.1f} BPM')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save segment plot
            segment_plot_path = os.path.join(plots_dir, f"segment_{segment_idx+1:02d}.png")
            fig.savefig(segment_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

    def _create_summary_report(self, export_dir, main_results, rr_data):
        """Create a comprehensive summary report."""
        report_path = os.path.join(export_dir, "analysis_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("BREAST CANCER CLASSIFIER - ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Prediction: {main_results['prediction'].iloc[0]}\n")
            f.write(f"Confidence: {main_results['confidence'].iloc[0]}\n")
            f.write(f"Data Duration: {main_results['data_duration_seconds'].iloc[0]:.1f} seconds\n")
            f.write(f"Total R-Peaks: {main_results['r_peaks_detected'].iloc[0]}\n")
            f.write(f"Sampling Rate: {main_results['sampling_rate_hz'].iloc[0]} Hz\n\n")
            
            f.write("HRV FEATURES:\n")
            f.write(f"  LF Power: {main_results['lf_power'].iloc[0]:.2f} ms²\n")
            f.write(f"  HF Power: {main_results['hf_power'].iloc[0]:.2f} ms²\n")
            f.write(f"  Total Power: {main_results['total_power'].iloc[0]:.2f} ms²\n\n")
            
            if rr_data is not None and not rr_data.empty:
                f.write("30-SECOND SEGMENT ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                for _, segment in rr_data.iterrows():
                    f.write(f"\nSegment: {segment['segment_start_time_s']:.1f}s - {segment['segment_end_time_s']:.1f}s\n")
                    f.write(f"  R-Peaks: {segment['r_peaks_detected']}\n")
                    f.write(f"  Mean RR: {segment['mean_rr_interval_ms']:.1f} ms\n")
                    f.write(f"  Mean HR: {segment['mean_heart_rate_bpm']:.1f} BPM\n")
                    f.write(f"  RR Std: {segment['std_rr_interval_ms']:.1f} ms\n")
            
            f.write(f"\nExported Files:\n")
            f.write(f"  - analysis_results.csv: Main results and features\n")
            if rr_data is not None:
                f.write(f"  - rr_interval_analysis.csv: Detailed RR interval data\n")
            f.write(f"  - power_spectral_density.png: PSD plot\n")
            f.write(f"  - segment_plots/: ECG plots for each 30-second segment\n")
            f.write(f"  - analysis_summary.txt: This summary file\n")

    @handle_errors(logger=None)
    def export_results(self):
        """Enhanced export with R-peak intervals and plot images every 30 seconds."""
        if self.cached_feats is None:
            raise RuntimeError("No analysis results to export")

        # Create export directory with timestamp
        export_dir = f"breast_cancer_analysis_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(export_dir, exist_ok=True)

        try:
            # Save main results CSV
            export_data = {
                'timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')],
                'prediction': [self.pred_var.get().replace('Prediction: ', '')],
                'confidence': [self.confidence_var.get().replace('Confidence: ', '')],
                'lf_power': [self.cached_feats[0]],
                'hf_power': [self.cached_feats[1]],
                'total_power': [self.cached_feats[2]],
                'processing_time_ms': [self.last_processing_time],
                'r_peaks_detected': [self.rp_var.get().replace('R-Peaks Detected: ', '').split(' ')[0]],
                'sampling_rate_hz': [self.fs_var.get().replace('Sampling Rate: ', '').replace(' Hz', '')],
                'data_duration_seconds': [self._get_export_duration()]
            }

            df_export = pd.DataFrame(export_data)
            
            # Add R-peak interval data if available
            # Add R-peak interval data if available
            if hasattr(self, '_full_data_ecg') and self._full_data_ecg is not None:
                rr_intervals_data = self._calculate_30s_rr_intervals()
                # FIX: Check if DataFrame exists and is not empty
                if rr_intervals_data is not None and not rr_intervals_data.empty:
                    df_export['rr_interval_data_available'] = [True]
                    # Save detailed RR interval data
                    rr_export_path = os.path.join(export_dir, "rr_interval_analysis.csv")
                    rr_intervals_data.to_csv(rr_export_path, index=False)
                else:
                    df_export['rr_interval_data_available'] = [False]
            else:
                df_export['rr_interval_data_available'] = [False]

            # Save main results
            main_export_path = os.path.join(export_dir, "analysis_results.csv")
            df_export.to_csv(main_export_path, index=False)

            # Save plot images
            self._save_plot_images(export_dir)

            # Create a summary report
            rr_data_for_report = rr_intervals_data if (rr_intervals_data is not None and not rr_intervals_data.empty) else None
            self._create_summary_report(export_dir, df_export, rr_data_for_report)

            self.status_var.set(f"Results exported to {export_dir}")
            self.logger.info(f"Complete analysis exported to {export_dir}")

            # Show success message with export location
            messagebox.showinfo("Export Complete", 
                              f"Analysis results exported to:\n{export_dir}\n\n"
                              f"Includes:\n• Main results CSV\n• RR interval analysis\n• Plot images\n• Summary report")

        except Exception as e:
            self.logger.error(f"Error during export: {e}")
            raise RuntimeError(f"Export failed: {str(e)}")

    # ========= ENHANCED UTILITIES =========

    @handle_errors(logger=None)
    def clear_analysis(self):
        """Clear current analysis and reset UI."""
        self.df_raw = None
        self.ts_col = None
        self.ecg_col = None
        self.cached_feats = None

        self.pred_var.set("Prediction: Awaiting data...")
        self.time_var.set("Processing Time: –")
        self.rp_var.set("R-Peaks Detected: –")
        self.fs_var.set("Sampling Rate: –")
        self.lf_var.set("LF Power: –")
        self.hf_var.set("HF Power: –")
        self.total_power_var.set("Total Power: –")
        self.confidence_var.set("Confidence: –")

        # Clear artists
        self.line_ecg.set_data([], [])
        self.rpk_line.set_data([], [])
        self.line_psd.set_data([], [])
        self.canvas.draw_idle()

        self.status_var.set("Analysis cleared")
        self.logger.info("Analysis cleared")

    def _extract_signal_from_df(self, df: pd.DataFrame, ts_col: str, ecg_col: str) -> Tuple[np.ndarray, float]:
        """Extract signal from DataFrame for batch processing."""
        t_vals = pd.to_numeric(df[ts_col], errors="coerce").values
        fs_est = estimate_fs_from_times(t_vals)
        fs = fs_est if fs_est and np.isfinite(fs_est) else 128.0
        ecg = pd.to_numeric(df[ecg_col], errors="coerce").astype(float).values
        return ecg, float(fs)

    def _show_settings(self):
        """Show application settings dialog."""
        settings_window = tk.Toplevel(self)
        settings_window.title("Application Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self)
        settings_window.grab_set()

        ttk.Label(settings_window, text="Settings", font=("Segoe UI", 14, "bold")).pack(pady=10)
        ttk.Label(settings_window, text="Settings interface to be implemented...").pack(pady=20)

        ttk.Button(settings_window, text="Close",
                   command=settings_window.destroy).pack(pady=10)

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'shimmer_reader') and self.shimmer_reader:
                self.shimmer_reader.disconnect()
            if hasattr(self, 'fig'):
                plt.close(self.fig)
        except Exception:
            pass

# ========= MAIN ENTRY POINT =========

if __name__ == "__main__":
    try:
        app = EnhancedApp()
        app.mainloop()
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        try:
            messagebox.showerror("Fatal Error", f"Application failed to start:\n{e}")
        except Exception:
            print(f"Fatal Error: {e}")