"""
Wavelet-Based Adaptive Filter for ECG Removal from EMGdi Signals
=================================================================
Implementation of:
    Zhan, C., Yeung, L.F., Yang, Z. (2010).
    "A wavelet-based adaptive filter for removing ECG interference
     in EMGdi signals."
    Journal of Electromyography and Kinesiology, 20, 542-549.

Key idea (Section 2.3 of the paper):
    Traditional wavelet denoising truncates SMALL coefficients
    (assumes signal is large, noise is small). For EMGdi+ECG the
    situation is inverted: EMGdi resembles band-limited noise
    (small coefficients) while ECG has large, sparse QRS spikes
    (large coefficients). So we truncate LARGE coefficients instead.

    The adaptive threshold T_j(x) = K * v_j(x) is computed from
    the local average of neighboring coefficients, excluding the
    immediate vicinity (to prevent an ECG spike from inflating
    its own threshold).

Algorithm A1 from the paper:
    Step 1: DWT decomposition -> wavelet coefficients d^j, a^j
    Step 2: Compute adaptive threshold T_j(x) per Eq. (10)-(11)
    Step 3: Apply large-coefficient-truncated shrinkage per Eq. (9)
    Step 4: Inverse DWT reconstruction

Requirements:
    pip install numpy scipy
    pip install PyWavelets  (optional - falls back to built-in DWT)
"""

from __future__ import annotations
import numpy as np

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


# ======================================================================
#  Built-in DWT when PyWavelets is not installed
#  Uses the exact db4 coefficients from the paper's Appendix
# ======================================================================

_DB4_H = np.array([
    (1 + np.sqrt(3)) / (4 * np.sqrt(2)),
    (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
    (3 - np.sqrt(3)) / (4 * np.sqrt(2)),
    (1 - np.sqrt(3)) / (4 * np.sqrt(2)),
])
_DB4_G = np.array([_DB4_H[3], -_DB4_H[2], _DB4_H[1], -_DB4_H[0]])

_WAVELET_FILTERS = {"db4": (_DB4_H, _DB4_G)}


def _get_filters(wavelet):
    if _HAS_PYWT:
        w = pywt.Wavelet(wavelet)
        return np.array(w.dec_lo), np.array(w.dec_hi)
    if wavelet in _WAVELET_FILTERS:
        return _WAVELET_FILTERS[wavelet]
    raise ValueError(f"Wavelet '{wavelet}' unavailable without PyWavelets.")


def _get_rec_filters(wavelet):
    if _HAS_PYWT:
        w = pywt.Wavelet(wavelet)
        return np.array(w.rec_lo), np.array(w.rec_hi)
    h, g = _get_filters(wavelet)
    return h[::-1].copy(), g[::-1].copy()


def _dwt_single(signal, h, g):
    filt_len = len(h)
    padded = np.pad(signal, filt_len - 1, mode="symmetric")
    a = np.convolve(padded, h, mode="valid")[::2]
    d = np.convolve(padded, g, mode="valid")[::2]
    return a, d


def _idwt_single(a, d, h_r, g_r, target_len):
    # Ensure a and d are the same length (pad shorter with zeros)
    max_len = max(len(a), len(d))
    a_pad = np.zeros(max_len); a_pad[:len(a)] = a
    d_pad = np.zeros(max_len); d_pad[:len(d)] = d
    up_a = np.zeros(2 * max_len); up_a[::2] = a_pad
    up_d = np.zeros(2 * max_len); up_d[::2] = d_pad
    rec = np.convolve(up_a, h_r, "full") + np.convolve(up_d, g_r, "full")
    delay = len(h_r) - 1
    return rec[delay : delay + target_len]


def _wavedec(signal, wavelet, level):
    if _HAS_PYWT:
        return pywt.wavedec(signal, wavelet, level=level), None
    h, g = _get_filters(wavelet)
    details = []
    lengths = [len(signal)]
    current = signal.copy()
    for _ in range(level):
        a, d = _dwt_single(current, h, g)
        details.append(d)
        lengths.append(len(current))
        current = a
    details.append(current)
    details.reverse()
    lengths.reverse()
    return details, lengths


def _waverec(coeffs, wavelet, orig_lengths=None):
    if _HAS_PYWT:
        return pywt.waverec(coeffs, wavelet)
    h_r, g_r = _get_rec_filters(wavelet)
    current = coeffs[0]
    for i in range(1, len(coeffs)):
        if orig_lengths is not None:
            target_len = orig_lengths[i]
        else:
            target_len = max(len(current), len(coeffs[i])) * 2
        current = _idwt_single(current, coeffs[i], h_r, g_r, target_len)
    return current


def _dwt_max_level(n, wavelet):
    if _HAS_PYWT:
        return pywt.dwt_max_level(n, wavelet)
    h, _ = _get_filters(wavelet)
    lev = 0
    while n >= len(h):
        n = (n + len(h) - 1) // 2
        lev += 1
    return max(lev - 1, 1)


# ======================================================================
#  Adaptive threshold (Eq. 10-11)
# ======================================================================

def zhan_adaptive_threshold(
    coefficients: np.ndarray,
    K: float = 1.5,
    Lb: int = 3,
    Ub: int = 15,
) -> np.ndarray:
    """
    Adaptive threshold T_j(x) from Equations (10) and (11).

    For coefficient at index i, the threshold is K times the average
    absolute value of neighbors from two windows that EXCLUDE the
    immediate vicinity:

        Right window:  indices  i + Lb  ..  i + Ub
        Left window:   indices  i - Ub  ..  i - Lb

    The exclusion zone prevents an ECG spike from inflating its
    own threshold.

    Parameters
    ----------
    coefficients : np.ndarray
        Wavelet coefficients at one decomposition level.
    K : float
        Gain factor (Eq. 10). Higher = less aggressive removal.
    Lb : int
        Inner exclusion radius of the neighbor window.
    Ub : int
        Outer reach of the neighbor window.

    Returns
    -------
    thresholds : np.ndarray
        T_j(x_i) for each coefficient.
    """
    n = len(coefficients)
    abs_c = np.abs(coefficients)
    thresholds = np.zeros(n)

    prefix = np.zeros(n + 1)
    prefix[1:] = np.cumsum(abs_c)

    def _rng(lo, hi):
        lo_c, hi_c = max(0, lo), min(n - 1, hi)
        if lo_c > hi_c:
            return 0.0, 0
        return float(prefix[hi_c + 1] - prefix[lo_c]), hi_c - lo_c + 1

    for i in range(n):
        r_sum, r_cnt = _rng(i + Lb, i + Ub)
        l_sum, l_cnt = _rng(i - Ub, i - Lb)
        total = r_cnt + l_cnt
        v_j = (r_sum + l_sum) / total if total > 0 else abs_c[i]
        thresholds[i] = K * v_j

    return thresholds


# ======================================================================
#  Large-coefficient-truncated shrinkage (Eq. 9)
# ======================================================================

def large_coefficient_shrinkage(
    coefficients: np.ndarray,
    thresholds: np.ndarray,
    mode: str = "hard",
) -> np.ndarray:
    """
    "Large-coefficient-truncated" shrinkage from Eq. (9).

    INVERSE of traditional denoising:
      Traditional (Eq. 7): keep large, zero small -> removes noise
      Zhan (Eq. 9):        keep small, zero large -> removes ECG

    Parameters
    ----------
    coefficients, thresholds : np.ndarray
    mode : 'hard' (exact Eq. 9) or 'soft' (smooth transition)
    """
    abs_c = np.abs(coefficients)
    if mode == "hard":
        return np.where(abs_c <= thresholds, coefficients, 0.0)
    elif mode == "soft":
        gain = np.clip(1.0 - abs_c / (thresholds + 1e-12), 0.0, 1.0)
        return coefficients * gain
    raise ValueError(f"Unknown mode: {mode}")


# ======================================================================
#  Main filter — Algorithm A1
# ======================================================================

def wavelet_adaptive_filter(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
    K: float = 1.5,
    Lb: int = 3,
    Ub: int = 15,
    shrinkage_mode: str = "hard",
    apply_to_approximation: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Wavelet-based adaptive filter for ECG removal (Algorithm A1).

    Works WITHOUT a reference ECG channel.

    Parameters
    ----------
    signal : np.ndarray (1-D)
        Raw EMGdi signal contaminated by ECG.
    wavelet : str
        Wavelet basis. Paper uses 'db4' (~1.92% PSD error).
    level : int or None
        DWT decomposition levels. None = auto (4 at 2000 Hz).
    K : float
        Gain for adaptive threshold (Eq. 10).
    Lb : int
        Inner exclusion radius for threshold averaging.
    Ub : int
        Outer reach for threshold averaging.
    shrinkage_mode : str
        'hard' (exact paper Eq. 9) or 'soft' (smooth variant).
    apply_to_approximation : bool
        Whether to also filter the deepest approximation coefficients.

    Returns
    -------
    cleaned : np.ndarray
        Filtered EMGdi signal.
    info : dict
        Diagnostic data.
    """
    n = len(signal)
    if level is None:
        level = min(_dwt_max_level(n, wavelet), 4)

    # Step 1: DWT decomposition (Eq. 3-4)
    coeffs, orig_lengths = _wavedec(signal, wavelet, level=level)

    coeffs_original = [c.copy() for c in coeffs]
    thresholds_all = []
    coeffs_filtered = []

    # Steps 2-3: threshold + shrinkage per level
    for idx, c in enumerate(coeffs):
        if idx == 0 and not apply_to_approximation:
            coeffs_filtered.append(c.copy())
            thresholds_all.append(np.zeros_like(c))
            continue

        T_j = zhan_adaptive_threshold(c, K=K, Lb=Lb, Ub=Ub)
        thresholds_all.append(T_j)
        coeffs_filtered.append(
            large_coefficient_shrinkage(c, T_j, mode=shrinkage_mode)
        )

    # Step 4: Inverse DWT (Eq. 5-6)
    cleaned = _waverec(coeffs_filtered, wavelet, orig_lengths=orig_lengths)[:n]

    return cleaned, {
        "coeffs_original": coeffs_original,
        "coeffs_filtered": coeffs_filtered,
        "thresholds": thresholds_all,
        "level": level,
        "wavelet": wavelet,
    }


def wavelet_adaptive_filter_multichannel(
    channels: list[np.ndarray] | np.ndarray,
    channel_names: list[str] | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Apply the filter to multiple EMGdi channels independently."""
    if isinstance(channels, np.ndarray) and channels.ndim == 2:
        ch_list = [channels[i] for i in range(channels.shape[0])]
    else:
        ch_list = list(channels)
    names = channel_names or [f"emgcol{i+1}" for i in range(len(ch_list))]
    return {
        nm: wavelet_adaptive_filter(ch, **kwargs)[0]
        for nm, ch in zip(names, ch_list)
    }


# ======================================================================
#  Evaluation metrics (Section 3)
# ======================================================================

def psd_relative_error(original, processed, fs=2000):
    """PSD relative error delta from Eq. (13)."""
    from scipy.signal import welch
    ml = min(len(original), len(processed))
    _, P = welch(original[:ml], fs=fs, nperseg=min(1024, ml))
    _, Pt = welch(processed[:ml], fs=fs, nperseg=min(1024, ml))
    return float(np.sum((P - Pt)**2) / (np.sum(P**2) + 1e-12) * 100)


def compute_psd_features(signal, fs=2000):
    """H/L ratio, total power, centroid freq from Table 2."""
    from scipy.signal import welch
    f, P = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    hp = np.sum(P[(f >= 125) & (f <= 150)])
    lp = np.sum(P[(f >= 25) & (f <= 50)])
    return {
        "hl_ratio": float(hp / (lp + 1e-12)),
        "total_power": float(np.sum(P)),
        "centroid_freq_hz": float(np.sum(f * P) / (np.sum(P) + 1e-12)),
    }


# ======================================================================
#  Synthetic data (Section 3.1)
# ======================================================================

def generate_paper_synthetic(duration=20.0, fs=2000, seed=42):
    from scipy.signal import butter, filtfilt
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)
    n = len(t)

    raw = rng.standard_normal(n)
    mod = raw * (0.5 + 0.5 * np.cos(2 * np.pi * 0.25 * t))
    b, a = butter(4, [0.5, 250], btype="band", fs=fs)
    emg = filtfilt(b, a, mod)
    emg = emg / np.std(emg) * 50

    ecg = np.zeros(n)
    bi = int(fs / 1.2)
    qw = int(0.08 * fs)
    for s in range(0, n - bi, bi):
        if s + qw < n:
            mid = qw // 2
            q = np.zeros(qw)
            q[:mid] = np.linspace(0, 1, mid)
            q[mid:] = np.linspace(1, -0.3, qw - mid)
            ecg[s:s+qw] += q * 400
        ps = max(0, s - int(0.16*fs))
        pl = int(0.08*fs)
        if ps + pl < n:
            ecg[ps:ps+pl] += 30 * np.sin(np.linspace(0, np.pi, pl))
        ts = s + qw + int(0.04*fs)
        tl = int(0.16*fs)
        if ts + tl < n:
            ecg[ts:ts+tl] += 60 * np.sin(np.linspace(0, np.pi, tl))

    return emg + ecg, emg, ecg, t


# ======================================================================
#  Demo
# ======================================================================

def demo():
    fs = 2000
    print("Generating synthetic EMGdi + ECG (Section 3.1)...")
    contaminated, emg_true, ecg_true, t = generate_paper_synthetic(20.0, fs)
    print(f"Signal: {len(contaminated)} samples ({len(contaminated)/fs:.1f}s)")
    print(f"Built-in DWT: {not _HAS_PYWT}\n")

    cleaned, info = wavelet_adaptive_filter(
        contaminated, wavelet="db4", level=4, K=1.5, Lb=3, Ub=15
    )

    delta = psd_relative_error(emg_true, cleaned, fs)
    print(f"PSD relative error delta = {delta:.2f}%")
    print(f"  (paper: 1.92%)\n")

    fc = compute_psd_features(contaminated, fs)
    fl = compute_psd_features(cleaned, fs)
    ft = compute_psd_features(emg_true, fs)
    print(f"{'':20s} {'H/L':>10s} {'PWR':>10s} {'fc(Hz)':>10s}")
    for lb, f in [("Contaminated", fc), ("Cleaned", fl), ("True EMGdi", ft)]:
        print(f"{lb:20s} {f['hl_ratio']:10.4f} {f['total_power']:10.1f} {f['centroid_freq_hz']:10.1f}")

    labels = ["a^M"] + [f"d^{info['level']-i}" for i in range(len(info['coeffs_original'])-1)]
    print(f"\nCoefficient stats (level={info['level']}):")
    for i, lb in enumerate(labels):
        o = info["coeffs_original"][i]
        f = info["coeffs_filtered"][i]
        z = np.sum(np.abs(f) < 1e-12) / len(o) * 100
        print(f"  {lb:5s}: {len(o):6d} coeffs, {z:5.1f}% zeroed")

    res = cleaned[:len(emg_true)] - emg_true[:len(cleaned)]
    sup = 10 * np.log10(np.mean(ecg_true**2) / (np.mean(res**2) + 1e-12))
    corr = np.corrcoef(
        cleaned[:len(emg_true)] / np.std(cleaned[:len(emg_true)]),
        emg_true[:len(cleaned)] / np.std(emg_true[:len(cleaned)])
    )[0, 1]
    print(f"\nECG suppression: {sup:.1f} dB")
    print(f"EMGdi correlation: {corr:.4f}")


if __name__ == "__main__":
    demo()
