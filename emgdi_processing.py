from zhan_wavelet_adaptive_filter import wavelet_adaptive_filter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
# from scipy.signal import stft, istft


def preprocess_signal(raw_signal, sampling_rate):
    """

    Additional pre-processing steps that might be needed.
    """
    raw_signal = np.nan_to_num(raw_signal, nan=1.0) # ensures values are not clipped if above voltage picked up by DAQ
    
    # Notch filter for power interference
    notch_freq = 60  # 60Hz in North America; 50Hz in EUrope
    quality_factor = 30.0
    b, a = signal.iirnotch(notch_freq, quality_factor, fs=sampling_rate)
    filtered = signal.filtfilt(b, a, raw_signal)
    
    # 2. Bandpass filter for EMG (10-500 Hz typical for surface EMG)
    nyquist = 0.5 * sampling_rate
    low = 20 / nyquist
    high = 499 / nyquist
    b, a = signal.butter(2, [low, high], btype='bandpass')
    filtered = signal.filtfilt(b, a, filtered)

    return filtered

def get_r_peaks(emgecgch, method, fs, ecgdistance, ecgprominence, ecgheight):

    if method == 'filtered':
        nyquist = 0.5 * fs
        low = 5 / nyquist  # 5 Hz
        high = 15 / nyquist  # 15 Hz
        b, a = signal.butter(3, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, emgecgch)
        peaks, _ = signal.find_peaks(filtered, distance=ecgdistance, prominence=max(filtered)/3.5, height=max(filtered)/6)
    else:
        peaks, _ = signal.find_peaks(emgecgch, distance=ecgdistance, prominence=ecgprominence, height=ecgheight)
    return peaks

def timeshift(inputarray, shift):
    if (shift == 0) or (shift>=len(inputarray)):
        return inputarray
    else:
        ret = np.empty_like(inputarray)
        if shift >= 0:
            ret[:shift] = 0
            ret[shift:] = inputarray[:-shift]
        else:
            ret[shift:] = 0
            ret[:shift] = inputarray[-shift:]   
    return ret

def timeshift_average(ecgdata, avgdata, tmin, tmax):
    
    lowest = np.inf
    tlowest = 0
    tavg = None
    
    for t in range(tmin,tmax+1):
        shiftedavg = timeshift(avgdata, t)
        
        if type(ecgdata) is np.ndarray:
            ecg = ecgdata
        else:
            ecg = np.array(ecgdata)
             
        newl = np.sum(np.square(ecg - shiftedavg))
        if newl<lowest:
            lowest = newl
            tlowest=t     
            tavg = shiftedavg
            
    return tlowest, tavg

def average_ecg(emg_signal, peaks, samplingfrequency, windowsize):
    ecgwindows = []
              
    ecgwindowstart = int(windowsize * samplingfrequency/2)
    ecgwindowend = int(windowsize * samplingfrequency/2)
    shiftinterval = int(0.2 * samplingfrequency)

    for peak in peaks:

        if (peak-ecgwindowstart >= 0) and (peak+ecgwindowend <= len(emg_signal)): #only average whole windows
            ecgwindow = np.array(emg_signal[peak-ecgwindowstart:peak+ecgwindowend])

            #align averaging windows:
            if len(ecgwindows)>0:
                _, ecgavgtime = timeshift_average(ecgwindows[0], ecgwindow, -shiftinterval, shiftinterval)
                ecgwindows += [ecgavgtime]
            else:    
                ecgwindows += [ecgwindow]

    ecgavg = np.mean(ecgwindows, axis=0)
    return ecgavg

def amplitude_average(ecgdata, avgdata, rangefactor, steps):
    lowest = np.inf
    alowest = 0
    aavg = None
    amplitudes = np.linspace(-1 * rangefactor, rangefactor, steps)
    for a in amplitudes:
        ampavg = avgdata * a
        newa = np.sum(np.square(ecgdata-ampavg))
        if newa<lowest:
            lowest = newa
            alowest=a   
            aavg = ampavg
        
    return alowest, aavg

def subtractecg(emgecgch, peaks, ecgavg, samplingfrequency, windowsize):
    """
    **Adapted almost directly from RespMech**

    Subtracts an average ECG template from each ECG in the signal

    Parameters
    ----------
    emgecgch : array
        The signal containing ECG artifacts.
    peaks : array-like
        Indices of detected R‑peaks.
    ecgavg : array-like
        Average ECG template.
    samplingfrequency : int
        Sampling frequency from settings.
    windowsize: float
        Window around each ECG to subtract, from settings.

    Returns
    -------
    (template_small, template_medium, template_large) : tuple of arrays or None
        Average ECG templates for peaks in the lower, middle, and upper tertiles.
        If a tertile contains fewer than `min_peaks_per_group` peaks, its entry is None.
    """
    
    ecgwindowstart = int(windowsize * samplingfrequency/2)
    ecgwindowend = int(windowsize * samplingfrequency/2)
    shiftinterval = int(0.2 * samplingfrequency)

    #Subtract average ECG from EMG
    peakno = 0
    retwindows = []
    for peak in peaks:
        peakno += 1

        partial = False
        if (peak-ecgwindowstart < 0) :
            ecgwindow = emgecgch[0:peak+ecgwindowend]
            emgecgch[0:peak+ecgwindowend] = list(np.array(ecgwindow) - np.array(ecgavg[len(ecgavg)-len(ecgwindow):len(ecgavg)]))
            retwindows += [[0, peak+ecgwindowend]]
            partial = True

        if (peak+ecgwindowend > len(emgecgch)):
            ecgwindow = emgecgch[peak-ecgwindowstart:len(emgecgch)-1]
            emgecgch[peak-ecgwindowstart:len(emgecgch)-1] = list(np.array(ecgwindow) - np.array(ecgavg[0:len(ecgwindow)]))
            retwindows += [[peak-ecgwindowstart, len(emgecgch)-1]]
            partial = True
            
        if partial == False:
            ecgwindow = np.array(emgecgch[peak-ecgwindowstart:peak+ecgwindowend])
            retwindows += [[peak-ecgwindowstart, peak+ecgwindowend]]
            
            _, ecgavgtime = timeshift_average(ecgwindow, ecgavg, -shiftinterval, shiftinterval)
            _, fittedavg = amplitude_average(ecgwindow, ecgavgtime, 1.25, 1000) 
            adjwindow = list(ecgwindow - fittedavg)
            
            emgecgch[peak-ecgwindowstart:peak+ecgwindowend] =  adjwindow 
            
   
    return emgecgch
def average_ecg_adj(emg_signal, r_peaks, window_samples=300, min_peaks_per_group=6):
    """
    Create three average ECG templates based on peak magnitude tertiles,
    provided each tertile contains at least `min_peaks_per_group` peaks.

    Parameters
    ----------
    emg_signal : array
        The signal containing ECG artifacts.
    r_peaks : array-like
        Indices of detected R‑peaks.
    window_samples : int
        Number of samples to extract around each R‑peak (default: 300).
    min_peaks_per_group : int
        Minimum number of peaks required in each tertile to compute an average.
        If a tertile has fewer peaks, its template is set to None (default: 6).

    Returns
    -------
    (template_small, template_medium, template_large) : tuple of arrays or None
        Average ECG templates for peaks in the lower, middle, and upper tertiles.
        If a tertile contains fewer than `min_peaks_per_group` peaks, its entry is None.
    """
    
    half_window = window_samples // 2
    segments = []
    magnitudes = []

    for peak in r_peaks:
        start_idx = max(0, peak - half_window)
        end_idx = min(len(emg_signal), peak + half_window)

        # Keep only segments of exact desired length (avoid edge artifacts)
        if end_idx - start_idx == window_samples:
            segments.append(emg_signal[start_idx:end_idx])
            magnitudes.append(emg_signal[peak])   # magnitude = amplitude at R‑peak

    if not segments:
        raise ValueError("No valid ECG segments found")

    segments = np.array(segments)
    magnitudes = np.array(magnitudes)

    # Determine tertile thresholds based on magnitude distribution
    p33 = np.percentile(magnitudes, 100/3)          # 33.33rd percentile
    p67 = np.percentile(magnitudes, 200/3)          # 66.67th percentile

    # Assign each peak to a tertile group
    group_indices = [[], [], []]
    for i, mag in enumerate(magnitudes):
        if mag <= p33:
            group_indices[0].append(i)
        elif mag <= p67:
            group_indices[1].append(i)
        else:
            group_indices[2].append(i)

    # Compute average for each tertile if enough peaks 
    templates = []
    for group in group_indices:
        if len(group) >= min_peaks_per_group:
            template = np.mean(segments[group], axis=0)
            template = template - np.mean(template)   # normalize (remove DC)
            templates.append(template)
        else:
            print(f"Warning: Tertile has only {len(group)} peaks (< {min_peaks_per_group}). Returning None for this group.")
            templates.append(None)

    # Unpack into small, medium, large
    template_small, template_medium, template_large = templates

    return template_small, template_medium, template_large

def subtractecg_adj(emgecgch, peaks, samplingfrequency, windowsize):
    """
    Subtract ECG artifacts from an EMG channel using three average templates
    selected based on the magnitude of each R‑peak.

    Parameters
    ----------
    emgecgch : list or array
        The raw signal (EMG + ECG) that will be modified in place.
    peaks : list of int
        Indices of detected R‑peaks.
    ecgavg_small, ecgavg_medium, ecgavg_large : array-like
        Average ECG templates for the smallest, middle, and largest third of peaks.
        Each should have length = int(windowsize * samplingfrequency).
    samplingfrequency : int, optional
        Sampling rate in Hz (default 1000).
    windowsize : float, optional
        Time window around each R‑peak (seconds) to extract/subtract (default 0.3).

    Returns
    -------
    emgecgch : array
        The signal with ECG artifacts reduced (same object, modified in place).
    """
    ecgavg_small, ecgavg_medium, ecgavg_large = average_ecg_adj(emgecgch, peaks, window_samples=int(windowsize*samplingfrequency))
    
    # Convert to numpy array
    emgecgch = np.array(emgecgch)
   
    window_len = int(windowsize * samplingfrequency)
    half_window = int(window_len / 2)

    # Compute peak magnitudes from the original signal
    peak_magnitudes = [emgecgch[p] for p in peaks]

    # Sort peaks by magnitude and split into three groups
    sorted_idx = np.argsort(peak_magnitudes)
    n_peaks = len(peaks)
    
    # Use array_split to get three groups
    groups = np.array_split(sorted_idx, 3)
    
    # Create a mapping for each peak index
    group_for_peak = np.empty(n_peaks, dtype=int)
    for group_id, group_indices in enumerate(groups):
        for orig_idx in group_indices:
            group_for_peak[orig_idx] = group_id

    # List of templates for access
    templates = [ecgavg_small, ecgavg_medium, ecgavg_large]

    # Process each peak (in original order) and subtract the appropriate template 
    for i, peak in enumerate(peaks):
        # Select the template corresponding to this peak's magnitude group
        ecgavg = templates[group_for_peak[i]]

        # Handle partial windows at the edges
        partial = False
        if peak - half_window < 0:
            # Window starts at beginning of signal
            ecgwindow = emgecgch[0:peak + half_window]
            # Subtract the trailing part of the template
            template_part = ecgavg[-len(ecgwindow):]
            emgecgch[0:peak + half_window] = ecgwindow - template_part
            partial = True

        if peak + half_window > len(emgecgch):
            # Window ends at end of signal
            ecgwindow = emgecgch[peak - half_window:len(emgecgch)]
            # Subtract the leading part of the template
            template_part = ecgavg[:len(ecgwindow)]
            emgecgch[peak - half_window:len(emgecgch)] = ecgwindow - template_part
            partial = True

        if not partial:
            # Full window – perform time alignment and amplitude scaling
            ecgwindow = emgecgch[peak - half_window:peak + half_window]

            # Time shift the template to best match this window
            shiftinterval = int(0.2 * samplingfrequency)
            _, ecgavgtime = timeshift_average(ecgwindow, ecgavg,
                                              -shiftinterval, shiftinterval)

            # Amplitude scale the time‑shifted template
            _, fittedavg = amplitude_average(ecgwindow, ecgavgtime, 3, 1000)

            # Subtract the fitted template
            adjwindow = ecgwindow - fittedavg
            
            # Return subtracted window back to EMG signal
            emgecgch[peak - half_window:peak + half_window] = adjwindow

    return emgecgch

# def spectral_denoise(signal, noise_segment, fs=1000, nperseg=256):
#     # STFT of signal
#     f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    
#     # STFT of noise
#     _, _, Zxx_noise = stft(noise_segment, fs=fs, nperseg=nperseg)
    
#     # Mean noise magnitude spectrum
#     noise_mag = np.mean(np.abs(Zxx_noise), axis=1, keepdims=True)
    
#     # Spectral subtraction
#     mag = np.abs(Zxx)
#     phase = np.angle(Zxx)
    
#     mag_clean = np.maximum(mag - noise_mag, 0)
#     Zxx_clean = mag_clean * np.exp(1j * phase)
    
#     # Reconstruct signal
#     _, clean_signal = istft(Zxx_clean, fs=fs, nperseg=nperseg)
    
#     return clean_signal[:len(signal)]

def rolling_rms(x, N):
    kernel = np.ones(N) / N
    power = np.convolve(x**2, kernel, mode='same')
    return np.sqrt(power)

def process_emgdi(emgcols, settings, pdf): 
    samplingfrequency = settings['samplingfrequency']
    windowsize = settings['ecgwindowsize']

    if settings["save_emg_proc_figures"]:
        fig_peaks, axes_peaks = plt.subplots(5,1,figsize=(12,10))
        fig_peaks.suptitle("Raw signal with r detection peaks")
        fig_peaks.tight_layout()

        # fig_ecg, axes_ecg = plt.subplots(5,1,figsize=(6,10))

        fig_sub, axes_sub = plt.subplots(5,1,figsize=(12,10))
        fig_sub.suptitle("ECG subtraction")
        fig_sub.tight_layout()

        fig_denoised, axes_denoised = plt.subplots(5,1,figsize=(12,10))
        fig_denoised.suptitle("Wavelet Filter Denoised")
        fig_denoised.tight_layout()

        # fig_rms, axes_rms = plt.subplots(5,1,figsize=(12,10))
        # fig_rms.suptitle("RMS")
        # fig_rms.tight_layout()

    count=0

    ecgprominence = settings["ecgprominence"] 
    ecgdistance = settings["ecgdistance"] * settings['samplingfrequency'] / 1000
    ecgheight = settings["ecgheight"]
    ecgmethod = settings["ecgmethod"]

    if settings['shareecgpeaksch'] != 0:
        share_peaks = get_r_peaks(emgcols["emgcol"+str(settings['shareecgpeaksch'])], ecgmethod, samplingfrequency,  ecgdistance, ecgprominence, ecgheight)


    for col in emgcols:
        if settings['shareecgpeaksch'] == 0:
            ## Get and plot R peaks for inspection
            peaks = get_r_peaks(emgcols[col], ecgmethod, samplingfrequency, ecgdistance, ecgprominence, ecgheight)
        else: peaks = share_peaks    

        ## Get and plot average ecg templates
        avg_ecg_rm = average_ecg(emgcols[col], peaks, samplingfrequency, windowsize)

        ## Subtract ECG from the raw signal
        if settings["ecgsubmethod"] == "adjusted":
            ecg_sub = subtractecg_adj(emgcols[col].copy(), peaks, samplingfrequency, windowsize)
        else:
            ecg_sub = subtractecg(emgcols[col].copy(), peaks, avg_ecg_rm, samplingfrequency, windowsize)
        

        emg_cleaned, info = wavelet_adaptive_filter(
            ecg_sub,
            wavelet="db4",   # paper's best basis
            level=4,         # paper uses 4 at 2000 Hz
            K=1.5,           # tune this: lower = more aggressive
            Lb=5,            # exclusion zone radius
            Ub=20,           # averaging reach
        )

        ## RMS signal
        signal_rms = rolling_rms(emg_cleaned, 200)

        if count == 0:
            emg_rms_df = pd.DataFrame({col: signal_rms})
        else:
            emg_rms_df[str(col)] =  signal_rms

        if settings["save_emg_proc_figures"]:
            axes_peaks[count].plot(emgcols[col])
            axes_peaks[count].plot(peaks, emgcols[col][peaks], marker='o', linestyle='None')
            axes_sub[count].plot(ecg_sub)
            axes_denoised[count].plot(emg_cleaned)
            # axes_rms[count].plot(signal_rms)
            # axes_rms[count].set_ylim((0, 0.2))
        count+=1

    for i in plt.get_fignums():
        pdf.savefig(i)
        plt.close(i)       
        
    return emg_rms_df