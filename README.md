![image](https://github.com/user-attachments/assets/e8a003f0-a32a-497d-b61a-87bc8c7d33eb)![image](https://github.com/user-attachments/assets/0c3a2896-a268-4013-9428-aed838838a7d)![image](https://github.com/user-attachments/assets/a05f6c64-a0c9-4acf-8457-7f2525f4a735)# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling

# AIM
  To perform experimental verification of signal sampling using various types.

# SOFTWARE REQUIRED
  Google Colab

# ALGORITHMS

1. Import Libraries and Define Original Signal:
Import necessary libraries: numpy and matplotlib.pyplot. Define original signal parameters: sampling frequency, time array, signal frequency, and signal amplitude.

2. Define Sampling Parameters:
Define sampling frequency and time array for sampling the original signal.

3. Sample Original Signal:
Sample the original signal using the defined sampling parameters to obtain the sampled signal.

4. Reconstruct Sampled Signal:
Reconstruct the sampled signal using a reconstruction technique, such as zero-order hold or linear interpolation.

5. Plot Results:
Plot the original signal, sampled signal, and reconstructed signal using matplotlib.pyplot to visualize the results.
# PROGRAM
**i) Ideal Sampling**

#Impulse Sampling or ideal sampling
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
fs = 100
t = np.arange(0, 1, 1/fs) 
f = 5
signal = np.sin(2 * np.pi * f * t)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
t_sampled = np.arange(0, 1, 1/fs)
signal_sampled = np.sin(2 * np.pi * f * t_sampled)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')
plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
reconstructed_signal = resample(signal_sampled, len(t))
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')
plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# OUTPUT
![image](https://github.com/user-attachments/assets/c601094b-04b7-4f33-b8f0-362dd83f0c6c)

# PROGRAM
  **ii) Natural Sampling**
  import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Parameters
fs = 1000  # Sampling frequency (samples per second)
T = 1  # Duration in seconds
t = np.arange(0, T, 1 / fs)  # Time vector

# Message Signal (sine wave message)
fm = 5  # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

# Pulse Train Parameters
pulse_rate = 50  # pulses per second
pulse_train = np.zeros_like(t)

# Construct Pulse Train (rectangular pulses)
pulse_width = int(fs / pulse_rate / 2)
for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:min(i + pulse_width, len(t))] = 1  # Corrected pulse width handling

# Natural Sampling
nat_signal = message_signal * pulse_train

# Reconstruction (Demodulation) Process

# Low-pass Filter (optional, smoother reconstruction)
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(nat_signal, 10, fs) # apply low pass filter to the naturally sampled signal.

plt.figure(figsize=(14, 10))

# Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

# Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)

# Natural Sampling
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)

# Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# OUTPUT
![image](https://github.com/user-attachments/assets/d94f25c8-c123-4d9e-9c85-24ade2359cc4)

# PROGRAM
 **iii) Flat Top Sampling**
 # PROGRAM
 import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Parameters
fs = 1000  # Sampling frequency (samples per second)
T = 1  # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector

# Message Signal (sine wave)
fm = 5  # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

# Pulse Train Parameters
pulse_rate = 50  # pulses per second
pulse_train = np.zeros_like(t)
pulse_width = int(fs / pulse_rate / 4)  # Flat-top width

# Construct Pulse Train (rectangular pulses for flat-top sampling)
for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i+pulse_width] = 1

# Flat-Top Sampling
flat_top_signal = np.copy(message_signal)
for i in range(0, len(t), int(fs / pulse_rate)):
    flat_top_signal[i:i+pulse_width] = message_signal[i]  # Hold value constant

# Reconstruction (Demodulation) Process
sampled_signal = flat_top_signal[pulse_train == 1]
sample_times = t[pulse_train == 1]

# Interpolation - Zero-Order Hold (for visualization)
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]

# Low-pass Filter for smoother reconstruction
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)

# Visualization
plt.figure(figsize=(14, 10))

# Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

# Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)

# Flat-Top Sampling
plt.subplot(4, 1, 3)
plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')
plt.legend()
plt.grid(True)

# Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Signal', color='green')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# OUTPUT
![image](https://github.com/user-attachments/assets/222f6d90-d4e7-4799-bfe8-0f0331bb7553)
 
# RESULT / CONCLUSIONS

Thus the given eperiment ideal sampling ,natural sampling,flat top sampling has been verified successfully by using python
