import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Load audio
data, sr = librosa.load("audioData/wooden_door_knock.wav", sr=None)

# Define clip start and end times (in seconds)
start_time = 1.1800   # start at 0.5 sec
end_time = 1.3000     # end at 1.5 sec

# Convert time to sample indices
start_sample = int(start_time * sr)
end_sample = int(end_time * sr)

# Clip the audio
clipped_audio = data[start_sample:end_sample]

# Save clipped audio
sf.write("wood_door_knock_clipped.wav", clipped_audio, sr)

print("Clipped audio saved successfully!")



# Load audio file
data, sr = librosa.load("wood_door_knock_clipped.wav", sr=None)

# Create time axis for waveform
time = np.linspace(0, len(data) / sr, num=len(data))

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(time, data)
plt.title("Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform.jpg")
plt.show()

# Compute Short-Time Fourier Transform (STFT)
D = librosa.stft(data)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Plot spectrogram
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram")
plt.tight_layout()
plt.savefig("spectrum.jpg")
plt.show()