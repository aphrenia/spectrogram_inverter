import numpy as np
import matplotlib.pyplot as plt
import scipy
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import PIL
from playsound import playsound
import os
rng = np.random.default_rng()


output_directory = os.path.dirname(os.path.abspath(__file__))


def generate_audio_file(image):
    #Image manipulations
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    image_gray = image.convert('L')
    image_resized = image_gray.resize((2000, 151), Image.ANTIALIAS)
    image_array = np.array(image_resized);
    magnitude_spectrogram = image_array / 255.0

    #Estimate phase and inverse short-time Fourier transform
    frames = magnitude_spectrogram.shape[1]
    spectrogram = magnitude_spectrogram*np.exp(1j*2*np.pi*np.random.rand(151,frames)) 
    
    N = 4
    K = 5
    for h in range(N*K - K+1):
        _,waveform = scipy.signal.istft(spectrogram, nperseg=300, input_onesided =True, boundary=True)
        waveform = np.real(waveform)
        _,_,spectrogram = scipy.signal.stft(waveform, nperseg=300, return_onesided=True)
        spectrogram = np.exp(1j*np.angle(spectrogram))*magnitude_spectrogram
        
    # Output sound
    sound = waveform / np.max(np.abs(waveform))
    sf.write('output.wav', sound, samplerate = 44100)


def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img_display = ImageTk.PhotoImage(img)
        image_label.config(image=img_display)
        image_label.image = img_display
        
        generate_audio_file(img)
        
    
def play_audio():
    playsound(output_directory + '/output.wav')
    

root = tk.Tk()
root.title("Spectrogram inverter")

select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

play_button = tk.Button(root, text="Play Audio", command=play_audio)
play_button.pack(pady=10)

root.geometry("300x400")
root.mainloop()



#### Code for generating spectrogram from a signal ###

# fs = 1024
# N = 10*fs
# nperseg = 512
# amp = 2 * np.sqrt(2)
# noise_power = 0.001 * fs / 2
# time = np.arange(N) / float(fs)
# carrier = amp * np.sin(2*np.pi*50*time)
# noise = rng.normal(scale=np.sqrt(noise_power),
#                    size=time.shape)
# x = carrier + noise

# f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg)
# plt.figure()
# plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
# plt.ylim([f[1], f[-1]])
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.yscale('log')
# plt.show()
# _, xrec = signal.istft(Zxx, fs)

# plt.figure()
# plt.plot(time, x, time, xrec, time, carrier)
# plt.xlim([2, 2.1])
# plt.xlabel('Time [sec]')
# plt.ylabel('Signal')
# plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
# plt.show()









