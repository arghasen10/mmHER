import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft
import scipy.signal as signal
from collections import deque

numADCSamples = 100
numRX = 4
numLanes = 2
isReal = 0


def bpf_vitalsign(Fc1=0.1, Fc2=2):
    # All frequency values are in Hz.
    Fs = 20  # Sampling Frequency

    N = 4  # Order
    Fc1 = Fc1  # First Cutoff Frequency
    Fc2 = Fc2  # Second Cutoff Frequency

    # Create the Butterworth bandpass filter
    b, a = signal.butter(N, [Fc1, Fc2], fs=Fs, btype='bandpass')

    return b, a


def readDCA1000(fileName, numADCSamples=100, numRX=4, numChirps=0, numTX=0):
    # Global variables
    global numLanes, isReal

    # Change based on sensor config
    numADCSamples = numADCSamples  # Number of ADC samples per chirp
    numADCBits = 16  # Number of ADC bits per sample
    numRX = numRX  # Number of receivers
    numLanes = 2  # Number of lanes is always 2
    isReal = 0  # Set to 1 if real only data, 0 if complex data
    numChirps = numChirps
    numTX = numTX
    if numTX != 0:
        pass

    # Read file
    # Read .bin file
    with open(fileName, 'rb') as fid:
        adcData = np.fromfile(fid, dtype=np.int16)

    # If 12 or 14 bits ADC per sample compensate for sign extension
    if numADCBits != 16:
        l_max = 2 ** (numADCBits - 1) - 1
        adcData[adcData > l_max] = adcData[adcData > l_max] - 2 ** numADCBits

    fileSize = adcData.shape[0]

    # For complex data
    numChirps = fileSize // (2 * numADCSamples * numRX)
    LVDS = np.zeros((fileSize // 2), dtype=np.complex64)
    counter = 0
    # Combine real and imaginary part into complex data
    LVDS[0::2] = adcData[0::4] + 1j * adcData[2::4]
    LVDS[1::2] = adcData[1::4] + 1j * adcData[3::4]
    # Create column for each chirp
    LVDS = LVDS.reshape(numChirps, numADCSamples * numRX).T

    # Organize data per RX
    adcData = np.zeros((numRX, numChirps * numADCSamples))
    for row in range(numRX):
        for i in range(numChirps):
            adcData[row, i * numADCSamples:(i + 1) * numADCSamples] = LVDS[
                                                                      (row * numADCSamples):(row + 1) * numADCSamples,
                                                                      i]

    # Return receiver data
    return adcData


def filter_RemoveImpulseNoise(dataPrev2, dataPrev1, dataCurr, thresh):
    pDataIn = [dataPrev2, dataPrev1, dataCurr]

    backwardDiff = pDataIn[1] - pDataIn[0]
    forwardDiff = pDataIn[1] - pDataIn[2]

    x1 = 0
    x2 = 2
    y1 = pDataIn[0]
    y2 = pDataIn[2]
    x = 1

    if (forwardDiff > thresh and backwardDiff > thresh) or (forwardDiff < -thresh and backwardDiff < -thresh):
        y = y1 + ((x - x1) * (y2 - y1)) / (x2 - x1)
    else:
        y = pDataIn[1]

    return y


def get_number_of_frames(filename):
    ADCBinFile = open(filename, 'rb')
    frame_count = 0
    while 1:
        try:
            bin_frame = np.frombuffer(ADCBinFile.read(12 * 128 * 256 * 4), dtype=np.int16)
            frame_count +=1
        except:
            pass
    return frame_count


def phase_unwraping(numFrame, signal_phase, ):
    # Phase unwrapping
    new_signal_phase = []
    for k in range(1, numFrame):
        diff = signal_phase[k] - signal_phase[k - 1]
        # np.where(diff > np.pi/2, diff - np.pi, )
        new_signal_phase.append(
            np.where(diff > np.pi / 2, diff - np.pi, np.where(diff < -np.pi / 2, diff + np.pi, diff)))
        # signal_phase[:, k:] -= np.cumsum(diff, axis=1)

    return np.array(new_signal_phase)


def get_vital_sign(adcData, numChirps, numTX, numFrame, numADCSamples, numFFTPoints, frame_period):
    RX1data = np.reshape(adcData, (numADCSamples, numChirps))  # RX1data

    range_win = np.hamming(numADCSamples)  # Generate hamming window
    din_win = np.zeros((numADCSamples, numFrame), dtype=complex)  # Array for storing windowed data
    datafft = np.zeros((numADCSamples, numFrame), dtype=complex)  # Array for storing FFT results
    for k in range(numFrame):
        din_win[:, k] = RX1data[:, 2 * k] * range_win  # Apply hamming window to the signal
        datafft[:, k] = fft(din_win[:, k])  # Perform FFT on the windowed signal

    # Find Range-bin peaks
    rangeBinStartIndex = 3  # Resolution 0.1m
    rangeBinEndIndex = 10
    data = np.zeros((numFrame,), dtype=complex)  # Array for storing data peaks

    for k in range(numFrame):
        peakIndex = np.argmax(np.abs(datafft[rangeBinStartIndex:rangeBinEndIndex, k]))
        data[k] = datafft[rangeBinStartIndex + peakIndex, k]

    # Get the real and imaginary parts of the signal
    data_real = np.real(data)
    data_imag = np.imag(data)

    # Calculate signal phase
    signal_phase = np.arctan2(data_imag, data_real)
    signal_phase = phase_unwraping(numFrame, signal_phase)
    delta_phase = np.diff(signal_phase)

    thresh = 0.8
    phaseUsedComputation = np.zeros((numFrame - 5))

    for k in range(numFrame - 5):
        phaseUsedComputation[k] = filter_RemoveImpulseNoise(delta_phase[k], delta_phase[k + 1], delta_phase[k + 2],
                                                            thresh)

    index = np.arange(0, (numFrame - 5) * frame_period, frame_period)
    bpf_filter_b, bpf_filter_a = bpf_vitalsign()
    vital_sign = signal.lfilter(bpf_filter_b, bpf_filter_a, phaseUsedComputation)
    plt.plot(index, vital_sign)
    plt.xlabel('Time(s)', fontweight='bold')
    plt.ylabel('Amplitude', fontweight='bold')
    plt.title('cardiopulmonary signal', fontweight='bold')
    plt.show()

    vital_sign_fft = np.fft.fft(vital_sign, numFFTPoints)  # Perform FFT on the original signal

    # Convert double sideband signal to single sideband
    freq = np.arange(0, numFFTPoints / 2 + 1) / frame_period / numFFTPoints  # Vital sign signal sampling rate frame number
    P2 = np.abs(vital_sign_fft / (numFFTPoints - 1))
    P1 = P2[:numFFTPoints // 2 + 1]  # Select the first half, as the spectrum is symmetric after FFT
    P1[1:-1] = 2 * P1[1:-1]

    # Original signal frequency domain plot
    plt.plot(freq, P1)
    plt.xlim([0, 2])
    plt.xlabel('Frequency (Hz)', fontweight='bold')
    plt.ylabel('Amplitude', fontweight='bold')
    plt.title('Cardiopulmonary signal spectrogram', fontweight='bold')
    plt.show()

    bpf_breathe_num, bpf_breathe_den = bpf_vitalsign(Fc2=0.6)
    filter_delta_phase_breathe = signal.lfilter(bpf_breathe_num, bpf_breathe_den, phaseUsedComputation)
    breathe = filter_delta_phase_breathe

    # Time Domain Diagram of Respiration Signal
    plt.plot(index, breathe)
    plt.xlabel('Time(s)', fontweight='bold')
    plt.ylabel('Amplitude', fontweight='bold')
    plt.title('Breathing signal', fontweight='bold')
    plt.show()

    breathe_fft = np.fft.fft(breathe, numFFTPoints)

    # Convert double sideband signal to single sideband
    P2_breathe = np.abs(breathe_fft / (numFFTPoints - 1))
    P1_breathe = P2_breathe[:numFFTPoints // 2 + 1]
    P1_breathe[1:-1] = 2 * P1_breathe[1:-1]

    # Respiratory signal frequency domain diagram
    plt.plot(freq, P1_breathe)
    plt.xlim([0, 2])
    plt.xlabel('Frequency(Hz)', fontweight='bold')
    plt.ylabel('Amplitude', fontweight='bold')
    plt.title('Respiratory signal spectrogram', fontweight='bold')
    plt.show()

    # Heartbeat signal bandpass filtering
    bpf_heart_num, bpf_heart_den = bpf_vitalsign(Fc1=0.9, Fc2=0.6)
    filter_delta_phase_heart = signal.lfilter(bpf_heart_num, bpf_heart_den, phaseUsedComputation)
    heart = filter_delta_phase_heart

    # Time Domain Diagram of Heartbeat Signal
    plt.plot(index, heart)
    plt.xlabel('Time(s)', fontweight='bold')
    plt.ylabel('Amplitude', fontweight='bold')
    plt.title('heartbeat signal', fontweight='bold')
    plt.show()
    # Do fft on the heartbeat signal
    heart_fft = np.fft.fft(heart, numFFTPoints)

    # Convert double sideband signal to single sideband
    P2_heart = np.abs(heart_fft / (numFFTPoints - 1))
    P1_heart = P2_heart[:numFFTPoints // 2 + 1]
    P1_heart[1:-1] = 2 * P1_heart[1:-1]

    # Heartbeat Harmonic Detection
    heart_peaks, _ = signal.find_peaks(P1_heart, height=0.9, distance=int(2 * numFFTPoints * frame_period))
    heart_harmonic_peaks, _ = signal.find_peaks(P1_heart, height=1.8, distance=int(4 * numFFTPoints * frame_period))
    heart_peaks = heart_peaks / numFFTPoints / frame_period
    heart_harmonic_peaks = heart_harmonic_peaks / numFFTPoints / frame_period

    num_heart_peaks = len(heart_peaks)
    num_heart_harmonic_peaks = len(heart_harmonic_peaks)

    for i in range(num_heart_peaks):
        if max(P1_heart) - P1_heart[int(round(heart_peaks[i] * numFFTPoints * frame_period))] < 0.3:
            for j in range(num_heart_harmonic_peaks):
                if heart_harmonic_peaks[j] / heart_peaks[i] == 2:
                    P1_heart[int(round(heart_peaks[i] * numFFTPoints * frame_period))] = 2 * P1_heart[int(round(heart_peaks[i] * numFFTPoints * frame_period))]

    # Heartbeat signal frequency domain diagram
    plt.plot(freq, P1_heart)
    plt.xlim([0, 4])
    plt.xlabel('Frequency(Hz)', fontweight='bold')
    plt.ylabel('Amplitude', fontweight='bold')
    plt.title('Spectrum diagram of heartbeat signal', fontweight='bold')
    plt.show()
    return vital_sign, P1, breathe, P1_breathe, heart, P1_heart


# adcData = readDCA1000('../demo.bin')
# vital_sign, P1, breathe, P1_breathe, heart, P1_heart = get_vital_sign(adcData[0], numChirps=800, numTX=0,
#                                                                       numFrame=400, numADCSamples=100,
#                                                                       numFFTPoints=1024, frame_period=0.05)


bin_filename = '/home/argha/1685517135.bin'
total_frame_number = os.path.getsize(bin_filename)/(12 * 128 * 256 * 4)
ADCBinFile = open(bin_filename, 'rb')

adcData = deque(maxlen=200)

for frame_no in range(int(total_frame_number)):
    bin_frame = np.frombuffer(ADCBinFile.read(12 * 128 * 256 * 4), dtype=np.int16)
    np_frame = np.zeros(shape=(len(bin_frame) // 2), dtype=np.complex_)
    np_frame[0::2] = bin_frame[0::4] + 1j * bin_frame[2::4]
    np_frame[1::2] = bin_frame[1::4] + 1j * bin_frame[3::4]
    frameWithChirp = np.reshape(np_frame, (128, 3, 4, -1))
    frameWithChirp = frameWithChirp.transpose(1, 2, 0, 3)
    frameWithChirp = frameWithChirp[0, 0, 0:2, :]
    print(adcData.append(frameWithChirp))
    if len(adcData) == 200:
        myData = np.array(adcData).transpose(2, 1, 0).reshape(256, -1)
        get_vital_sign(myData, numChirps=400, numTX=3, numFrame=200, numADCSamples=256, numFFTPoints=1024,
                       frame_period=0.1)
