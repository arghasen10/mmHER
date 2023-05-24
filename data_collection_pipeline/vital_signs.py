import sys
import os
import struct
import time
import numpy as np
import array as arr
import configuration as cfg
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
from pcd_generation import *

IIR_FILTER_BREATH_NUM_STAGES = 2
IIR_FILTER_COEFS_SECOND_ORDER = 6
BREATHING_WFM_IIR_FILTER_TAPS_LENGTH = (IIR_FILTER_COEFS_SECOND_ORDER * IIR_FILTER_BREATH_NUM_STAGES) + 2
IIR_FILTER_HEART_NUM_STAGES = 4
HEART_WFM_IIR_FILTER_TAPS_LENGTH = (IIR_FILTER_COEFS_SECOND_ORDER * IIR_FILTER_HEART_NUM_STAGES) + 2
pVitalSigns_Breath_CircularBuffer = []
guiFlag_MotionDetection = 1
motionDetection_BlockSize = 20
pMotionCircularBuffer = np.zeros(shape=(20))
gFrameCount = 0
motionDetection_Thresh = 1
pVitalSigns_Heart_CircularBuffer = []
MAX_PEAKS_ALLOWED_WFM = 128              # Maximum number of peaks allowed in the breathing and cardiac waveforms
pPeakLocsHeart = np.zeros(shape=(MAX_PEAKS_ALLOWED_WFM))
MAX_ALLOWED_PEAKS_SPECTRUM = 128              # Maximum allowed peaks in the Vital Signs Spectrum
pPeakValues = np.zeros(shape= MAX_ALLOWED_PEAKS_SPECTRUM)
pPeakLocsValid = np.zeros(shape=(MAX_PEAKS_ALLOWED_WFM))    # Peak locations after only retaining the valid peaks
framePeriodicity_ms = 50                                # 50 ms each frame
samplingFreq_Hz = 1000 / framePeriodicity_ms        # 1000 to convert from ms to seconds
heart_startFreq_Hz = 0.8    # Heart-Rate peak search Start-Frequency
heart_endFreq_Hz   = 2.0    # Heart-Rate peak search End-Frequency
peakDistanceHeart_Min  = samplingFreq_Hz/heart_endFreq_Hz
peakDistanceHeart_Max  = samplingFreq_Hz/heart_startFreq_Hz
CONVERT_HZ_BPM = 60.0             # Converts Hz to Beats per minute
FIR_FILTER_SIZE = 10        # Filter size for the Moving average filter
pFilterCoefs = np.full(shape=FIR_FILTER_SIZE, fill_value=0.1)
pPeakLocsBreath = np.zeros(shape=(MAX_PEAKS_ALLOWED_WFM))   # Peak locations (indices) of the Breathing Waveform
breath_startFreq_Hz = 0.1    # Breathing-Rate peak search Start-Frequency
breath_endFreq_Hz   = 0.6   # Breathing-Rate peak search End-Frequency
peakDistanceBreath_Min = samplingFreq_Hz/breath_endFreq_Hz
peakDistanceBreath_Max = samplingFreq_Hz/breath_startFreq_Hz
PHASE_FFT_SIZE = 1024       # FFT size for each of the Breathing and Cardiac waveform
breathingWfm_Spectrum_FftSize = PHASE_FFT_SIZE
pVitalSignsBuffer_Cplx = np.zeros(breathingWfm_Spectrum_FftSize, dtype=np.complex128)
scale_breathingWfm = 100000
pVitalSignsSpectrumTwiddle32x32 = np.zeros(shape=(breathingWfm_Spectrum_FftSize), dtype=np.complex128)
pVitalSigns_SpectrumCplx = np.zeros(shape=(breathingWfm_Spectrum_FftSize), dtype=np.complex128)
pVitalSigns_Breath_AbsSpectrum = np.zeros(shape=(breathingWfm_Spectrum_FftSize), dtype=np.float128)
guiFlag_GainControl = 1
PERFORM_XCORR = 1
XCORR_NUM_LAGS = 200
pXcorr = np.zeros(shape=(XCORR_NUM_LAGS))
xCorr_minLag = samplingFreq_Hz/2.1    # (Fs/Freq)  Corresponding to f = 2.1 Hz at a sampling Frequency of 20 Hz
xCorr_maxLag = samplingFreq_Hz/0.8    # (Fs/Freq)  Corresponding to f = 0.8 Hz at a sampling Frequency of 20 Hz
pPeakIndex = np.zeros(shape=(MAX_ALLOWED_PEAKS_SPECTRUM))        # Indices of the Peaks in the Cardiac/Breathing spectrum
xCorr_Breath_minLag = samplingFreq_Hz/breath_endFreq_Hz    # (Fs/Freq)  Corresponding to f = 0.6 Hz at a sampling Frequency of 20 Hz
xCorr_Breath_maxLag = samplingFreq_Hz/breath_startFreq_Hz    # (Fs/Freq)  Corresponding to f = 0.1 Hz at a sampling Frequency of 20 Hz



def unwrap(phase, phasePrev, diffPhaseCorrectionCum):
    modFactorF = 0.0
    diffPhase = phase - phasePrev
    PI = np.pi

    if diffPhase > PI:
        modFactorF = 1.0
    elif diffPhase < -PI:
        modFactorF = -1.0

    diffPhaseMod = diffPhase - modFactorF * 2 * PI

    if diffPhaseMod == -PI and diffPhase > 0:
        diffPhaseMod = PI

    diffPhaseCorrection = diffPhaseMod - diffPhase

    if (diffPhaseCorrection < PI and diffPhaseCorrection > 0) or (diffPhaseCorrection > -PI and diffPhaseCorrection < 0):
        diffPhaseCorrection = 0.0

    diffPhaseCorrectionCum = diffPhaseCorrectionCum + diffPhaseCorrection
    phaseOut = phase + diffPhaseCorrectionCum

    return phaseOut, diffPhaseCorrectionCum


def clutter_removal_C(rangeResult):
    outputResult = np.zeros_like(rangeResult)
    for i in range(rangeResult.shape[0]):
        alphaClutter = 0.01
        prevRangeResult = alphaClutter * rangeResult[i] + (1 - alphaClutter) * rangeResult[i]
        outputResult[i] = np.abs(rangeResult[i] - prevRangeResult)
    return outputResult

def process_vitalsigns(rangeResult):
    gFrameCount+=1
    res_across_virtual_antenna = []
    for i in range(rangeResult.shape[0]):
        for j in range(rangeResult.shape[1]):
            for k in range(rangeResult.shape[2]):
                res_across_virtual_antenna.append(process_indiv_vitalsigns(rangeResult[i][j][k]))


def filter_RemoveImpulseNoise(dataPrev2, dataPrev1, dataCurr, thresh):
    forwardDiff = dataPrev1 - dataCurr
    backwardDiff = dataPrev1 - dataPrev2
    x1, x2, y1, y2, x, y = 0, 2, dataPrev2, dataCurr, 1, 0
    pDataIn = [dataPrev2, dataPrev1, dataCurr]

    if (
        (forwardDiff > thresh and backwardDiff > thresh)
        or (forwardDiff < -thresh and backwardDiff < -thresh)
    ):
        y = y1 + ((x - x1) * (y2 - y1)) / (x2 - x1)
    else:
        y = dataPrev1

    return y

def filter_FIR(pDataIn, filterCoefs, numCoefs):
    sum = 0.0
    for temp in range(numCoefs):
        sum += pDataIn[temp] * filterCoefs[temp]
    return sum

def filter_IIR_BiquadCascade(DataIn, pFilterCoefs, pScaleVals, pDelay, numStages):
    numCoefsStage = 6
    input_val = DataIn

    for indexStage in range(numStages):
        indexTemp = numCoefsStage * indexStage
        b0 = pFilterCoefs[indexTemp]
        b1 = pFilterCoefs[indexTemp + 1]
        b2 = pFilterCoefs[indexTemp + 2]
        a1 = pFilterCoefs[indexTemp + 4]
        a2 = pFilterCoefs[indexTemp + 5]
        scaleVal = pScaleVals[indexStage]

        pDelay[indexTemp] = scaleVal * input_val - a1 * pDelay[indexTemp + 1] - a2 * pDelay[indexTemp + 2]
        y = b0 * pDelay[indexTemp] + b1 * pDelay[indexTemp + 1] + b2 * pDelay[indexTemp + 2]

        pDelay[indexTemp + 2] = pDelay[indexTemp + 1]
        pDelay[indexTemp + 1] = pDelay[indexTemp]

        input_val = y
    output_val = y
    return output_val


def computeAGC(pDataIn, dataInLength, lenBlock, thresh):
    scaleValueSum = 0
    for indexTemp in range(lenBlock, dataInLength):
            sumEnergy = 0
            for indexInner in range(0, lenBlock+1):
                indexCurr = indexTemp - lenBlock + indexInner
                sumEnergy += pDataIn[indexCurr] * pDataIn[indexCurr]


            if sumEnergy > thresh:
                scaleValue = np.sqrt(thresh/sumEnergy)
                scaleValueSum += 1
                for indexInner in range(0, lenBlock+1):
                    indexCurr = indexTemp - lenBlock + indexInner
                    pDataIn[indexCurr]= pDataIn[indexCurr] * scaleValue
    return scaleValueSum


def find_Peaks(pDataIn, pPeakIndex, pPeakValues, startIndex, endIndex):
    numPeaks = 0
    for temp in range(startIndex + 1, endIndex - 1):
        if pDataIn[temp] > pDataIn[temp - 1] and pDataIn[temp] > pDataIn[temp + 1]:
            pPeakIndex[numPeaks] = temp
            pPeakValues[numPeaks] = float(pDataIn[temp])
            numPeaks += 1
    return numPeaks

def DIG_REV(i, m, j):
    _ = i
    _ = ((_ & 0x33333333) << 2) | ((_ & ~0x33333333) >> 2)
    _ = ((_ & 0x0F0F0F0F) << 4) | ((_ & ~0x0F0F0F0F) >> 4)
    _ = ((_ & 0x00FF00FF) << 8) | ((_ & ~0x00FF00FF) >> 8)
    _ = ((_ & 0x0000FFFF) << 16) | ((_ & ~0x0000FFFF) >> 16)
    j[0] = _ >> m

def SMPY_32(x, y):
    return int((x * y) >> 31)

def DSP_fft32x32(ptr_w, npoints, ptr_x, ptr_y):
    w = ptr_w
    x, x2, x0 = ptr_x, [], []
    y0, y1, y2, y3 = [], [], [], []
    i, j, l1, l2, h2, predj, tw_offset, stride, fft_jmp = 0, 0, 0, 0, 0, 0, 0, 0, 0
    xt0_0, yt0_0, xt1_0, yt1_0, xt2_0, yt2_0 = 0, 0, 0, 0, 0, 0
    xh0_0, xh1_0, xh20_0, xh21_0, xl0_0, xl1_0, xl20_0, xl21_0 = 0, 0, 0, 0, 0, 0, 0, 0
    xh0_1, xh1_1, xl0_1, xl1_1 = 0, 0, 0, 0
    x_0, x_1, x_2, x_3, x_l1_0, x_l1_1, x_l2_0, x_l2_1 = 0, 0, 0, 0, 0, 0, 0, 0
    xh0_2, xh1_2, xl0_2, xl1_2, xh0_3, xh1_3, xl0_3, xl1_3 = 0, 0, 0, 0, 0, 0, 0, 0
    x_4, x_5, x_6, x_7, x_h2_0, x_h2_1 = 0, 0, 0, 0, 0, 0
    x_8, x_9, x_a, x_b, x_c, x_d, x_e, x_f = 0, 0, 0, 0, 0, 0, 0, 0
    si10, si20, si30, co10, co20, co30 = 0, 0, 0, 0, 0, 0
    n00, n10, n20, n30, n01, n11, n21, n31 = 0, 0, 0, 0, 0, 0, 0, 0
    n02, n12, n22, n32, n03, n13, n23, n33 = 0, 0, 0, 0, 0, 0, 0, 0
    n0, j0, radix, norm, m = 0, 0, 0, 0, 0

    for i in range(31, -1, -1):
        if npoints & (1 << i) == 0:
            break
        m += 1
    radix = 2 if m & 1 else 4
    norm = m - 2

    stride = npoints
    tw_offset = 0
    fft_jmp = 6 * stride

    while stride > radix:
        j =  0
        fft_jmp >>= 2
        h2 = stride >> 1
        l1 = stride
        l2 = stride + (stride >> 1)
        x = ptr_x
        w = ptr_w + tw_offset
        tw_offset += fft_jmp
        stride >>=  2

        for i in range(0, npoints,4):
            co10, si10, co20, si20, co30, si30 = w[j+1], w[j+0], w[j+3], w[j+2], w[j+5], w[j+4]
            x_0, x_1, x_l1_0, x_l1_1, x_l2_0, x_l2_1, x_h2_0, x_h2_1 = x[0], x[1], x[l1], x[l1+1], x[l2], x[l2+1], x[h2], x[h2+1]
            xh0_0  = x_0    + x_l1_0      
            xh1_0  = x_1    + x_l1_1
            xl0_0  = x_0    - x_l1_0      
            xl1_0  = x_1    - x_l1_1
            xh20_0 = x_h2_0 + x_l2_0      
            xh21_0 = x_h2_1 + x_l2_1
            xl20_0 = x_h2_0 - x_l2_0      
            xl21_0 = x_h2_1 - x_l2_1
            x0 = x
            x2 = x0
            j    += 6
            x    += 2
            predj = (j - fft_jmp)
            if not predj:
                x += fft_jmp
                j = 0
            x0[0] = xh0_0 + xh20_0;       x0[1] = xh1_0 + xh21_0
            xt0_0 = xh0_0 - xh20_0;       yt0_0 = xh1_0 - xh21_0
            xt1_0 = xl0_0 + xl21_0;       yt2_0 = xl1_0 + xl20_0
            xt2_0 = xl0_0 - xl21_0;       yt1_0 = xl1_0 - xl20_0
            x2[h2  ] = SMPY_32(si10,yt1_0) + SMPY_32(co10, xt1_0)
            x2[h2+1] = SMPY_32(co10,yt1_0) - SMPY_32(si10, xt1_0)
            x2[l1]   = SMPY_32(si20,yt0_0) + SMPY_32(co20, xt0_0)
            x2[l1+1] = SMPY_32(co20,yt0_0) - SMPY_32(si20, xt0_0)
            x2[l2]   = SMPY_32(si30,yt2_0) + SMPY_32(co30, xt2_0)
            x2[l2+1] = SMPY_32(co30,yt2_0) - SMPY_32(si30, xt2_0)
    
    y0 = ptr_y
    y2 = ptr_y + npoints
    x0 = ptr_x
    x2 = ptr_x + (npoints>>1)

    if radix is 2:
        y1  = y0 + (npoints >> 2)
        y3  = y2 + (npoints >> 2)
        l1  = norm + 1
        j0  = 8
        n0  = npoints >> 1
    else:
        y1  = y0 + (npoints >> 1)
        y3  = y2 + (npoints >> 1)
        l1  = norm + 2
        j0  = 4
        n0  = npoints >> 2
    j = 0

    for i in range(0, npoints, 8):
        # Digit reverse the index
        DIG_REV(i, l1, h2)

        # Read input data
        x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7 = x0[0:8]
        x0 += 8

        # Perform computations
        xh0_0 = x_0 + x_4
        xh1_0 = x_1 + x_5
        xl0_0 = x_0 - x_4
        xl1_0 = x_1 - x_5
        xh0_1 = x_2 + x_6
        xh1_1 = x_3 + x_7
        xl0_1 = x_2 - x_6
        xl1_1 = x_3 - x_7

        n00 = xh0_0 + xh0_1
        n01 = xh1_0 + xh1_1
        n10 = xl0_0 + xl1_1
        n11 = xl1_0 - xl0_1
        n20 = xh0_0 - xh0_1
        n21 = xh1_0 - xh1_1
        n30 = xl0_0 - xl1_1
        n31 = xl1_0 + xl0_1
        if radix == 2:
            n00 = x_0 + x_2
            n01 = x_1 + x_3
            n20 = x_0 - x_2
            n21 = x_1 - x_3
            n10 = x_4 + x_6
            n11 = x_5 + x_7
            n30 = x_4 - x_6
            n31 = x_5 - x_7
        y0[2*h2] = n00
        y0[2*h2 + 1] = n01
        y1[2*h2] = n10
        y1[2*h2 + 1] = n11
        y2[2*h2] = n20
        y2[2*h2 + 1] = n21
        y3[2*h2] = n30
        y3[2*h2 + 1] = n31
        x_8 = x2[0]
        x_9 = x2[1]
        x_a = x2[2]
        x_b = x2[3]
        x_c = x2[4]
        x_d = x2[5]
        x_e = x2[6]
        x_f = x2[7]
        x2 += 8

        xh0_2 = x_8 + x_c
        xh1_2 = x_9 + x_d
        xl0_2 = x_8 - x_c
        xl1_2 = x_9 - x_d
        xh0_3 = x_a + x_e
        xh1_3 = x_b + x_f
        xl0_3 = x_a - x_e
        xl1_3 = x_b - x_f

        n02 = xh0_2 + xh0_3
        n03 = xh1_2 + xh1_3
        n12 = xl0_2 + xl1_3
        n13 = xl1_2 - xl0_3
        n22 = xh0_2 - xh0_3
        n23 = xh1_2 - xh1_3
        n32 = xl0_2 - xl1_3
        n33 = xl1_2 + xl0_3

        if radix == 2:
            n02 = x_8 + x_a
            n03 = x_9 + x_b
            n22 = x_8 - x_a
            n23 = x_9 - x_b
            n12 = x_c + x_e
            n13 = x_d + x_f
            n32 = x_c - x_e
            n33 = x_d - x_f

        y0[2*h2+2] = n02
        y0[2*h2+3] = n03
        y1[2*h2+2] = n12
        y1[2*h2+3] = n13
        y2[2*h2+2] = n22
        y2[2*h2+3] = n23
        y3[2*h2+2] = n32
        y3[2*h2+3] = n33
        j += j0

        if j == n0: 
            j += n0
            x0 += npoints >> 1
            x2 += npoints >> 1
    return w, x, y0


def MmwDemo_magnitudeSquared(inpBuff, magSqrdBuff, numSamples):
    for i in range(numSamples):
        magSqrdBuff[i] = float(inpBuff[i].real) * float(inpBuff[i].real) + float(inpBuff[i].imag) * float(inpBuff[i].imag)
    return magSqrdBuff

def filterPeaksWfm(pPeakLocsIn, pPeakLocsOut, numPeaksIn, winMin, winMax):
    numPeaksOutValid = 1  # The first peak is assumed to be valid

    pPeakLocsOut[0] = pPeakLocsIn[0]
    for tempIndex in range(1, numPeaksIn):
        pkDiff = pPeakLocsIn[tempIndex] - pPeakLocsOut[numPeaksOutValid - 1]
        if pkDiff > winMin:
            pPeakLocsOut[numPeaksOutValid] = pPeakLocsIn[tempIndex]
            numPeaksOutValid += 1

    return numPeaksOutValid

def computeAutoCorrelation(pDataIn, dataInLength, pDataOut, minLag, maxLag):
    for indexLag in range(minLag, maxLag):
        sum = 0
        for index in range(0, dataInLength - indexLag):
            sum += pDataIn[index]* pDataIn[index + indexLag]
        pDataOut[indexLag] = sum

def computeMaxIndex(pDataIn, startIndex, endIndex):
    MaxVal = pDataIn[startIndex]
    MaxValIndex = startIndex
    for temp in range(startIndex+1, endIndex):
        if pDataIn[temp] > MaxVal:
            MaxVal = pDataIn[temp]
            MaxValIndex = temp
    return MaxValIndex


def process_indiv_vitalsigns(range_indv_bin):
    range_indv_bin = clutter_removal_C(range_indv_bin)
    rangeBinPhase = np.arctan(range_indv_bin.imag/range_indv_bin.real)
    phasePrevFrame = 0
    diffPhaseCorrectionCum = 0
    phaseUsedComputationPrev = 0
    dataCurr = 0
    dataPrev2 = 0
    dataPrev1 = 0
    noiseImpulse_Thresh = 1.5
    pDataIn = np.zeros(shape=FIR_FILTER_SIZE)

    for i in range(rangeBinPhase):
        unwrapPhasePeak, diffPhaseCorrectionCum = unwrap(rangeBinPhase[i], phasePrevFrame, diffPhaseCorrectionCum)
        phasePrevFrame = rangeBinPhase[i]
        phaseUsedComputation = unwrapPhasePeak - phaseUsedComputationPrev
        phaseUsedComputationPrev = unwrapPhasePeak
        dataPrev2 = dataPrev1
        dataPrev1 = dataCurr
        dataCurr = phaseUsedComputation
        phaseUsedComputation = filter_RemoveImpulseNoise(dataPrev2, dataPrev1, dataCurr, noiseImpulse_Thresh)
        pFilterCoefsBreath = np.array([1.0000, 0, -1.0000, 1.0000, -1.9632, 0.9644, 1.0000, 0, -1.0000, 1.0000, -1.8501, 0.8681])
        pScaleValsBreath = np.array([0.0602, 0.0602, 1.0000])
        pDelayBreath = np.zeros(shape=(BREATHING_WFM_IIR_FILTER_TAPS_LENGTH,))
        pFilterCoefsHeart_4Hz = np.array([1.0000, 0, -1.0000, 1.0000, -0.5306, 0.5888, 
                                          1.0000, 0, -1.0000, 1.0000, -1.8069, 0.8689, 
                                          1.0000, 0, -1.0000, 1.0000, -1.4991, 0.5887, 
                                          1.0000, 0, -1.0000, 1.0000, -0.6654, 0.2099])
        pScaleValsHeart_4Hz = np.array([0.4188, 0.4188, 0.3611, 0.3611, 1.0000])
        pDelayHeart = np.zeros(shape=(HEART_WFM_IIR_FILTER_TAPS_LENGTH,))
        outputFilterBreathOut = filter_IIR_BiquadCascade(phaseUsedComputation, pFilterCoefsBreath, pScaleValsBreath, pDelayBreath, IIR_FILTER_BREATH_NUM_STAGES)
        outputFilterHeartOut  = filter_IIR_BiquadCascade(phaseUsedComputation, pFilterCoefsHeart_4Hz, pScaleValsHeart_4Hz, pDelayHeart, IIR_FILTER_HEART_NUM_STAGES)

        for loopIndexBuffer in range(1, min(10, len(pVitalSigns_Breath_CircularBuffer))):
            pVitalSigns_Breath_CircularBuffer[loopIndexBuffer - 1] = pVitalSigns_Breath_CircularBuffer[loopIndexBuffer]
        pVitalSigns_Breath_CircularBuffer[- 1] = outputFilterBreathOut

        # Detection of Motion corrupted Segments
        if guiFlag_MotionDetection == 1:
            # Update the Motion Removal Circular Buffer
            for loopIndexBuffer in range(1, motionDetection_BlockSize):
                pMotionCircularBuffer[loopIndexBuffer - 1] = pMotionCircularBuffer[loopIndexBuffer]
            pMotionCircularBuffer[motionDetection_BlockSize - 1] = outputFilterHeartOut
            indexMotionDetection = gFrameCount % motionDetection_BlockSize

            # Only perform these steps for every motionDetection_BlockSize sample
            if indexMotionDetection == 0:
                # Check if the current segment is "Noisy"
                sumEnergy = 0
                for loopIndexBuffer in range(motionDetection_BlockSize):
                    sumEnergy += pMotionCircularBuffer[loopIndexBuffer] ** 2

                if sumEnergy > motionDetection_Thresh:
                    print('Mtion detected')
                else:
                    print('Perfect Frame')
                    tempEndIndex = None
                    # Shift the current contents of the circular Buffer
                    for loopIndexBuffer in range(motionDetection_BlockSize, min(500, len(pVitalSigns_Heart_CircularBuffer))):
                        pVitalSigns_Heart_CircularBuffer[loopIndexBuffer - motionDetection_BlockSize] = pVitalSigns_Heart_CircularBuffer[loopIndexBuffer]
                    # Copy the current data segment to the end of the Circular Buffer
                    for loopIndexBuffer in range(motionDetection_BlockSize):
                        tempEndIndex = len(pVitalSigns_Heart_CircularBuffer) - motionDetection_BlockSize
                        pVitalSigns_Heart_CircularBuffer[tempEndIndex + loopIndexBuffer] = pMotionCircularBuffer[loopIndexBuffer]
        # If Motion DETECTED then don't UPDATE or SHIFT the values in the buffer
        else:  # Regular processing
            # Copies the "Cardiac Waveform" in a circular Buffer
            for loopIndexBuffer in range(1, len(pVitalSigns_Heart_CircularBuffer)):
                pVitalSigns_Heart_CircularBuffer[loopIndexBuffer - 1] = pVitalSigns_Heart_CircularBuffer[loopIndexBuffer]
            pVitalSigns_Heart_CircularBuffer[- 1] = outputFilterHeartOut

        # Spectral Estimation based on the Inter-Peaks Distance
        numPeaksHeart = find_Peaks(pVitalSigns_Heart_CircularBuffer, pPeakLocsHeart, pPeakValues, 0, len(pVitalSigns_Heart_CircularBuffer) - 1)
        if numPeaksHeart != 0:
            numPeaksHeart = filterPeaksWfm(pPeakLocsHeart, pPeakLocsValid, numPeaksHeart, peakDistanceHeart_Min, peakDistanceHeart_Max)
        heartRateEst_peakCount = CONVERT_HZ_BPM * ((numPeaksHeart * samplingFreq_Hz) / len(pVitalSigns_Heart_CircularBuffer))

        for loopIndexBuffer in range(1, FIR_FILTER_SIZE):
            pDataIn[loopIndexBuffer - 1] = pDataIn[loopIndexBuffer]
        pDataIn[FIR_FILTER_SIZE - 1] = heartRateEst_peakCount
        heartRateEst_peakCount_filtered = filter_FIR(pDataIn, pFilterCoefs, FIR_FILTER_SIZE)

        numPeaksBreath = find_Peaks(pVitalSigns_Breath_CircularBuffer, pPeakLocsBreath, pPeakValues, 0, le(pVitalSigns_Breath_CircularBuffer) - 1)
        if numPeaksBreath != 0:
            numPeaksBreath = filterPeaksWfm(pPeakLocsBreath, pPeakLocsValid, numPeaksBreath, peakDistanceBreath_Min, peakDistanceBreath_Max)

        breathingRateEst_peakCount = CONVERT_HZ_BPM * ((numPeaksBreath * samplingFreq_Hz) / len(pVitalSigns_Heart_CircularBuffer))
        heartRateEst_peakCount = CONVERT_HZ_BPM * ((numPeaksHeart * samplingFreq_Hz) / len(pVitalSigns_Breath_CircularBuffer))
        import numpy as np

        # Input to the FFT needs to be complex
        for loopIndexBuffer in range(len(pVitalSigns_Breath_CircularBuffer)):
            pVitalSignsBuffer_Cplx[loopIndexBuffer] = scale_breathingWfm * pVitalSigns_Breath_CircularBuffer[loopIndexBuffer]

        # Input is overwritten by the DSP_fft32x32 function
        pVitalSignsSpectrumTwiddle32x32, pVitalSignsBuffer_Cplx, pVitalSigns_SpectrumCplx = DSP_fft32x32(
            pVitalSignsSpectrumTwiddle32x32,
            breathingWfm_Spectrum_FftSize,
            pVitalSignsBuffer_Cplx,
            pVitalSigns_SpectrumCplx
        )

        pVitalSigns_Breath_AbsSpectrum = MmwDemo_magnitudeSquared(
            pVitalSigns_SpectrumCplx,
            pVitalSigns_Breath_AbsSpectrum,
            breathingWfm_Spectrum_FftSize
        )

        # Pre-Processing Steps for the Cardiac Waveform
        # Perform Automatic Gain Control if enabled from the GUI
        if guiFlag_GainControl == 1:
            computeAGC(pVitalSigns_Heart_CircularBuffer, len(pVitalSigns_Heart_CircularBuffer), motionDetection_BlockSize, motionDetection_Thresh)

        if guiFlag_MotionDetection == 1:
            outputFilterHeartOut = pMotionCircularBuffer[motionDetection_BlockSize - 1]
        else:
            outputFilterHeartOut = pVitalSigns_Heart_CircularBuffer[- 1]

        # Perform Autocorrelation on the Waveform=
        if PERFORM_XCORR:
            computeAutoCorrelation (pVitalSigns_Heart_CircularBuffer,  len(pVitalSigns_Heart_CircularBuffer) , pXcorr, xCorr_minLag, xCorr_maxLag)
            xCorr_numPeaks  = find_Peaks(pXcorr, pPeakIndex, pPeakValues, xCorr_minLag, xCorr_maxLag)
            maxIndex_lag    = computeMaxIndex(pXcorr, xCorr_minLag, xCorr_maxLag)
            temp = (1.0)/(maxIndex_lag/samplingFreq_Hz)
            heartRateEst_xCorr = CONVERT_HZ_BPM *temp
            
            confidenceMetricHeartOut_xCorr = 0 if (xCorr_numPeaks == 0 ) else pXcorr[maxIndex_lag]

            # Auto-correlation on the Breathing Waveform
            computeAutoCorrelation (pVitalSigns_Breath_CircularBuffer, len(pVitalSigns_Breath_CircularBuffer), 
                                  pXcorr, xCorr_Breath_minLag, xCorr_Breath_maxLag)
            xCorr_numPeaks  = find_Peaks(pXcorr, pPeakIndex, pPeakValues, xCorr_Breath_minLag, xCorr_Breath_maxLag)
            maxIndex_lag    = computeMaxIndex(pXcorr, xCorr_Breath_minLag, xCorr_Breath_maxLag)
            temp = (float) (1.0)/(maxIndex_lag/samplingFreq_Hz)
            breathRateEst_xCorr = CONVERT_HZ_BPM *temp

            confidenceMetricBreathOut_xCorr = 0 if (xCorr_numPeaks == 0 ) else pXcorr[maxIndex_lag]

#  Start from line number 2988 
    


if __name__ == "__main__":
    raw_poincloud_data_for_plot = []
    bin_filename='1684598876.bin'
    total_frame_number=799
    pointCloudProcessCFG = PointCloudProcessCFG()
    shift_arr=cfg.MMWAVE_RADAR_LOC
    bin_reader=RawDataReader(bin_filename)
    frame_no = 0
    raw_adc_data = np.fromfile(bin_filename, dtype=np.uint16)
    raw_adc_data = raw_adc_data.reshape(total_frame_number, -1)

    for frame_no in range(total_frame_number):
        bin_frame=bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame=bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame,frameConfig)
        rangeResult = rangeFFT(reshapedFrame,frameConfig)
        if pointCloudProcessCFG.enableStaticClutterRemoval:
            rangeResult = clutter_removal(rangeResult,axis=2)

        dopplerResult = dopplerFFT(rangeResult,frameConfig)
        pointCloud = frame2pointcloud(dopplerResult, pointCloudProcessCFG)
        frame_no+=1
        print('Frame %d:'%(frame_no), pointCloud.shape)
        raw_poincloud_data_for_plot.append(pointCloud)
