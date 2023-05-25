import numpy as np
import configuration as cfg
import math
from pcd_generation import PointCloudProcessCFG, RawDataReader, bin2np_frame, frameReshape, rangeFFT, clutter_removal, \
    dopplerFFT, frame2pointcloud

IIR_FILTER_BREATH_NUM_STAGES = 2
IIR_FILTER_COEFS_SECOND_ORDER = 6
BREATHING_WFM_IIR_FILTER_TAPS_LENGTH = (IIR_FILTER_COEFS_SECOND_ORDER * IIR_FILTER_BREATH_NUM_STAGES) + 2
IIR_FILTER_HEART_NUM_STAGES = 4
HEART_WFM_IIR_FILTER_TAPS_LENGTH = (IIR_FILTER_COEFS_SECOND_ORDER * IIR_FILTER_HEART_NUM_STAGES) + 2
guiFlag_MotionDetection = 1
motionDetection_BlockSize = 20
pMotionCircularBuffer = np.zeros(shape=(20,))
gFrameCount = 0
motionDetection_Thresh = 1
MAX_PEAKS_ALLOWED_WFM = 128  # Maximum number of peaks allowed in the breathing and cardiac waveforms
pPeakLocsHeart = np.zeros(shape=MAX_PEAKS_ALLOWED_WFM)
MAX_ALLOWED_PEAKS_SPECTRUM = 128  # Maximum allowed peaks in the Vital Signs Spectrum
pPeakValues = np.zeros(shape=MAX_ALLOWED_PEAKS_SPECTRUM)
pPeakLocsValid = np.zeros(shape=MAX_PEAKS_ALLOWED_WFM)  # Peak locations after only retaining the valid peaks
framePeriodicity_ms = 50  # 50 ms each frame
samplingFreq_Hz = 1000 / framePeriodicity_ms  # 1000 to convert from ms to seconds
heart_startFreq_Hz = 0.8  # Heart-Rate peak search Start-Frequency
heart_endFreq_Hz = 2.0  # Heart-Rate peak search End-Frequency
peakDistanceHeart_Min = samplingFreq_Hz / heart_endFreq_Hz
peakDistanceHeart_Max = samplingFreq_Hz / heart_startFreq_Hz
CONVERT_HZ_BPM = 60.0  # Converts Hz to Beats per minute
FIR_FILTER_SIZE = 10  # Filter size for the Moving average filter
pFilterCoefs = np.full(shape=FIR_FILTER_SIZE, fill_value=0.1)
pPeakLocsBreath = np.zeros(shape=MAX_PEAKS_ALLOWED_WFM)  # Peak locations (indices) of the Breathing Waveform
breath_startFreq_Hz = 0.1  # Breathing-Rate peak search Start-Frequency
breath_endFreq_Hz = 0.6  # Breathing-Rate peak search End-Frequency
peakDistanceBreath_Min = samplingFreq_Hz / breath_endFreq_Hz
peakDistanceBreath_Max = samplingFreq_Hz / breath_startFreq_Hz
PHASE_FFT_SIZE = 1024  # FFT size for each of the Breathing and Cardiac waveform
breathingWfm_Spectrum_FftSize = PHASE_FFT_SIZE
pVitalSignsBuffer_Cplx = np.zeros(breathingWfm_Spectrum_FftSize, dtype=np.complex128)
scale_breathingWfm = 100000
pVitalSignsSpectrumTwiddle32x32 = np.zeros(shape=breathingWfm_Spectrum_FftSize, dtype=np.complex128)
pVitalSigns_SpectrumCplx = np.zeros(shape=breathingWfm_Spectrum_FftSize, dtype=np.complex128)
pVitalSigns_Breath_AbsSpectrum = np.zeros(shape=breathingWfm_Spectrum_FftSize, dtype=np.float128)
guiFlag_GainControl = 1
PERFORM_XCORR = 1
XCORR_NUM_LAGS = 200
pXcorr = np.zeros(shape=XCORR_NUM_LAGS)
xCorr_minLag = samplingFreq_Hz / 2.1  # (Fs/Freq)  Corresponding to f = 2.1 Hz at a sampling Frequency of 20 Hz
xCorr_maxLag = samplingFreq_Hz / 0.8  # (Fs/Freq)  Corresponding to f = 0.8 Hz at a sampling Frequency of 20 Hz
pPeakIndex = np.zeros(shape=MAX_ALLOWED_PEAKS_SPECTRUM)  # Indices of the Peaks in the Cardiac/Breathing spectrum
xCorr_Breath_minLag = samplingFreq_Hz / breath_endFreq_Hz  # (Fs/Freq)  f = 0.6 Hz at a sampling Frequency of 20 Hz
xCorr_Breath_maxLag = samplingFreq_Hz / breath_startFreq_Hz  # (Fs/Freq)  f = 0.1 Hz at a sampling Frequency of 20 Hz
FLAG_APPLY_WINDOW = 1
DOPPLER_WINDOW_SIZE = 16
pDopplerWindow = np.zeros(shape=DOPPLER_WINDOW_SIZE, dtype=float)
scale_heartWfm = 300000
heartWfm_Spectrum_FftSize = PHASE_FFT_SIZE
pVitalSigns_Heart_AbsSpectrum = np.zeros(shape=heartWfm_Spectrum_FftSize, dtype=float)
freqIncrement_Hz = samplingFreq_Hz / PHASE_FFT_SIZE
breath_startFreq_Index = math.floor(breath_startFreq_Hz / freqIncrement_Hz)
breath_endFreq_Index = math.ceil(breath_endFreq_Hz / freqIncrement_Hz)
heart_startFreq_Index = math.floor(heart_startFreq_Hz / freqIncrement_Hz)
heart_endFreq_Index = math.ceil(heart_endFreq_Hz / freqIncrement_Hz)
heart_startFreq_Index_1p6Hz = math.floor(1.6 / freqIncrement_Hz)
heart_endFreq_Index_4Hz = math.ceil(4.0 / freqIncrement_Hz)
MAX_NUM_PEAKS_SPECTRUM = 4  # Maximum number of peaks selected in the Vital Signs Spectrum
pPeakSortOutIndex = np.zeros(shape=MAX_NUM_PEAKS_SPECTRUM)  # Sorted Peaks in the Spectrum
confidenceMetricBreath = np.zeros(shape=MAX_NUM_PEAKS_SPECTRUM,
                                  dtype=float)  # Confidence Metric associated with each Breathing Spectrum Peak
confMetric_spectrumHeart_IndexStart = math.floor(heart_startFreq_Hz / freqIncrement_Hz)
confMetric_spectrumHeart_IndexEnd = math.ceil(heart_endFreq_Hz / freqIncrement_Hz)
confMetric_spectrumBreath_IndexStart = math.floor(breath_startFreq_Hz / freqIncrement_Hz)
confMetric_spectrumBreath_IndexEnd = math.ceil(breath_endFreq_Hz / freqIncrement_Hz)
confMetric_spectrumHeart_IndexStart_1p6Hz = heart_startFreq_Index_1p6Hz
confMetric_spectrumHeart_IndexStart_4Hz = heart_endFreq_Index_4Hz
CONF_METRIC_BANDWIDTH_PEAK_HEART_HZ = 0.1  # Bandwidth around the max peak to include in the signal power estimation
CONF_METRIC_BANDWIDTH_PEAK_BREATH_HZ = 0.2  # Bandwidth around the max peak to include in the signal power estimation
confMetric_numIndexAroundPeak_heart = math.floor(CONF_METRIC_BANDWIDTH_PEAK_HEART_HZ / freqIncrement_Hz)
confMetric_numIndexAroundPeak_breath = math.floor(CONF_METRIC_BANDWIDTH_PEAK_BREATH_HZ / freqIncrement_Hz)
WAVELENGTH_MM = 3.9  # Wavelength in millimeter
scaleFactor_PhaseToDisp = WAVELENGTH_MM / (4 * math.pi)
MAX_HEART_RATE_BPM = 120  # Maximum Heart-rate allowed
HEART_HAMRONIC_THRESH_BPM = 4.0  # Threshold
confidenceMetricHeart = np.zeros(
    shape=MAX_NUM_PEAKS_SPECTRUM)  # Confidence Metric associated with each Cardiac Spectrum Peak
FLAG_HARMONIC_CANCELLATION = 1
BREATHING_HARMONIC_NUM = 2  # Breathing Harmonic
BREATHING_HAMRONIC_THRESH_BPM = 4.0  # Threshold
FLAG_MEDIAN_FILTER = 1
MEDIAN_WINDOW_LENGTH = 20  # Median window length
pBufferHeartRate = np.zeros(shape=MEDIAN_WINDOW_LENGTH,
                            dtype=float)  # Maintains a history of the Previous heart rate measurements
pBufferBreathingRate = np.zeros(shape=MEDIAN_WINDOW_LENGTH,
                                dtype=float)  # Maintains a history of the Previous heart rate measurements
pBufferHeartRate_4Hz = np.zeros(shape=MEDIAN_WINDOW_LENGTH)
alpha_breathing = 0.1
alpha_heart = 0.05
first_time = 1
guiFlag_ClutterRemoval = 1
pFilterCoefsBreath = np.array(
    [1.0000, 0, -1.0000, 1.0000, -1.9632, 0.9644, 1.0000, 0, -1.0000, 1.0000, -1.8501, 0.8681])
pScaleValsBreath = np.array([0.0602, 0.0602, 1.0000])
pDelayBreath = np.zeros(shape=(BREATHING_WFM_IIR_FILTER_TAPS_LENGTH,))
pDelayHeart = np.zeros(shape=(HEART_WFM_IIR_FILTER_TAPS_LENGTH,))
circularBufferSizeBreath = 256
circularBufferSizeHeart = 512
pVitalSigns_Breath_CircularBuffer = np.empty(circularBufferSizeBreath, dtype=np.float32)
pVitalSigns_Heart_CircularBuffer = np.empty(circularBufferSizeHeart, dtype=np.float32)
rangeStart_meter = 0.3
rangeEnd_meter = 1.1
rangeBinStartIndex = math.floor(
    rangeStart_meter / cfg.RANGE_RESOLUTION)  # Range-bin index corresponding to the Starting range in meters
rangeBinEndIndex = math.floor(
    rangeEnd_meter / cfg.RANGE_RESOLUTION)  # Range-bin index corresponding to the Ending range in meters
pFilterCoefsHeart_4Hz = np.array([1.0000, 0, -1.0000, 1.0000, -0.5306, 0.5888,
                                  1.0000, 0, -1.0000, 1.0000, -1.8069, 0.8689,
                                  1.0000, 0, -1.0000, 1.0000, -1.4991, 0.5887,
                                  1.0000, 0, -1.0000, 1.0000, -0.6654, 0.2099])
pScaleValsHeart_4Hz = np.array([0.4188, 0.4188, 0.3611, 0.3611, 1.0000])
prevRangeBins = np.zeros(shape=(rangeBinEndIndex - rangeBinStartIndex), dtype=np.complex128)
breathWfmOutUpdated = 0.0
heartWfmOutUpdated = 0.0
phasePrevFrame = 0
diffPhaseCorrectionCum = 0
phaseUsedComputationPrev = 0
dataCurr = 0
dataPrev2 = 0
dataPrev1 = 0


def unwrap(phase, phasePrev, phaseCorrectionCum):
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

    if (PI > diffPhaseCorrection > 0) or (-PI < diffPhaseCorrection < 0):
        diffPhaseCorrection = 0.0

    phaseCorrectionCum = phaseCorrectionCum + diffPhaseCorrection
    phaseOut = phase + phaseCorrectionCum

    return phaseOut, phaseCorrectionCum


def clutter_removal_C(range_indv_bin):
    global prevRangeBins
    alphaClutter = 0.01
    prevRangeBins = alphaClutter * range_indv_bin + (1 - alphaClutter) * prevRangeBins
    rangeBinMax = np.argmax(np.absolute(prevRangeBins - range_indv_bin))
    return rangeBinMax


def process_vitalsigns(rangeBuffer):
    global gFrameCount
    gFrameCount += 1
    res_across_virtual_antenna = []
    for i in range(rangeBuffer.shape[0]):
        for j in range(rangeBuffer.shape[1]):
            for k in range(rangeBuffer.shape[2]):
                res_across_virtual_antenna.append(process_indiv_vitalsigns(rangeBuffer[i][j][k]))


def filter_RemoveImpulseNoise(dataprev2, dataprev1, datacurr, thresh):
    forwardDiff = dataprev1 - dataCurr
    backwardDiff = dataprev1 - dataprev2
    return (dataprev2 + (datacurr - dataprev2) / 2) if (forwardDiff > thresh and backwardDiff > thresh) or (
            forwardDiff < -thresh and backwardDiff < -thresh) else dataprev1


def filter_FIR(pDataIn, filterCoefs, numCoefs):
    return sum((pDataIn * filterCoefs)[0:numCoefs])


def filter_IIR_BiquadCascade(DataIn, filterCoefs, pScaleVals, pDelay, numStages):
    numCoefsStage = 6
    input_val = DataIn
    y = 0
    for indexStage in range(numStages):
        indexTemp = numCoefsStage * indexStage
        b0 = filterCoefs[indexTemp]
        b1 = filterCoefs[indexTemp + 1]
        b2 = filterCoefs[indexTemp + 2]
        a1 = filterCoefs[indexTemp + 4]
        a2 = filterCoefs[indexTemp + 5]
        scaleVal = pScaleVals[indexStage]

        pDelay[indexTemp] = scaleVal * input_val - a1 * pDelay[indexTemp + 1] - a2 * pDelay[indexTemp + 2]
        y = b0 * pDelay[indexTemp] + b1 * pDelay[indexTemp + 1] + b2 * pDelay[indexTemp + 2]

        pDelay[indexTemp + 2] = pDelay[indexTemp + 1]
        pDelay[indexTemp + 1] = pDelay[indexTemp]

        input_val = y
    return y, pDelay


def computeAGC(pDataIn, dataInLength, lenBlock, thresh):
    scaleValueSum = 0
    for indexTemp in range(lenBlock, dataInLength):
        sumEnergy = 0
        for indexInner in range(0, lenBlock + 1):
            indexCurr = indexTemp - lenBlock + indexInner
            sumEnergy += pDataIn[indexCurr] * pDataIn[indexCurr]

        if sumEnergy > thresh:
            scaleValue = np.sqrt(thresh / sumEnergy)
            scaleValueSum += 1
            for indexInner in range(0, lenBlock + 1):
                indexCurr = indexTemp - lenBlock + indexInner
                pDataIn[indexCurr] = pDataIn[indexCurr] * scaleValue
    return scaleValueSum


def find_Peaks(pDataIn, peakIndex, peakValues, startIndex, endIndex):
    numPeaks = 0
    for temp in range(startIndex + 1, endIndex - 1):
        if pDataIn[temp] > pDataIn[temp - 1] and pDataIn[temp] > pDataIn[temp + 1]:
            peakIndex[numPeaks] = temp
            peakValues[numPeaks] = float(pDataIn[temp])
            numPeaks += 1
    return numPeaks, peakValues


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
    i, j, l1, l2, h2, predj, tw_offset, stride, fft_jmp = 0, 0, 0, 0, 0, 0, 0, 0, 0
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
        j = 0
        fft_jmp >>= 2
        h2 = stride >> 1
        l1 = stride
        l2 = stride + (stride >> 1)
        x = ptr_x
        w = ptr_w + tw_offset
        tw_offset += fft_jmp
        stride >>= 2

        for i in range(0, npoints, 4):
            co10, si10, co20, si20, co30, si30 = w[j + 1], w[j + 0], w[j + 3], w[j + 2], w[j + 5], w[j + 4]
            x_0, x_1, x_l1_0, x_l1_1, x_l2_0, x_l2_1, x_h2_0, x_h2_1 = x[0], x[1], x[l1], x[l1 + 1], x[l2], x[l2 + 1], \
                x[h2], x[h2 + 1]
            xh0_0 = x_0 + x_l1_0
            xh1_0 = x_1 + x_l1_1
            xl0_0 = x_0 - x_l1_0
            xl1_0 = x_1 - x_l1_1
            xh20_0 = x_h2_0 + x_l2_0
            xh21_0 = x_h2_1 + x_l2_1
            xl20_0 = x_h2_0 - x_l2_0
            xl21_0 = x_h2_1 - x_l2_1
            x0 = x
            x2 = x0
            j += 6
            x += 2
            predj = (j - fft_jmp)
            if not predj:
                x += fft_jmp
                j = 0
            x0[0] = xh0_0 + xh20_0
            x0[1] = xh1_0 + xh21_0
            xt0_0 = xh0_0 - xh20_0
            yt0_0 = xh1_0 - xh21_0
            xt1_0 = xl0_0 + xl21_0
            yt2_0 = xl1_0 + xl20_0
            xt2_0 = xl0_0 - xl21_0
            yt1_0 = xl1_0 - xl20_0
            x2[h2] = SMPY_32(si10, yt1_0) + SMPY_32(co10, xt1_0)
            x2[h2 + 1] = SMPY_32(co10, yt1_0) - SMPY_32(si10, xt1_0)
            x2[l1] = SMPY_32(si20, yt0_0) + SMPY_32(co20, xt0_0)
            x2[l1 + 1] = SMPY_32(co20, yt0_0) - SMPY_32(si20, xt0_0)
            x2[l2] = SMPY_32(si30, yt2_0) + SMPY_32(co30, xt2_0)
            x2[l2 + 1] = SMPY_32(co30, yt2_0) - SMPY_32(si30, xt2_0)

    y0 = ptr_y
    y2 = ptr_y + npoints
    x0 = ptr_x
    x2 = ptr_x + (npoints >> 1)

    if radix == 2:
        y1 = y0 + (npoints >> 2)
        y3 = y2 + (npoints >> 2)
        l1 = norm + 1
        j0 = 8
        n0 = npoints >> 1
    else:
        y1 = y0 + (npoints >> 1)
        y3 = y2 + (npoints >> 1)
        l1 = norm + 2
        j0 = 4
        n0 = npoints >> 2
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
        y0[2 * h2] = n00
        y0[2 * h2 + 1] = n01
        y1[2 * h2] = n10
        y1[2 * h2 + 1] = n11
        y2[2 * h2] = n20
        y2[2 * h2 + 1] = n21
        y3[2 * h2] = n30
        y3[2 * h2 + 1] = n31
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

        y0[2 * h2 + 2] = n02
        y0[2 * h2 + 3] = n03
        y1[2 * h2 + 2] = n12
        y1[2 * h2 + 3] = n13
        y2[2 * h2 + 2] = n22
        y2[2 * h2 + 3] = n23
        y3[2 * h2 + 2] = n32
        y3[2 * h2 + 3] = n33
        j += j0

        if j == n0:
            j += n0
            x0 += npoints >> 1
            x2 += npoints >> 1
    return w, x, y0


def MmwDemo_magnitudeSquared(inpBuff, magSqrdBuff, numSamples):
    for i in range(numSamples):
        magSqrdBuff[i] = float(inpBuff[i].real) * float(inpBuff[i].real) + float(inpBuff[i].imag) * float(
            inpBuff[i].imag)
    return magSqrdBuff


def filterPeaksWfm(pPeakLocsIn, pPeakLocsOut, numPeaksIn, winMin):
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
        sumVal = 0
        for index in range(0, dataInLength - indexLag):
            sumVal += pDataIn[index] * pDataIn[index + indexLag]
        pDataOut[indexLag] = sumVal


def computeMaxIndex(pDataIn, startIndex, endIndex):
    MaxVal = pDataIn[startIndex]
    MaxValIndex = startIndex
    for temp in range(startIndex + 1, endIndex):
        if pDataIn[temp] > MaxVal:
            MaxVal = pDataIn[temp]
            MaxValIndex = temp
    return MaxValIndex


def heapsort_index(pDataIn, dataLength):
    l = dataLength >> 1
    ir = dataLength - 1

    # Initialize pSortOutIndex with indices from 0 to dataLength - 1
    pSortOutIndex = list(range(dataLength))

    if dataLength == 1:
        return pSortOutIndex

    while True:
        if l > 0:
            indxt = pSortOutIndex[l - 1]
            q = pDataIn[indxt]
            l -= 1
        else:
            indxt = pSortOutIndex[ir]
            q = pDataIn[indxt]
            pSortOutIndex[ir] = pSortOutIndex[0]
            ir -= 1
            if ir == 0:
                pSortOutIndex[0] = indxt
                return pSortOutIndex

        i = l
        j = (l << 1) + 1

        while j <= ir:
            if j < ir and pDataIn[pSortOutIndex[j]] < pDataIn[pSortOutIndex[j + 1]]:
                j += 1

            if q < pDataIn[pSortOutIndex[j]]:
                pSortOutIndex[i] = pSortOutIndex[j]
                j = 2 * j + 1
            else:
                break

        pSortOutIndex[i] = indxt


def computeConfidenceMetric(pDataSpectrum, spectrumIndexStart, spectrumIndexEnd, peakIndex, numIndexAroundPeak):
    startInd = peakIndex - numIndexAroundPeak
    endInd = peakIndex + numIndexAroundPeak

    if startInd < 0:
        startInd = 0
    if endInd >= spectrumIndexEnd:
        endInd = spectrumIndexEnd - 1

    # Energy of the complete Spectrum
    sumSignal = sum(pDataSpectrum[spectrumIndexStart: spectrumIndexEnd + 1])

    # Energy of the frequency Bins including (and around) the peak of interest
    sumPeak = sum(pDataSpectrum[startInd: endInd + 1])

    if abs(sumSignal - sumPeak) < 0.0001:  # This condition would arise if the input signal amplitude is very low
        confidenceMetric = 0
    else:
        confidenceMetric = sumPeak / (sumSignal - sumPeak)

    return confidenceMetric


def computeEnergyHarmonics(pAbsSpectrum, spectrumStartIndex, spectrumEndIndex, freqWindowSize):
    pDataOut = []
    for index in range(spectrumStartIndex, spectrumEndIndex):
        window1StartIndex = index - freqWindowSize
        window2StartIndex = 2 * index - freqWindowSize

        sum_value = 0
        for indexInnerLoop in range(freqWindowSize):
            sum_value += pAbsSpectrum[window1StartIndex + indexInnerLoop] + pAbsSpectrum[
                window2StartIndex + indexInnerLoop]
        pDataOut.append(sum_value)
    return pDataOut


def process_indiv_vitalsigns(range_indv_bin):
    global first_time, pPeakValues, pDelayBreath, pDelayHeart, pVitalSignsBuffer_Cplx, \
        pVitalSignsSpectrumTwiddle32x32, pVitalSigns_SpectrumCplx, pVitalSigns_Breath_AbsSpectrum, \
        pVitalSigns_Heart_AbsSpectrum, breathWfmOutUpdated, heartWfmOutUpdated, phasePrevFrame, \
        diffPhaseCorrectionCum, phaseUsedComputationPrev, dataCurr, dataPrev2, dataPrev1
    range_indv_bin = range_indv_bin[rangeBinStartIndex:rangeBinEndIndex]
    if first_time == 1:
        # static variables taken from c
        first_time = 0
    if guiFlag_ClutterRemoval == 1:
        rangeBinIndexPhase = clutter_removal_C(range_indv_bin)

    rangeBinIndexPhase = np.argmax(np.absolute(range_indv_bin))
    maxVal = np.abs(range_indv_bin[rangeBinIndexPhase])
    rangeBinPhase = np.arctan(range_indv_bin[rangeBinIndexPhase].imag / range_indv_bin[rangeBinIndexPhase].real)
    noiseImpulse_Thresh = 1.5
    pDataIn = np.zeros(shape=FIR_FILTER_SIZE)
    # Processing only the maximum phase value 

    unwrapPhasePeak, diffPhaseCorrectionCum = unwrap(rangeBinPhase, phasePrevFrame, diffPhaseCorrectionCum)
    phasePrevFrame = rangeBinPhase
    phaseUsedComputation = unwrapPhasePeak - phaseUsedComputationPrev
    phaseUsedComputationPrev = unwrapPhasePeak
    dataPrev2 = dataPrev1
    dataPrev1 = dataCurr
    dataCurr = phaseUsedComputation
    phaseUsedComputation = filter_RemoveImpulseNoise(dataPrev2, dataPrev1, dataCurr, noiseImpulse_Thresh)

    outputFilterBreathOut, pDelayBreath = filter_IIR_BiquadCascade(phaseUsedComputation, pFilterCoefsBreath,
                                                                   pScaleValsBreath, pDelayBreath,
                                                                   IIR_FILTER_BREATH_NUM_STAGES)
    outputFilterHeartOut, pDelayHeart = filter_IIR_BiquadCascade(phaseUsedComputation, pFilterCoefsHeart_4Hz,
                                                                 pScaleValsHeart_4Hz, pDelayHeart,
                                                                 IIR_FILTER_HEART_NUM_STAGES)

    for loopIndexBuffer in range(1, circularBufferSizeBreath):
        pVitalSigns_Breath_CircularBuffer[loopIndexBuffer - 1] = pVitalSigns_Breath_CircularBuffer[loopIndexBuffer]
    pVitalSigns_Breath_CircularBuffer[- 1] = outputFilterBreathOut
    motionDetected = 0
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
                print('Motion detected')
                motionDetected = 1
            else:
                print('Perfect Frame')
                motionDetected = 0
                # Shift the current contents of the circular Buffer
                for loopIndexBuffer in range(motionDetection_BlockSize, circularBufferSizeHeart):
                    pVitalSigns_Heart_CircularBuffer[loopIndexBuffer - motionDetection_BlockSize] = \
                        pVitalSigns_Heart_CircularBuffer[loopIndexBuffer]
                # Copy the current data segment to the end of the Circular Buffer
                for loopIndexBuffer in range(motionDetection_BlockSize):
                    tempEndIndex = circularBufferSizeHeart - motionDetection_BlockSize
                    pVitalSigns_Heart_CircularBuffer[tempEndIndex + loopIndexBuffer] = pMotionCircularBuffer[
                        loopIndexBuffer]
    # If Motion DETECTED then don't UPDATE or SHIFT the values in the buffer
    else:  # Regular processing
        # Copies the "Cardiac Waveform" in a circular Buffer
        for loopIndexBuffer in range(1, circularBufferSizeHeart):
            pVitalSigns_Heart_CircularBuffer[loopIndexBuffer - 1] = pVitalSigns_Heart_CircularBuffer[loopIndexBuffer]
        pVitalSigns_Heart_CircularBuffer[- 1] = outputFilterHeartOut

    # Spectral Estimation based on the Inter-Peaks Distance
    numPeaksHeart, pPeakValues = find_Peaks(pVitalSigns_Heart_CircularBuffer, pPeakLocsHeart, pPeakValues, 0,
                                            circularBufferSizeHeart - 1)
    if numPeaksHeart != 0:
        numPeaksHeart = filterPeaksWfm(pPeakLocsHeart, pPeakLocsValid, numPeaksHeart, peakDistanceHeart_Min)
    heartRateEst_peakCount = CONVERT_HZ_BPM * ((numPeaksHeart * samplingFreq_Hz) / circularBufferSizeHeart)

    for loopIndexBuffer in range(1, FIR_FILTER_SIZE):
        pDataIn[loopIndexBuffer - 1] = pDataIn[loopIndexBuffer]
    pDataIn[FIR_FILTER_SIZE - 1] = heartRateEst_peakCount
    heartRateEst_peakCount_filtered = filter_FIR(pDataIn, pFilterCoefs, FIR_FILTER_SIZE)

    numPeaksBreath, pPeakValues = find_Peaks(pVitalSigns_Breath_CircularBuffer, pPeakLocsBreath, pPeakValues, 0,
                                             circularBufferSizeBreath - 1)
    if numPeaksBreath != 0:
        numPeaksBreath = filterPeaksWfm(pPeakLocsBreath, pPeakLocsValid, numPeaksBreath, peakDistanceBreath_Min)

    breathingRateEst_peakCount = CONVERT_HZ_BPM * ((numPeaksBreath * samplingFreq_Hz) / circularBufferSizeHeart)
    heartRateEst_peakCount = CONVERT_HZ_BPM * ((numPeaksHeart * samplingFreq_Hz) / circularBufferSizeBreath)

    # Input to the FFT needs to be complex
    for loopIndexBuffer in range(circularBufferSizeBreath):
        pVitalSignsBuffer_Cplx[loopIndexBuffer] = scale_breathingWfm * pVitalSigns_Breath_CircularBuffer[
            loopIndexBuffer]

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
        computeAGC(pVitalSigns_Heart_CircularBuffer, circularBufferSizeHeart, motionDetection_BlockSize,
                   motionDetection_Thresh)

    if guiFlag_MotionDetection == 1:
        outputFilterHeartOut = pMotionCircularBuffer[motionDetection_BlockSize - 1]
    else:
        outputFilterHeartOut = pVitalSigns_Heart_CircularBuffer[- 1]

    # Perform Autocorrelation on the Waveform=
    confidenceMetricHeartOut_xCorr = 0
    heartRateEst_xCorr = 0
    breathRateEst_xCorr = 0
    confidenceMetricBreathOut_xCorr = 0
    if PERFORM_XCORR:
        computeAutoCorrelation(pVitalSigns_Heart_CircularBuffer, circularBufferSizeHeart, pXcorr, xCorr_minLag,
                               xCorr_maxLag)
        xCorr_numPeaks, pPeakValues = find_Peaks(pXcorr, pPeakIndex, pPeakValues, xCorr_minLag, xCorr_maxLag)
        maxIndex_lag = computeMaxIndex(pXcorr, xCorr_minLag, xCorr_maxLag)
        temp = 1.0 / (maxIndex_lag / samplingFreq_Hz)
        heartRateEst_xCorr = CONVERT_HZ_BPM * temp

        confidenceMetricHeartOut_xCorr = 0 if (xCorr_numPeaks == 0) else pXcorr[maxIndex_lag]

        # Auto-correlation on the Breathing Waveform
        computeAutoCorrelation(pVitalSigns_Breath_CircularBuffer, circularBufferSizeBreath,
                               pXcorr, xCorr_Breath_minLag, xCorr_Breath_maxLag)
        xCorr_numPeaks, pPeakValues = find_Peaks(pXcorr, pPeakIndex, pPeakValues, xCorr_Breath_minLag,
                                                 xCorr_Breath_maxLag)
        maxIndex_lag = computeMaxIndex(pXcorr, xCorr_Breath_minLag, xCorr_Breath_maxLag)
        temp = float(1.0) / (maxIndex_lag / samplingFreq_Hz)
        breathRateEst_xCorr = CONVERT_HZ_BPM * temp

        confidenceMetricBreathOut_xCorr = 0 if (xCorr_numPeaks == 0) else pXcorr[maxIndex_lag]

    if FLAG_APPLY_WINDOW:
        index_WinEnd = circularBufferSizeHeart - 1
        for index_win in range(DOPPLER_WINDOW_SIZE):
            tempFloat = pDopplerWindow[index_win]
            pVitalSignsBuffer_Cplx[index_win].real = scale_heartWfm * tempFloat * pVitalSigns_Heart_CircularBuffer[
                index_win]
            pVitalSignsBuffer_Cplx[index_WinEnd].real = scale_heartWfm * tempFloat * pVitalSigns_Heart_CircularBuffer[
                index_WinEnd]
            pVitalSignsBuffer_Cplx[index_win].imag = 0
            pVitalSignsBuffer_Cplx[index_WinEnd].imag = 0
            index_WinEnd -= 1

        for loopIndexBuffer in range(DOPPLER_WINDOW_SIZE, circularBufferSizeHeart - DOPPLER_WINDOW_SIZE):
            pVitalSignsBuffer_Cplx[loopIndexBuffer].real = scale_heartWfm * pVitalSigns_Heart_CircularBuffer[
                loopIndexBuffer]
            pVitalSignsBuffer_Cplx[loopIndexBuffer].imag = 0

    else:
        for loopIndexBuffer in range(circularBufferSizeHeart):
            pVitalSignsBuffer_Cplx[loopIndexBuffer].real = scale_heartWfm * pVitalSigns_Heart_CircularBuffer[
                loopIndexBuffer]
            pVitalSignsBuffer_Cplx[loopIndexBuffer].imag = 0

    pVitalSignsSpectrumTwiddle32x32, pVitalSignsBuffer_Cplx, pVitalSigns_SpectrumCplx = DSP_fft32x32(
        pVitalSignsSpectrumTwiddle32x32, heartWfm_Spectrum_FftSize, pVitalSignsBuffer_Cplx, pVitalSigns_SpectrumCplx)

    pVitalSigns_Heart_AbsSpectrum = MmwDemo_magnitudeSquared(pVitalSigns_SpectrumCplx, pVitalSigns_Heart_AbsSpectrum,
                                                             heartWfm_Spectrum_FftSize)

    # Pick the Peaks in the Breathing Spectrum
    numPeaks_BreathSpectrum, pPeakValues = find_Peaks(pVitalSigns_Breath_AbsSpectrum, pPeakIndex, pPeakValues,
                                                      breath_startFreq_Index, breath_endFreq_Index)
    indexNumPeaks = numPeaks_BreathSpectrum if (
            numPeaks_BreathSpectrum < MAX_NUM_PEAKS_SPECTRUM) else MAX_NUM_PEAKS_SPECTRUM

    maxIndexBreathSpect = 0
    confidenceMetricBreathOut = 0
    if indexNumPeaks != 0:
        pPeakIndexSorted = heapsort_index(pPeakValues, numPeaks_BreathSpectrum)
        for indexTemp in range(indexNumPeaks):
            pPeakSortOutIndex[indexTemp] = pPeakIndex[pPeakIndexSorted[numPeaks_BreathSpectrum - indexTemp - 1]]
            confidenceMetricBreath[indexTemp] = computeConfidenceMetric(pVitalSigns_Breath_AbsSpectrum,
                                                                        confMetric_spectrumBreath_IndexStart,
                                                                        confMetric_spectrumBreath_IndexEnd,
                                                                        pPeakSortOutIndex[indexTemp],
                                                                        confMetric_numIndexAroundPeak_breath)
            maxIndexBreathSpect = pPeakSortOutIndex[0]  # The maximum peak
            confidenceMetricBreathOut = confidenceMetricBreath[0]

    else:
        maxIndexBreathSpect = computeMaxIndex(pVitalSigns_Breath_AbsSpectrum, breath_startFreq_Index,
                                              breath_endFreq_Index)
        confidenceMetricBreathOut = computeConfidenceMetric(pVitalSigns_Breath_AbsSpectrum,
                                                            0,
                                                            PHASE_FFT_SIZE / 4,
                                                            maxIndexBreathSpect,
                                                            confMetric_numIndexAroundPeak_breath)
    peakValueBreathSpect = pVitalSigns_Breath_AbsSpectrum[maxIndexBreathSpect] / (10 * scale_breathingWfm)

    # Pick the Peaks in the Heart Spectrum [1.6 - 4.0 Hz]
    numPeaks_heartSpectrum, pPeakValues = find_Peaks(pVitalSigns_Heart_AbsSpectrum, pPeakIndex, pPeakValues,
                                                     heart_startFreq_Index_1p6Hz, heart_endFreq_Index_4Hz)
    indexNumPeaks = numPeaks_heartSpectrum if (
            numPeaks_heartSpectrum < MAX_NUM_PEAKS_SPECTRUM) else MAX_NUM_PEAKS_SPECTRUM

    if indexNumPeaks != 0:
        pPeakIndexSorted = heapsort_index(pPeakValues, numPeaks_heartSpectrum)
        for indexTemp in range(indexNumPeaks):
            pPeakSortOutIndex[indexTemp] = pPeakIndex[pPeakIndexSorted[numPeaks_heartSpectrum - indexTemp - 1]]
        maxIndexHeartBeatSpect_4Hz = pPeakSortOutIndex[0]  # The maximum peak
        confidenceMetricHeartOut_4Hz = computeConfidenceMetric(pVitalSigns_Heart_AbsSpectrum,
                                                               confMetric_spectrumHeart_IndexStart_1p6Hz,
                                                               confMetric_spectrumHeart_IndexStart_4Hz,
                                                               maxIndexHeartBeatSpect_4Hz,
                                                               confMetric_numIndexAroundPeak_heart)
    else:
        maxIndexHeartBeatSpect_4Hz = computeMaxIndex(pVitalSigns_Heart_AbsSpectrum, heart_startFreq_Index_1p6Hz,
                                                     heart_endFreq_Index_4Hz)
        confidenceMetricHeartOut_4Hz = computeConfidenceMetric(pVitalSigns_Heart_AbsSpectrum,
                                                               0,
                                                               PHASE_FFT_SIZE / 4,
                                                               maxIndexHeartBeatSpect_4Hz,
                                                               confMetric_numIndexAroundPeak_heart)
    heartRateEst_FFT_4Hz = CONVERT_HZ_BPM * maxIndexHeartBeatSpect_4Hz * freqIncrement_Hz

    # If a peak is within [1.6 2.0] Hz then check if a harmonic is present is the cardiac spectrum region [0.8 - 2.0] Hz
    if heartRateEst_FFT_4Hz < MAX_HEART_RATE_BPM:
        for indexTemp in range(1, numPeaks_heartSpectrum):
            if abs(heartRateEst_FFT_4Hz - CONVERT_HZ_BPM * freqIncrement_Hz * pPeakSortOutIndex[indexTemp]) < \
                    HEART_HAMRONIC_THRESH_BPM:
                heartRateEst_FFT_4Hz = CONVERT_HZ_BPM * freqIncrement_Hz * pPeakSortOutIndex[indexTemp]
                break

    # Pick the Peaks in the Cardiac Spectrum
    numPeaks_heartSpectrum, pPeakValues = find_Peaks(pVitalSigns_Heart_AbsSpectrum, pPeakIndex, pPeakValues,
                                                     heart_startFreq_Index,
                                                     heart_endFreq_Index)
    indexNumPeaks = numPeaks_heartSpectrum if (
            numPeaks_heartSpectrum < MAX_NUM_PEAKS_SPECTRUM) else MAX_NUM_PEAKS_SPECTRUM

    if indexNumPeaks != 0:
        pPeakIndexSorted = heapsort_index(pPeakValues, numPeaks_heartSpectrum)
        for indexTemp in range(indexNumPeaks):
            pPeakSortOutIndex[indexTemp] = pPeakIndex[pPeakIndexSorted[numPeaks_heartSpectrum - indexTemp - 1]]
            confidenceMetricHeart[indexTemp] = computeConfidenceMetric(pVitalSigns_Heart_AbsSpectrum,
                                                                       confMetric_spectrumHeart_IndexStart,
                                                                       confMetric_spectrumHeart_IndexEnd,
                                                                       pPeakSortOutIndex[indexTemp],
                                                                       confMetric_numIndexAroundPeak_heart)
        maxIndexHeartBeatSpect = pPeakSortOutIndex[0]  # The maximum peak
        confidenceMetricHeartOut = confidenceMetricHeart[0]

    else:
        maxIndexHeartBeatSpect = computeMaxIndex(pVitalSigns_Heart_AbsSpectrum, heart_startFreq_Index,
                                                 heart_endFreq_Index)
        confidenceMetricHeartOut = computeConfidenceMetric(pVitalSigns_Heart_AbsSpectrum,
                                                           0,
                                                           PHASE_FFT_SIZE / 4,
                                                           maxIndexHeartBeatSpect,
                                                           confMetric_numIndexAroundPeak_heart)

    # Remove the First Breathing Harmonic (if present in the cardiac Spectrum)
    if FLAG_HARMONIC_CANCELLATION:
        diffIndex = abs(maxIndexHeartBeatSpect - BREATHING_HARMONIC_NUM * maxIndexBreathSpect)
        # Only cancel the 2nd Breathing Harmonic
        if diffIndex * freqIncrement_Hz * CONVERT_HZ_BPM < BREATHING_HAMRONIC_THRESH_BPM:
            maxIndexHeartBeatSpect = pPeakSortOutIndex[1]  # Pick the 2nd Largest peak in the cardiac-spectrum
            confidenceMetricHeartOut = confidenceMetricHeart[1]

    heartRateEst_FFT = CONVERT_HZ_BPM * maxIndexHeartBeatSpect * freqIncrement_Hz
    breathingRateEst_FFT = CONVERT_HZ_BPM * maxIndexBreathSpect * freqIncrement_Hz

    pDataOutTemp = np.zeros(heartWfm_Spectrum_FftSize, dtype=np.float32)

    # Compute energy harmonics for heart
    pDataOutTemp = computeEnergyHarmonics(pVitalSigns_Heart_AbsSpectrum,
                                          confMetric_spectrumHeart_IndexStart,
                                          confMetric_spectrumHeart_IndexEnd,
                                          confMetric_numIndexAroundPeak_heart)
    maxIndexHeartBeatSpect_temp = computeMaxIndex(pDataOutTemp,
                                                  confMetric_spectrumHeart_IndexStart,
                                                  confMetric_spectrumHeart_IndexEnd)
    heartRateEst_HarmonicEnergy = CONVERT_HZ_BPM * maxIndexHeartBeatSpect_temp * freqIncrement_Hz

    pDataOutTemp = np.zeros(heartWfm_Spectrum_FftSize, dtype=np.float32)

    # Compute energy harmonics for breath
    pDataOutTemp = computeEnergyHarmonics(pVitalSigns_Breath_AbsSpectrum,
                                          confMetric_spectrumBreath_IndexStart,
                                          confMetric_spectrumBreath_IndexEnd,
                                          confMetric_numIndexAroundPeak_breath)
    maxIndexBreathSpect_temp = computeMaxIndex(pDataOutTemp,
                                               confMetric_spectrumBreath_IndexStart,
                                               confMetric_spectrumBreath_IndexEnd)
    breathRateEst_HarmonicEnergy = CONVERT_HZ_BPM * maxIndexBreathSpect_temp * freqIncrement_Hz

    #  Median Value for Heart Rate and Breathing Rate based on 'MEDIAN_WINDOW_LENGTH' previous estimates
    if FLAG_MEDIAN_FILTER:
        for loopIndexBuffer in range(1, MEDIAN_WINDOW_LENGTH):
            pBufferHeartRate[loopIndexBuffer - 1] = pBufferHeartRate[loopIndexBuffer]
            pBufferBreathingRate[loopIndexBuffer - 1] = pBufferBreathingRate[loopIndexBuffer]
            pBufferHeartRate_4Hz[loopIndexBuffer - 1] = pBufferHeartRate_4Hz[loopIndexBuffer]

        pBufferHeartRate[MEDIAN_WINDOW_LENGTH - 1] = heartRateEst_FFT
        pBufferBreathingRate[MEDIAN_WINDOW_LENGTH - 1] = breathingRateEst_FFT
        pBufferHeartRate_4Hz[MEDIAN_WINDOW_LENGTH - 1] = heartRateEst_FFT_4Hz

        pPeakSortTempIndex = heapsort_index(pBufferHeartRate, MEDIAN_WINDOW_LENGTH)
        heartRateEst_FFT = pBufferHeartRate[pPeakSortTempIndex[int(MEDIAN_WINDOW_LENGTH / 2) + 1]]

        pPeakSortTempIndex = heapsort_index(pBufferHeartRate_4Hz, MEDIAN_WINDOW_LENGTH)
        heartRateEst_FFT_4Hz = pBufferHeartRate_4Hz[pPeakSortTempIndex[int(MEDIAN_WINDOW_LENGTH / 2) + 1]]

        pPeakSortTempIndex = heapsort_index(pBufferBreathingRate, MEDIAN_WINDOW_LENGTH)
        breathingRateEst_FFT = pBufferBreathingRate[pPeakSortTempIndex[int(MEDIAN_WINDOW_LENGTH / 2) + 1]]

    # Exponential Smoothing
    breathWfmOutPrev = breathWfmOutUpdated
    breathWfmOutUpdated = alpha_breathing * (outputFilterBreathOut * outputFilterBreathOut) + (
            1 - alpha_breathing) * breathWfmOutPrev  # Exponential Smoothing
    sumEnergyBreathWfm = breathWfmOutUpdated * 10000

    heartWfmOutPrev = heartWfmOutUpdated
    heartWfmOutUpdated = alpha_heart * (outputFilterHeartOut * outputFilterHeartOut) + (
            1 - alpha_heart) * heartWfmOutPrev  # Exponential Smoothing
    sumEnergyHeartWfm = heartWfmOutUpdated * 10000

    VitalSigns_Output = {
        "unwrapPhasePeak_mm": unwrapPhasePeak,
        "outputFilterBreathOut": outputFilterBreathOut,
        "outputFilterHeartOut": outputFilterHeartOut,
        "rangeBinIndexPhase": rangeBinIndexPhase,
        "maxVal": maxVal,
        "sumEnergyHeartWfm": sumEnergyHeartWfm,
        "sumEnergyBreathWfm": sumEnergyBreathWfm * peakValueBreathSpect,

        "confidenceMetricBreathOut": confidenceMetricBreathOut,
        "confidenceMetricHeartOut": confidenceMetricHeartOut,  # Confidence Metric associated with the estimates
        "confidenceMetricHeartOut_4Hz": confidenceMetricHeartOut_4Hz,
        "confidenceMetricHeartOut_xCorr": confidenceMetricHeartOut_xCorr,

        "breathingRateEst_FFT": breathingRateEst_FFT,
        "breathingRateEst_peakCount": breathingRateEst_peakCount,
        "heartRateEst_peakCount_filtered": heartRateEst_peakCount_filtered,
        "heartRateEst_xCorr": heartRateEst_xCorr,
        "heartRateEst_FFT_4Hz": heartRateEst_FFT_4Hz,
        "heartRateEst_FFT": heartRateEst_FFT,
        "rangeBinStartIndex": rangeBinStartIndex,
        "rangeBinEndIndex": rangeBinEndIndex,
        "motionDetectedFlag": motionDetected,
        "breathingRateEst_xCorr": breathRateEst_xCorr,  # breathRateEst_HarmonicEnergy;
        "confidenceMetricBreathOut_xCorr": confidenceMetricBreathOut_xCorr,  # breathRateEst_HarmonicEnergy;
        "breathingRateEst_harmonicEnergy": breathRateEst_HarmonicEnergy,  # heartRateEst_HarmonicEnergy;
        "heartRateEst_harmonicEnergy": heartRateEst_HarmonicEnergy
    }


if __name__ == "__main__":
    pcd = []
    bin_filename = '1684598876.bin'
    total_frame_number = 799
    pointCloudProcessCFG = PointCloudProcessCFG()
    shift_arr = cfg.MMWAVE_RADAR_LOC
    bin_reader = RawDataReader(bin_filename)
    frame_no = 0
    raw_adc_data = np.fromfile(bin_filename, dtype=np.uint16)
    raw_adc_data = raw_adc_data.reshape(total_frame_number, -1)

    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        if pointCloudProcessCFG.enableStaticClutterRemoval:
            rangeResult = clutter_removal(rangeResult, axis=2)

        dopplerResult = dopplerFFT(rangeResult, frameConfig)
        pointCloud = frame2pointcloud(dopplerResult, pointCloudProcessCFG)
        frame_no += 1
        print('Frame %d:' % frame_no, pointCloud.shape)
        pcd.append(pointCloud)
