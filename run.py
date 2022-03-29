import librosa
import numpy as np
import soundfile
import sys

def get_frames(data, window_length=2048, hop_length=512):
    return librosa.core.spectrum.util.frame(
        np.pad(data, int(window_length // 2), mode='reflect'),
        frame_length=window_length, hop_length=hop_length)

def get_STFT(data, window_length=2048, hop_length=512):
    # разбиваем амплитуды на пересекающиеся фреймы
    frames = get_frames(data, window_length, hop_length)
    
    # получаем веса для Фурье
    fft_weights = librosa.core.spectrum.get_window('hann', window_length, fftbins=True)
    
    # оконное преобразование Фурье
    stft = np.fft.rfft(frames * fft_weights[:, None], axis=0)
    return stft

def phase_shift(data, ratio, window_length=2048, hop_length=512):
    stft = get_STFT(data)
    bins_phase = np.linspace(0, np.pi * hop_length, stft.shape[0]) # частоты фаз бинов
    phase_shift = np.angle(stft[:, 0]) # разность фаз между двумя фреймами
    delta_t = np.arange(0, stft.shape[1], ratio)

    syn_phase = np.zeros(len(phase_shift), dtype=stft.dtype)
    stft_out = np.empty((stft.shape[0], len(delta_t)), dtype=stft.dtype)

    stft = np.concatenate((stft, np.zeros((stft.shape[0], 2))), axis=-1)

    for i, k in enumerate(delta_t):
        
        frequency_dev = np.angle(stft[:, int(k)]) - np.angle(stft[:, int(k)+1]) - bins_phase # сдвиг частоты для двух последовательных бинов
        wrapped_fd = np.mod(frequency_dev + np.pi, 2*np.pi) - np.pi # диапозон -pi/pi
        true_frequency = frequency_dev + wrapped_fd # истинная частота компоненты

        # Пересчитывем разбиение на фреймы(окна) так, чтоб частота компоненты сигнала попадала точно на частоту бина
        # И разность фаз была 0
        
        bin_energy = (1 - k%1) * np.abs(stft[:, int(k)]) + k%1 * np.abs(stft[:, int(k)+1]) # энергия компоненты сигнала
        syn_phase.real, syn_phase.imag = np.cos(phase_shift), np.sin(phase_shift)
        stft_out[:, i] = bin_energy * syn_phase


        phase_shift = phase_shift + true_frequency # накопление сдвига фазы

        wav_new = librosa.istft(stft_out, hop_length=hop_length, window='hann')

    return wav_new

if __name__ == '__main__':
    [inputFile, outputFile, ratio] = sys.argv[1:]

    data, sr = librosa.load(inputFile)
    wav_new = phase_shift(data, 1/float(ratio))
    soundfile.write(outputFile, wav_new, sr)