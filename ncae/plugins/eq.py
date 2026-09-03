import numpy as np
from scipy.signal import sosfilt

from ncae.audio import Audio

# 10-band graphic EQ, standard octave-spaced center frequencies
BAND_FREQS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
BAND_Q = 1.41


def _peak_sos(f0, gain_db, fs):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * BAND_Q)
    cw = np.cos(w0)
    b = np.array([1 + alpha * a, -2 * cw, 1 - alpha * a])
    a_ = np.array([1 + alpha / a, -2 * cw, 1 - alpha / a])
    return np.concatenate([b / a_[0], [1.0], a_[1:] / a_[0]])[None, :]


def _apply_sos(audio: Audio, sos):
    wave = audio.wave
    if wave.ndim == 1:
        audio.wave = sosfilt(sos, wave)
    else:
        for ch in range(wave.shape[1]):
            wave[:, ch] = sosfilt(sos, wave[:, ch])


class NCAEPluginEQ:

    def __init__(self, data: dict):

        assert len(data["eqs"]) == 10
        self.eqs = data["eqs"]

    def applyto(self, audio: Audio):

        preamp = -max(max(self.eqs), 0)
        if preamp:
            audio.wave = audio.wave * 10 ** (preamp / 20)
        sos = np.concatenate(
            [
                _peak_sos(f0, gain, audio.sr)
                for f0, gain in zip(BAND_FREQS, self.eqs)
                if gain != 0
            ]
        )
        if sos.shape[0]:
            _apply_sos(audio, sos)
