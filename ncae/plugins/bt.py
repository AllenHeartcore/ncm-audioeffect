import numpy as np
from scipy.signal import sosfilt

from ncae.audio import Audio

BASS_FREQ = 120
TREBLE_FREQ = 8000


def _shelf_sos(f0, gain_db, fs, high=False):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    cw = np.cos(w0)
    sw = np.sin(w0)
    # RBJ shelf, S = 1
    alpha = sw / 2 * np.sqrt((a + 1 / a) * (1 / 1 - 1) + 2)
    sa = 2 * np.sqrt(a) * alpha
    if high:
        b = np.array([a * ((a + 1) + (a - 1) * cw + sa),
                      -2 * a * ((a - 1) + (a + 1) * cw),
                      a * ((a + 1) + (a - 1) * cw - sa)])
        a_ = np.array([(a + 1) - (a - 1) * cw + sa,
                       2 * ((a - 1) - (a + 1) * cw),
                       (a + 1) - (a - 1) * cw - sa])
    else:
        b = np.array([a * ((a + 1) - (a - 1) * cw + sa),
                      2 * a * ((a - 1) - (a + 1) * cw),
                      a * ((a + 1) - (a - 1) * cw - sa)])
        a_ = np.array([(a + 1) + (a - 1) * cw + sa,
                       -2 * ((a - 1) + (a + 1) * cw),
                       (a + 1) + (a - 1) * cw - sa])
    return np.concatenate([b / a_[0], [1.0], a_[1:] / a_[0]])[None, :]


def _apply_sos(audio: Audio, sos):
    wave = audio.wave
    if wave.ndim == 1:
        audio.wave = sosfilt(sos, wave)
    else:
        for ch in range(wave.shape[1]):
            wave[:, ch] = sosfilt(sos, wave[:, ch])


class NCAEPluginBT:

    def __init__(self, data: dict):

        self.bass = data["bass"]
        self.treble = data["treble"]

    def applyto(self, audio: Audio):

        preamp = -max(self.bass, self.treble, 0)
        if preamp:
            audio.wave = audio.wave * 10 ** (preamp / 20)
        sos = []
        if self.bass:
            sos.append(_shelf_sos(BASS_FREQ, self.bass, audio.sr, high=False))
        if self.treble:
            sos.append(_shelf_sos(TREBLE_FREQ, self.treble, audio.sr, high=True))
        if sos:
            _apply_sos(audio, np.concatenate(sos))
