import numpy as np
from scipy.signal import lfilter

def filter_adaptively(ma_coeff: np.ndarray, ar_coeff: np.ndarray, signal:np.ndarray, filter_state_in: np.ndarray =None) -> [np.ndarray, np.ndarray]:
    """
    Allows for segment-wise filtering of a signal with changing filters.

    Example for LPC filtering:
    Call

        segment_out, filter_state = filter_adaptively(np.array([1]), LPCs, segment, filter_state_in)

    for every signal segment, using the corresponding (time varying) LPCs for this frame. 'filter_adaptively' will
    ensure a correct initialization of the time varying filter for each segment.
    For the first segment, you do not need to provide a filter_state_in as  'filter_adaptively' will then initialize
    and return the first filter state.


    :param ma_coeff: the moving average filter coefficientsF
    :param ar_coeff: the autoregressive filter coefficients (e.g. LPCs)
    :param signal: the input signal
    :param filter_state: the initial conditions to be used when filtering
    :return: a numpy array containing the filtered version of the input signal
    """
    if np.all(filter_state_in == None):
        filter_state_in = np.zeros(np.max(ar_coeff.shape) - 1)
        signal_out, filter_state_out = lfilter(ma_coeff, ar_coeff, signal, -1, filter_state_in)
    else:
        signal_out, filter_state_out = lfilter(ma_coeff, ar_coeff, signal, -1, filter_state_in)

    return signal_out, filter_state_out
