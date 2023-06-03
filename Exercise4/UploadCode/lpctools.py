#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for conversion of LPC coefficients to reflection coefficients and log-area ratios.

    The conversions are based on Section 6.3.1.3 in [1].


    [1] P. Vary and R. Martin, Digital speech transmission:
    Enhancement, Coding and Error Concealment. John Wiley & Sons, Ltd,
    2006.
"""

import numpy as np
from scipy import io as spio


def poly2rc(v_a):
    """Converts LPC coefficients to reflections coefficients.

    :v_a: LPC coefficients
    :returns: reflection coefficients

    """
    # get order
    order = len(v_a) - 1

    # normalize LPC coefficients
    v_alpha = v_a / v_a[0]

    # only LPC coefficients with i >= 1 required
    v_alpha = v_alpha[1:]

    # initialize reflection coefficients
    v_rc = np.zeros(order)

    # backward Levinson-Durbin iterations
    for p in range(order, 0, -1):
        # obtain new reflection coefficient (see p. 185, (a))
        v_rc[p - 1] = v_alpha[-1]

        # update alphas (see p. 185, (b))
        v_alpha = (v_alpha[:-1] - v_rc[p - 1]*v_alpha[-2::-1]) / \
                (1 - v_rc[p - 1]**2)

    return v_rc


def rc2lar(v_rc):
    """Converts reflection coefficients to log-area ratios.

    :v_rc: reflection coefficients
    :returns: log-area ratios

    """
    if np.any(np.abs(v_rc) > 1):
        raise ValueError('All reflection coefficients should have a magnitude less than unity.')

    return np.log((1 + v_rc)/(1 - v_rc))


def lar2rc(v_lar):
    """Converts log-area ratios to reflection coefficients.

    :v_lar: log-area ratios
    :returns: reflection coefficients

    """
    return (np.exp(v_lar) - 1) / (np.exp(v_lar) + 1)


def rc2poly(v_rc):
    """Converts reflection coefficients to LPC.

    :v_rc: reflections coefficients
    :returns: linear prediction coefficients

    """
    # get order
    order = len(v_rc)

    # pre allocate coefficient vector
    v_a = np.zeros(order + 1)

    # first coefficient is always 1
    v_a[0] = 1

    for p in range(1, order + 1):
        v_a[1:p] = v_a[1:p] + v_rc[p-1] * v_a[p-1:0:-1]
        v_a[p] = v_rc[p-1]

    return v_a

