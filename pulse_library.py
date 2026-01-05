# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Dict
import warnings

import numpy as np
from numpy.typing import ArrayLike
from laboneq.core.utilities.pulse_sampler import _pulse_samplers, _pulse_factories
from laboneq.dsl.experiment.pulse import (
    PulseFunctional,
    PulseSampled,
)

# deprecated alias for _pulse_samples, use pulse_library.pulse_sampler(...) instead:
pulse_function_library = _pulse_samplers


def register_pulse_functional(sampler: Callable, name: str | None = None):
    """Build & register a new pulse type from a sampler function.

    The sampler function must have the following signature:

    ``` py

        def sampler(x: ndarray, **pulse_params: Dict[str, Any]) -> ndarray:
            pass
    ```

    The vector ``x`` marks the points where the pulse function is to be evaluated. The
    values of ``x`` range from -1 to +1. The argument ``pulse_params`` contains all
    the sweep parameters, evaluated for the current iteration.
    In addition, ``pulse_params``  also contains the following keys:

    - ``length``: the true length of the pulse
    - ``amplitude``: the true amplitude of the pulse
    - ``sampling_rate``: the sampling rate

    Typically, the sampler function should discard ``length`` and ``amplitude``, and
    instead assume that the pulse extends from -1 to 1, and that it has unit
    amplitude. LabOne Q will automatically rescale the sampler's output to the correct
    amplitude and length.


    Args:
        sampler:
            the function used for sampling the pulse
        name:
            the name used internally for referring to this pulse type

    Returns:
        pulse_factory (function):
            A factory function for new ``Pulse`` objects.
            The return value has the following signature:
            ``` py

                def <name>(
                    uid: str = None,
                    length: float = 100e-9,
                    amplitude: float = 1.0,
                    **pulse_parameters: Dict[str, Any],
                ):
                    pass
            ```
    """
    if name is None:
        function_name = sampler.__name__
    else:
        function_name = name

    def factory(
        uid: str | None = None,
        length: float = 100e-9,
        amplitude: float = 1.0,
        can_compress=False,
        **pulse_parameters: Dict[str, Any],
    ):
        if pulse_parameters == {}:
            pulse_parameters = None
        if uid is None:
            return PulseFunctional(
                function=function_name,
                length=length,
                amplitude=amplitude,
                pulse_parameters=pulse_parameters,
                can_compress=can_compress,
            )
        else:
            return PulseFunctional(
                function=function_name,
                uid=uid,
                length=length,
                amplitude=amplitude,
                pulse_parameters=pulse_parameters,
                can_compress=can_compress,
            )

    factory.__name__ = function_name
    factory.__doc__ = sampler.__doc__
    # we do not wrap __qualname__, it throws off the documentation generator

    _pulse_samplers[function_name] = sampler
    _pulse_factories[function_name] = factory
    return factory


@register_pulse_functional
def gaussian(
    x,
    sigma=1 / 3,
    order=2,
    zero_boundaries=False,
    **_,
):
    """Create a Gaussian pulse.

    Returns a generalised Gaussian pulse with order parameter $n$, defined by:

    $$ g(x, \\sigma, n) = e^{-\\left(\\frac{x^2}{2\\sigma^2}\\right)^{\\frac{n}{2}}} $$.

    When the order $n = 2$, the formula simplifies to the standard Gaussian:

    $$ g(x, \\sigma_0) = e^{-\\frac{x^2}{2\\sigma_0^2}} $$

    For higher orders ($n > 2$), the value of $\\sigma$ is adjusted so that the
    pulse has the same near-zero values at the edges as the ordinary Gaussian.

    In general, for $x \\in [-L, L]$, the adjusted $\\sigma$ can be written as:

    $$\\sigma = \\frac{\\sigma_0^{\\frac{2}{n}}}{2^{\\left(\\frac{n-2}{2 n}\\right)} L^{\\left(\\frac{2-n}{n}\\right)}}$$

    Considering here $x \\in [-1, 1]$, the adjusted $\\sigma$ simplifies to:

    $$\\sigma = \\frac{\\sigma_0^{\\frac{2}{n}}}{2^{\\left(\\frac{n-2}{2 n}\\right)}}$$

    Arguments:
        sigma (float):
            Standard deviation relative to the interval the pulse is sampled from, here [-1, 1]. Defaults
                to 1/3.
        order (int):
            Order of the Gaussian pulse, must be positive and even, default is 2 (standard Gaussian), order > 2 will create a super Gaussian pulse
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries, default is False

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse


    Returns:
        pulse (Pulse): Gaussian pulse.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("The order must be a positive and even integer.")
    elif order == 2:
        gauss = np.exp(-(x**2 / (2 * sigma**2)))
    elif order > 2:
        sigma_updated = (sigma ** (2 / order)) / (2 ** ((order - 2) / (2 * order)))
        gauss = np.exp(-((x**2 / (2 * sigma_updated**2)) ** (order / 2)))

    if zero_boundaries:
        dt = x[0] - (x[1] - x[0])
        dt = np.abs(dt)
        if order == 2:
            delta = np.exp(-(dt**2 / (2 * sigma**2)))
        else:
            sigma_updated = (sigma ** (2 / order)) / (
                2 ** ((order - 2) / (2 * order)) * dt ** ((2 - order) / order)
            )
            delta = np.exp(-((dt**2 / (2 * sigma_updated**2)) ** (order / 2)))
        gauss -= delta
        gauss /= 1 - delta
    return gauss


@register_pulse_functional
def gaussian_square(x, sigma=1 / 3, width=None, zero_boundaries=False, *, length, **_):
    """Create a gaussian square waveform with a square portion of length
    ``width`` and Gaussian shaped sides.

    Arguments:
        length (float):
            Length of the pulse in seconds
        width (float):
            Width of the flat portion of the pulse in seconds. Dynamically set to 90% of `length` if not provided.
        sigma (float):
            Std. deviation of the Gaussian rise/fall portion of the pulse
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Gaussian square pulse.
    """

    if width is not None and width >= length:
        raise ValueError(
            "The width of the flat portion of the pulse must be smaller than the total length."
        )

    if width is None:
        width = 0.9 * length

    risefall_in_samples = round(len(x) * (1 - width / length) / 2)
    flat_in_samples = len(x) - 2 * risefall_in_samples
    gauss_x = np.linspace(-1.0, 1.0, 2 * risefall_in_samples)
    gauss_part = np.exp(-(gauss_x**2) / (2 * sigma**2))
    gauss_sq = np.concatenate(
        (
            gauss_part[:risefall_in_samples],
            np.ones(flat_in_samples),
            gauss_part[risefall_in_samples:],
        )
    )
    if zero_boundaries:
        t_left = gauss_x[0] - (gauss_x[1] - gauss_x[0])
        delta = np.exp(-(t_left**2) / (2 * sigma**2))
        gauss_sq -= delta
        gauss_sq /= 1 - delta
    return gauss_sq

@register_pulse_functional
def gaussian_rise(x, sigma=1 / 3, width=None, zero_boundaries=False, **_):

    gauss_rise = np.exp(-((x-1)**2) / (2 * sigma**2))

    if zero_boundaries:
        dt = x[0] - (x[1] - x[0])
        dt = np.abs(dt)
        delta = np.exp(-(dt**2) / (2 * sigma**2))
        gauss_rise -= delta
        gauss_rise /= 1 - delta

    return gauss_rise

@register_pulse_functional
def gaussian_fall(x, sigma=1 / 3, width=None, zero_boundaries=False, **_):

    gauss_fall = np.exp(-((x+1)**2) / (2 * sigma**2))

    if zero_boundaries:
        dt = x[0] - (x[1] - x[0])
        dt = np.abs(dt)
        delta = np.exp(-(dt**2) / (2 * sigma**2))
        gauss_fall -= delta
        gauss_fall /= 1 - delta

    return gauss_fall

'''
t = x*length/2

a = np.pi * t/length

return np.cos(a)**3
'''

@register_pulse_functional
def integration_weight_gaussian_const(
    x,
    length,
    chi,
    kappa,
    sigma=1/3, # total length 2 -> sigma : 1/3
    width=None,
    **_
): 
    # stabilization of Bosonic codes in superconducting circuits by Steven Touzard (Readout cavity trajectory reference)
    # gaussian rise + constant (without gaussian fall)
    def gaussian_constant(t, T_rise, T_flat, A, sigma):
        """
        Gaussian-rise / constant pulse (no fall)
        t       : 1D numpy array [s]
        T_rise  : Gaussian rise time [s]
        T_flat  : constant duration [s]
        A       : amplitude
        sigma   : Gaussian width for rise [s]
        Total effective pulse length = T_rise + T_flat
        """
        pulse = np.zeros_like(t)
        # Gaussian rise
        idx_rise = (t >= 0) & (t < T_rise)
        pulse[idx_rise] = A * np.exp(
            - (t[idx_rise] - T_rise)**2 / (2 * sigma**2)
        )
        # Constant region
        idx_flat = (t >= T_rise) & (t <= T_rise + T_flat)
        pulse[idx_flat] = A
        return pulse
    
    def readout_cavity_trajectory(t, a_in, chi, kappa, state):

        if state == 'g':
            chi = chi
        else:
            chi = -chi
            
        coeff = 2*np.sqrt(kappa)*a_in/(1j*chi - kappa)

        return coeff * (1 - np.exp((-kappa/2 + 1j*chi/2)*t))
    
    if width is None:
        width = 0.95 * length
    
    t = (x+1)*length/2 # x is [-1,1], t is [0,length]
    T_rise = length - width
    sigma = T_rise / 2 * sigma

    a_in = gaussian_constant(t, T_rise = length-width , T_flat = width, A=1, sigma=sigma)

    a_g = readout_cavity_trajectory(t, a_in, chi, kappa, state='g')
    a_e = readout_cavity_trajectory(t, a_in, chi, kappa, state='e')

    function = np.imag(a_e - a_g)
    function_norm = function / np.max(np.abs(function))

    return function_norm


@register_pulse_functional
def const(x, **_):
    """Create a constant pulse.

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Constant pulse.
    """
    return np.ones_like(x)


@register_pulse_functional
def triangle(x, **_):
    """Create a triangle pulse.

    A triangle pulse varies linearly from a starting amplitude of
    zero, to a maximum amplitude of one in the middle of the pulse,
    and then back to a final amplitude of zero.

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Triangle pulse.
    """
    return 1 - np.abs(x)


@register_pulse_functional
def sawtooth(x, **_):
    """Create a sawtooth pulse.

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Sawtooth pulse.
    """

    return 0.5 * (1 - x)


@register_pulse_functional
def drag(x, sigma=1 / 3, beta=0.0, zero_boundaries=False, **_):
    """Create a DRAG pulse.

    Arguments:
        sigma (float):
            Standard deviation relative to the interval the pulse is sampled from, here [-1, 1]. Defaults
        beta (float):
            Relative amplitude of the quadrature component
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): DRAG pulse.
    """
    gauss = np.exp(-(x**2) / (2 * sigma**2))
    delta = 0
    if zero_boundaries:
        dt = x[0] - (x[1] - x[0])
        delta = np.exp(-(dt**2) / (2 * sigma**2))
    d_gauss = -x / sigma**2 * gauss
    gauss -= delta
    return (gauss + 1j * beta * d_gauss) / (1 - delta)


@register_pulse_functional
def cos2(x, **_):
    """Create a raised cosine pulse.

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Raised cosine pulse.
    """
    return np.cos(x * np.pi / 2) ** 2

@register_pulse_functional
def cond_disp_pulse(
    x,
    length,
    frequency, # input unit is Hz
    detuning,
    sigma=1 / 3,
    zero_boundaries=False,
    **_,
):
    """Create a conditional displacement pulse.

    The conditional displacement pulse is defined as:
       S_RF(t) = Re{[I(t)+iQ(t)]e^(-1j*2pi*f_IF*t)e^(-1j*2pi*f_RF*t)e^(1j*phi)}

    where:
       I(t)+iQ(t) = exp(-(t^2 / (2 * sigma^2))) * 
       (exp(-1j*(frequency + detuning)*2*pi*t) - exp(-1j*(frequency - detuning)*2*pi*t))

       f_IF + f_RF = f_d(driving freq), f_d+frequency = f_e or f_g, 
       and nonzero detuning is asymmetric pulse.

       x is an array of points in the range [-1, 1] where the pulse is evaluated.
       len(x) is the number of samples in the pulse. 
       sampling_rate is 2GHz/sample. So, 200ns pulse will have 400 samples.

       Unit of frequency should be matched due to "x"
    """

    _frequency = frequency * length/2
    _detuning = detuning * length/2

    pulse = np.exp(-(x**2 / (2 * sigma**2)))*(np.exp(-1j*(_frequency + _detuning)*2*np.pi*x) - np.exp(-1j*(_frequency - _detuning)*2*np.pi*x))


    if zero_boundaries:
        dt = x[0] - (x[1] - x[0])
        dt = np.abs(dt)
        
        delta = np.exp(-(dt**2 / (2 * sigma**2)))
        
        pulse -= delta
        pulse /= 1 - delta
    return pulse


@register_pulse_functional
def sidebands_pulse(
    x,
    length,
    frequency_l, # input unit is Hz
    frequency_h,
    amp_l,
    amp_h,
    phase, # in radian
    extra_phase=0, # in radian
    is_gauss_rise = False,
    is_gauss_fall = False,
    sigma=1 / 3,
    zero_boundaries=False,
    **_,
):
    """Create a sideband pulses.

    The sideband pulse is defined as:
       S_RF(t) = Re{[I(t)+iQ(t)]e^(-1j*2pi*f_IF*t)e^(-1j*2pi*f_RF*t)e^(1j*phi)}
               = I(t)*cos(2*pi*(f_IF+f_RF)*t) + Q(t)*sin(2*pi*(f_IF+f_RF)*t)
    where:
       I(t)+iQ(t) = amp_l*exp(-1j*2pi*(_frequency_l*t + phase)) + amp_h*exp(1j*2pi*(_frequency_h*t + phase))

       x is an array of points in the range [-1, 1] where the pulse is evaluated.
       len(x) is the number of samples in the pulse. 
       sampling_rate is 2GHz/sample. So, 200ns pulse will have 400 samples.

       Unit of frequency should be matched due to "x"
    """

    _frequency_l = frequency_l * length/2
    _frequency_h = frequency_h * length/2

    pulse = amp_h*np.exp(-1j*(_frequency_h*x*2*np.pi + phase)) + amp_l*np.exp(1j*(_frequency_l*x*2*np.pi + phase + extra_phase))

    if is_gauss_rise:
        gauss_rise = np.exp(-((x-1)**2) / (2 * sigma**2))

        if zero_boundaries:
            dt = x[0] - (x[1] - x[0])
            dt = np.abs(dt)
            delta = np.exp(-(dt**2) / (2 * sigma**2))
            gauss_rise -= delta
            gauss_rise /= 1 - delta
        
        pulse *= gauss_rise

    if is_gauss_fall:
        gauss_fall = np.exp(-((x+1)**2) / (2 * sigma**2))

        if zero_boundaries:
            dt = x[0] - (x[1] - x[0])
            dt = np.abs(dt)
            delta = np.exp(-(dt**2) / (2 * sigma**2))
            gauss_fall -= delta
            gauss_fall /= 1 - delta

        pulse *= gauss_fall

    return pulse


@register_pulse_functional
def gauss_sidebands_pulse(
    x,
    length,
    frequency_l, # input unit is Hz
    frequency_h,
    amp_l,
    amp_h,
    phase, # in radian
    width=None,
    is_gauss_rise = False,
    is_gauss_fall = False,
    sigma=1 / 3,
    zero_boundaries=False,
    **_,
):
    """Create a sideband pulses.

    The sideband pulse is defined as:
       S_RF(t) = Re{[I(t)+iQ(t)]e^(-1j*2pi*f_IF*t)e^(-1j*2pi*f_RF*t)e^(1j*phi)}
               = I(t)*cos(2*pi*(f_IF+f_RF)*t) + Q(t)*sin(2*pi*(f_IF+f_RF)*t)
    where:
       I(t)+iQ(t) = amp_l*exp(-1j*2pi*(_frequency_l*t + phase)) + amp_h*exp(1j*2pi*(_frequency_h*t + phase))

       x is an array of points in the range [-1, 1] where the pulse is evaluated.
       len(x) is the number of samples in the pulse. 
       sampling_rate is 2GHz/sample. So, 200ns pulse will have 400 samples.

       Unit of frequency should be matched due to "x"
    """

    _frequency_l = frequency_l * length/2
    _frequency_h = frequency_h * length/2

    pulse = amp_h*np.exp(-1j*(_frequency_h*x*2*np.pi + phase)) + amp_l*np.exp(1j*(_frequency_l*x*2*np.pi + phase))


    if width is not None and width >= length:
        raise ValueError(
            "The width of the flat portion of the pulse must be smaller than the total length."
        )

    if width is None:
        width = 0.9 * length

    risefall_in_samples = round(len(x) * (1 - width / length) / 2)
    flat_in_samples = len(x) - 2 * risefall_in_samples
    gauss_x = np.linspace(-1.0, 1.0, 2 * risefall_in_samples)
    gauss_part = np.exp(-(gauss_x**2) / (2 * sigma**2))
    gauss_sq = np.concatenate(
        (
            gauss_part[:risefall_in_samples],
            np.ones(flat_in_samples),
            gauss_part[risefall_in_samples:],
        )
    )
    if zero_boundaries:
        t_left = gauss_x[0] - (gauss_x[1] - gauss_x[0])
        delta = np.exp(-(t_left**2) / (2 * sigma**2))
        gauss_sq -= delta
        gauss_sq /= 1 - delta

    return gauss_sq * pulse



@register_pulse_functional
def drachma_readout_pulse(
    x,
    length,
    kappa,
    chi_list,
    zeta_list,
    amp_trial=1,
    **_,
):
    """Create a Drachma readout pulse.

    Arguments:
        length (float):
            Length of the pulse in seconds
        kappa (float):
            Cavity decay rate
    """

    def trial_field(t, length, amp_trial=1):

        a = np.pi * t/length

        return amp_trial * np.cos(a)**3 # t is [-length/2, length/2]
    
    def input_pulse_wo_kerr(t, trial_field, kappa, chi_j):

        dadt = np.gradient(trial_field(t, length), t[1]-t[0])

        a_tilde_j = ((kappa/2)+1j*chi_j)*trial_field(t, length) + dadt
        a_tilde_j /= np.sqrt(kappa)

        return a_tilde_j
    
    t = x*length/2  # x is [-1,1], t is [-length/2, length/2]

    a_tilde = []
    for chi_j in chi_list:
        a_tilde_j = input_pulse_wo_kerr(t, trial_field, kappa, chi_j)/np.sqrt(kappa)
        a_tilde.append(a_tilde_j)
    
    f = trial_field(t, length, amp_trial)
    
    for j in range(len(chi_list)):
        chi_j = chi_list[j]
        zeta_j = zeta_list[j]
        a_tilde_j = a_tilde[j]

        dt = t[1] - t[0]
        dfdt = np.gradient(f, dt)

        # time-dependent nonlinear frequency shift due to Kerr effect
        delta_nl = chi_j + 4*zeta_j*np.abs(a_tilde_j)**2

        f = (kappa/2 + 1j*delta_nl)*f + dfdt
    
    a_in = f / kappa ** (len(chi_list))

    return a_in

@register_pulse_functional
def drachma_readout_weighting_pulse(
    x,
    length,
    **_,
):  
    
    t = x*length/2

    a = np.pi * t/length

    return np.cos(a)**3



def sampled_pulse(
    samples: ArrayLike, uid: str | None = None, can_compress: bool = False
):
    """Create a pulse based on a array of waveform values.

    Arguments:
        samples (numpy.ndarray): waveform envelope data.
        uid (str): Unique identifier of the created pulse.

    Returns:
        pulse (Pulse): Pulse based on the provided sample values.
    """
    if uid is None:
        return PulseSampled(samples=samples, can_compress=can_compress)
    else:
        return PulseSampled(uid=uid, samples=samples, can_compress=can_compress)


def sampled_pulse_real(
    samples: ArrayLike, uid: str | None = None, can_compress: bool = False
):
    """Create a pulse based on a array of real values.

    Arguments:
        samples (numpy.ndarray): Real valued data.
        uid (str): Unique identifier of the created pulse.

    Returns:
        pulse (Pulse): Pulse based on the provided sample values.

    !!! version-changed "Deprecated in version 2.51.0"
        Use `sampled_pulse` instead.

    """
    warnings.warn(
        "The `sampled_pulse_real` function, along with `PulseSampledReal`, is deprecated. "
        "Please use `sampled_pulse` instead, as `sampled_pulse_real` now calls `sampled_pulse` internally.",
        FutureWarning,
        stacklevel=2,
    )

    return sampled_pulse(samples, uid=uid, can_compress=can_compress)


def sampled_pulse_complex(
    samples: ArrayLike, uid: str | None = None, can_compress: bool = False
):
    """Create a pulse based on a array of complex values.

    Args:
        samples (numpy.ndarray): Complex valued data.
        uid (str): Unique identifier of the created pulse.

    Returns:
        pulse (Pulse): Pulse based on the provided sample values.

    !!! version-changed "Deprecated in version 2.51.0"
        Use `sampled_pulse` instead.
    """
    warnings.warn(
        "The `sampled_pulse_complex` function, along with `PulseSampledComplex`, is deprecated. "
        "Please use `sampled_pulse` instead, as `sampled_pulse_complex` now calls `sampled_pulse` internally.",
        FutureWarning,
        stacklevel=2,
    )

    return sampled_pulse(samples, uid=uid, can_compress=can_compress)


def pulse_sampler(name: str) -> Callable:
    """Return the named pulse sampler.

    The sampler is the original function used to define the pulse.

    For example in:

        ```python
        @register_pulse_functional
        def const(x, **_):
            return numpy.ones_like(x)
        ```

    the sampler is the *undecorated* function `const`. Calling
    `pulse_sampler("const")` will return this undecorated function.

    This undecorate function is called a "sampler" because it is used by
    the LabOne Q compiler to generate the samples played by a pulse.

    Arguments:
        name: The name of the sampler to return.

    Return:
        The sampler function.
    """
    return _pulse_samplers[name]


def pulse_factory(name: str) -> Callable:
    """Return the named pules factory.

    The pulse factory returns the description of the pulse used to specify
    a pulse when calling LabOne Q DSl commands such as `.play(...)` and
    `.measure(...)`.

    For example, in:

        ```python
        @register_pulse_functional
        def const(x, **_):
            return numpy.ones_like(x)
        ```

    the factory is the *decorated* function `const`. Calling
    `pulse_factory("const")` will return this decorated function. This is
    the same function one calls when calling `pulse_library.const(...)`.

    Arguments:
        name: The name of the factory to return.

    Return:
        The factory function.
    """
    return _pulse_factories[name]
