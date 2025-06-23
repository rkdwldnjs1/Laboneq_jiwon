# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for definition of device setup and calibration settings"""

import numpy as np

from laboneq.dsl.quantum.qubit import Qubit, QubitParameters
from laboneq.dsl.quantum.transmon import Transmon, TransmonParameters, Transmon_cr, TransmonParameters_cr, Transmon_mult, TransmonParameters_mult


# function to generate a set of base transmon qubit parameters, used as input to the create_transmon function
def generate_dummy_transmon_parameters(
    number_of_qubits,
    drive_centre_frequency=4e9,
    readout_centre_frequency=6e9,
):
    return {
        "resonance_frequency_ge": [
            drive_centre_frequency + it * 100e6
            for _ in range(int(np.ceil(number_of_qubits / 6)))
            for it in np.linspace(-2.5, 2.5, 6)
        ],
        "resonance_frequency_ef": [
            drive_centre_frequency - 250e6 + it * 100e6
            for _ in range(int(np.ceil(number_of_qubits / 6)))
            for it in np.linspace(-2.5, 2.5, 6)
        ],
        "drive_lo_frequency": [
            drive_centre_frequency for _ in range(int(np.ceil(number_of_qubits / 2)))
        ],
        # readout resonatro frequency
        "readout_resonator_frequency": [
            readout_centre_frequency + it * 100e6
            for _ in range(int(np.ceil(number_of_qubits / 6)))
            for it in np.linspace(-2.5, 2.5, 6)
        ],
        # readout LO Frequency
        "readout_lo_frequency": [
            readout_centre_frequency for _ in range(int(np.ceil(number_of_qubits / 6)))
        ],
        # collection of pulse parameters
        "readout_length": 2e-6,
        "readout_amplitude": 0.6,
        "readout_integration_delay": 40e-9,
        "pulse_length_spectroscopy": 2e-6,
        "amplitude_pi": 0.66,
        "pulse_length": 100e-9,
        # range settings
        "readout_range_out": 5,
        "readout_range_in": 10,
        "drive_range": 5,
        #  delay inserted after every readout
        "reset_delay_length": 100e-9,
    }


# function to create a transmon qubit object from entries in a parameter dictionary
def create_dummy_transmon(index, base_parameters, device_setup):
    q_name = "q" + str(index)
    qubit = Transmon.from_logical_signal_group(
        q_name,
        lsg=device_setup.logical_signal_groups[q_name],
        parameters=TransmonParameters(
            resonance_frequency_ge=base_parameters["resonance_frequency_ge"][index],
            resonance_frequency_ef=base_parameters["resonance_frequency_ef"][index],
            drive_lo_frequency=base_parameters["drive_lo_frequency"][
                int(np.floor(index / 2))
            ],
            readout_resonator_frequency=base_parameters["readout_resonator_frequency"][
                index
            ],
            readout_lo_frequency=base_parameters["readout_lo_frequency"][
                int(np.floor(index / 6))
            ],
            readout_integration_delay=base_parameters["readout_integration_delay"],
            drive_range=base_parameters["drive_range"],
            readout_range_out=base_parameters["readout_range_out"],
            readout_range_in=base_parameters["readout_range_in"],
            user_defined={
                "amplitude_pi": base_parameters["amplitude_pi"],
                "pulse_length": base_parameters["pulse_length"],
                "readout_length": base_parameters["readout_length"],
                "readout_amplitude": base_parameters["readout_amplitude"],
                "reset_delay_length": base_parameters["reset_delay_length"],
            },
        ),
    )
    return qubit


def generate_dummy_transmon_cr_parameters(
    number_of_qubits,
    drive_centre_frequency=4e9,
    readout_centre_frequency=6e9,
):
    output = {
        "resonance_frequency_ge": [
            drive_centre_frequency + it * 100e6
            for _ in range(int(np.ceil(number_of_qubits / 6)))
            for it in np.linspace(-2.5, 2.5, 6)
        ],
        "resonance_frequency_ef": [
            drive_centre_frequency - 250e6 + it * 100e6
            for _ in range(int(np.ceil(number_of_qubits / 6)))
            for it in np.linspace(-2.5, 2.5, 6)
        ],
        "drive_lo_frequency": [
            drive_centre_frequency for _ in range(int(np.ceil(number_of_qubits / 2)))
        ],
        # readout resonatro frequency
        "readout_resonator_frequency": [
            readout_centre_frequency + it * 100e6
            for _ in range(int(np.ceil(number_of_qubits / 6)))
            for it in np.linspace(-2.5, 2.5, 6)
        ],
        # readout LO Frequency
        "readout_lo_frequency": [
            readout_centre_frequency for _ in range(int(np.ceil(number_of_qubits / 6)))
        ],
        # collection of pulse parameters
        "readout_length": 2e-6,
        "readout_amplitude": 0.6,
        "readout_integration_delay": 40e-9,
        "pulse_length_spectroscopy": 2e-6,
        "amplitude_pi": 0.66,
        "pulse_length": 100e-9,
        # range settings
        "readout_range_out": 5,
        "readout_range_in": 10,
        "drive_range": 5,
        #  delay inserted after every readout
        "reset_delay_length": 100e-9,
    }

    for i in range(number_of_qubits):
        output.update({
            "resonance_frequency_aux" + str(i): [
                drive_centre_frequency + it * 100e6
                for _ in range(int(np.ceil(number_of_qubits / 6)))
                for it in np.linspace(-2.5, 2.5, 6)
            ]
        })

    return output

def generate_dummy_transmon_mult_parameters(
    number_of_qubits,
    drive_centre_frequency=4e9,
    readout_centre_frequency=6e9,
):
    output = {
        "resonance_frequency_ge": [
            drive_centre_frequency + it * 100e6
            for _ in range(int(np.ceil(number_of_qubits / 6)))
            for it in np.linspace(-2.5, 2.5, 6)
        ],
        "drive_lo_frequency": [
            drive_centre_frequency for _ in range(int(np.ceil(number_of_qubits / 2)))
        ],
        # readout resonatro frequency
        "readout_resonator_frequency": [
            readout_centre_frequency + it * 100e6
            for _ in range(int(np.ceil(number_of_qubits / 6)))
            for it in np.linspace(-2.5, 2.5, 6)
        ],
        # readout LO Frequency
        "readout_lo_frequency": [
            readout_centre_frequency for _ in range(int(np.ceil(number_of_qubits / 6)))
        ],
        # collection of pulse parameters
        "readout_length": 2e-6,
        "readout_amplitude": 0.6,
        "readout_integration_delay": 40e-9,
        "pulse_length_spectroscopy": 2e-6,
        "amplitude_pi": 0.66,
        "pulse_length": 100e-9,
        # range settings
        "readout_range_out": 5,
        "readout_range_in": 10,
        "drive_range": 5,
        #  delay inserted after every readout
        "reset_delay_length": 100e-9,
    }


    return output


# function to create a transmon qubit object from entries in a parameter dictionary
def create_dummy_transmon_cr(index, base_parameters, device_setup):
    q_name = "q" + str(index)
    qubit = Transmon_cr.from_logical_signal_group(
        q_name,
        lsg=device_setup.logical_signal_groups[q_name],
        parameters=TransmonParameters_cr(
            resonance_frequency_ge=base_parameters["resonance_frequency_ge"][index],
            resonance_frequency_ef=base_parameters["resonance_frequency_ef"][index],
            resonance_frequency_aux0=base_parameters["resonance_frequency_aux0"][index],
            resonance_frequency_aux1=base_parameters["resonance_frequency_aux1"][index],
            resonance_frequency_aux2=base_parameters["resonance_frequency_aux2"][index],
            resonance_frequency_aux3=base_parameters["resonance_frequency_aux3"][index],
            resonance_frequency_aux4=base_parameters["resonance_frequency_aux4"][index],
            resonance_frequency_aux5=base_parameters["resonance_frequency_aux5"][index],
            drive_lo_frequency=base_parameters["drive_lo_frequency"][
                int(np.floor(index / 2))
            ],
            readout_resonator_frequency=base_parameters["readout_resonator_frequency"][
                index
            ],
            readout_lo_frequency=base_parameters["readout_lo_frequency"][
                int(np.floor(index / 6))
            ],
            readout_integration_delay=base_parameters["readout_integration_delay"],
            drive_range=base_parameters["drive_range"],
            readout_range_out=base_parameters["readout_range_out"],
            readout_range_in=base_parameters["readout_range_in"],
            user_defined={
                "amplitude_pi": base_parameters["amplitude_pi"],
                "pulse_length": base_parameters["pulse_length"],
                "readout_length": base_parameters["readout_length"],
                "readout_amplitude": base_parameters["readout_amplitude"],
                "reset_delay_length": base_parameters["reset_delay_length"],
            },
        ),
    )
    return qubit


# function to create a transmon qubit object from entries in a parameter dictionary
def create_dummy_transmon_mult(index, base_parameters, device_setup):
    q_name = "q" + str(index)
    qubit = Transmon_mult.from_logical_signal_group(
        q_name,
        lsg=device_setup.logical_signal_groups[q_name],
        parameters=TransmonParameters_mult(
            resonance_frequency_ge=base_parameters["resonance_frequency_ge"][index],
            drive_lo_frequency=base_parameters["drive_lo_frequency"][
                int(np.floor(index / 2))
            ],
            readout_resonator_frequency=base_parameters["readout_resonator_frequency"][
                index
            ],
            readout_lo_frequency=base_parameters["readout_lo_frequency"][
                int(np.floor(index / 6))
            ],
            readout_integration_delay=base_parameters["readout_integration_delay"],
            drive_range=base_parameters["drive_range"],
            readout_range_out=base_parameters["readout_range_out"],
            readout_range_in=base_parameters["readout_range_in"],
            user_defined={
                "amplitude_pi": base_parameters["amplitude_pi"],
                "pulse_length": base_parameters["pulse_length"],
                "readout_length": base_parameters["readout_length"],
                "readout_amplitude": base_parameters["readout_amplitude"],
                "reset_delay_length": base_parameters["reset_delay_length"],
            },
        ),
    )
    return qubit


# function to create a transmon qubit object from entries in a parameter dictionary
def create_dummy_qubit(index, base_parameters, device_setup):
    q_name = "q" + str(index)
    qubit = Qubit.from_logical_signal_group(
        q_name,
        lsg=device_setup.logical_signal_groups[q_name],
        parameters=QubitParameters(
            resonance_frequency=base_parameters["resonance_frequency_ge"][index],
            drive_lo_frequency=base_parameters["drive_lo_frequency"][
                int(np.floor(index / 2))
            ],
            readout_resonator_frequency=base_parameters["readout_resonator_frequency"][
                index
            ],
            readout_lo_frequency=base_parameters["readout_lo_frequency"][
                int(np.floor(index / 6))
            ],
            readout_integration_delay=base_parameters["readout_integration_delay"],
            drive_range=base_parameters["drive_range"],
            readout_range_out=base_parameters["readout_range_out"],
            readout_range_in=base_parameters["readout_range_in"],
            user_defined={
                "amplitude_pi": base_parameters["amplitude_pi"],
                "pulse_length": base_parameters["pulse_length"],
                "readout_length": base_parameters["readout_length"],
                "readout_amplitude": base_parameters["readout_amplitude"],
                "reset_delay_length": base_parameters["reset_delay_length"],
            },
        ),
    )
    return qubit
