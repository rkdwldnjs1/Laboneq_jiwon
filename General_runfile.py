# In[]
import matplotlib
# matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt

import numpy as np
import qutip as qt
import sys
# import os
# sys.path.append("D:/Software/SHFQC/")
from pathlib import Path

# Helpers:
from laboneq.contrib.example_helpers.generate_device_setup import (
    generate_device_setup_qubits,
)

# LabOne Q:
from laboneq.simple import *

# from Basic_qubit_characterization import Basic_qubit_characterization_experiments

from Bosonic_experiments import Bosonic_experiments

from laboneq.contrib.bloch_simulator_pulse_plotter.inspector.update_inspect import (
    pulse_update,
)

from laboneq.dsl.experiment import pulse_library
from laboneq.contrib.bloch_simulator_pulse_plotter.inspector.update_inspect import (
    pulse_inspector,
)


# In[] initial parameters for experiment (6 components available)

# 같은 physical port를 공유하면, 초기 logical signal line calibration할 때
# LO oscillator와 physical port와 관련한 변수들이 나중에 선언한 것으로 덮어씌워짐.
# (IF oscillator만 별도로 선언 가능)
# drive line과 drive_ef line이 서로 공유하므로, 나중에 선언한 drive_ef line의 변수들로 덮어씌워짐.
# "q0"의 measure, acquire line과 "q1"의 measure, acquire line이 서로 공유하므로, 
# 나중에 선언한 "q1"의 measure, acquire line의 변수들로 덮어씌워짐.
# In[] physical_ports
physical_ports = {
    "measure":{
        "freq_LO": 7.4e9,   # f = f_LO + f_IF (positive sideband)
        "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
        "port_mode": None,
        "delay_signal": None, # Global delay for the signal line, implemented by adjusting the waveform.
        "threshold": None,
        "added_outputs": None, # Added outputs to the signal line's physical channel port. (RTR option enabled)
        "automute": True, # Mute output channel when no waveform is played on it i.e for the duration of delays.
        "range": 10,
        "amplitude": 1.0, # it does not work!
                          # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
    },
    "acquire":{
        "freq_LO": 7.4e9,   # f = f_LO + f_IF (positive sideband) (should be same with measure line)
        "port_delay": (280)*1e-9, # Because readout pulse shape is gussian square, for eliminating the rising edge.
        "port_mode": None,
        "delay_signal": None, # Global delay for the signal line, implemented by adjusting the waveform.
        "threshold": None,
        "added_outputs": None, # Added outputs to the signal line's physical channel port. (RTR option enabled)
        "automute": False, # Mute output channel when no waveform is played on it i.e for the duration of delays.
        "range": -5,
        "amplitude": 1.0, # it does not work!
                          # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
    },
    "drive":{
        "q0":{
            "freq_LO": 4e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
        "q1":{ # 2nd control line port
            "freq_LO": 6e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
        "q2":{ # 3rd control line port
            "freq_LO": 4.2e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
        "m0":{ # mapped to port4 
            "freq_LO": 5e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
        "m1":{ # mapped to port5
            "freq_LO": 6.6e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
        "m2":{ # mapped to port6 
            "freq_LO": 5e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
    }
}
# In[] # qubits_parameters, cavity_parameters
### These parameters will be utilized in defining experimental sequences.
qubits_parameters = {
    # two(cr) qubit gate : each parameter is assigned to each qubit except ef_frequency;

    "q0": {

        #port 1 (15-1)

        "ge_frequency" : 4.3077e9 + 0.23e6,
        "ef_frequency" : 4.2975e9, #4.275e9, #4.3865e9,
        "readout_frequency" : 7.66475e9 - 0.6e6,   # 7.6642e9 - 0.1e6, 
        ### readout parameters ###
        "readout_amp" : 0.18, #0.25, #0.18,
        'readout_pulse_length': 1400e-9,
        "readout_integration_length": (1120)*1e-9,
        "readout_integration_amp": 1,

        "drachma_readout_pulse_length": 1400e-9,
        "drachma_readout_integration_length": (1120)*1e-9,
        "drachma_readout_amp": 0.23,

        "readout_phase": (25) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.52, 
        "drive_pulse_length": 64e-9,
        "drive_beta": 0.035,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.26,
        "pi2_beta": 0.035,

        "pi_length": 64e-9,
        "pi_amp": 0.52,
        "pi_beta": 0.035,

        "cond_pi_length": 2048e-9,
        "cond_pi_amp": 0.01406,
        "cond_pi_beta": 0.035,

        "rabi_drive_amp": 0.1852,
        "rabi_ramp_length": 120e-9,

        ### extra parameters ###
        "reset_delay_length": 300e-6,
        "readout_kappa": 2*np.pi*0.68e6,
        "readout_chi": 2*np.pi*0.93e6,

        "is_cr" : False,
        "cr" : {
            "control_qubit_frequency": 4.9341e9+0.039e6,
            "target_qubit_frequency_1": 4.7e9+0.12e6,
            "target_qubit_frequency_2": 4.7e9,  # temporally arbitrary value
        }

    },

    "q1": {

        #port 1 (15-1)

        "ge_frequency" : 6.0125e9,
        "ef_frequency" : 6.0125e9, #4.275e9, #4.3865e9,
        "readout_frequency" : 7.66457e9 - 0.0e6,
        ### readout parameters ###
        "readout_amp" : 0.1, #0.25, #0.18,
        'readout_pulse_length': 1400e-9,
        "readout_integration_length": (1240)*1e-9,
        "readout_integration_amp": 1,

        "drachma_readout_pulse_length": 1600e-9,
        "drachma_readout_integration_length": (1400)*1e-9,
        "drachma_readout_amp": 0.5,

        "readout_phase": (-35) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 1.7, 
        "drive_pulse_length": 64e-9,
        "drive_beta": 0.015,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.14,
        "pi2_beta": 0.015,

        "pi_length": 64e-9,
        "pi_amp": 0.28,
        "pi_beta": 0.015,

        "cond_pi_length": 1024e-9,
        "cond_pi_amp": 0.02,
        "cond_pi_beta": 0.0,

        "rabi_drive_amp": 0.1852,
        "ramp_length": 120e-9,

        ### extra parameters ###
        "reset_delay_length": 300e-6,
        "readout_kappa": 2*np.pi*1e6,
        "readout_chi": 2*np.pi*0.93e6,

        "is_cr" : False,
        "cr" : {
            "control_qubit_frequency": 4.9341e9+0.039e6,
            "target_qubit_frequency_1": 4.7e9+0.12e6,
            "target_qubit_frequency_2": 4.7e9,  # temporally arbitrary value
        }

    },

    "q2": {

        #port 1 (15-1)

        "ge_frequency" : 4.3077e9 + 0.23e6,
        "ef_frequency" : 4.2975e9, #4.275e9, #4.3865e9,
        "readout_frequency" : 7.66475e9 - 0.6e6,   # 7.6642e9 - 0.1e6, 
        ### readout parameters ###
        "readout_amp" : 0.18, #0.25, #0.18,
        'readout_pulse_length': 1400e-9,
        "readout_integration_length": (1120)*1e-9,
        "readout_integration_amp": 1,

        "drachma_readout_pulse_length": 1400e-9,
        "drachma_readout_integration_length": (1120)*1e-9,
        "drachma_readout_amp": 0.23,

        "readout_phase": (-50) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.49, 
        "drive_pulse_length": 64e-9,
        "drive_beta": 0.035,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.245,
        "pi2_beta": 0.035,

        "pi_length": 64e-9,
        "pi_amp": 0.49,
        "pi_beta": 0.035,

        "cond_pi_length": 2048e-9,
        "cond_pi_amp": 0.01406,
        "cond_pi_beta": 0.035,

        "rabi_drive_amp": 0.1852,
        "ramp_length": 120e-9,

        ### extra parameters ###
        "reset_delay_length": 1e-6,
        "readout_kappa": 2*np.pi*0.68e6,
        "readout_chi": 2*np.pi*0.93e6,

        "is_cr" : False,
        "cr" : {
            "control_qubit_frequency": 4.9341e9+0.039e6,
            "target_qubit_frequency_1": 4.7e9+0.12e6,
            "target_qubit_frequency_2": 4.7e9,  # temporally arbitrary value
        }

    },

}

cavity_parameters = {

    'm0': { # mm2 - qubit 4.394GHz
        "mode_frequency": 5.152159e9 + 0.092e6 - 0.227e6/2, # center of g and e freq
        "cavity_drive_length": 40e-9, # this should be as short as possible to cover wide frequency range
        "cavity_drive_amp": 0.071*4, #(this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_cavity_drive_amp": 0.071, # 1 when it is not found yet. (This is amp when photon = 1) (this should be same with scaling_factor in disp_amp_calibration_parity

        "cond_disp_pulse_length": 200e-9, #200e-9,
        "cond_disp_pulse_amp": 0.0957, #0.0446j, # (this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_CNOD_amp": 0.0957, # 1 when it is not found yet. (this should be same with scaling_factor in CNOD calibration)

        # (omega_d : w_LO + w_IF , omega_e or g : w_LO + w_IF + cond_disp_pulse_frequency)
        "cavity_mode_chi": -0.227e6, 
        "chi_for_cnod": -0.227e6/2, #-0.227e6/2, # it depends on the driving freq frame.
        "cond_disp_pulse_frequency": -0.227e6/2, #-0.227e6/2,
        # driving only cavity mode of ground state in driving freq(mode_frequency) frame.
        "cond_disp_pulse_detuning": 0.5e6,
        "cond_disp_pulse_sigma": 2,
        "geophase_correction_coeff": 0.033/2, # after calibrating geometric phase.

        "sideband_length": 1e-6,
        "sideband_chunk_length": 1e-6,
        "sideband_rise_fall_length": 120e-9,

        # "sideband_frequency_l" : 15e6, #30e6, # input unit is Hz
        # "sideband_frequency_h" : 15e6, #30e6,
        # "sideband_amp_l" : 0.15,
        # "sideband_amp_h" : 0.15,
        # "sideband_phase" : 0, # in radians
        # "sideband_att_l" : 416.4209, # The larger value, the smaller attenuation
        # "sideband_att_h" : 354.3604, #239, # The larger value, the smaller attenuation
        # "sideband_extra_phase": np.pi, # in radians

        "sideband_frequency_l" : 10e6, # input unit is Hz
        "sideband_frequency_h" : 10e6,
        "sideband_amp_l" : 0.1,
        "sideband_amp_h" : 0.1,
        "sideband_phase" : 0, # in radians
        "sideband_att_l" : 384.85,  # The larger value, the smaller attenuation
        "sideband_att_h" : 346.3965,  # The larger value, the smaller attenuation
        "sideband_extra_phase": np.pi, # in radians

        "steady_time": 5e-6,

        "reset_delay_length": 300e-6,

    },


    'm1': { # mm2 - qubit 4.394GHz
        "mode_frequency": 6.795e9 - 1.2e6, #5.152209e9 - 0.1e6, # center of g and e freq
        "cavity_drive_length": 40e-9, # this should be as short as possible to cover wide frequency range
        "cavity_drive_amp": 0.0482*5, #(this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_cavity_drive_amp": 0.0482, # 1 when it is not found yet. (This is amp when photon = 1) (this should be same with scaling_factor in disp_amp_calibration_parity

        "cond_disp_pulse_length": 280e-9, #200e-9,
        "cond_disp_pulse_amp": 0.0943, #0.0446j, # (this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_CNOD_amp": 0.0943, # 1 when it is not found yet. (this should be same with scaling_factor in CNOD calibration)
        # (omega_d : w_LO + w_IF , omega_e or g : w_LO + w_IF + cond_disp_pulse_frequency)
        "cavity_mode_chi": -0.06e6, 
        "chi_for_cnod": -0.06e6, #-0.2e6/2, # it depends on the driving freq frame.
        "cond_disp_pulse_frequency": -0.06e6, 
        # driving only cavity mode of ground state in driving freq(mode_frequency) frame.
        "cond_disp_pulse_detuning": 0.5e6,
        "cond_disp_pulse_sigma": 2,
        "geophase_correction_coeff": 0.01/2, # after calibrating geometric phase.

        "sideband_length": 1e-6,
        "sideband_chunk_length": 1e-6,
        "sideband_rise_fall_length": 120e-9,

        "sideband_frequency_l" : 30e6, #30e6, # input unit is Hz
        "sideband_frequency_h" : 30e6, #30e6,
        "sideband_amp_l" : 0.1,
        "sideband_amp_h" : 0.1,
        "sideband_phase" : 0, # in radians
        "sideband_att_l" : 531, #448, # The larger value, the smaller attenuation
        "sideband_att_h" : 211, #239, # The larger value, the smaller attenuation
        "sideband_extra_phase": 0, # in radians

        "reset_delay_length": 300e-6,
    },

    # 'm2': { 
    #     "mode_frequency": 5.151637e9 + 0.45e6, # center of g and e freq
    #     "cavity_drive_length": 40e-9, # this should be as short as possible to cover wide frequency range
    #     "cavity_drive_amp": 0.0362*3, #(this is the maximum amplitude range, and this goes to amp of experiment)
    #     "alpha_1_cavity_drive_amp": 0.0362, # 1 when it is not found yet. (This is amp when photon = 1) (this should be same with scaling_factor in disp_amp_calibration_parity

    #     "cond_disp_pulse_length": 160e-9, #200e-9,
    #     "cond_disp_pulse_amp": 0.083, #0.0446j, # (this is the maximum amplitude range, and this goes to amp of experiment)
    #     "alpha_1_CNOD_amp": 0.083, # 1 when it is not found yet. (this should be same with scaling_factor in CNOD calibration)

    #     # (omega_d : w_LO + w_IF , omega_e or g : w_LO + w_IF + cond_disp_pulse_frequency)
    #     "cavity_mode_chi": -0.22e6, #-0.824e6,
    #     "cond_disp_pulse_frequency": -0.11e6, # driving only cavity mode of ground state in driving freq(mode_frequency) frame.
    #     "cond_disp_pulse_detuning": 0.5e6,
    #     "cond_disp_pulse_sigma": 2,

    #     "sideband_length": 1e-6,
    #     "sideband_chunk_length": 1e-6,
    #     "sideband_rise_fall_length": 120e-9,

    #     "sideband_frequency_l" : 20e6, # input unit is Hz
    #     "sideband_frequency_h" : 20e6,
    #     "sideband_amp_l" : 0.1,
    #     "sideband_amp_h" : 0.1,
    #     "sideband_phase" : 0, # in radians
    #     "sideband_att_l" : 1105, # The larger value, the smaller attenuation
    #     "sideband_att_h" : 576, # The larger value, the smaller attenuation

    #     "reset_delay_length": 500e-6,
    # },

}

# In[]

my_run = Bosonic_experiments(physical_ports, qubits_parameters, cavity_parameters, 
                 number_of_qubits=3,
                 number_of_memory_modes=2,
                 is_memory_mode = True, 
                 which_qubit=0,
                 which_mode =0,
                 which_data= "I", 
                 cr_drive_lines=False, 
                 multiplex_drive_lines = False,
                 is_cavity_trajectory_weighting_function = True,
                 use_emulation = False)


# %% propagation delay calibration for readout and drive line

my_run.prop_delay_calibration(line = "readout", average_exponent=12, readout_pulse_type = "const")

# %% 
my_run.nopi_pi(average_exponent=12, phase = 25, readout_pulse_type="gaussian_square",
                            readout_weighting_type ="cavity_trajectory", is_plot_simulation=False)
## readout weighting is not important for above experiment ; result is obtained through RAW data
my_run.single_shot_nopi_pi(npts_exponent = 12, phase = 25, readout_pulse_type="gaussian_square",
                            readout_weighting_type ="cavity_trajectory", is_plot_simulation=False)
my_run.plot_nopi_pi(npts = 112)

# %%
my_run.nopi_pi(average_exponent=12, phase = 25, readout_pulse_type="drachma",
                            readout_weighting_type ="drachma", is_plot_simulation=False) 
## readout weighting is not important for above experiment ; result is obtained through RAW data
my_run.single_shot_nopi_pi(npts_exponent = 12, phase = 25, readout_pulse_type="drachma",
                            readout_weighting_type ="drachma", is_plot_simulation=False)
my_run.plot_nopi_pi(npts = 112)


#In[]

my_run.T1(average_exponent=10, duration = 150e-6, npts = 51, is_plot_simulation=False)
my_run.plot_T1()

# In[]
my_run.qubits_parameters['q0']["pi2_amp"] = 0.26 

my_run.Pi2_cal(average_exponent=11, start = 0, npts = 12, is_plot_simulation=False)
my_run.plot_Pi2_cal()

# In[]
my_run.qubits_parameters['q0']["pi_amp"] = 0.52
my_run.qubits_parameters['q0']["cond_pi_amp"] = 0.0147

my_run.Pi_cal(average_exponent=11, npts = 12, is_plot_simulation=False, is_cond_pulse= False)
my_run.plot_Pi_cal()

# In[]
my_run.Ramsey(is_echo = False, n_pi_pulse=1, # for CP or CPMG
              qubit_phase = 0, # 0 CP, pi/2 CPMG
              detuning = 0.2e6, 
              average_exponent=10, duration = 20e-6, npts = 40,
              is_zz_interaction= False,
              control_qubit= 2,
              is_plot_simulation= False)
my_run.plot_Ramsey(is_fourier_transform=True, is_fit = True)


# In[]
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.02

my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.1
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi


my_run.Ramsey_with_photon(is_echo = False,
            detuning = 0.0e6,
            cavity_freq_detuning= 0e6, # memory mode weak driving (when sideband driving, it should be zero)
            average_exponent=10, duration = 5e-6, npts = 50, #duration/npts = integer
            steady_time = 1.025e-6,
            delay_after_disp_drive = 0e-6,
            qubit_extra_phase= 0,
            is_constant_drive= False,
            is_sideband_drive= True,
            is_displacement_drive = False, ##### cautious
            is_plot_simulation= False)

my_run.plot_Ramsey(is_ramsey_with_photon = True, is_fourier_transform=True)

# In[]
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi

my_run.Ramsey_amp_sweep(is_echo = False, detuning = 0.2e6, 
                        cavity_freq_detuning= -10e6, # memory mode weak driving (when sideband driving, it should be zero)
              average_exponent=9, duration = 10e-6, npts = 50, #duration/npts = integer
              sweep_start = 0.0, sweep_stop = 0.2, sweep_npts = 11, # sweep_start = -np.pi/16, sweep_stop = np.pi/16, sweep_npts = 11,
              fixed_sideband_amp_l = 0, # if not zero, sideband_amp_l is fixed during amp sweep. only sideband_amp_h is swept.
              # in case is_sideband_phase_sweep = True, this parameter is used as fixed amplitude for amp_l and amp_h 
              kappa = 0.0047, chi = 0.227/2, # in MHz unit # chi = chi_ge/2
              steady_time = 1.0e-6,
              is_constant_drive= True,
              is_sideband_drive= False, # both sidebands (do not forget making cavity_freq_detuning zero)
              is_sideband_phase_sweep = False, # when it wants to be swept, is_sideband_drive should be True either
              is_fourier_transform = True,
              is_plot_figure=False,
              is_plot_simulation= False)



# In[] Ramsey_echo_sweep_with_SB
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.4
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.4
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2 # sweep variable if is_amp_sweep = False
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.1 # sweep variable if is_amp_sweep = True

my_run.Ramsey_echo_sweep_with_SB(detuning = 0.2e6,
               average_exponent = 9, duration = 30e-6, npts = 60,
               sweep_start = -0.4, sweep_stop = 0.4, sweep_npts = 21, # sweep_start = 0, sweep_stop = np.pi, sweep_npts = 11, 
               is_disp_amp_real = True, # if True, disp_amp is real number sweep. if False, imaginary number sweep.
               steady_time = 5e-6,
               delay_after_disp_drive = 0e-9,
               is_amp_sweep = True, # else sideband_phase sweep with const disp_amp
               is_plot_figure = False,
               is_plot_simulation = False)


# In[] Ramsey_echo_disp_amp_2Dsweep_with_SB
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.4
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.4
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2

my_run.Ramsey_echo_disp_amp_2Dsweep_with_SB(detuning = 0.2e6,
               average_exponent = 9, duration = 30e-6, npts = 60,
               amp_x_start = -0.4, amp_x_stop = 0.4, amp_x_npts = 9,
               amp_y_start = -0.4, amp_y_stop = 0.4, amp_y_npts = 9,
               steady_time = 5e-6,
               delay_after_disp_drive = 0,
               is_plot_figure = False,
               is_plot_simulation = False)

# In[] long_time ramsey
my_run.long_time_Ramsey(time_end= 30, time_gap=0.5, average_exponent=10, detuning=0.2e6, duration=20e-6, npts=41)

# In[] RamseyEcho_to_chi
my_run.RamseyEcho_to_chi(detuning = 0.2e6,
               average_exponent = 8, duration = 20e-6, npts = 40,
               steady_time = 5e-6,
               wait_time = 0.5e-6,
               cavity_freq_start = -2e6, cavity_freq_stop = 2e6, cavity_freq_npts = 81,
               is_plot_simulation = False)
my_run.plot_RamseyEcho_to_chi()

# In[]
my_run.readout_chi_spectroscopy(average_exponent = 10, npts = 61, 
                                freq_start = -2e6,
                                freq_stop = 1e6,
                                is_plot_simulation = False)
my_run.plot_readout_chi_spectroscopy()


# %%
my_run.error_amplification(average_exponent= 10, pulse_npts = 12, amp_npts = 21, 
                           start_amp = -0.05, end_amp = 0.05,
                           is_drag_beta_calibration = True, #if True, beta will be swept
                           is_plot_simulation=False)
my_run.plot_error_amplification(is_drag_beta_calibration = True)
 # In[]
# my_run.Rabi_amplitude(average_exponent=12, npts = 101, is_plot_simulation=True)

# In[]
my_run.qubits_parameters['q0']["rabi_drive_amp"] = 0.2688

my_run.Rabi_length(average_exponent=9, duration = 1000e-9, start_time = 16e-9, npts = 100,
                   detuning_freq = 0e6, # detuned freq for qubit drive
                   is_single_shot = False, is_plot_simulation=False)
my_run.plot_Rabi_length(is_normalize = True)

# In[]
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.1

my_run.cavity_parameters['m0']["sideband_frequency_l"] = 10e6 # 15e6
my_run.cavity_parameters['m0']["sideband_frequency_h"] = 10e6 # 15e6
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.1 # 0.15
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.1 # 0.15
my_run.cavity_parameters['m0']["sideband_att_l"] = 384.85
my_run.cavity_parameters['m0']["sideband_att_h"] = 346.3965

# my_run.cavity_parameters['m0']["sideband_frequency_l"] = 15e6 # 15e6
# my_run.cavity_parameters['m0']["sideband_frequency_h"] = 15e6 # 15e6
# my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.15 # 0.15
# my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.15 # 0.15
# my_run.cavity_parameters['m0']["sideband_att_l"] = 397.4415 
# my_run.cavity_parameters['m0']["sideband_att_h"] = 339.234

# my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
# my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi

my_run.qubits_parameters['q0']["rabi_drive_amp"] = 0.2969

my_run.Rabi_length_with_photon(average_exponent=8, duration = 5000e-9, start_time = 16e-9, npts = 500,
                   detuning_freq = -0.473e6, #-0.464e6, # detuned freq for qubit drive
                   steady_time = 1.025e-6, is_sideband_drive= True,
                   is_single_shot = False, is_plot_simulation=False)
my_run.plot_Rabi_length(is_normalize = True)

# In[]
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.1
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi
my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi

my_run.qubits_parameters['q0']["rabi_drive_amp"] = 0.2

my_run.Rabi_length_spin_locking(average_exponent = 8, duration = 1e-6, start_time = 16e-9,
                                detuning_freq = -0.0e6, # detuned freq for qubit drive
                                npts = 100, rabi_phase = 0, is_init_pi2=True, init_pi2_phase=np.pi/2,
                                is_sideband_drive= False, steady_time = 1.025e-6, 
                                is_plot_simulation = False)
my_run.plot_Rabi_length_spin_locking(is_xyz_plot = True, is_purity_plot = False)



# In[] cr_calibration_amp
my_run.cr_calibration_amp(average_exponent=12, npts = 101, control_qubit = 0, target_qubit = 1, is_plot_simulation=False)
my_run.plot_cr_calibration_amp()

# In[] cr_calibration_length

my_run.cr_calibration_length(average_exponent=12, npts = 101, control_qubit = 0, target_qubit = 1,
                             duration = 1e-6, target_qubit_amp = 0.5,
                             is_plot_simulation=False)
my_run.plot_cr_calibration_length()


# In[]
my_run.All_XY(average_exponent=12, is_plot_simulation=False)
my_run.plot_All_XY()


# In[] drag_calibration
my_run.drag_calibration(average_exponent=11, beta_start = -0.2, beta_stop = 0.2, beta_count = 81, is_plot_simulation=False)
my_run.plot_drag_calibration()

# In[] cavity_T1 (crosskerr effect : it requires large photon number => it could have side-effect)

my_run.cavity_T1(average_exponent=10, start = 0e-6, duration = 200e-6, npts = 51, is_plot_simulation=False)
my_run.plot_cavity_T1(init_guess = [6, 10, 30000, 0.0])

# In[] cavity_mode_spectroscopy

my_run.cavity_mode_spectroscopy(average_exponent=11, freq_start = -100e6,
                                freq_stop = 100e6, npts = 201, is_plot_simulation=False)

my_run.plot_cavity_mode_spectroscopy()

# In[] finding chi between cavity mode and qubit (qubit_state_revival)
my_run.cavity_parameters['m0']['cavity_drive_length'] = 40e-9
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.15

my_run.qubit_state_revival(average_exponent=11, wait_time= 12e-6, wait_npts = 121, is_plot_simulation=False)
my_run.plot_qubit_state_revival()


# In[] cavity_pi_nopi
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.2
my_run.cavity_parameters['m0']["cond_disp_pulse_length"] = 280e-9 #320e-9
my_run.cavity_parameters['m0']['cond_disp_pulse_detuning'] = 0.5e6
my_run.cavity_parameters['m0']['cond_disp_pulse_sigma'] = 2

my_run.cavity_pi_nopi(average_exponent=9, freq_start = -0.5e6, freq_stop = 0.5e6, 
                      freq_npts = 80, 
                      is_qubit2 = False, 
                      qubit2 = 1,
                      auto_chunking=True,
                      is_plot_simulation=False)
# not accurate for finding chi with cavity_pi_nopi experiment

my_run.plot_cavity_pi_nopi()


# In[]
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.4 # sweep variable
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.071

my_run.CNOD_calibration(average_exponent=10, amp_range=1j,  npts= 61, qubit_phase = 0, 
                        is_calibrated_geo_phase = False,
                        is_displaced_state= True, is_plot_simulation=False)
my_run.plot_CNOD_calibration(scaling_factor= 0.0957)

# In[]
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.0957*5

# CNOD 방향 dependance 잇는지 체크

my_run.CNOD_geophase_calibration(average_exponent=11, amp_start= 0.0, amp_stop = 1, npts = 11, 
                                 is_calibrated_geo_phase = False,
                                 is_plot_simulation=False)
my_run.plot_CNOD_geophase_calibration(is_normalize=True)


# In[] before doing it, "alpha_1_CNOD_amp" needs to be calibrated. 
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.3 # sweep variable
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 1 # 이게 1이어야 함.. 코드 내부에서 amp 자리에 CNOD_alpha_1_amp가 들어가도록 되어 있음.
# 이제까지 중복해서 곱해지는 현상 떄문에 원했던 amp보다 덜 들어가는 상황이었음.
my_run.cavity_parameters['m0']['alpha_1_CNOD_amp'] = 0.0954

my_run.disp_pulse_calibration_geophase(average_exponent=10, amp_sweep = 1, amp_npts=51, is_plot_simulation=False)
my_run.plot_disp_pulse_calibration_geophase()

# In[] Not proper in small chi regime
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.2 # sweep variable

my_run.disp_pulse_calibration_parity(average_exponent=10, amp_start = -1, amp_stop=1, amp_npts = 101, is_plot_simulation=False)
my_run.plot_disp_pulse_calibration_parity(is_fit=True, scaling_factor=0.0458) # scaling factor -> alpha_1_cavity_drive_amp 
                                                        # sigma should be 0.5 by adjusting scaling factor

# %%
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.9 # sweep variable
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.0458 # assigned constant from above experiment

my_run.out_and_back_measurement(average_exponent=8, init_state = "e",
                                 phase_start= 180, phase_stop=250, # deg
                                 cavity_drive_amp_start = 0, cavity_drive_amp_stop = 1, # final amp will be multiplying it by "cavity_drive_amp"
                                 phase_npts=30, amp_npts=51,
                                 wait_time=1e-6,
                                 is_plot_simulation=False)

my_run.plot_out_and_back_measurement(fitting=False, 
                                     x_threshold=30, 
                                     y_threshold=False, 
                                     wait_time=1e-6, init_state="e")


 # In[]
my_run.cavity_parameters['m0']['cavity_drive_amp'] = my_run.cavity_parameters['m0']["alpha_1_cavity_drive_amp"]*2
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = my_run.cavity_parameters['m0']['alpha_1_CNOD_amp'] * 0.3j

my_run.storage_mode_characterization(average_exponent=10, 
                                    wait_time = 10e-6,
                                    wait_npts = 101,
                                    detuning = -0.15e6,#-0.0e6,
                                    init_state = "e",
                                    is_plot_simulation=False)
# init_guess = [amplitude,omega,T1,freq=2*detuning,offset]
my_run.plot_storage_mode_characterization(is_fit=True,init_guess=[-0.69, 5.279, 60e-6, 0.0e6, -0.024], is_fourier_transform=True)

# In[] calibrate_sideband_pulse_phase

my_run.cavity_parameters['m0']["sideband_chunk_length"] = 1e-6
my_run.cavity_parameters['m0']["sideband_rise_fall_length"] = 1e-6
my_run.cavity_parameters['m0']["sideband_frequency_l"] = 10e6
my_run.cavity_parameters['m0']["sideband_frequency_h"] = 10e6

my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.05
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.05
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
my_run.cavity_parameters['m0']["steady_time"] = 1.025e-6
my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi

my_run.qubits_parameters['q0']["rabi_drive_amp"] = 0.2721


# rabi_pulse_length:200ns 배수
# sidebands_pulse_length:1us 배수로 설정

# m = np.linspace(0.21, 0.22, 11)
# for i in m :
#     my_run.qubits_parameters['q0']["rabi_drive_amp"] = i

my_run.calibrate_sideband_pulse_phase(average_exponent=10, sidebands_pulse_length= 10e-6, rabi_pulse_length= 5e-6,
                                    qubit_drive_detuning_freq= -0.0e6,
                                    npts_phase_sweep= 81, is_sideband_phase_sweep = True, # else rabi phase sweep
                                    phase_sweep_stop = 2*np.pi,
                                    rabi_phase = 0,
                                    auto_chunking = True,
                                    is_init_qubit_pi2=False,
                                    is_plot_simulation = True)

my_run.plot_calibrate_sideband_pulse_phase(is_normalize=True, is_fig_plot=True)


# In[] Dirac simulation 1D
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.071*4 # sweep variable for wigner
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.072
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.0956 * 5
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.0956

my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.1
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi

my_run.qubits_parameters['q0']["rabi_drive_amp"] = 0.2739

my_run.Dirac_simul_1D(average_exponent=10, npts = 61, amp_range = 1, acquire_delay = 200e-9, detuning_freq = -0.476e6, 
                      steady_time = 1.0e-6, evolution_time = 2.1e-6, qubit_phase = 0, is_post_selection=True, is_plot_simulation=True)
my_run.plot_characteristic_function_measurement_1D(is_normalize = True, is_plot_G=True, is_plot_E=True)

'''trajectory = []
for evolution_time in np.linspace(0, 2.1e-6, 21):
    my_run.Dirac_simul_1D(average_exponent=10, npts = 61, amp_range = 1, acquire_delay = 200e-9, detuning_freq = -0.476e6, 
                      steady_time = 1.0e-6, evolution_time = evolution_time, qubit_phase = 0, is_post_selection=True, is_plot_simulation=False)
    x_grid, y_grid, G_data, E_data, se_G_data = my_run.plot_characteristic_function_measurement_1D(is_normalize = True, is_plot_G=True, is_plot_E=True)
    trajectory.append((x_grid, y_grid, G_data))'''

# In[] with post selection (singleshot measure를 해야해서 measure point 수 한계가 있음)

## before running below code, drachma nopi-pi code should be conducted to get threshold for post-selection

my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.071*4 # sweep variable for wigner
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.071
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.0957 * 5
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.0957

my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.1
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
my_run.cavity_parameters['m0']["steady_time"] = 1.025e-6
my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi


# 'cavity_drive_amp' * amplitude에 해당하는 displaced state 형성

my_run.characteristic_function_1D(average_exponent=10, npts = 61, amp_range = 1, amplitude = 0.5j,
                                  acquire_delay = 120e-9, 
                                  alpha = 1, # for cat state 
                                  beta = 1, # for cat state
                                  qubit_phase = -np.pi/2, chunk_count = 61, # -np.pi/2 : Imag of characteristic function
                                    is_coherent_state=True,
                                    is_coherent_state_pi = False,
                                    is_schrodinger_cat_state=False,
                                    is_schrodinger_cat_state_2=False,
                                    is_cat_state=False,
                                    is_cat_state_2=False,
                                    is_cat_and_back=False,
                                    is_sideband_drive=False,
                                    is_post_selection=False,
                                    is_plot_simulation=False)

my_run.plot_characteristic_function_measurement_1D(is_normalize = True, is_store_values = True)
my_run.get_state_info()


# In[] with post selection (singleshot measure를 해야해서 measure point 수 한계가 있음)

## before running below code, drachma nopi-pi code should be conducted to get threshold for post-selection

my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.071*6 # sweep variable for wigner
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.071
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.0911 * 5
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.0911

my_run.cavity_parameters['m0']["sideband_frequency_l"] = 10e6 # 15e6
my_run.cavity_parameters['m0']["sideband_frequency_h"] = 10e6 # 15e6
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.1
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi
my_run.cavity_parameters['m0']["steady_time"] = 1.075e-6

my_run.cavity_parameters['m0']["sideband_att_l"] = 396.738
my_run.cavity_parameters['m0']["sideband_att_h"] = 356.827


# 'cavity_drive_amp' * amplitude에 해당하는 displaced state 형성
# 'cavity_drive_amp'/(2*'alpha_1_cavity_drive_amp') 사이즈에 해당하는 Re-Im plane plot in case of wigner

my_run.wigner_characteristic_function_2D(average_exponent=8, npts_x = 21, npts_y = 21, amplitude = 0.166, #amplitude for coherent state
                          acquire_delay= 200e-9,
                          alpha= 0.5, beta= -0.05,
                          qubit_phase = -np.pi/2,
                          chunk_count = 21,
                          is_wigner_function= True,
                          is_coherent_state = False,
                          is_coherent_state_pi = False, # coherent drive after pi pulse (for measuring chi through wigner)
                          is_schrodinger_cat_state= False, 
                          is_schrodinger_cat_state_2=False,
                          is_cat_state=False,
                          is_cat_state_2=False,
                          is_cat_and_back=False,
                          is_sideband_drive=True,
                          is_init_displaced_state=False,
                          is_Dirac_simul = False,
                          evolution_time_for_Dirac_simul = 1.5e-6,
                          is_post_selection= False,
                          is_plot_simulation=False)

x_grid, y_grid, data_wo_post_selection_im, std_data_wo_post_selection_im = my_run.plot_wigner_characteristic_function_measurement_2D(is_normalize = True, 
                                                                                           is_plot_G=True, is_plot_E=True)

# x_grid, y_grid, G_data, E_data, se_G, se_E = my_run.plot_wigner_characteristic_function_measurement_2D(is_normalize = True, 
#                                                                                            is_plot_G=True, is_plot_E=True)

# my_run.wigner_characteristic_function_2D(average_exponent=10, npts_x = 21, npts_y = 21, amplitude = 0.375, #amplitude for coherent state
#                           is_wigner_function= False,
#                           is_coherent_state = False, is_schrodinger_cat_state= False, 
#                           is_schrodinger_cat_state_2=False,
#                           is_cat_state=False,
#                           is_cat_state_2=False,
#                           is_cat_and_back=False,
#                           is_sideband_drive=True,
#                           acquire_delay= 40e-9,
#                           alpha= 0.34, beta= -0.05,
#                           qubit_phase = np.pi/2,
#                           chunk_count = 21,
#                           is_post_selection= False,
#                           is_plot_simulation=False)

# x_grid, y_grid, data_wo_post_selection_im, std_data_wo_post_selection_im = my_run.plot_wigner_characteristic_function_measurement_2D(is_normalize = True)

# In[] with post selection (singleshot measure를 해야해서 measure point 수 한계가 있음)

## before running below code, drachma nopi-pi code should be conducted to get threshold for post-selection

my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.073*4 # sweep variable for wigner
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.073
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.087 * 8
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.087

# 'cavity_drive_amp' * amplitude에 해당하는 displaced state 형성
# 'cavity_drive_amp'/(2*'alpha_1_cavity_drive_amp') 사이즈에 해당하는 Re-Im plane plot in case of wigner

my_run.wigner_characteristic_function_2D(average_exponent=9, npts_x = 11, npts_y = 11, amplitude = 0.375, #amplitude for coherent state
                          is_wigner_function= True,
                          is_coherent_state = False, 
                          is_coherent_state_pi = True,
                          is_schrodinger_cat_state= False, 
                          is_schrodinger_cat_state_2=False,
                          is_cat_state=False,
                          is_cat_state_2=False,
                          is_cat_and_back=False,
                          is_sideband_drive=False,
                          acquire_delay= 10000e-9,
                          alpha= 0.4, beta= -0.05,
                          qubit_phase = 0, # np.pi/2,
                          chunk_count = 11,
                          is_post_selection= True,
                          is_plot_simulation=False)

x_grid, y_grid, G_data, E_data, se_G, se_E = my_run.plot_wigner_characteristic_function_measurement_2D(is_normalize = True, 
                                                                                           is_plot_G=True, is_plot_E=True)

# np.savez(
#     "wigner_cf_run_cat_2.npz",  # 저장 파일 이름
#     x_grid=x_grid,
#     y_grid=y_grid,
#     data_post_selection_G=G_data,
#     std_data_post_selection_G=se_G,
#     data_post_selection_E=E_data,
#     std_data_post_selection_E=se_E
# )



# In[]

my_run.wigner_characteristic_function_2D(average_exponent=12, npts_x = 31, npts_y = 31, amplitude = 0.375, #amplitude for coherent state
                          is_wigner_function= False,
                          is_coherent_state = False, is_schrodinger_cat_state= False, 
                          is_schrodinger_cat_state_2=False,
                          is_cat_state=False,
                          is_cat_state_2=True,
                          is_cat_and_back=False,
                          is_sideband_drive=True,
                          acquire_delay= 120e-9,
                          alpha= 0.3, beta= -0.05,
                          qubit_phase = 0, # np.pi/2,
                          chunk_count = 31,
                          is_post_selection= True,
                          is_plot_simulation=False)

x_grid, y_grid, G_data, E_data, se_G, se_E = my_run.plot_wigner_characteristic_function_measurement_2D(is_normalize = True, 
                                                                                           is_plot_G=True, is_plot_E=True)

np.savez(
    "wigner_cf_run_cat_3.npz",  # 저장 파일 이름
    x_grid=x_grid,
    y_grid=y_grid,
    data_post_selection_G=G_data,
    std_data_post_selection_G=se_G,
    data_post_selection_E=E_data,
    std_data_post_selection_E=se_E
)
# In[]

loaded = np.load("wigner_cf_run_re.npz")

x_grid = loaded["x_grid"]
y_grid = loaded["y_grid"]
data_wo_post_selection_re = loaded["data_wo_post_selection_re"]
std_data_wo_post_selection_re = loaded["std_data_wo_post_selection_re"]

loaded = np.load("wigner_cf_run_im.npz")

x_grid = loaded["x_grid"]
y_grid = loaded["y_grid"]
data_wo_post_selection_im = loaded["data_wo_post_selection_im"]
std_data_wo_post_selection_im = loaded["std_data_wo_post_selection_im"]

# In[]

loaded = np.load("wigner_cf_run_re.npz")

x_grid = loaded["x_grid"]
y_grid = loaded["y_grid"]
data_wo_post_selection_re = loaded["data_wo_post_selection_re"]
std_data_wo_post_selection_re = loaded["std_data_wo_post_selection_re"]

# In[]
psi_init = qt.coherent(N = 30, alpha = 2) # + qt.coherent(N = 30, alpha = -1.5)
psi_init = psi_init.unit()
rho_init = qt.ket2dm(psi_init)

# In[]

rho = my_run.density_matrix_reconstruction(N = 30, rho_init = rho_init, data = G_data, sigmas = se_G,
                                           x_grid = x_grid, y_grid = y_grid,
                                           maxiter=2000, maxfun=2000000)
my_run.state_fidelity(rho_ideal = rho_init, rho = rho)

my_run.plot_characteristic_function(x_grid, y_grid, rho)
my_run.plot_characteristic_function(x_grid, y_grid, rho_init)

# In[]

rho = my_run.density_matrix_reconstruction_re_im(N = 20, rho_init = rho_init, 
                                                 data_re = data_wo_post_selection_re, data_im = data_wo_post_selection_im, 
                                                 sigmas_re = std_data_wo_post_selection_re, sigmas_im = std_data_wo_post_selection_im,
                                                 x_grid = x_grid, y_grid = y_grid,
                                                 maxiter=2000, maxfun=2000000)
my_run.state_fidelity(rho_ideal = rho_init, rho = rho)

my_run.plot_characteristic_function(x_grid, y_grid, rho)
my_run.plot_characteristic_function(x_grid, y_grid, rho_init)


# In[]
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.077
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.077 * 4

my_run.disentangling_power_sweep(average_exponent = 10,
                                  alpha = 1,
                                  beta_sweep_start = -0.5,
                                  beta_sweep_stop = 0.5,
                                  beta_sweep_count = 21,
                                  is_xyz=True)
my_run.plot_disentangling_power_sweep(is_normalize=True)
















# In[]

my_run.continuous_wave(average_exponent=19, freq_l=5e6, freq_h=10e6, amp_l=0.5, amp_h=0.1, phase = np.pi/2,
                       amp_cont = 0.1,
                       is_sideband_pulse = True, is_plot_simulation=False)






# In[]
## define pulse
my_pulse = pulse_library.cond_disp_pulse(uid="my_pulse", length=100e-9, 
                                         amplitude=1.0, sigma = 1,
                                         chi = 1e6, detuning = 0.001e6,
                                         zero_boundaries=False)


_pulse = pulse_library.gaussian(
                uid="cavity_drive_pulse",
                length= 2048e-9, #cavity_parameters["m1"]["cavity_drive_length"],
                amplitude=cavity_parameters["m1"]["cavity_drive_amp"],
                zero_boundaries=False,
                sigma = 1/3,
            )

_pulse = pulse_library.drachma_readout_pulse(
                uid="drachma_readout_pulse",
                length = qubits_parameters["q0"]["readout_pulse_length"],
                amplitude=qubits_parameters["q0"]["readout_amp"],
                kappa=qubits_parameters["q0"]["readout_kappa"],
                chi_list=[-qubits_parameters["q0"]["readout_chi"]/2, qubits_parameters["q0"]["readout_chi"]/2],
                zeta_list = [-2*np.pi*1000,-2*np.pi*1000],
            )

_pulse = pulse_library.sidebands_pulse(
                uid="sidebands_drive_pulse",
                length=cavity_parameters["m0"]["sideband_chunk_length"],
                frequency_l=cavity_parameters["m0"]["sideband_frequency_l"],
                frequency_h=cavity_parameters["m0"]["sideband_frequency_h"],
                amp_l=0.1,
                amp_h=0.2,
                phase=np.pi/2,
            )


pulse = pulse_update(
            _pulse,
            spectral_window = None,
            flip_angle = None,
            pulse_parameters=_pulse.pulse_parameters,
        )

my_run.analyze_pulse(_pulse, high_res = False)




# %%
