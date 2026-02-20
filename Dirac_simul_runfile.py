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
        "port_delay": (120+160)*1e-9, # Because readout pulse shape is gussian square, for eliminating the rising edge.
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

        "readout_phase": (-30) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.47, 
        "drive_pulse_length": 64e-9,
        "drive_beta": 0.035,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.235,
        "pi2_beta": 0.035,

        "pi_length": 64e-9,
        "pi_amp": 0.47,
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
        "cavity_drive_amp": 0.072*4, #(this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_cavity_drive_amp": 0.072, # 1 when it is not found yet. (This is amp when photon = 1) (this should be same with scaling_factor in disp_amp_calibration_parity

        "cond_disp_pulse_length": 200e-9, #200e-9,
        "cond_disp_pulse_amp": 0.0964, #0.0446j, # (this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_CNOD_amp": 0.0964, # 1 when it is not found yet. (this should be same with scaling_factor in CNOD calibration)
        # (omega_d : w_LO + w_IF , omega_e or g : w_LO + w_IF + cond_disp_pulse_frequency)
        "cavity_mode_chi": -0.227e6, 
        "chi_for_cnod": -0.227e6/2, #-0.227e6/2, # it depends on the driving freq frame.
        "cond_disp_pulse_frequency": -0.227e6/2, #-0.227e6/2,
        # driving only cavity mode of ground state in driving freq(mode_frequency) frame.
        "cond_disp_pulse_detuning": 0.5e6,
        "cond_disp_pulse_sigma": 2,
        "geophase_correction_coeff": 0.034/2, # after calibrating geometric phase.

        "sideband_length": 1e-6,
        "sideband_chunk_length": 1e-6,
        "sideband_rise_fall_length": 120e-9,

        "sideband_frequency_l" : 15e6, #30e6, # input unit is Hz
        "sideband_frequency_h" : 15e6, #30e6,
        "sideband_amp_l" : 0.15,
        "sideband_amp_h" : 0.15,
        "sideband_phase" : 0, # in radians
        "sideband_att_l" : 398.97, # The larger value, the smaller attenuation
        "sideband_att_h" : 340.18, #239, # The larger value, the smaller attenuation
        "sideband_extra_phase": np.pi, # in radians

        # "sideband_frequency_l" : 10e6, # input unit is Hz
        # "sideband_frequency_h" : 10e6,
        # "sideband_amp_l" : 0.1,
        # "sideband_amp_h" : 0.1,
        # "sideband_phase" : 0, # in radians
        # "sideband_att_l" : 385.39,  # The larger value, the smaller attenuation
        # "sideband_att_h" : 346.77,  # The larger value, the smaller attenuation
        # "sideband_extra_phase": np.pi, # in radians

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

# In[]
# 아래 순서로 Calibration 하고 Dirac_1D simul code 실험하면 됨
# general한 calibration은 General_runfile에서 전부 진행하고 값이 맞춰져 있을 때 이 파일 사용하여 실험 진행
# 1) 주기적 pi/2, pi calibration, (Ramsey로는 qubit freq 맞는지 확인 : 자주 할 필요 없음)
# 2) CNOD alpha calibration, Displacement beta calibration (이것도 그렇게까지 자주 할 필요 없음)
# 3) Manually rabi freq calibration in stark shift frame with fake sideband driving 
# 4) evolution time dependent Dirac 1D simulation experiment

# 특히 3과 4번을 반복적으로 번갈아가며 실험하여 rabi freq가 틀어지지 않도록 하는 것이 중요함.
# qubit에게 driving하는 amplitude fluctuation이 생각보다 민감함.


# %% 
my_run.nopi_pi(average_exponent=12, phase = -30, readout_pulse_type="gaussian_square",
                            readout_weighting_type ="cavity_trajectory", is_plot_simulation=False)
## readout weighting is not important for above experiment ; result is obtained through RAW data
my_run.single_shot_nopi_pi(npts_exponent = 12, phase = -30, readout_pulse_type="gaussian_square",
                            readout_weighting_type ="cavity_trajectory", is_plot_simulation=False)
my_run.plot_nopi_pi(npts = 112)

# %%
my_run.nopi_pi(average_exponent=12, phase = -40, readout_pulse_type="drachma",
                            readout_weighting_type ="drachma", is_plot_simulation=False) 
## readout weighting is not important for above experiment ; result is obtained through RAW data
my_run.single_shot_nopi_pi(npts_exponent = 12, phase = -40, readout_pulse_type="drachma",
                            readout_weighting_type ="drachma", is_plot_simulation=False)
my_run.plot_nopi_pi(npts = 112)






# In[]
my_run.qubits_parameters['q0']["pi2_amp"] = 0.235

my_run.Pi2_cal(average_exponent=11, start = 0, npts = 12, is_plot_simulation=False)
my_run.plot_Pi2_cal()

# In[]
my_run.qubits_parameters['q0']["pi_amp"] = 0.47
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
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.4 # sweep variable
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.072

my_run.CNOD_calibration(average_exponent=10, amp_range=1j,  npts= 61, qubit_phase = 0, 
                        is_calibrated_geo_phase = False,
                        is_displaced_state= True, is_plot_simulation=False)
my_run.plot_CNOD_calibration(scaling_factor= 0.095)

# In[]
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.0917*5

# CNOD 방향 dependance 잇는지 체크

my_run.CNOD_geophase_calibration(average_exponent=11, amp_start= 0.0, amp_stop = 1, npts = 11, 
                                 is_calibrated_geo_phase = False,
                                 is_plot_simulation=False)
my_run.plot_CNOD_geophase_calibration(is_normalize=True)






# In[]
my_run.qubits_parameters['q0']["rabi_drive_amp"] = 0.2571

my_run.Rabi_length(average_exponent=10, duration = 1000e-9, start_time = 1e-6, npts = 200,
                   detuning_freq = 0e6, # detuned freq for qubit drive
                   is_single_shot = False, is_plot_simulation=True)
my_run.plot_Rabi_length(is_normalize = True)

# In[]
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.1

my_run.cavity_parameters['m0']["sideband_frequency_l"] = 10e6 # 15e6
my_run.cavity_parameters['m0']["sideband_frequency_h"] = 10e6 # 15e6
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.05 # 0.15
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.05 # 0.15
my_run.cavity_parameters['m0']["sideband_att_l"] = 385.39 # 398.349
my_run.cavity_parameters['m0']["sideband_att_h"] = 346.77 # 339.855

# my_run.cavity_parameters['m0']["sideband_frequency_l"] = 15e6 # 15e6
# my_run.cavity_parameters['m0']["sideband_frequency_h"] = 15e6 # 15e6
# my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.15 # 0.15
# my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.15 # 0.15
# my_run.cavity_parameters['m0']["sideband_att_l"] = 398.97
# my_run.cavity_parameters['m0']["sideband_att_h"] = 340.18


my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi

my_run.qubits_parameters['q0']["rabi_drive_amp"] = 0.2485

my_run.Rabi_length_with_photon(average_exponent=9, duration = 10000e-9, start_time = 16e-9, npts = 1000,
                   detuning_freq = -0.482e6, # detuned freq for qubit drive
                   steady_time = 1.0e-6, is_sideband_drive= True,
                   is_single_shot = False, is_plot_simulation=False)
my_run.plot_Rabi_length(is_normalize = True)




# In[] Dirac simulation 1D
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.072*4 # sweep variable for wigner
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.072
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.095 * 5
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.095

my_run.cavity_parameters['m0']["sideband_frequency_l"] = 10e6 # 15e6
my_run.cavity_parameters['m0']["sideband_frequency_h"] = 10e6 # 15e6
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.1
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi

my_run.cavity_parameters['m0']["sideband_att_l"] = 385.39
my_run.cavity_parameters['m0']["sideband_att_h"] = 346.77

my_run.qubits_parameters['q0']["rabi_drive_amp"] = 0.2505

my_run.Dirac_simul_1D(average_exponent=10, npts = 61, amp_range = 1, acquire_delay = 200e-9, detuning_freq = -0.482e6, 
                      steady_time = 1.025e-6, evolution_time = 1e-6, qubit_phase = 0, is_post_selection=True, is_plot_simulation=True)
my_run.plot_characteristic_function_measurement_1D(is_normalize = True, is_plot_G=True, is_plot_E=True)

# In[] with post selection (singleshot measure를 해야해서 measure point 수 한계가 있음)

## before running below code, drachma nopi-pi code should be conducted to get threshold for post-selection

my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.072*4 # sweep variable for wigner
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.072
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.095 * 5
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.095


my_run.cavity_parameters['m0']["sideband_frequency_l"] = 10e6 # 15e6
my_run.cavity_parameters['m0']["sideband_frequency_h"] = 10e6 # 15e6
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.1
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
my_run.cavity_parameters['m0']['sideband_extra_phase'] = np.pi
my_run.cavity_parameters['m0']["steady_time"] = 1.025e-6

my_run.cavity_parameters['m0']["sideband_att_l"] = 385.39
my_run.cavity_parameters['m0']["sideband_att_h"] = 346.77

my_run.qubits_parameters['q0']["rabi_drive_amp"] = 0.2505

# 'cavity_drive_amp' * amplitude에 해당하는 displaced state 형성
# 'cavity_drive_amp'/(2*'alpha_1_cavity_drive_amp') 사이즈에 해당하는 Re-Im plane plot in case of wigner

my_run.wigner_characteristic_function_2D(average_exponent=8, npts_x = 21, npts_y = 21, amplitude = 0.5, #amplitude for coherent state
                          acquire_delay= 200e-9,
                          alpha= 0.5, beta= -0.05,
                          qubit_phase = 0,
                          chunk_count = 21,
                          is_wigner_function= True,
                          is_coherent_state = False,
                          is_coherent_state_pi = False, # coherent drive after pi pulse (for measuring chi through wigner)
                          is_schrodinger_cat_state= False, 
                          is_schrodinger_cat_state_2=False,
                          is_cat_state=False,
                          is_cat_state_2=False,
                          is_cat_and_back=False,
                          is_sideband_drive= False,
                          is_init_displaced_state=False,
                          is_Dirac_simul = True,
                          evolution_time_for_Dirac_simul = 0.575e-6,
                          is_post_selection= True,
                          is_plot_simulation=True)

x_grid, y_grid, G_data, E_data, se_G, se_E = my_run.plot_wigner_characteristic_function_measurement_2D(is_normalize = True, 
                                                                                           is_plot_G=True, is_plot_E=True)
# %%
