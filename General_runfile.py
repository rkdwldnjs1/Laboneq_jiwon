# In[]

import matplotlib.pyplot as plt
%matplotlib qt5
import numpy as np
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
        "port_delay": (80+80)*1e-9, # Because readout pulse shape is gussian square, for eliminating the rising edge.
        "port_mode": None,
        "delay_signal": None, # Global delay for the signal line, implemented by adjusting the waveform.
        "threshold": None,
        "added_outputs": None, # Added outputs to the signal line's physical channel port. (RTR option enabled)
        "automute": False, # Mute output channel when no waveform is played on it i.e for the duration of delays.
        "range": -10,
        "amplitude": 1.0, # it does not work!
                          # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
    },
    "drive":{
        "q0":{
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
        "q1":{ # 2nd control line port
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
            "voltage_offset": 5,
        },
        "m0":{ # mapped to port4 # 현재 안 쓰는 포트
            "freq_LO": 6.6e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
        },
        "m1":{ # mapped to port5
            "freq_LO": 5e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
        }
    }
}

### These parameters will be utilized in defining experimental sequences.
qubits_parameters = {
    # two(cr) qubit gate : each parameter is assigned to each qubit except ef_frequency;

    "q0": {

        #port 1 (15-1)

        "ge_frequency" : 4.383093e9,
        "ef_frequency" : 4.394e9, #4.275e9, #4.3865e9,
        "readout_frequency" : 7.66437e9-0.1e6, 
        ### readout parameters ###
        "readout_amp" : 0.36, #0.18,
        'readout_pulse_length': 1400e-9,
        "readout_integration_length": (1200)*1e-9,
        "readout_integration_amp": 1,
        "readout_phase": (-95) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.165*2, 
        "drive_pulse_length": 64e-9,
        "drive_beta": 0.04,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.165,
        "pi2_beta": 0.04,

        "pi_length": 64e-9,
        "pi_amp": 0.165*2,
        "pi_beta": 0.04,

        "cond_pi_length": 2048e-9,
        "cond_pi_amp": 0.0116,
        "cond_pi_beta": 0.0,

        "rabi_drive_amp": 0.1778,
        "ramp_length": 0e-9,
        ### extra parameters ###
        "reset_delay_length": 200e-6,

        "is_cr" : False,
        "cr" : {
            "control_qubit_frequency": 4.9341e9+0.039e6,
            "target_qubit_frequency_1": 4.7e9+0.12e6,
            "target_qubit_frequency_2": 4.7e9,  # temporally arbitrary value
        }

    },

    "q1": {

        #port 1 (15-1)

        "ge_frequency" : 4.383093e9,
        "ef_frequency" : 4.394e9, #4.275e9, #4.3865e9,
        "readout_frequency" : 7.66437e9, 
        ### readout parameters ###
        "readout_amp" : 0.34, #0.18,
        'readout_pulse_length': 1400e-9,
        "readout_integration_length": (1200)*1e-9,
        "readout_integration_amp": 1,
        "readout_phase": (-90) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.026, 
        "drive_pulse_length": 64e-9,
        "drive_beta": 0.0,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.013,
        "pi2_beta": 0.0,

        "pi_length": 64e-9,
        "pi_amp": 0.026,
        "pi_beta": 0.0,

        "cond_pi_length": 2048e-9,
        "cond_pi_amp": 0.009,
        "cond_pi_beta": 0.0,

        "rabi_drive_amp": 0.4,
        "ramp_length": 0e-9,
        ### extra parameters ###
        "reset_delay_length": 200e-6,

        "is_cr" : False,
        "cr" : {
            "control_qubit_frequency": 4.9341e9+0.039e6,
            "target_qubit_frequency_1": 4.7e9+0.12e6,
            "target_qubit_frequency_2": 4.7e9,  # temporally arbitrary value
        }

    },

    "q2": { #control out port 1 (8-2)
        ### frequency parameters ###
        "ge_frequency" : 4.4017e9, #4.275e9, #4.3865e9,
        "ef_frequency" : 4.4017e9, #4.275e9, #4.3865e9,
        "readout_frequency" : 7.426323e9, 
        ### readout parameters ###
        "readout_amp" : 0.25, #0.18,
        'readout_pulse_length': 1800e-9,
        "readout_integration_length": 1600e-9,
        "readout_integration_amp": 1,
        "readout_phase": (77) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.29, 
        "drive_pulse_length": 64e-9,
        # "drive_ef_amp" : 0.6,
        # "drive_ef_pulse_length": 64e-9,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.145,
        "pi2_beta": 0.03,

        "pi_length": 64e-9,
        "pi_amp": 0.286,
        "pi_beta": 0.03,

        "cond_pi_length": 2*512e-9,
        "cond_pi_amp": 0.06,
        "cond_pi_beta": 0.0,

        "rabi_drive_amp": 0.2,
        "ramp_length": 0e-9,
        ### extra parameters ###
        "reset_delay_length": 200e-6,

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
        "mode_frequency": 6.795139e9 - 1.22e6 - 0.05e6, # center of g and e freq
        "cavity_drive_length": 40e-9, # this should be as short as possible to cover wide frequency range
        "cavity_drive_amp": 0.17*4.5, #(this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_cavity_drive_amp": 0.17, # 1 when it is not found yet. (This is amp when photon = 1) (this should be same with scaling_factor in disp_amp_calibration_parity

        "cond_disp_pulse_length": 320e-9, #200e-9,
        "cond_disp_pulse_amp": 0.8, #0.0446j, # (this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_CNOD_amp": 0.2080, # 1 when it is not found yet. (this should be same with scaling_factor in CNOD calibration)

        # (omega_d : w_LO + w_IF , omega_e or g : w_LO + w_IF + cond_disp_pulse_frequency)
        "cavity_mode_chi": -0.1e6, #-0.824e6,
        "cond_disp_pulse_frequency": -0.05e6, # driving only cavity mode of ground state in driving freq(mode_frequency) frame.
        "cond_disp_pulse_detuning": 0.5e6,
        "cond_disp_pulse_sigma": 2,

        "sideband_length": 1e-6,
        "sideband_frequency_l" : 10e6, # input unit is Hz
        "sideband_frequency_h" : 10e6,
        "sideband_amp_l" : 0.0,
        "sideband_amp_h" : 0.0,
        "sideband_phase" : np.pi/2, # in radians
        "sideband_att_l" : 155, # The larger value, the smaller attenuation
        "sideband_att_h" : 147, # The larger value, the smaller attenuation

        "reset_delay_length": 400e-6,
    },

    'm1': { # mm1
        "mode_frequency": 5.153319e9 - 0.24e6, # omega_d
        "cavity_drive_length": 40e-9, # this should be as short as possible to cover wide frequency range
        "cavity_drive_amp": 0.0895*4.5, #(this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_cavity_drive_amp": 0.0895, # 1 when it is not found yet. (This is amp when photon = 1) (this should be same with scaling_factor in disp_amp_calibration_parity

        "cond_disp_pulse_length": 200e-9, #200e-9,
        "cond_disp_pulse_amp": 0.8, #0.0446j, # (this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_CNOD_amp": 0.0964, # 1 when it is not found yet. (this should be same with scaling_factor in CNOD calibration)

        # (omega_d : w_LO + w_IF , omega_e or g : w_LO + w_IF + cond_disp_pulse_frequency)
        "cavity_mode_chi": -0.18e6, #-0.824e6,
        "cond_disp_pulse_frequency": -0.18e6, # driving only cavity mode of ground state in driving freq(mode_frequency) frame.
        "cond_disp_pulse_detuning": 0.5e6,
        "cond_disp_pulse_sigma": 2,

        "sideband_length": 1e-6,
        "sideband_frequency_l" : 10e6, # input unit is Hz
        "sideband_frequency_h" : 10e6,
        "sideband_amp_l" : 0.1,
        "sideband_amp_h" : 0.1,
        "sideband_phase" : 0, # in radians

        "reset_delay_length": 400e-6,
    },

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
                 use_emulation = False)

# %% propagation delay calibration for readout and drive line

my_run.prop_delay_calibration(line = "readout", average_exponent=12)


# %%
my_run.nopi_pi(average_exponent=12, phase = -95, is_plot_simulation=False) # used for 2^n averages, n=average_exponent, maximum: n = 19
my_run.single_shot_nopi_pi(npts_exponent = 12, phase = -95)
my_run.plot_nopi_pi(npts = 100)

# In[]

my_run.test_consecutive_measurements(npts_exponent = 12, phase = -100, acquire_delay = 480e-9,
                                     first_amp=1,
                                     second_amp=1,
                                     is_plot_simulation = True)
my_run.plot_test_consecutive_measure()

#In[]

my_run.T1(average_exponent=11, duration = 100e-6, npts = 51, is_plot_simulation=False)
my_run.plot_T1()

# In[]

my_run.Pi2_cal(average_exponent=12, start = 0, npts = 12, is_plot_simulation=False)
my_run.plot_Pi2_cal()

# In[]

my_run.Pi_cal(average_exponent=12, npts = 12, is_plot_simulation=False, is_cond_pulse=True)
my_run.plot_Pi_cal()

# In[]
my_run.Ramsey(is_echo = False, n_pi_pulse=1, # for CP or CPMG
              qubit_phase = 0, # 0 CP, pi/2 CPMG
              detuning = 0.0e6, 
              average_exponent=10, duration = 10e-6, npts = 51,
              is_zz_interaction= False,
              control_qubit= 2,
              is_plot_simulation= False)
my_run.plot_Ramsey()

# In[]
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.8

my_run.Ramsey_with_photon(is_echo = False,
              detuning = 0.5e6,
              cavity_freq_detuning= 40e6, # memory mode weak driving
              average_exponent=11, duration = 5e-6, npts = 50, #duration/npts = integer
              steady_time = 5e-6,
              is_plot_simulation= False)

my_run.plot_Ramsey(is_ramsey_with_photon = True)

# In[]
my_run.Ramsey_amp_sweep(is_echo = False, detuning = 0.5e6, cavity_freq_detuning= -10e6, # memory mode weak driving
              average_exponent=10, duration = 5e-6, npts = 50, #duration/npts = integer
              amp_start = 0, amp_stop = 0.3, amp_npts = 11,
              kappa = 0.003, chi = 0.1, # in MHz unit
              steady_time = 20e-6,
              is_plot_figure=True,
              is_plot_simulation= False)

# In[] long_time ramsey
my_run.long_time_Ramsey(time_end=180, time_gap=0.5, average_exponent=10, detuning=0.2e6, duration=10e-6, npts=51)


# %%
my_run.error_amplification(average_exponent= 11, pulse_npts = 12, amp_npts = 21, 
                           start_amp = -0.1, end_amp = 0.1,
                           is_drag_beta_calibration = True, #if True, beta will be swept
                           is_plot_simulation=False)
my_run.plot_error_amplification(is_drag_beta_calibration = True)
 # In[]
# my_run.Rabi_amplitude(average_exponent=12, npts = 101, is_plot_simulation=True)

# In[]
my_run.Rabi_length(average_exponent=10, duration = 1e-6, start_time = 1e-6, npts = 200, is_single_shot = False, is_plot_simulation=False)
my_run.plot_Rabi_length()

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

my_run.cavity_T1(average_exponent=10, start = 0e-6, duration = 350e-6, npts = 51, is_plot_simulation=False)
my_run.plot_cavity_T1(init_guess = [1.5, 10, 20000, 0.8])

# In[]

# In[] cavity_mode_spectroscopy

my_run.cavity_mode_spectroscopy(average_exponent=11, freq_start = 100e6,
                                freq_stop = 200e6, npts = 101, is_plot_simulation=False)

my_run.plot_cavity_mode_spectroscopy()

# In[] finding chi between cavity mode and qubit
my_run.cavity_parameters['m0']['cavity_drive_length'] = 40e-9
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.15

my_run.qubit_state_revival(average_exponent=11, wait_time= 12e-6, wait_npts = 121, is_plot_simulation=False)
my_run.plot_qubit_state_revival()


# In[]
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.5
my_run.cavity_parameters['m0']["cond_disp_pulse_length"] = 320e-9
my_run.cavity_parameters['m0']['cond_disp_pulse_detuning'] = 0.5e6
my_run.cavity_parameters['m0']['cond_disp_pulse_sigma'] = 2

my_run.cavity_pi_nopi(average_exponent=9, freq_start = -0.3e6, freq_stop = 0.3e6, 
                      freq_npts = 100, 
                      is_qubit2 = False, 
                      qubit2 = 1,
                      auto_chunking=True,
                      is_plot_simulation=False)
# not accurate for finding chi with cavity_pi_nopi experiment

my_run.plot_cavity_pi_nopi()


# In[]
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.7 # sweep variable
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.17

my_run.CNOD_calibration(average_exponent=10, amp_range=1j, npts= 61, qubit_phase = 0, is_displaced_state= False, is_plot_simulation=False)
my_run.plot_CNOD_calibration(scaling_factor=0.2080)

# In[]

# my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.2 # sweep variable
# my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.046*2

# my_run.Characteristic_function_2D(average_exponent=10, npts=61, qubit_phase = 90, is_plot_simulation=False)
# my_run.plot_Characteristic_function_2D()

# In[] before doing it, "alpha_1_CNOD_amp" needs to be calibrated. 
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.5 # sweep variable
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 1 # 이게 1이어야 함.. 코드 내부에서 amp 자리에 CNOD_alpha_1_amp가 들어가도록 되어 있음.
# 이제까지 중복해서 곱해지는 현상 떄문에 원했던 amp보다 덜 들어가는 상황이었음.
my_run.cavity_parameters['m0']['alpha_1_CNOD_amp'] = 0.2

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
my_run.cavity_parameters['m0']['cavity_drive_amp'] = my_run.cavity_parameters['m0']["alpha_1_cavity_drive_amp"]*2.5
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = my_run.cavity_parameters['m0']['alpha_1_CNOD_amp'] * 1j

my_run.storage_mode_characterization(average_exponent=10, 
                                    wait_time = 60e-6,
                                    wait_npts = 121,
                                    detuning = 0.2e6,#-0.0e6,
                                    init_state = "g",
                                    is_plot_simulation=False)
# init_guess = [amplitude,omega,T1,freq=2*detuning,offset]
my_run.plot_storage_mode_characterization(is_fit=True,init_guess=[-0.69, 5.279, 30e-6, 0.4e6, -0.024])
# In[]
my_run.plot_storage_mode_characterization(is_fit=True,init_guess=[-2, 2.5, 10e-6, 0.0e6, 0.5])


# In[] with post selection (singleshot measure를 해야해서 measure point 수 한계가 있음)

my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.176 # sweep variable
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.17
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.2 * 4
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.2

# 'cavity_drive_amp' * amplitude에 해당하는 displaced state 형성
# 'cavity_drive_amp'/(2*'alpha_1_cavity_drive_amp') 사이즈에 해당하는 Re-Im plane plot in case of wigner

my_run.wigner_characteristic_function_2D(average_exponent=10, npts_x = 11, npts_y = 21, amplitude = 1, #amplitude for coherent state
                          is_wigner_function=False,
                          is_coherent_state = True, is_schrodinger_cat_state= False, 
                          is_schrodinger_cat_state_2=False,
                          is_cat_state=False,
                          is_cat_state_2=False,
                          acquire_delay=480e-9,
                          alpha=0.25, beta= 0.1,
                          is_plot_simulation=True)
my_run.plot_wigner_characteristic_function_measurement_2D(vmax = 0.68, vmin = -1.34, threshold = -0.21,
                                                          is_g_smaller_than_e=True, is_plot_G=False, is_plot_E=False,
                                                          is_wo_post_selection=True)


# In[] without post selection

my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.17 # sweep variable
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.17
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.2 * 4
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.2

# 'cavity_drive_amp' * amplitude에 해당하는 displaced state 형성
# 'cavity_drive_amp'/(2*'alpha_1_cavity_drive_amp') 사이즈에 해당하는 Re-Im plane plot in case of wigner

my_run._wigner_characteristic_function_2D(average_exponent=10, npts_x = 11, npts_y = 21, amplitude = 1, #amplitude for coherent state
                          wait_length = 520e-9,
                          is_wigner_function=False,
                          is_coherent_state = True, is_schrodinger_cat_state= False, 
                          is_schrodinger_cat_state_2=False,
                          is_cat_state=False,
                          alpha=0.25, beta= 0.1,
                          is_autochunking=True,
                          is_plot_simulation=True)
my_run._plot_wigner_characteristic_function_measurement_2D(vmax = -0.4, vmin = 1.0)


# In[]
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.046
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.046*4

my_run.disentangling_power_sweep(average_exponent=11,
                                  alpha = 1,
                                  beta_sweep_start = -0.5,
                                  beta_sweep_stop = 0.5,
                                  beta_sweep_count = 21,
                                  is_xyz=True)
my_run.plot_disentangling_power_sweep()



# In[]

my_run.calibrate_sideband_pulse_phase(average_exponent=10, sidebands_pulse_length=10e-6, rabi_pulse_length=10e-6,
                                       npts_phase_sweep=21, is_sideband_phase_sweep = False, # else rabi phase sweep
                                       auto_chunking = False,
                                       is_init_qubit_pi2=False,
                                       is_plot_simulation = True)

my_run.plot_calibrate_sideband_pulse_phase()


















# In[]

my_run.continuous_wave(average_exponent=19, freq_l=10e6, freq_h=10e6, amp_l=0.5, amp_h=0.5, is_sideband_pulse = True, is_plot_simulation=True)






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

pulse = pulse_update(
            _pulse,
            spectral_window = None,
            flip_angle = None,
            pulse_parameters=_pulse.pulse_parameters,
        )

my_run.analyze_pulse(_pulse, high_res = True)




# %%
