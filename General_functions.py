# In[]

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time as time_module
import scipy.optimize
import qutip as qt
from scipy.optimize import minimize

import sys
sys.path.append("D:/Software/SHFQC/")
from smart_fit import sFit

# Helpers:
from laboneq.contrib.example_helpers.generate_device_setup import (
    generate_device_setup_qubits,
)

# LabOne Q:
from laboneq.simple import *
from laboneq.dsl.experiment.builtins import *
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_results
from datetime import datetime
from laboneq.contrib.example_helpers.randomized_benchmarking_helper import (
    clifford_parametrized,
    generate_play_rb_pulses,
    make_pauli_gate_map,
)
# from laboneq_applications.analysis.fitting_helpers import exponential_decay_fit
from laboneq.contrib.bloch_simulator_pulse_plotter.inspector.update_inspect import (
    pulse_update,
)


# In[] device_setup calibration based on initial parameters

class ZI_QCCS(object):

    def __init__(self, physical_ports, qubits_parameters, cavity_parameters,
                 number_of_qubits,
                 number_of_memory_modes,
                 which_qubit, 
                 which_mode,
                 which_data, 
                 use_emulation,
                 multiplex_drive_lines = True,
                 cr_drive_lines = True,
                 is_memory_mode = True,
                 is_cavity_trajectory_weighting_function = False):

        # specify the number of qubits you want to use

        if (number_of_qubits + number_of_memory_modes) > 6:
            raise ValueError("Number of qubits and memory modes should be less than or equal to 6") 
        

        # generate the device setup and the qubit objects using a helper function - remove either the UHFQA or the SHFQC to make this work
        self.device_setup, self.qubits = generate_device_setup_qubits(
            number_qubits=number_of_qubits,
            number_memory_modes = number_of_memory_modes,
            
            shfqc=[
                {
                    "serial": "DEV12256",
                    "zsync": 1,
                    "number_of_channels": number_of_qubits + number_of_memory_modes,
                    "readout_multiplex": number_of_qubits,
                    "options": 'SHFQC/PLUS/QC6CH',

                }
            ],
            multiplex_drive_lines=multiplex_drive_lines,
            cr_drive_lines=cr_drive_lines,
            is_memory_mode=is_memory_mode,

            include_flux_lines=False,
            server_host="localhost",
            setup_name=f"my_{number_of_qubits}_fixed_qubit_setup",
        )
        # use emulation mode - no connection to instruments
        use_emulation = use_emulation

        # create and connect to a LabOne Q session
        self.session = Session(device_setup=self.device_setup, server_log=True)
        self.session.disconnect()
        self.session.connect(do_emulation=use_emulation)

        self.physical_ports = physical_ports
        self.qubits_parameters = qubits_parameters
        self.cavity_parameters = cavity_parameters

        self.multiplex_drive_lines = multiplex_drive_lines
        self.cr_drive_lines = cr_drive_lines
        self.is_memory_mode = is_memory_mode

        self.which_qubit = which_qubit # which qubit to use "0 is 'q0', 1 is 'q1' and so on."
        self.which_mode = which_mode
        self.which_data = which_data # which data to use "I" or "Q"

        self.is_cavity_trajectory_weighting_function = is_cavity_trajectory_weighting_function

        self.device_setup_calibration()

    def close_session(self):
        self.session.disconnect()

# In[] device_setup calibration based on initial parameters

    def device_setup_calibration(self):

        physical_ports = self.physical_ports
        qubits_parameters = self.qubits_parameters
        if self.is_memory_mode:
            cavity_parameters = self.cavity_parameters
        device_setup = self.device_setup

        def new_oscillator(port, type, modulation_type = None, sg = None):
            oscillator = Oscillator()

            if sg is None:
                oscillator.uid = f"{port}" +f"_{type}" + "_osc"
            else :
                oscillator.uid = f"{port}" +f"_{type}" + f"_{sg}" + "_osc"
            oscillator.modulation_type = modulation_type

            if type == "lo":
                if port == "drive":
                    oscillator.frequency = physical_ports[port][sg]["freq_LO"]
                else :
                    oscillator.frequency = physical_ports[port]["freq_LO"]

            elif type == "if":
                if port == "measure":
                    freq = qubits_parameters[sg]["readout_frequency"] - physical_ports["measure"]["freq_LO"]
                    self.qubits_parameters[sg]["readout_freq_IF"] = freq
                elif port == "acquire":
                    freq = qubits_parameters[sg]["readout_frequency"] - physical_ports["acquire"]["freq_LO"]
                elif port == "drive":
                    freq = qubits_parameters[sg]["ge_frequency"] - physical_ports["drive"][sg]["freq_LO"]
                    self.qubits_parameters[sg]["ge_freq_IF"] = freq
                elif port == "drive_ef":
                    freq = qubits_parameters[sg]["ef_frequency"] - physical_ports["drive"][sg]["freq_LO"]
                    self.qubits_parameters[sg]["ef_freq_IF"] = freq
            
                elif port == "control_qubit_drive":
                    
                    if qubits_parameters[sg]["is_cr"]:
                        freq = qubits_parameters[sg]["cr"]["control_qubit_frequency"] - physical_ports["drive"][sg]["freq_LO"]
                        self.qubits_parameters[sg]["cr"]["control_qubit_frequency_IF"] = freq
                        
                elif port == "target_drive_line_1":
                    if qubits_parameters[sg]["is_cr"]:
                        freq = qubits_parameters[sg]["cr"]["target_qubit_frequency_1"] - physical_ports["drive"][sg]["freq_LO"]
                        self.qubits_parameters[sg]["cr"]["target_qubit_frequency_1_IF"] = freq
                        
                elif port == "target_drive_line_2":
                    if qubits_parameters[sg]["is_cr"]:
                        freq = qubits_parameters[sg]["cr"]["target_qubit_frequency_2"] - physical_ports["drive"][sg]["freq_LO"]
                        self.qubits_parameters[sg]["cr"]["target_qubit_frequency_2_IF"] = freq

                elif port == "m0_drive":
                    freq = cavity_parameters[sg]["mode_frequency"] - physical_ports["drive"][sg]["freq_LO"]
                    self.cavity_parameters[sg]["m0_freq_IF"] = freq
                
                elif port == "m1_drive":
                    freq = cavity_parameters[sg]["mode_frequency"] - physical_ports["drive"][sg]["freq_LO"]
                    self.cavity_parameters[sg]["m1_freq_IF"] = freq

                elif port == "m2_drive":
                    freq = cavity_parameters[sg]["mode_frequency"] - physical_ports["drive"][sg]["freq_LO"]
                    self.cavity_parameters[sg]["m2_freq_IF"] = freq

                else:
                    raise ValueError("Invalid port")
                
                if (freq > 0) and (freq < 500e6):
                    oscillator.frequency = freq
                    
                else :
                    raise ValueError("Invalid IF frequency")
            else:
                raise ValueError("Invalid oscillator type")
            return oscillator
        
        calibration = Calibration()

        device_setup.set_calibration(calibration)
        
        #### Calibration for LO and physical ports ##########################################

        for port in physical_ports:
            for component in qubits_parameters:
            ## only 1 port for measure and acquire line 
                if port == "measure":
                    calibration[device_setup.logical_signal_groups[component].logical_signals["measure"]] = SignalCalibration(
                            oscillator = new_oscillator(port, "if", modulation_type=ModulationType.SOFTWARE, sg = component),
                            local_oscillator = new_oscillator(port, "lo", modulation_type=ModulationType.AUTO),
                            range = physical_ports[port]["range"],
                            amplitude = physical_ports[port]["amplitude"],
                            port_delay = physical_ports[port]["port_delay"],
                            port_mode = physical_ports[port]["port_mode"],
                            delay_signal = physical_ports[port]["delay_signal"],
                            threshold = physical_ports[port]["threshold"],
                            added_outputs = physical_ports[port]["added_outputs"],
                            automute = physical_ports[port]["automute"],
                        )
                elif port == "acquire":
                    calibration[device_setup.logical_signal_groups[component].logical_signals["acquire"]] = SignalCalibration(
                            oscillator = new_oscillator(port, "if", modulation_type=ModulationType.SOFTWARE, sg = component),
                            local_oscillator = new_oscillator(port, "lo", modulation_type=ModulationType.AUTO),
                            range = physical_ports[port]["range"],
                            amplitude = physical_ports[port]["amplitude"],
                            port_delay = physical_ports[port]["port_delay"],
                            port_mode = physical_ports[port]["port_mode"],
                            delay_signal = physical_ports[port]["delay_signal"],
                            threshold = physical_ports[port]["threshold"],
                            added_outputs = physical_ports[port]["added_outputs"],
                            automute = physical_ports[port]["automute"],
                        )
                elif port == "drive":

                    calibration[device_setup.logical_signal_groups[component].logical_signals["drive"]] = SignalCalibration(
                        oscillator = new_oscillator(port, "if", modulation_type=ModulationType.HARDWARE, sg = component),
                        local_oscillator = new_oscillator(port, "lo", modulation_type=ModulationType.AUTO, sg=component),
                        range = physical_ports[port][component]["range"],
                        amplitude = physical_ports[port][component]["amplitude"],
                        port_delay = physical_ports[port][component]["port_delay"],
                        port_mode = physical_ports[port][component]["port_mode"],
                        delay_signal = physical_ports[port][component]["delay_signal"],
                        threshold = physical_ports[port][component]["threshold"],
                        added_outputs = physical_ports[port][component]["added_outputs"],
                        automute = physical_ports[port][component]["automute"],
                        voltage_offset= physical_ports[port][component]["voltage_offset"],
                    )

                    ## Once physical port sets, no more required, it will be overwritten by the next calibration

                    if self.multiplex_drive_lines:

                        calibration[device_setup.logical_signal_groups[component].logical_signals["drive_line_ef"]] = SignalCalibration(
                            oscillator = new_oscillator("drive_ef", "if", modulation_type=ModulationType.HARDWARE, sg = component),
                        )
                    
                    if self.cr_drive_lines:

                        calibration[device_setup.logical_signal_groups[component].logical_signals["control_drive_line"]] = SignalCalibration(
                            oscillator = new_oscillator("control_qubit_drive", "if", modulation_type=ModulationType.HARDWARE, sg = component),
                        )
                        calibration[device_setup.logical_signal_groups[component].logical_signals["target_drive_line_1"]] = SignalCalibration(
                            oscillator = new_oscillator("target_drive_line_1", "if", modulation_type=ModulationType.HARDWARE, sg = component),
                        )
                        calibration[device_setup.logical_signal_groups[component].logical_signals["target_drive_line_2"]] = SignalCalibration(
                            oscillator = new_oscillator("target_drive_line_2", "if", modulation_type=ModulationType.HARDWARE, sg = component),
                        )
                    
                else:
                    raise ValueError("Invalid port")
        
        if self.is_memory_mode:

            for port in physical_ports:
                for component in cavity_parameters:

                    if port == "drive":

                        calibration[device_setup.logical_signal_groups[component].logical_signals["cavity_drive_line"]] = SignalCalibration(
                            oscillator = new_oscillator(f"{component}_drive", "if", modulation_type=ModulationType.HARDWARE, sg = component),
                            local_oscillator = new_oscillator(port, "lo", modulation_type=ModulationType.AUTO, sg=component),
                            range = physical_ports[port][component]["range"],
                            amplitude = physical_ports[port][component]["amplitude"],
                            port_delay = physical_ports[port][component]["port_delay"],
                            port_mode = physical_ports[port][component]["port_mode"],
                            delay_signal = physical_ports[port][component]["delay_signal"],
                            threshold = physical_ports[port][component]["threshold"],
                            added_outputs = physical_ports[port][component]["added_outputs"],
                            automute = physical_ports[port][component]["automute"],
                            voltage_offset= physical_ports[port][component]["voltage_offset"],
                        )

        device_setup.set_calibration(calibration)
# In[]
    def IF_demodulation(self, I_data, Q_data, IF_freq):

        I_data = np.array(I_data)
        Q_data = np.array(Q_data)

        coswt = np.cos(2 * np.pi * IF_freq * np.arange(len(I_data))/2 * 1e-9) # reason of 2 ; 2GHz sampling rate
        sinwt = np.sin(2 * np.pi * IF_freq * np.arange(len(Q_data))/2 * 1e-9) 

        # demodulate the data
        I_demod = I_data * coswt + Q_data * sinwt
        Q_demod = -I_data * sinwt + Q_data * coswt

        return I_demod, Q_demod
    
    

# In[] simulation plot

    def simulation_plot(self, compiled_experiment, start_time, length, component = None, is_snippet = False):  

        plot_simulation(
            compiled_experiment,
            start_time=start_time,
            length=length,
            plot_width=10,
            plot_height=3,
        )

        if is_snippet: # Not completed yet
            # Get physical channel references via the logical signals
            drive_iq_port = self.device_setup.logical_signal_by_uid(
                component + "/drive"
            ).physical_channel
            measure_iq_port = self.device_setup.logical_signal_by_uid(
                component + "/measure"
            ).physical_channel
            acquire_port = self.device_setup.logical_signal_by_uid(
                component + "/acquire"
            ).physical_channel

            # Get waveform snippets from the simulation
            simulation = OutputSimulator(compiled_experiment)

            drive_snippet = simulation.get_snippet(drive_iq_port, start=0, output_length=250e-9)

            measure_snippet = simulation.get_snippet(
                measure_iq_port, start=200e-9, output_length = self.parameters[component]['readout']["user_defined"]["pulse_length"]
            )

            acquire_snippet = simulation.get_snippet(
                acquire_port, start=200e-9, output_length = self.parameters[component]['readout']["user_defined"]["pulse_length"]
            )

            fig = plt.figure(figsize=(15, 5))
            plt.plot(drive_snippet.time, drive_snippet.wave.real, label="drive I")
            plt.plot(drive_snippet.time, drive_snippet.wave.imag, label="drive Q")
            plt.plot(measure_snippet.time, measure_snippet.wave.real, label="measure I")
            plt.plot(measure_snippet.time, measure_snippet.wave.imag, label="measure Q")
            plt.plot(acquire_snippet.time, acquire_snippet.wave.real, label="acquire start")
            plt.legend()
            plt.show()

# In[]
    # It needs to be updated for multiqubit handling
    def signal_map(self, component, which_qubit = None):
        device_setup = self.device_setup

        if which_qubit is None:
            signal_map = {
                    "measure": device_setup.logical_signal_groups[component].logical_signals["measure"],
                    "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire"],
                    "drive": device_setup.logical_signal_groups[component].logical_signals["drive"],
                }
        elif which_qubit == "control" :
            signal_map = {
                    "control_drive": device_setup.logical_signal_groups[component].logical_signals["control_drive_line"],
                    "target_drive_1": device_setup.logical_signal_groups[component].logical_signals["target_drive_line_1"],
                }
        # elif which_qubit == "target" :
        #     signal_map = {
        #             "measure": device_setup.logical_signal_groups[component].logical_signals["measure_line"],
        #             "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire_line"],
        #         }
        return signal_map

# In[]
    def continuous_wave(self, is_sideband_pulse = False, average_exponent=19, freq_l=10e6, freq_h=10e6, amp_l=1.0, amp_h=1.0, phase = 0, amp_cont=0.5,
                        is_plot_simulation = False):

        device_setup = self.device_setup    
        component = list(self.qubits_parameters.keys())[self.which_qubit]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        drive_pulse = pulse_library.const(uid="drive_pulse", 
                                            length=40e-6, 
                                            amplitude=amp_cont)
        
        sideband_pulse = pulse_library.sidebands_pulse(uid = "sideband_pulse", 
                                                       length=10e-6,
                                                        frequency_l = freq_l,
                                                        frequency_h = freq_h,
                                                        amp_l = amp_l,
                                                        amp_h = amp_h,
                                                        phase = phase)
        

        if is_sideband_pulse:
            pulse = sideband_pulse
        else :
            pulse = drive_pulse

        exp_cont_wave = Experiment(
                uid="cont_wave",
                signals=[
                    ExperimentSignal(uid="cavity_drive"),
                ],
            )
            
        with exp_cont_wave.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
        ):
            with exp_cont_wave.section(uid="cavity_drive"):
                exp_cont_wave.play(signal="cavity_drive", pulse=pulse, length= 40e-6)
                
              
        signal_map = {
                "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
            }
        
        exp_cont_wave.set_signal_map(signal_map)

        compiled_exp_cont_wave = self.session.compile(exp_cont_wave)
        self.session.run(compiled_exp_cont_wave)

        if is_plot_simulation:
            self.simulation_plot(compiled_exp_cont_wave, start_time=0, length=20e-6)


# In[]
    def analyze_pulse(self, pulse, high_res = False):
        
        pulse = pulse_update(
            pulse,
            spectral_window = None,
            flip_angle = None,
            pulse_parameters=pulse.pulse_parameters,
        )
        
        if high_res:
            added_npts = len(pulse.t)
            array_for_high_res = np.zeros(added_npts)
            
            pulse_I = np.append(array_for_high_res, np.append(pulse.i_wave, array_for_high_res))
            pulse_Q = np.append(array_for_high_res, np.append(pulse.q_wave, array_for_high_res))
            
            time_step = pulse.t[-1]/(len(pulse.t)-1)
            pulse_t = np.linspace(-added_npts*time_step, pulse.t[-1] + added_npts*time_step, len(pulse_I))
        else:
            pulse_I = pulse.i_wave
            pulse_Q = pulse.q_wave
            pulse_t = pulse.t
            
        fft_pulse_I = np.fft.fft(pulse_I)
        fft_pulse_Q = np.fft.fft(pulse_Q)
        
        fft_freqs = np.fft.fftfreq(len(pulse_t), pulse_t[1] - pulse_t[0])
        
        fig, axs = plt.subplots(2,2)
        fig.suptitle(r"$\mathcal{I}$ and $\mathcal{Q}$")

        ylab1 = [r"I", r"Q"]

        for i in range(2):
            axs[i,0].set_ylabel(ylab1[i])
            axs[i,1].set_ylabel(rf"$|F({ylab1[i]})|$")

        axs[0,0].plot(pulse_t * 1e9, pulse_I)
        axs[0,0].set(xlabel = "Time (ns)")
        
        axs[0,1].plot(fft_freqs * 1e-6, np.abs(fft_pulse_I))
        axs[0,1].set(xlabel = "Frequency (MHz)")

        axs[1,0].plot(pulse_t * 1e9, pulse_Q)
        axs[1,0].set(xlabel = "Time (ns)")
        
        axs[1,1].plot(fft_freqs * 1e-6, np.abs(fft_pulse_Q))
        axs[1,1].set(xlabel = "Frequency (MHz)") 

        plt.show()
        
    # %% propagation delay calibration for readout line

    def prop_delay_calibration(self, line, average_exponent, readout_pulse_type = "const"):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        
        
        ## define pulses used for experiment

        if readout_pulse_type == "const":

            readout_pulse = pulse_library.const(
                uid="readout_pulse", 
                length=qubits_parameters[component]["readout_pulse_length"], 
                amplitude=qubits_parameters[component]["readout_amp"], 
            )

        elif readout_pulse_type == "drachma":
            readout_pulse = pulse_library.drachma_readout_pulse(
                uid="drachma_readout_pulse",
                length = qubits_parameters[component]["readout_pulse_length"],
                amplitude=qubits_parameters[component]["readout_amp"],
                kappa=qubits_parameters[component]["readout_kappa"],
                chi_list=[-qubits_parameters[component]["readout_chi"]/2, qubits_parameters[component]["readout_chi"]/2],
                zeta_list = [0, 0],
            )

        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.const(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )

        if line == "readout":
        # Create Experiment & Define experiment signals
            exp_prop_delay = Experiment(
                uid="Optimal weights",
                signals=[
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
            
            with exp_prop_delay.acquire_loop_rt(
                uid="shots",
                count=pow(2, average_exponent),
                averaging_mode=AveragingMode.CYCLIC,
                acquisition_type=AcquisitionType.RAW,
            ):
                # qubit readout and data acquisition
                with exp_prop_delay.section(uid="measure_line_calibration"):
                    exp_prop_delay.play(signal="measure", pulse=readout_pulse)
                    exp_prop_delay.acquire(
                        signal="acquire", handle="ac_0", kernel=readout_weighting_function
                    )
                # relax time after readout - for signal processing and qubit relaxation to ground state
                with exp_prop_delay.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_prop_delay.reserve(signal="measure")

            # experiment signal - logical signal mapping
            signal_map = {
                "measure": device_setup.logical_signal_groups[component].logical_signals["measure"],
                "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire"],
            }
        
        elif line == "drive":

            exp_prop_delay = Experiment(
                uid="Optimal weights",
                signals=[
                    ExperimentSignal(uid="drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
            
            with exp_prop_delay.acquire_loop_rt(
                uid="shots",
                count=pow(2, average_exponent),
                averaging_mode=AveragingMode.CYCLIC,
                acquisition_type=AcquisitionType.RAW,
            ):
                # qubit readout and data acquisition
                with exp_prop_delay.section(uid="drive_line_calibration"):
                    exp_prop_delay.play(signal="drive", pulse=readout_pulse)
                    exp_prop_delay.play(signal="measure", pulse=readout_pulse, amplitude=0) # amplitude = 0 to avoid interference
                    exp_prop_delay.acquire(
                        signal="acquire", handle="ac_0", kernel=readout_weighting_function # can be acquired only there is a measure signal
                    )
                # relax time after readout - for signal processing and qubit relaxation to ground state
                with exp_prop_delay.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_prop_delay.reserve(signal="drive")

            # experiment signal - logical signal mapping
            signal_map = {
                "drive": device_setup.logical_signal_groups[component].logical_signals["drive"],
                "measure": device_setup.logical_signal_groups[component].logical_signals["measure"],
                "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire"],
            }

            ## send readout pulse through drive port to check flight time

            exp_calibration = Calibration()
            # sets the oscillator of the experimental measure signal
            # for spectroscopy, set the sweep parameter as frequency
            readout_if_oscillator = Oscillator(
                "readout_if_osc",
                frequency=self.qubits_parameters[component]["readout_freq_IF"],
                modulation_type=ModulationType.SOFTWARE,
            )
            readout_lo_oscillator = Oscillator(
                "readout_lo_osc",
                frequency=self.physical_ports["measure"]["freq_LO"],
            )

            exp_calibration["drive"] = SignalCalibration( # experimental signal line 이름으로 signal calibration : 해당 실험 일시적 적용
                oscillator=readout_if_oscillator,
                local_oscillator=readout_lo_oscillator,
            )

            exp_prop_delay.set_calibration(exp_calibration)

        elif line == "cavity_drive":

            exp_prop_delay = Experiment(
                uid="Optimal weights",
                signals=[
                    ExperimentSignal(uid="cavity_drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
            
            with exp_prop_delay.acquire_loop_rt(
                uid="shots",
                count=pow(2, average_exponent),
                averaging_mode=AveragingMode.CYCLIC,
                acquisition_type=AcquisitionType.RAW,
            ):
                # qubit readout and data acquisition
                with exp_prop_delay.section(uid="drive_line_calibration"):
                    exp_prop_delay.play(signal="cavity_drive", pulse=readout_pulse)
                    exp_prop_delay.play(signal="measure", pulse=readout_pulse, amplitude=0) # amplitude = 0 to avoid interference
                    exp_prop_delay.acquire(
                        signal="acquire", handle="ac_0", kernel=readout_weighting_function # can be acquired only there is a measure signal
                    )
                # relax time after readout - for signal processing and qubit relaxation to ground state
                with exp_prop_delay.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_prop_delay.reserve(signal="cavity_drive")

            # experiment signal - logical signal mapping
            signal_map = {
                "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
                "measure": device_setup.logical_signal_groups[component].logical_signals["measure"],
                "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire"],
            }

            ## send readout pulse through drive port to check flight time

            exp_calibration = Calibration()
            # sets the oscillator of the experimental measure signal
            # for spectroscopy, set the sweep parameter as frequency
            readout_if_oscillator = Oscillator(
                "readout_if_osc",
                frequency=self.qubits_parameters[component]["readout_freq_IF"],
                modulation_type=ModulationType.SOFTWARE,
            )
            readout_lo_oscillator = Oscillator(
                "readout_lo_osc",
                frequency=self.physical_ports["measure"]["freq_LO"],
            )

            exp_calibration["cavity_drive"] = SignalCalibration( # experimental signal line 이름으로 signal calibration : 해당 실험 일시적 적용
                oscillator=readout_if_oscillator,
                local_oscillator=readout_lo_oscillator,
            )

            exp_prop_delay.set_calibration(exp_calibration)

        exp_prop_delay.set_signal_map(signal_map)

        results = self.session.run(exp_prop_delay)
        raw = results.get_data("ac_0")
        
        time = np.linspace(0, len(raw) / 2, len(raw)) # 2GSa/s acquisition rate
        
        I_demod, Q_demod = self.IF_demodulation(np.real(raw), np.imag(raw), self.qubits_parameters[component]["readout_freq_IF"])

        plt.figure()

        plt.plot(time, I_demod, "b")
        plt.plot(time, Q_demod, "r")

        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude (a.u.)")

        self.save_results(experiment_name = "prop_delay_calibration")

        plt.show()

#################### Frequency Domain Measurements #######################################################################################
# In[]

    def resonator_spectroscopy(self, 
                               average_exponent = 10, 
                               start_freq = -10e6, 
                               stop_freq = 10e6, 
                               npts = 101, 
                               integration_time = 1e-3,
                               is_plot_simulation = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        freq_sweep = LinearSweepParameter(uid="res_freq", start=start_freq, stop=stop_freq, count=npts)

        exp_spec = Experiment(
            uid="Resonator Spectroscopy",
            signals=[
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
                ExperimentSignal("drive"),
            ],
        )

        ## define experimental sequence
        # loop - average multiple measurements for each frequency - measurement in spectroscopy mode
        with exp_spec.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp_spec.sweep(uid="res_freq", parameter=freq_sweep):
                # readout pulse and data acquisition
                with exp_spec.section(uid="spectroscopy"):
                    # resonator signal readout
                    exp_spec.acquire(
                        signal="acquire",
                        handle="res_spec",
                        length=integration_time,
                    )
                # relax time after readout - for signal processing and qubit relaxation to ground state
                with exp_spec.section(uid="relax", length=1e-6):
                    exp_spec.reserve(signal="measure")
                    exp_spec.reserve(signal="acquire")

        signal_map = self.signal_map(component)

        exp_spec.set_signal_map(signal_map)

        compiled_res_spec = self.session.compile(exp_spec)

        res_spec_results = self.session.run(compiled_res_spec)

        self.res_spec = res_spec_results.get_data("res_spec")

    def plot_resonator_spectroscopy(self):

        res_spec = self.res_spec

        freqs = res_spec.frequency
        mags = res_spec.magnitude
        phases = res_spec.phase

        fig, ax = plt.subplots(2, 1, figsize=(20, 20))

        ax[0].plot(freqs * 1e-6, mags, "b")
        ax[0].set_xlabel("Frequency (MHz)", fontsize=20)
        ax[0].set_ylabel("Magnitude (a.u.)", fontsize=20)
        ax[0].tick_params(axis='both', which='major', labelsize=20)
        ax[0].tick_params(axis='both', which='minor', labelsize=15)

        ax[1].plot(freqs * 1e-6, phases, "r")
        ax[1].set_xlabel("Frequency (MHz)", fontsize=20)
        ax[1].set_ylabel("Phase (deg)", fontsize=20)
        ax[1].tick_params(axis='both', which='major', labelsize=20)
        ax[1].tick_params(axis='both', which='minor', labelsize=15)

        plt.show()

##################### useful functions ################################################################################################
# In[] useful functions

    def hist_fidelity(self, hist1, hist2):

        mean1 = np.mean(hist1)
        mean2 = np.mean(hist2)
        thresh = np.mean([mean1, mean2])
        fdlty=1
        
        for i in range(len(hist1)):
            if mean1 > mean2:
                if hist1[i] <= thresh: fdlty = fdlty - 1/(2*len(hist1))
                if hist2[i] >= thresh: fdlty = fdlty - 1/(2*len(hist2))
            else:
                if hist1[i] >= thresh: fdlty = fdlty - 1/(2*len(hist1))
                if hist2[i] <= thresh: fdlty = fdlty - 1/(2*len(hist2))

        return fdlty, thresh

    def gaussian(self, x, a, x0, sigma, offset):

        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

    def double_gaussian(self, x, a1, x01, sigma1, a2, x02, sigma2):

        return a1 * np.exp(-((x - x01) ** 2) / (2 * sigma1 ** 2)) + a2 * np.exp(-((x - x02) ** 2) / (2 * sigma2 ** 2))
    
    def cosine(self, x, amp, freq, phi, offset):

        return amp * np.cos(freq * x + phi) + offset

# In[]
    def pulse_generator(self, type, qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component, length = None, npts = None):

        if type == "readout":

            readout_pulse = pulse_library.gaussian_square(
                uid="readout_pulse",
                length=qubits_parameters[qubits_component]["readout_pulse_length"],
                amplitude=qubits_parameters[qubits_component]["readout_amp"]
            )

            if self.is_cavity_trajectory_weighting_function:

                readout_weighting_function = pulse_library.integration_weight_gaussian_const(
                    uid="readout_weighting_function", 
                    length=qubits_parameters[qubits_component]["readout_integration_length"],
                    chi=qubits_parameters[qubits_component]["readout_chi"],
                    kappa=qubits_parameters[qubits_component]["readout_kappa"],
                    amplitude=qubits_parameters[qubits_component]["readout_integration_amp"], 
                )
            else:
                readout_weighting_function = pulse_library.gaussian_square(
                    uid="readout_weighting_function",
                    length=qubits_parameters[qubits_component]["readout_integration_length"],
                    amplitude=qubits_parameters[qubits_component]["readout_integration_amp"]
                )

            return readout_pulse, readout_weighting_function
        
        elif type == "special_readout":

            drachma_readout_pulse = pulse_library.drachma_readout_pulse(
                uid="drachma_readout_pulse",
                length = qubits_parameters[qubits_component]["drachma_readout_pulse_length"],
                amplitude=qubits_parameters[qubits_component]["drachma_readout_amp"],
                kappa=qubits_parameters[qubits_component]["readout_kappa"],
                chi_list=[-qubits_parameters[qubits_component]["readout_chi"]/2, qubits_parameters[qubits_component]["readout_chi"]/2],
                zeta_list = [0,0],
            )

            drachma_readout_weighting_function = pulse_library.drachma_readout_weighting_pulse(
                uid="readout_weighting_function", 
                length=qubits_parameters[qubits_component]["drachma_readout_integration_length"],
                amplitude=qubits_parameters[qubits_component]["readout_integration_amp"], 
            )

            return drachma_readout_pulse, drachma_readout_weighting_function
        
        elif type == "qubit_control":

            pi2_pulse = pulse_library.drag(
                uid="pi2_pulse",
                length=qubits_parameters[qubits_component]["pi2_length"],
                amplitude=qubits_parameters[qubits_component]["pi2_amp"],
                beta=qubits_parameters[qubits_component]["pi2_beta"]
            )

            pi_pulse = pulse_library.drag(
                uid="pi_pulse",
                length=qubits_parameters[qubits_component]["pi_length"],
                amplitude=qubits_parameters[qubits_component]["pi_amp"],
                beta=qubits_parameters[qubits_component]["pi_beta"]
            )

            cond_pi_pulse = pulse_library.drag(
                uid="cond_pi_pulse",
                length=qubits_parameters[qubits_component]["cond_pi_length"],
                amplitude=qubits_parameters[qubits_component]["cond_pi_amp"],
                beta=qubits_parameters[qubits_component]["cond_pi_beta"]
            )

            return pi2_pulse, pi_pulse, cond_pi_pulse
        
        elif type == "cavity_control":

            cond_disp_pulse = pulse_library.cond_disp_pulse(
                uid="cond_disp_pulse",
                length=cavity_parameters[cavity_component]["cond_disp_pulse_length"],
                amplitude=cavity_parameters[cavity_component]["cond_disp_pulse_amp"],
                sigma=cavity_parameters[cavity_component]["cond_disp_pulse_sigma"],
                frequency=cavity_parameters[cavity_component]["cond_disp_pulse_frequency"],
                detuning=cavity_parameters[cavity_component]["cond_disp_pulse_detuning"],
            )

            cavity_drive_pulse = pulse_library.gaussian(
                uid="cavity_drive_pulse",
                length=cavity_parameters[cavity_component]["cavity_drive_length"],
                amplitude=cavity_parameters[cavity_component]["cavity_drive_amp"]
            )

            sideband_att_h = cavity_parameters[cavity_component]["sideband_att_h"]/np.sqrt(cavity_parameters[cavity_component]["sideband_att_h"]**2 + cavity_parameters[cavity_component]["sideband_att_l"]**2)
            sideband_att_l = cavity_parameters[cavity_component]["sideband_att_l"]/np.sqrt(cavity_parameters[cavity_component]["sideband_att_h"]**2 + cavity_parameters[cavity_component]["sideband_att_l"]**2)
            
            sidebands_drive_pulse_chunk = pulse_library.sidebands_pulse(
                uid="sidebands_drive_pulse",
                length=cavity_parameters[cavity_component]["sideband_chunk_length"],
                frequency_l=cavity_parameters[cavity_component]["sideband_frequency_l"],
                frequency_h=cavity_parameters[cavity_component]["sideband_frequency_h"],
                amp_l=cavity_parameters[cavity_component]["sideband_amp_l"]*sideband_att_h,
                amp_h=cavity_parameters[cavity_component]["sideband_amp_h"]*sideband_att_l,
                phase=cavity_parameters[cavity_component]["sideband_phase"],
                extra_phase = cavity_parameters[cavity_component]["sideband_extra_phase"],
            )

            sidebands_drive_gaussian_rise = pulse_library.sidebands_pulse(
                uid="sidebands_drive_gaussian_rise",
                length=cavity_parameters[cavity_component]["sideband_rise_fall_length"],
                frequency_l=cavity_parameters[cavity_component]["sideband_frequency_l"],
                frequency_h=cavity_parameters[cavity_component]["sideband_frequency_h"],
                amp_l=cavity_parameters[cavity_component]["sideband_amp_l"]*sideband_att_h,
                amp_h=cavity_parameters[cavity_component]["sideband_amp_h"]*sideband_att_l,
                phase=cavity_parameters[cavity_component]["sideband_phase"],
                is_gauss_rise = True,
            )

            sidebands_drive_gaussian_fall = pulse_library.sidebands_pulse(
                uid="sidebands_drive_gaussian_fall",
                length=cavity_parameters[cavity_component]["sideband_rise_fall_length"],
                frequency_l=cavity_parameters[cavity_component]["sideband_frequency_l"],
                frequency_h=cavity_parameters[cavity_component]["sideband_frequency_h"],
                amp_l=cavity_parameters[cavity_component]["sideband_amp_l"]*sideband_att_h,
                amp_h=cavity_parameters[cavity_component]["sideband_amp_h"]*sideband_att_l,
                phase=cavity_parameters[cavity_component]["sideband_phase"],
                is_gauss_fall = True,
            )

            if length is None:
                length = cavity_parameters[cavity_component]["cavity_drive_length"]
            else :
                length = length/npts

            cavity_drive_pulse_constant_chunk = pulse_library.gaussian_square(
                uid="cavity_drive_pulse",
                length=length,
                amplitude=cavity_parameters[cavity_component]["cavity_drive_amp"]
            )
            

            return cond_disp_pulse, cavity_drive_pulse, sidebands_drive_pulse_chunk, \
                   sidebands_drive_gaussian_rise, sidebands_drive_gaussian_fall, cavity_drive_pulse_constant_chunk
        
        elif type == "rabi":

            rabi_drive_chunk = pulse_library.const(uid="drive_pulse", 
                                                length=length, 
                                                amplitude = qubits_parameters[qubits_component]["rabi_drive_amp"])
            
            rabi_drive = pulse_library.gaussian_square(uid="drive_pulse", 
                                                length = length,
                                                zero_boundaries=True,
                                                amplitude = qubits_parameters[qubits_component]["rabi_drive_amp"])
            
            rabi_ramp_up = pulse_library.gaussian_rise(uid="ramp_up", 
                                        length=qubits_parameters[qubits_component]["ramp_length"], 
                                        amplitude=qubits_parameters[qubits_component]["rabi_drive_amp"])
            
            rabi_ramp_down = pulse_library.gaussian_fall(uid="ramp_down", 
                                            length=qubits_parameters[qubits_component]["ramp_length"], 
                                            amplitude=qubits_parameters[qubits_component]["rabi_drive_amp"])
            
            return rabi_drive_chunk, rabi_drive, rabi_ramp_up, rabi_ramp_down

        else:
            raise ValueError("Invalid pulse type. Choose from 'readout', 'qubit_control', or 'cavity_control'.")
        
# In[]

    def data_to_p_e(self, data):
        """Normalize the data to the range [g, e]."""

        e_state = self.pi_value
        g_state = self.nopi_value

        data_to_p_e = (data - g_state) / (e_state - g_state)

        return data_to_p_e, e_state, g_state

    def data_to_sigma_z(self, data):
        """Normalize the data to the range [-1, 1]."""

        e_state = self.pi_value
        g_state = self.nopi_value

        data_to_sigma_z = 1 - 2 * (data - g_state) / (e_state - g_state)

        return data_to_sigma_z, e_state, g_state

################### Time Domain Measurements #################################################################################################

# In[]

    def save_results(self, experiment_name, detail = None):

        from datetime import datetime
        import os

        now = datetime.now()
        run_date = now.strftime("%Y%m%d")  # e.g. 20251022
        run_timestamp = now.strftime("%H%M%S")  # e.g. 20251022_153045

        # simple console log

        save_dir = f"results/{run_date}/{experiment_name}"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{run_timestamp}_{detail}.png"), dpi=300, bbox_inches='tight')

# In[]

    def fourier_transform(self, data, time):

        # time, data 리스트가 이미 주어져 있다고 가정
        time = np.array(time)    # [us]
        data = np.array(data)

        # 샘플링 간격 및 FFT 준비
        dt = time[1] - time[0]              # time step (us)
        Fs = 1.0 / dt                        # sampling frequency (MHz 단위가 아님!)
        # time 단위가 us이므로 freq는 MHz 단위가 되도록 보정
        Fs_MHz = Fs                          # MHz

        # 평균 제거
        data_detrend = data - np.mean(data)

        # 윈도우 적용 (Hann window)
        window = np.hanning(len(data))
        data_win = data_detrend * window

        # FFT 수행
        fft_vals = np.fft.fft(data_win)
        fft_freq = np.fft.fftfreq(len(data_win), d=dt)   # [MHz]

        # 양의 주파수 부분만 추출
        idx = np.where(fft_freq > 0)
        freqs = fft_freq[idx]
        amps = np.abs(fft_vals[idx])

        # 가장 큰 두 개의 peak 찾기
        sorted_indices = np.argsort(amps)[::-1]   # 큰 순서대로 정렬
        peak1_idx = sorted_indices[0]
        peak2_idx = sorted_indices[1]
        peak3_idx = sorted_indices[2]
        peak4_idx = sorted_indices[3]

        peak1_freq = freqs[peak1_idx]
        peak2_freq = freqs[peak2_idx]
        peak3_freq = freqs[peak3_idx]
        peak4_freq = freqs[peak4_idx]
        peak1_amp = amps[peak1_idx]
        peak2_amp = amps[peak2_idx]
        peak3_amp = amps[peak3_idx]
        peak4_amp = amps[peak4_idx]

        print(f"1st peak  : {peak1_freq*1e-6:.6f} MHz  (amp={peak1_amp:.3f})")
        print(f"2nd peak : {peak2_freq*1e-6:.6f} MHz  (amp={peak2_amp:.3f})")
        print(f"3rd peak  : {peak3_freq*1e-6:.6f} MHz  (amp={peak3_amp:.3f})")
        print(f"4th peak : {peak4_freq*1e-6:.6f} MHz  (amp={peak4_amp:.3f})")

        return freqs, amps
    
# In[] Density matrix reconstruction toolkits

# Refer to Conditional-NOT Displacement: Fast Multioscillator Control with a Single Qubit (Appendix F8)

    # Characteristic function 계산
    def compute_char(self, rho, grid, N):
        chi = np.zeros_like(grid, dtype=np.complex128)
        a = qt.destroy(N)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                D = qt.displace(N, grid[i, j])
                chi[i, j] = (rho * D).tr()
        return chi
    
    def precompute_displacements(self, alphas: np.ndarray, N: int):
        """
        Precompute D(alpha) as QuTiP operators for all alpha on the grid.
        This greatly speeds up the cost evaluation, since D(alpha) creation is expensive.
        """
        a = qt.destroy(N)
        D_ops = [qt.displace(N, complex(alpha)) for alpha in alphas]
        return D_ops
    
    # Density matrix parametrization from theta (upper-triangular T)
    def rho_from_theta_upper(self, theta, N):
        # cholesky method with upper-triangular T

        T = np.zeros((N, N), dtype=np.complex128)
        idx = 0
        # diagonal > 0
        for i in range(N):
            T[i, i] = np.exp(theta[idx])
            idx += 1
        # strictly upper (i < j)
        for i in range(N):
            for j in range(i+1, N):
                T[i, j] = theta[idx] + 1j*theta[idx+1]
                idx += 2
        rho = T.conj().T @ T
        rho /= np.trace(rho)
        return qt.Qobj(rho, dims=[[N],[N]])
    
    # Initial theta0 from given rho_init (for initial guess of minimization)
    def theta0_from_rho_init(self, rho_init, N, eps=1e-8):
        """
        rho_init: QuTiP Qobj or numpy (NxN). Returns theta0 for rho_from_theta_upper().
        Adds eps*I/N regularization to make it PD.
        """
        rho_np = rho_init.full() if hasattr(rho_init, "full") else np.array(rho_init, dtype=np.complex128)
        # make Hermitian (numerical safety)
        rho_np = 0.5 * (rho_np + rho_np.conj().T)
        # regularize to be strictly PD
        rho_reg = (1 - eps) * rho_np + eps * np.eye(N) / N
        # Cholesky: rho = L L† (L lower)
        L = np.linalg.cholesky(rho_reg)
        # Then rho = (L†)† (L†) = T† T with T = L† (UPPER)
        T0 = L.conj().T
        # extract theta (upper-triangular convention)
        theta = []
        for i in range(N):
            theta.append(np.log(np.real(T0[i, i])))
        for i in range(N):
            for j in range(i+1, N):
                theta.append(np.real(T0[i, j]))
                theta.append(np.imag(T0[i, j]))
        return np.array(theta, dtype=float)

    # TODO : sigma value 추가

    # def cost_qutip_upper(theta, D_ops, G_meas, N):
    #     rho = rho_from_theta_upper(theta, N)
    #     res = 0.0
    #     for D, g in zip(D_ops, G_meas):
    #         pred = (D * rho).tr()
    #         res += (abs(pred - g)**2)
    #     return float(np.real(res))
    
    def cost_numpy(self, theta, D_mats, G_meas, N):
        # rho from theta (numpy로)
        T = np.zeros((N, N), dtype=np.complex128)
        idx = 0
        for i in range(N):
            T[i, i] = np.exp(theta[idx]); idx += 1
        for i in range(N):
            for j in range(i+1, N):  # upper 컨벤션 예시
                T[i, j] = theta[idx] + 1j*theta[idx+1]
                idx += 2
        rho = T.conj().T @ T
        rho /= np.trace(rho)
        # pred_k = Tr[D_k rho] for all k at once
        preds = np.einsum('kij,ji->k', D_mats, rho)   # D_mats : (K,N,N), rho : (N,N) -> preds : (K,) = (D_k)_ij * rho_ji 
        r = (preds - G_meas)
        return float(np.real(np.vdot(r, r)))          # sum |r|^2

    def density_matrix_reconstruction(self, N, rho_init, data, x_grid, y_grid, maxiter=2000, maxfun=200000):

        theta0 = self.theta0_from_rho_init(rho_init, N=N) # initial guess

        X, Y = np.meshgrid(x_grid, y_grid)
        alphas = (X + 1j * Y).ravel()

        # Precompute D(alpha) operators 
        D_ops = self.precompute_displacements(alphas, N)
        D_mats = np.stack([D.full() for D in D_ops], axis=0)

        res = minimize(
            self.cost_numpy,
            theta0,  
            args=(D_mats, data, N), # data should be 1D
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxfun": maxfun}
        )
        print("success:", res.success)
        print("message:", res.message)
        print("number of iterations:", res.nit)
        reconstruncted_rho = self.rho_from_theta_upper(res.x, N)
        print("trace(rho) =", reconstruncted_rho.tr())

        return reconstruncted_rho
    
    def plot_characteristic_function(self, grid_x, grid_y, rho):

        alpha_x , alpha_y = np.meshgrid(grid_x, grid_y)
        
        characteristic_func = self.compute_char(rho, alpha_x + 1j * alpha_y, rho.shape[0])

        plt.figure(figsize=(12, 5))
        plt.pcolormesh(grid_x, grid_y, np.real(characteristic_func), cmap='bwr', vmin = -1, vmax=1)
        
        plt.colorbar(label='|χ(ξ)|')
        plt.xlabel('Re(ξ)')
        plt.ylabel('Im(ξ)')
        plt.title('Characteristic Function (Real Part)')
        plt.gca().set_aspect('equal')

        self.save_results(experiment_name="characteristic_reconstructed_rho")

        plt.show()
    
    def state_fidelity(self, rho_ideal, rho):
        """Compute the fidelity between two density matrices rhos."""

        print("state fidelity:", qt.metrics.fidelity(rho_ideal, rho))

        return qt.metrics.fidelity(rho_ideal, rho)
        