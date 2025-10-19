# In[]

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time as time_module
import scipy.optimize

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
from laboneq.contrib.example_helpers.randomized_benchmarking_helper import (
    clifford_parametrized,
    generate_play_rb_pulses,
    make_pauli_gate_map,
)
from laboneq_applications.analysis.fitting_helpers import exponential_decay_fit
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
                 is_memory_mode = True,):

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
        self.session = Session(device_setup=self.device_setup)
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
    def continuous_wave(self, is_sideband_pulse = False, average_exponent=19, freq_l=10e6, freq_h=10e6, amp_l=1.0, amp_h=1.0,
                        is_plot_simulation = False):

        device_setup = self.device_setup    
        component = list(self.qubits_parameters.keys())[self.which_qubit]

        drive_pulse = pulse_library.const(uid="drive_pulse", 
                                            length=40e-6, 
                                            amplitude=1)
        
        sideband_pulse = pulse_library.sidebands_pulse(uid = "sideband_pulse", 
                                                       length=10e-6,
                                                        frequency_l = freq_l,
                                                        frequency_h = freq_h,
                                                        amp_l = amp_l,
                                                        amp_h = amp_h,
                                                        phase = 0)

        if is_sideband_pulse:
            pulse = sideband_pulse
        else :
            pulse = drive_pulse

        exp_cont_wave = Experiment(
                uid="cont_wave",
                signals=[
                    ExperimentSignal(uid="drive"),
                ],
            )
            
        with exp_cont_wave.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
        ):
            with exp_cont_wave.section(uid="drive"):
                exp_cont_wave.play(signal="drive", pulse=pulse)
                
              
        signal_map = {
                "drive": device_setup.logical_signal_groups[component].logical_signals["drive"],
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

    def prop_delay_calibration(self, line, average_exponent):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.const(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
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

            readout_weighting_function = pulse_library.gaussian_square(
                uid="readout_weighting_function",
                length=qubits_parameters[qubits_component]["readout_integration_length"],
                amplitude=qubits_parameters[qubits_component]["readout_integration_amp"]
            )

            return readout_pulse, readout_weighting_function
        
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

            sidebands_drive_pulse_chunk = pulse_library.sidebands_pulse(
                uid="sidebands_drive_pulse",
                length=cavity_parameters[cavity_component]["sideband_length"],
                frequency_l=cavity_parameters[cavity_component]["sideband_frequency_l"],
                frequency_h=cavity_parameters[cavity_component]["sideband_frequency_h"],
                amp_l=cavity_parameters[cavity_component]["sideband_amp_l"]*cavity_parameters[cavity_component]["sideband_att_h"]/cavity_parameters[cavity_component]["sideband_att_l"],
                amp_h=cavity_parameters[cavity_component]["sideband_amp_h"],
                phase=cavity_parameters[cavity_component]["sideband_phase"],
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
            

            return cond_disp_pulse, cavity_drive_pulse, sidebands_drive_pulse_chunk, cavity_drive_pulse_constant_chunk

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

