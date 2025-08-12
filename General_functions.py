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
    def continuous_wave(self, average_exponent=19, N=3):

        device_setup = self.device_setup    
        component = list(self.qubits_parameters.keys())[self.which_qubit]

        drive_pulse = pulse_library.const(uid="drive_pulse", 
                                            length=40e-6, 
                                            amplitude=1)

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
                exp_cont_wave.play(signal="drive", pulse=drive_pulse)
                
              
        signal_map = {
                "drive": device_setup.logical_signal_groups[component].logical_signals["drive_line"],
            }
        
        exp_cont_wave.set_signal_map(signal_map)

        results = self.session.run(exp_cont_wave)


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

            exp_calibration["drive"] = SignalCalibration(
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

            exp_calibration["cavity_drive"] = SignalCalibration(
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
                        qubits_component, cavity_component):
        
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

            return cond_disp_pulse, cavity_drive_pulse
        
        else:
            raise ValueError("Invalid pulse type. Choose from 'readout', 'qubit_control', or 'cavity_control'.")
        

    def CNOD(self, exp, cond_disp_pulse, pi_pulse, amp, prev_uid, uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid = "pi_pulse_1"):
        
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        fix_angle = cavity_parameters[cavity_component]["cavity_mode_chi"] * (qubits_parameters[qubits_component]["pi_length"]/2 + cavity_parameters[cavity_component]["cond_disp_pulse_length"]) # in radians

        with exp.section(uid=uid1, play_after=prev_uid):
            exp.play(signal="cavity_drive", pulse=cond_disp_pulse, amplitude=amp)

        with exp.section(uid=pi_pulse_uid, play_after=uid1):
            exp.play(signal="drive", pulse=pi_pulse)

        with exp.section(uid=uid2, play_after=pi_pulse_uid):
            exp.play(signal="cavity_drive", pulse=cond_disp_pulse, 
                     amplitude=amp, phase = np.pi, increment_oscillator_phase = 2*np.pi*fix_angle)

    def wigner_function(self, exp, cavity_drive_pulse, pi2_pulse, amplitude_sweep, sweep_case):

        def correction(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)
    
            return decorator

        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        with exp.section(uid="alpha_sweep", play_after="preparation"):
            # D^+(alpha)
            exp.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude_sweep, phase=np.pi)

        with exp.section(uid="qubit_excitation_1", play_after="alpha_sweep"):
            exp.play(signal="drive", pulse=pi2_pulse)
            exp.delay(signal="drive", time=np.abs(1/(2*cavity_parameters[cavity_component]["cavity_mode_chi"]))) # delay for cross Kerr effect

        with exp.section(uid="qubit_excitation_2", play_after="qubit_excitation_1"):
            @correction(sweep_case, exp=exp)
            def play_correction(v):
                if v == 0:
                    exp.play(signal="drive", pulse=pi2_pulse, phase = 0)
                elif v == 1:
                    exp.play(signal="drive", pulse=pi2_pulse, phase = np.pi)

    def characteristic_function(self, exp, pi2_pulse, pi_pulse, cond_disp_pulse, amplitude_sweep, qubit_phase=0):

        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        with exp.section(uid="qubit_excitation_1", play_after="preparation"):
            exp.play(signal="drive", pulse=pi2_pulse)

        self.CNOD(exp=exp, cond_disp_pulse=cond_disp_pulse, 
                pi_pulse = pi_pulse, amp = amplitude_sweep, prev_uid="qubit_excitation_1", 
                uid1 = "char_cond_disp_pulse_1", uid2 = "char_cond_disp_pulse_2", pi_pulse_uid = "char_pi_pulse_1")
        
        with exp.section(uid="qubit_excitation_2", play_after="char_cond_disp_pulse_2"):
            exp.play(signal="drive", pulse=pi2_pulse, phase=qubit_phase)

################### Time Domain Measurements #################################################################################################
# In[] Time Domain Measurements
    
    def nopi_pi(self, average_exponent, phase = 0, is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        drive_pulse = pulse_library.drag(uid="drive_pulse", 
                                             length = qubits_parameters[component]['drive_pulse_length'], 
                                             amplitude = qubits_parameters[component]["drive_amp"],
                                             beta = qubits_parameters[component]["drive_beta"],
                                             )
        

        phase = phase * np.pi / 180
        # nopi --------------------------------------------------------------------------------------------
        exp_nopi = Experiment(
                uid="nopi_experiment",
                signals=[
                    ExperimentSignal(uid="drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
            
        with exp_nopi.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.RAW,
            reset_oscillator_phase = True,
        ):
            with exp_nopi.section(uid="nopi"):
                with exp_nopi.section(uid="drive"):
                    exp_nopi.reserve(signal="drive")
                with exp_nopi.section(uid="measure", play_after="drive"):
                    exp_nopi.play(signal="measure", pulse=readout_pulse, phase = phase)
                    exp_nopi.acquire(
                        signal="acquire", handle="ac_nopi", kernel=readout_weighting_function # can be acquired only there is a measure signal
                    )
            # relax time after readout - for signal processing and qubit relaxation to ground state
            with exp_nopi.section(uid="relax_nopi", length=qubits_parameters[component]["reset_delay_length"]):
                exp_nopi.reserve(signal="measure")

        signal_map = self.signal_map(component)

        exp_nopi.set_signal_map(signal_map)
        
        compiled_experiment_nopi = self.session.compile(exp_nopi)

        self.nopi_results = self.session.run(compiled_experiment_nopi)


        # pi --------------------------------------------------------------------------------------------

        exp_pi = Experiment(
                uid="pi_experiment",
                signals=[
                    ExperimentSignal(uid="drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
            
        with exp_pi.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.RAW,
            reset_oscillator_phase = True,
        ):
            with exp_pi.section(uid="pi"):
                with exp_pi.section(uid="drive"):
                    exp_pi.play(signal="drive", pulse=drive_pulse)
                with exp_pi.section(uid="measure", play_after="drive"):
                    exp_pi.play(signal="measure", pulse=readout_pulse, phase = phase)
                    exp_pi.acquire(
                        signal="acquire", handle="ac_pi", kernel=readout_weighting_function # can be acquired only there is a measure signal
                    )
            # relax time after readout - for signal processing and qubit relaxation to ground state
            with exp_pi.section(uid="relax_pi", length=qubits_parameters[component]["reset_delay_length"]):
                exp_pi.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[component].logical_signals["measure"],
            "drive": device_setup.logical_signal_groups[component].logical_signals["drive"],
            "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire"],
        }

        exp_pi.set_signal_map(signal_map)
        
        compiled_experiment_pi = self.session.compile(exp_pi)

        self.pi_results = self.session.run(compiled_experiment_pi)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_nopi, start_time=0, length=20e-6, component=component)
            self.simulation_plot(compiled_experiment_pi, start_time=0, length=20e-6, component=component)

    def single_shot_nopi_pi(self, npts_exponent, phase = 0, is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        drive_pulse = pulse_library.drag(uid="drive_pulse", 
                                             length = qubits_parameters[component]['drive_pulse_length'], 
                                             amplitude = qubits_parameters[component]["drive_amp"],
                                             beta = qubits_parameters[component]["drive_beta"],
                                             )
        
        phase = phase * np.pi / 180
        # nopi --------------------------------------------------------------------------------------------

        exp_ss_nopi = Experiment(
                uid="nopi_experiment",
                signals=[
                    ExperimentSignal(uid="drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
        
        with exp_ss_nopi.acquire_loop_rt(
            uid="shots",
            count=pow(2, npts_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION
        ):
            with exp_ss_nopi.section(uid="nopi"):
                with exp_ss_nopi.section(uid="drive"):
                    exp_ss_nopi.reserve(signal="drive")
                with exp_ss_nopi.section(uid="measure", play_after="drive"):
                    exp_ss_nopi.play(signal="measure", pulse=readout_pulse, phase = phase)
                    exp_ss_nopi.acquire(
                        signal="acquire", handle="ac_nopi", kernel=readout_weighting_function # can be acquired only there is a measure signal
                    )
            # relax time after readout - for signal processing and qubit relaxation to ground state
            with exp_ss_nopi.section(uid="relax_nopi", length=qubits_parameters[component]["reset_delay_length"]):
                exp_ss_nopi.reserve(signal="measure")
        
        signal_map = self.signal_map(component)

        exp_ss_nopi.set_signal_map(signal_map)
        
        compiled_experiment_nopi = self.session.compile(exp_ss_nopi)

        nopi_results = self.session.run(compiled_experiment_nopi)

        self.ss_nopi_results = nopi_results.get_data("ac_nopi")

        # pi --------------------------------------------------------------------------------------------

        exp_ss_pi = Experiment(
                uid="pi_experiment",
                signals=[
                    ExperimentSignal(uid="drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
        
        with exp_ss_pi.acquire_loop_rt(
            uid="shots",
            count=pow(2, npts_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION
        ):
            with exp_ss_pi.section(uid="pi"):
                with exp_ss_pi.section(uid="drive"):
                    exp_ss_pi.play(signal="drive", pulse=drive_pulse)
                with exp_ss_pi.section(uid="measure", play_after="drive"):
                    exp_ss_pi.play(signal="measure", pulse=readout_pulse, phase = phase)
                    exp_ss_pi.acquire(
                        signal="acquire", handle="ac_pi", kernel=readout_weighting_function # can be acquired only there is a measure signal
                    )
            # relax time after readout - for signal processing and qubit relaxation to ground state
            with exp_ss_pi.section(uid="relax_pi", length=qubits_parameters[component]["reset_delay_length"]):
                exp_ss_pi.reserve(signal="measure")
        
        signal_map = self.signal_map(component)

        exp_ss_pi.set_signal_map(signal_map)

        compiled_experiment_pi = self.session.compile(exp_ss_pi)

        pi_results = self.session.run(compiled_experiment_pi)

        self.ss_pi_results = pi_results.get_data("ac_pi")

        
    def plot_nopi_pi(self, npts = 100, num_of_bins = 300, is_gaussian_fit = True):

        self.raw_nopi = self.nopi_results.get_data("ac_nopi")
        self.raw_pi = self.pi_results.get_data("ac_pi")

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]

        ### data acquisition and processing ##########################################

        I_demod_nopi, Q_demod_nopi = self.IF_demodulation(np.real(self.raw_nopi), np.imag(self.raw_nopi), self.qubits_parameters[component]["readout_freq_IF"])
        I_demod_pi, Q_demod_pi = self.IF_demodulation(np.real(self.raw_pi), np.imag(self.raw_pi), self.qubits_parameters[component]["readout_freq_IF"])
        
        num_of_pnts = npts
        data_len = len(self.raw_nopi)
        num_avg_pnts = data_len // num_of_pnts

        time = np.linspace(0, data_len/2 - data_len/2/num_of_pnts, num_of_pnts) # 2GSa/s acquisition rate

        average_I_demod_nopi = [np.mean(I_demod_nopi[i:i+num_avg_pnts])* data_len for i in range(0, len(I_demod_nopi), num_avg_pnts)]
        average_Q_demod_nopi = [np.mean(Q_demod_nopi[i:i+num_avg_pnts])* data_len for i in range(0, len(Q_demod_nopi), num_avg_pnts)]
        
        average_I_demod_pi = [np.mean(I_demod_pi[i:i+num_avg_pnts])* data_len for i in range(0, len(I_demod_pi), num_avg_pnts)]
        average_Q_demod_pi = [np.mean(Q_demod_pi[i:i+num_avg_pnts])* data_len for i in range(0, len(Q_demod_pi), num_avg_pnts)]
        
        ### single shot data ##########################
        ss_nopi_results = self.ss_nopi_results
        ss_pi_results = self.ss_pi_results

        I_nopi_data = np.real(ss_nopi_results)
        Q_nopi_data = np.imag(ss_nopi_results)

        I_nopi_data_mean = np.mean(I_nopi_data)
        Q_nopi_data_mean = np.mean(Q_nopi_data)

        nopi_noise = np.var(np.sqrt((I_nopi_data-I_nopi_data_mean)**2 + 
                                    (Q_nopi_data-Q_nopi_data_mean)**2))

        I_pi_data = np.real(ss_pi_results)
        Q_pi_data = np.imag(ss_pi_results)

        I_pi_data_mean = np.mean(I_pi_data)
        Q_pi_data_mean = np.mean(Q_pi_data)

        SNR = np.sqrt((I_nopi_data_mean-I_pi_data_mean)**2 + (Q_nopi_data_mean-Q_pi_data_mean)**2)/np.sqrt(nopi_noise)

        #### figure plot ###############################################################

        measurement_length = qubits_parameters[component]["readout_pulse_length"]

        fig, ax = plt.subplots(2, 2, figsize=(16, 10))

        # fig.suptitle(f"nopi and pi \n r_amp:{qubits_parameters[component]["readout_amp"]}, r_freq:{qubits_parameters[component]["readout_frequency"]}", fontsize=20)
        fig.suptitle(f"\n r_amp:{qubits_parameters[component]["readout_amp"]}, r_freq:{qubits_parameters[component]["readout_frequency"]}", fontsize=20)
        fig.text(0.45, 0.965, "NoPi", fontsize=20, color = "blue")
        fig.text(0.49, 0.965, "and", fontsize=20, color = "black")
        fig.text(0.52, 0.965, "Pi", fontsize=20, color = "red")

        ax[0,0].scatter(time, average_I_demod_nopi, color="b", marker = 'o')
        ax[0,0].scatter(time, average_I_demod_pi, color="r", marker = 'o')
        ax[0,0].tick_params(axis='both', which='major', labelsize=10)
        ax[0,0].tick_params(axis='both', which='minor', labelsize=10)
        ax[0,0].set_xlabel("Time (ns)", fontsize=20)
        ax[0,0].set_ylabel("I (a.u.)", fontsize=20)

        ax[0,1].scatter(time, average_Q_demod_nopi, color="b", marker = 'o')
        ax[0,1].scatter(time, average_Q_demod_pi, color="r", marker = 'o')
        ax[0,1].tick_params(axis='both', which='major', labelsize=10)
        ax[0,1].tick_params(axis='both', which='minor', labelsize=10)
        ax[0,1].set_xlabel("Time (ns)", fontsize=20)
        ax[0,1].set_ylabel("Q (a.u.)", fontsize=20)

        axis_range = np.max([np.abs(average_I_demod_nopi), np.abs(average_Q_demod_nopi), np.abs(average_I_demod_pi), np.abs(average_Q_demod_pi)])
        ax[1,0].plot(average_I_demod_nopi, average_Q_demod_nopi, color="b", marker = 'o', markeredgecolor='black')
        ax[1,0].plot(average_I_demod_pi, average_Q_demod_pi, color="r", marker = 'o', markeredgecolor='black')
        ax[1,0].plot(I_nopi_data_mean, Q_nopi_data_mean, color="k", alpha = 0.3, marker = 'x')
        ax[1,0].plot(I_pi_data_mean, Q_pi_data_mean, color="k", alpha = 0.3, marker = 'x')

        ax[1,0].scatter(I_nopi_data, Q_nopi_data, color="b", alpha = 0.3, marker = '.')
        ax[1,0].scatter(I_pi_data, Q_pi_data, color="r", alpha = 0.3, marker = '.')
        ax[1,0].set_xlim([-np.max([np.sqrt(I_nopi_data**2 + Q_nopi_data**2), np.sqrt(I_pi_data**2 + Q_pi_data**2)])*1.3,
                    np.max([np.sqrt(I_nopi_data**2 + Q_nopi_data**2), np.sqrt(I_pi_data**2 + Q_pi_data**2)])*1.3])
        ax[1,0].set_ylim([-np.max([np.sqrt(I_nopi_data**2 + Q_nopi_data**2), np.sqrt(I_pi_data**2 + Q_pi_data**2)])*1.3,
                    np.max([np.sqrt(I_nopi_data**2 + Q_nopi_data**2), np.sqrt(I_pi_data**2 + Q_pi_data**2)])*1.3])
        ax[1,0].set_aspect('equal', 'box')

        ax[1,0].tick_params(axis='both', which='major', labelsize=10)
        ax[1,0].tick_params(axis='both', which='minor', labelsize=10)
        ax[1,0].set_xlabel("I (a.u.)", fontsize=20)
        ax[1,0].set_ylabel("Q (a.u.)", fontsize=20)

        an1 = ax[1,0].annotate((f'SNR = {SNR:.3f}'), xy = (0,0))
        an1.draggable()

        # np.save(f"nopi_pi_data_{self.which_qubit}", [I_nopi_data, Q_nopi_data, I_pi_data, Q_pi_data])

        if self.which_data == "I":
            nopi_data = I_nopi_data
            pi_data = I_pi_data
            IQ = "I"
        else:
            nopi_data = Q_nopi_data
            pi_data = Q_pi_data
            IQ = "Q"

        nopi_hist_data = ax[1,1].hist(nopi_data, bins = num_of_bins, color = "b", alpha = 0.5)
        pi_hist_data = ax[1,1].hist(pi_data, bins = num_of_bins, color = "r", alpha = 0.5)

        self.nopi_value = np.mean(nopi_data)
        self.pi_value = np.mean(pi_data)

        ax[1,1].plot([np.mean(nopi_data), np.mean(nopi_data)], [0, max([max(nopi_hist_data[0]),max(pi_hist_data[0])]) + 5], '-k')
        ax[1,1].plot([np.mean(pi_data), np.mean(pi_data)], [0, max([max(nopi_hist_data[0]),max(pi_hist_data[0])]) + 5], '--k')

        ax[1,1].tick_params(axis='both', which='major', labelsize=10)

        ax[1,1].set_xlabel(f"{IQ} (a.u.)", fontsize=20)
        ax[1,1].set_ylabel("Counts", fontsize=20)

        fdlty, thresh = self.hist_fidelity(nopi_data, pi_data)

        an2 = ax[1,1].annotate((f'Fidelity = {fdlty:.3f}'), xy = (np.average(nopi_data), max(nopi_hist_data[0])))
        an2.draggable()

        plt.show()

        if is_gaussian_fit:

            ground_state = np.mean(nopi_data)
            excited_state = np.mean(pi_data)

            nopi_h = max(nopi_hist_data[0])
            pi_h = max(pi_hist_data[0])

            plt.figure()
            plt.plot(nopi_hist_data[1][:-1], nopi_hist_data[0], color = "b", alpha = 0.5, label = "nopi")
            plt.plot(pi_hist_data[1][:-1], pi_hist_data[0], color = "r", alpha = 0.5, label = "pi")

            init_guess_1 = [nopi_h, ground_state, 0.1, 0, excited_state, 0.1]

            popt1, pcov1 = scipy.optimize.curve_fit(self.double_gaussian, nopi_hist_data[1][:-1], nopi_hist_data[0], p0=init_guess_1)
            plt.plot(nopi_hist_data[1][:-1], self.double_gaussian(nopi_hist_data[1][:-1], *popt1), color = "b", alpha = 0.5, label = "nopi_fit")
            plt.plot([popt1[1],popt1[1]], [0, nopi_h], color = "b", linestyle = "--", alpha = 0.5)

            init_guess_2 = [pi_h*0.2, ground_state, 0.1, pi_h, excited_state, 0.1]

            popt2, pcov2 = scipy.optimize.curve_fit(self.double_gaussian, pi_hist_data[1][:-1], pi_hist_data[0], p0=init_guess_2)
            plt.plot(pi_hist_data[1][:-1], self.double_gaussian(pi_hist_data[1][:-1], *popt2), color = "r", alpha = 0.5, label = "pi_fit")
            plt.plot([popt2[4],popt2[4]], [0, pi_h], color = "r", linestyle = "--", alpha = 0.5)
            # plt.annotate(f"ground state value :{popt1[1]:.2f}", xy = (popt1[1], nopi_h))
            # plt.annotate(f"excited state value :{popt2[4]:.2f}", xy = (popt2[4], pi_h))

            print(f"nopi : {popt1}")
            print(f"pi : {popt2}")
            print(f"Thermal population : {min(popt1[0]/popt1[3], popt1[3]/popt1[0])}")
            print(f"residual population : {min(popt2[0]/popt2[3], popt2[3]/popt2[0])}")

            plt.legend()

# In[]        
    def T1(self, average_exponent = 12, duration = 100e-6, npts = 101, is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        self.exp_T1_dict = {
            "duration": duration,
            "npts": npts,
        }
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        drive_pulse = pulse_library.gaussian(uid="drive_pulse", 
                                             length = qubits_parameters[component]['drive_pulse_length'], 
                                             amplitude = qubits_parameters[component]["drive_amp"])
        
        phase = qubits_parameters[component]["readout_phase"]

        delay_sweep = LinearSweepParameter(uid="delay", start=0, stop=duration, count=npts)

        exp_T1 = Experiment(
            uid="T1 experiment",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        
        with exp_T1.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - sweep over delay between qubit excitation and readout pulse
            # right alignment makes sure the readout section follows a fixed timing, being the last element in each shot
            with exp_T1.sweep(
                uid="sweep", parameter=delay_sweep
            ):
                # qubit drive pulse followed by variable delay
                with exp_T1.section(uid="qubit_excitation", alignment=SectionAlignment.RIGHT):
                    exp_T1.play(signal="drive", pulse=drive_pulse)
                    exp_T1.delay(signal="drive", time=delay_sweep)
                # qubit readout pulse and data acquisition
                with exp_T1.section(uid="qubit_readout", play_after="qubit_excitation"):
                    # play readout pulse
                    exp_T1.play(signal="measure", pulse=readout_pulse, phase = phase)
                    # signal data acquisition
                    exp_T1.acquire(
                        signal="acquire",
                        handle="ac_T1",
                        kernel=readout_weighting_function,
                    )
                # relax time after readout - for signal processing and qubit relaxation to ground state
                with exp_T1.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_T1.reserve(signal="measure")
                
        
        signal_map = self.signal_map(component)
        
        exp_T1.set_signal_map(signal_map)
        
        compiled_experiment_T1 = self.session.compile(exp_T1)

        self.T1_results = self.session.run(compiled_experiment_T1)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_T1, start_time=0, length=20e-6, component=component)
            show_pulse_sheet("T1", compiled_experiment_T1)
    
    def plot_T1(self, is_fit = True):

        ### data processing ###############################################################

        averaged_nums = len(self.T1_results.acquired_results['ac_T1'].axis[0])
        # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

        self.T1_data = self.T1_results.get_data("ac_T1") # (2^N, npts) array
        time = self.T1_results.acquired_results['ac_T1'].axis[1]

        if self.which_data == "I":
            
            data = np.real(np.mean(self.T1_data, axis = 0))
            std_data = np.real(np.std(self.T1_data, axis = 0)/np.sqrt(averaged_nums))

        else:
            data = np.imag(np.mean(self.T1_data, axis = 0))
            std_data = np.imag(np.std(self.T1_data, axis = 0)/np.sqrt(averaged_nums))


        ### data plot ######################################################################

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        ax.errorbar(time*1e6, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))

        if is_fit :
            sfit1 = sFit('Exp', time, data)
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            _,decay_rate,_ = popt
            _,decay_rate_err,_ = np.sqrt(np.diag(pcov))

            ax.plot(time*1e6, sfit1.func(time, *popt))
            an = ax.annotate((f'T1 = {(1/decay_rate*1e6):.2f}±{(1/(decay_rate)**2*decay_rate_err*1e6):.2f}[us]'), 
                             xy = (np.average(time*1e6), np.average(data[0:10]) ),
                             size = 16)
            an.draggable()

        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=15)

        ax.set_title("T1 measurement", fontsize=20)
        ax.set_xlabel("Time (us)", fontsize=20)
        ax.set_ylabel(f"{self.which_data} (a.u.)", fontsize=20)

        plt.show()
# In[]            
    def Pi2_cal(self, average_exponent = 12, start = 0, npts = 12, is_plot_simulation = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        pi2_pulse = pulse_library.drag(
            uid="drive_pulse", 
            length = qubits_parameters[component]['pi2_length'], 
            amplitude = qubits_parameters[component]["pi2_amp"],
            beta = qubits_parameters[component]["pi2_beta"]
        )
        
        phase = qubits_parameters[component]["readout_phase"]

        pulse_count = LinearSweepParameter(uid="pulses", start = start, stop= start+npts-1, count=npts)
        
        def repeat(count: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                for _ in range(int(v)):
                                    f()
                else:
                    for _ in range(count):
                        f()

            return decorator
        
                            
        exp_pi2_cal = Experiment(
            uid="Pi2 calibration",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        
        
        with exp_pi2_cal.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_pi2_cal.sweep(uid="sweep", parameter = pulse_count, alignment=SectionAlignment.RIGHT):
                with exp_pi2_cal.section(uid="pi2_pulses", alignment=SectionAlignment.RIGHT):
                    @repeat(pulse_count, exp_pi2_cal)
                    def play_pi2():
                        exp_pi2_cal.play(signal = "drive", 
                                        pulse = pi2_pulse)
            
                with exp_pi2_cal.section(uid="measure", play_after="pi2_pulses"):
                    exp_pi2_cal.play(signal="measure", pulse=readout_pulse, phase = phase)
                    exp_pi2_cal.acquire(
                        signal="acquire", handle="ac_pi2_cal", kernel=readout_weighting_function
                    )
                
                with exp_pi2_cal.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_pi2_cal.reserve(signal="measure")
        
        signal_map = self.signal_map(component)
        
        exp_pi2_cal.set_signal_map(signal_map)
        
        compiled_experiment_pi2_cal = self.session.compile(exp_pi2_cal)

        self.pi2_cal_results = self.session.run(compiled_experiment_pi2_cal)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_pi2_cal, start_time=0, length=20e-6, component=component)
    
    def plot_Pi2_cal(self):

        qubits_parameters = self.qubits_parameters
        component = list(qubits_parameters.keys())[self.which_qubit]

        ### data processing ###############################################################

        averaged_nums = len(self.pi2_cal_results.acquired_results['ac_pi2_cal'].axis[0])
        pi2_nums = self.pi2_cal_results.acquired_results['ac_pi2_cal'].axis[1]

        self.pi2_cal_data = self.pi2_cal_results.get_data("ac_pi2_cal") # (2^N, npts) array

        if self.which_data == "I":
            data = np.real(np.mean(self.pi2_cal_data, axis = 0))
            std_data = np.real(np.std(self.pi2_cal_data, axis = 0)/np.sqrt(averaged_nums))

        else:
            data = np.imag(np.mean(self.pi2_cal_data, axis = 0))
            std_data = np.imag(np.std(self.pi2_cal_data, axis = 0)/np.sqrt(averaged_nums))


        ### data plot ######################################################################
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        middle_points = []
        middle_indexes = []
        for i in range(len(data)//2):
            middle_indexes.append(pi2_nums[0] + 2*i+1)
            middle_points.append(data[2*i+1])

        ax.errorbar(pi2_nums, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, 
                    ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
        ax.plot(middle_indexes, middle_points, color="r", marker="o", linestyle = '--', markeredgecolor='black')
        
        an = ax.annotate((f'pi2_amp = {qubits_parameters[component]["pi2_amp"]}, length = {qubits_parameters[component]["pi2_length"]*1e9}[ns]'),
                            xy = (0, np.average(data)),
                            size = 16)
        an.draggable()

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title("Pi/2 calibration", fontsize=20)
        ax.set_xlabel("Number of pi/2 pulses", fontsize=20)
        ax.set_ylabel(f"{self.which_data} (a.u.)", fontsize=20)

        plt.show()
    
    # In[]            
    def Pi_cal(self, average_exponent = 12, npts = 12, is_plot_simulation = False, is_cond_pulse = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        self.is_cond_pulse = is_cond_pulse
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        pi2_pulse = pulse_library.drag(
            uid="pi2_pulse", 
            length = qubits_parameters[component]['pi2_length'], 
            amplitude = qubits_parameters[component]["pi2_amp"],
            beta = qubits_parameters[component]["pi2_beta"]
        )

        if is_cond_pulse:
            pi_pulse = pulse_library.drag(
                uid="pi_pulse", 
                length = qubits_parameters[component]['cond_pi_length'], 
                amplitude = qubits_parameters[component]["cond_pi_amp"],
                beta = qubits_parameters[component]["cond_pi_beta"],
                conditional=True
            )
        else :
            pi_pulse = pulse_library.drag(
                uid="pi_pulse", 
                length = qubits_parameters[component]['pi_length'], 
                amplitude = qubits_parameters[component]["pi_amp"],
                beta = qubits_parameters[component]["pi_beta"]
            )

        phase = qubits_parameters[component]["readout_phase"]
        
        pulse_count = LinearSweepParameter(uid="pulses", start=0, stop=npts-1, count=npts)
        
        def repeat(count: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                if v%2 == 0:
                                    for _ in range(int(v)//2):
                                        f()
                                else:
                                    exp.play(signal = "drive", pulse = pi2_pulse)
                                    for _ in range(int(v)//2):
                                        f()
                                
            
            return decorator
        
        exp_pi_cal = Experiment(
            uid="Pi calibration",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        
        with exp_pi_cal.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_pi_cal.sweep(uid="sweep", parameter = pulse_count, alignment=SectionAlignment.RIGHT):
                with exp_pi_cal.section(uid="pi_pulses", alignment=SectionAlignment.RIGHT):
                    @repeat(pulse_count, exp_pi_cal)
                    def play_pi():
                        exp_pi_cal.play(signal = "drive", 
                                         pulse = pi_pulse)
                            
                with exp_pi_cal.section(uid="measure", play_after="pi_pulses"):
                    exp_pi_cal.play(signal="measure", pulse=readout_pulse, phase = phase)
                    exp_pi_cal.acquire(
                        signal="acquire", handle="ac_pi_cal", kernel=readout_weighting_function
                    )
                
                with exp_pi_cal.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_pi_cal.reserve(signal="measure")
        
        signal_map = self.signal_map(component)
        
        exp_pi_cal.set_signal_map(signal_map)
        
        compiled_experiment_pi_cal = self.session.compile(exp_pi_cal)

        self.pi_cal_results = self.session.run(compiled_experiment_pi_cal)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_pi_cal, start_time=0, length=20e-6, component=component)
    
    def plot_Pi_cal(self):

        qubits_parameters = self.qubits_parameters
        component = list(qubits_parameters.keys())[self.which_qubit]

        ### data processing ###############################################################

        averaged_nums = len(self.pi_cal_results.acquired_results['ac_pi_cal'].axis[0])
        pi_nums = self.pi_cal_results.acquired_results['ac_pi_cal'].axis[1]

        self.pi_cal_data = self.pi_cal_results.get_data("ac_pi_cal") # (2^N, npts) array

        if self.which_data == "I":
            
            data = np.real(np.mean(self.pi_cal_data, axis = 0))
            std_data = np.real(np.std(self.pi_cal_data, axis = 0)/np.sqrt(averaged_nums))

        else:
            data = np.imag(np.mean(self.pi_cal_data, axis = 0))
            std_data = np.imag(np.std(self.pi_cal_data, axis = 0)/np.sqrt(averaged_nums))


        ### data plot ######################################################################
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        middle_points = []
        middle_indexes = []
        for i in range(len(data)//2):
            middle_indexes.append(2*i+1)
            middle_points.append(data[2*i+1])

        ax.errorbar(pi_nums, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, 
                    ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
        ax.plot(middle_indexes, middle_points, color="r", marker="o", linestyle = '--', markeredgecolor='black')
        if self.is_cond_pulse:
            ax.set_title("Cond Pi calibration", fontsize=20)
        else:
            ax.set_title("Pi calibration", fontsize=20)
        ax.set_xlabel("Number of Pi pulses", fontsize=20)
        ax.set_ylabel(f'{self.which_data} (a.u.)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)

        if self.is_cond_pulse:
            an = ax.annotate((f'cond_pi_amp = {qubits_parameters[component]["cond_pi_amp"]}, length = {qubits_parameters[component]["cond_pi_length"]*1e9}[ns]'),
                            xy = (0, np.average(data)),
                            size = 16)
        else:
            an = ax.annotate((f'pi_amp = {qubits_parameters[component]["pi_amp"]}, length = {qubits_parameters[component]["pi_length"]*1e9}[ns]'),
                            xy = (0, np.average(data)),
                            size = 16)
        an.draggable()

        plt.show()
        
            
# In[]

    def Ramsey(self, detuning = 0, is_echo = False, n_pi_pulse = 1, qubit_phase = 0, # (phase : 0 CP, phase : pi/2 CPMG)
               average_exponent = 12, duration = 100e-6, npts = 101,
               is_zz_interaction = False,
               control_qubit = None,
               is_plot_simulation = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]

        if is_zz_interaction:
            control_component = list(qubits_parameters.keys())[control_qubit]
            control_drive_pulse_pi = pulse_library.gaussian(uid="control_drive_pulse", 
                                             length = qubits_parameters[control_component]['pi_length'], 
                                             amplitude = qubits_parameters[control_component]["pi_amp"])
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"]
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"]
        )
        drive_pulse_pi = pulse_library.gaussian(uid="drive_pulse", 
                                             length = qubits_parameters[component]['pi_length'], 
                                             amplitude = qubits_parameters[component]["pi_amp"])
        
        drive_pulse_pi2 = pulse_library.gaussian(uid="drive_pulse_2",
                                                length = qubits_parameters[component]['pi2_length'], 
                                                amplitude = qubits_parameters[component]["pi2_amp"])
        
        time_sweep = LinearSweepParameter(uid="time_sweep", start=0, stop=duration, count=npts)

        phase = qubits_parameters[component]["readout_phase"]


        if is_zz_interaction:
            exp_ramsey = Experiment(
                uid="Ramsey experiment",
                signals=[
                    ExperimentSignal("drive"),
                    ExperimentSignal("control_drive"),
                    ExperimentSignal("measure"),
                    ExperimentSignal("acquire"),
                ],
            )
        else :
            exp_ramsey = Experiment(
                uid="Ramsey experiment",
                signals=[
                    ExperimentSignal("drive"),
                    ExperimentSignal("measure"),
                    ExperimentSignal("acquire"),
                ],
            )

        self.is_echo = is_echo
        self.n_pi_pulse = n_pi_pulse
        self.qubit_phase = qubit_phase
        
        def _CPMG(n_pi_pulse, qubit_phase):
            """CPMG sequence for Ramsey experiment"""
            if n_pi_pulse == 1:
                exp_ramsey.play(signal="drive", pulse=drive_pulse_pi2)
                exp_ramsey.delay(signal="drive", time=time_sweep/2)
                exp_ramsey.play(signal="drive", pulse=drive_pulse_pi, phase = qubit_phase)
                exp_ramsey.delay(signal="drive", time=time_sweep/2)
                exp_ramsey.play(signal="drive", 
                                pulse=drive_pulse_pi2,
                                phase = 2*np.pi*detuning*time_sweep)
            
            else :
                exp_ramsey.play(signal="drive", pulse=drive_pulse_pi2)
                exp_ramsey.delay(signal="drive", time=time_sweep/(2*n_pi_pulse))
                for _ in range(n_pi_pulse-1):
                    exp_ramsey.play(signal="drive", pulse=drive_pulse_pi, phase = qubit_phase)
                    exp_ramsey.delay(signal="drive", time=time_sweep/(n_pi_pulse))
                exp_ramsey.play(signal="drive", pulse=drive_pulse_pi, phase = qubit_phase)
                exp_ramsey.delay(signal="drive", time=time_sweep/(2*n_pi_pulse))
                exp_ramsey.play(signal="drive", 
                                pulse=drive_pulse_pi2,
                                phase = 2*np.pi*detuning*time_sweep)

        with exp_ramsey.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase = False,
        ):
            # inner loop - real time sweep of Ramsey time delays
            with exp_ramsey.sweep(
                uid="ramsey_sweep", parameter=time_sweep, alignment=SectionAlignment.RIGHT
            ):
                with exp_ramsey.section(uid="ZZ_interaction", alignment=SectionAlignment.RIGHT):
                    if is_zz_interaction:
                        exp_ramsey.play(signal="control_drive", pulse=control_drive_pulse_pi)
                    else:
                        pass
                # play qubit excitation pulse - pulse amplitude is swept
                with exp_ramsey.section(
                    uid="qubit_excitation", alignment=SectionAlignment.RIGHT, play_after="ZZ_interaction"
                ):  
                    if is_echo:
                        _CPMG(n_pi_pulse= n_pi_pulse, qubit_phase = qubit_phase)
                    else:
                        exp_ramsey.play(signal="drive", pulse=drive_pulse_pi2)
                        exp_ramsey.delay(signal="drive", time=time_sweep)
                        exp_ramsey.play(signal="drive", 
                                        pulse=drive_pulse_pi2, 
                                        phase = 2*np.pi*detuning*time_sweep)
                # readout pulse and data acquisition
                with exp_ramsey.section(
                    uid="readout_section", play_after="qubit_excitation"
                ):
                    # play readout pulse on measure line
                    exp_ramsey.play(signal="measure", pulse=readout_pulse, phase = phase)
                    # trigger signal data acquisition
                    exp_ramsey.acquire(
                        signal="acquire",
                        handle="ramsey",
                        kernel=readout_weighting_function,
                    )

                # relax time after readout - for qubit relaxation to groundstate and signal processing
                with exp_ramsey.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_ramsey.reserve(signal="measure")
        
        if is_zz_interaction:
            signal_map_1 = self.signal_map(component)
            exp_ramsey.set_signal_map(signal_map_1)

            signal_map_2 = {
                "control_drive": device_setup.logical_signal_groups[control_component].logical_signals["drive_line"],
            }
            exp_ramsey.set_signal_map(signal_map_2)

        else:
            signal_map = self.signal_map(component)
            exp_ramsey.set_signal_map(signal_map)
        
        
        compiled_experiment_ramsey = self.session.compile(exp_ramsey)
        
        self.ramsey_results = self.session.run(compiled_experiment_ramsey)
        
        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_ramsey, start_time=0, length=20e-6, component=component)
            
    def plot_Ramsey(self, is_fit = True):
        
        ### data processing ###############################################################

        averaged_nums = len(self.ramsey_results.acquired_results['ramsey'].axis[0])
        # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

        self.T2_data = self.ramsey_results.get_data("ramsey") # (2^N, npts) array
        time = self.ramsey_results.acquired_results['ramsey'].axis[1]

        if self.which_data == "I":
            data = np.real(np.mean(self.T2_data, axis = 0))
            std_data = np.real(np.std(self.T2_data, axis = 0)/np.sqrt(averaged_nums))

        else:
            data = np.imag(np.mean(self.T2_data, axis = 0))
            std_data = np.imag(np.std(self.T2_data, axis = 0)/np.sqrt(averaged_nums))

        ### data plot ######################################################################
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.errorbar(time*1e6, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, 
                    ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))

        if is_fit :
            sfit1 = sFit('ExpCos', time, data)
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            _,freq,decay_rate,_,_ = popt
            _,freq_err,decay_rate_err,_,_ = np.sqrt(np.diag(pcov))

            # Var(1/X) = Var(X)/X^4 => std(1/X) = std(X)/X^2

            ax.plot(time*1e6, sfit1.func(time, *popt))
            an = ax.annotate((f'T2 = {(1/decay_rate*1e6):.2f}±{(1/(decay_rate)**2*decay_rate_err*1e6):.2f}[us], freq = {(freq*1e-6):.3f}±{(freq_err*1e-6):.3f}[MHz]'), 
                                xy = (np.average(time), np.average(data)+(np.max(data)-np.min(data))*0.3),
                                size = 16)
            an.draggable()
            ax.tick_params(axis='both', which='major', labelsize=16)
            if self.is_echo:
                if self.qubit_phase == 0:
                    ax.set_title(f"Ramsey measurement with CP : n_pi_pulse = {self.n_pi_pulse}", fontsize=20)
                else:
                    ax.set_title(f"Ramsey measurement with CPMG : n_pi_pulse = {self.n_pi_pulse}", fontsize=20)
            else:
                ax.set_title("Ramsey measurement", fontsize=20)
            ax.set_xlabel("Time (us)", fontsize=20)
            ax.set_ylabel(f'{self.which_data} (a.u.)', fontsize=20)

    
# In[]
    def Rabi_amplitude(self, average_exponent = 12, npts = 100, is_plot_simulation = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        
        drive_pulse = pulse_library.gaussian(uid="drive_pulse", 
                                             length = qubits_parameters[component]['drive_pulse_length'], 
                                             amplitude = qubits_parameters[component]["drive_amp"])
        
        amplitude_sweep = LinearSweepParameter(uid="amplitude_sweep", start=0, stop=1, count=npts)

        exp_rabi = Experiment(
            uid="Amplitude Rabi",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        
        with exp_rabi.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - real time sweep of Rabi ampitudes
            with exp_rabi.sweep(uid="rabi_sweep", parameter=amplitude_sweep):
                # play qubit excitation pulse - pulse amplitude is swept
                with exp_rabi.section(
                    uid="qubit_excitation", alignment=SectionAlignment.RIGHT
                ):
                    exp_rabi.play(
                        signal="drive", pulse=drive_pulse, amplitude=amplitude_sweep
                    )
                # readout pulse and data acquisition
                with exp_rabi.section(uid="readout_section", play_after="qubit_excitation"):
                    # play readout pulse on measure line
                    exp_rabi.play(signal="measure", pulse=readout_pulse)
                    # trigger signal data acquisition
                    exp_rabi.acquire(
                        signal="acquire",
                        handle="amp_rabi",
                        kernel=readout_weighting_function,
                    )
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                with exp_rabi.section(uid="reserve", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_rabi.reserve(signal="measure")
    
        signal_map = self.signal_map(component)
        
        exp_rabi.set_signal_map(signal_map)
        
        compiled_experiment_rabi = self.session.compile(exp_rabi)
        
        self.rabi_amp_results = self.session.run(compiled_experiment_rabi)
        
        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_rabi, start_time=0, length=20e-6, component=component)
    
    def plot_Rabi_amplitude(self):

        self.rabi_amp_data = self.rabi_amp_results.get_data("amp_rabi")
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))

        ax[0].plot(self.rabi_amp_results.acquired_results['amp_rabi'].axis[0], np.real(self.rabi_amp_data), color="r", marker="o")
        ax[1].plot(self.rabi_amp_results.acquired_results['amp_rabi'].axis[0], np.imag(self.rabi_amp_data), color="r", marker="o")

        plt.show()
            

# In[]
    def Rabi_length(self, average_exponent = 12, duration = 100e-6, npts = 100, is_single_shot = True, is_plot_simulation = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        self.exp_Rabi_length_dict = {
            "duration": duration,
            "npts": npts,
        }
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        
        rabi_drive_chunk = pulse_library.const(uid="drive_pulse", 
                                             length = duration/npts, 
                                             amplitude = qubits_parameters[component]["rabi_drive_amp"])
        
        ramp_up = pulse_library.gaussian_rise(uid="ramp_up", 
                                        length=qubits_parameters[component]["ramp_length"], 
                                        amplitude=qubits_parameters[component]["rabi_drive_amp"])
        ramp_down = pulse_library.gaussian_fall(uid="ramp_down", 
                                        length=qubits_parameters[component]["ramp_length"], 
                                        amplitude=qubits_parameters[component]["rabi_drive_amp"])
        
        rabi_length_sweep = LinearSweepParameter(uid="pulses", start=0, stop=npts-1, count=npts)

        
        def repeat(count: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                if v == 0:
                                    exp.play(signal="drive", pulse=ramp_up)
                                    exp.play(signal="drive", pulse=ramp_down)
                                else:
                                    exp.play(signal="drive", pulse=ramp_up)
                                    for _ in range(int(v)):
                                        f()
                                    exp.play(signal="drive", pulse=ramp_down)
                else:
                    for _ in range(count):
                        f()

            return decorator
        
        if is_single_shot :
            averaging_mode = AveragingMode.SINGLE_SHOT
            self.is_single_shot = True
        else:
            averaging_mode = AveragingMode.CYCLIC
            self.is_single_shot = False
        
        exp_rabi_length = Experiment(
            uid="Rabi_length",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_rabi_length.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=averaging_mode,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            
            with exp_rabi_length.sweep(uid="rabi_length_sweep", parameter= rabi_length_sweep, auto_chunking=True):
                with exp_rabi_length.section(uid="rabi_drives", alignment=SectionAlignment.RIGHT):
                    @repeat(rabi_length_sweep, exp_rabi_length)
                    def play_rabi():
                        exp_rabi_length.play(signal = "drive", 
                                            pulse = rabi_drive_chunk)
                                             
                # readout pulse and data acquisition
                with exp_rabi_length.section(uid="readout_section", play_after="rabi_drives"):
                    # play readout pulse on measure line
                    exp_rabi_length.play(signal="measure", pulse=readout_pulse, phase = qubits_parameters[component]["readout_phase"])
                    # trigger signal data acquisition
                    exp_rabi_length.acquire(
                        signal="acquire",
                        handle="rabi_length",
                        kernel=readout_weighting_function,
                    )
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                with exp_rabi_length.section(uid="reserve", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_rabi_length.reserve(signal="measure")
    
        signal_map = self.signal_map(component)
        
        exp_rabi_length.set_signal_map(signal_map)
        
        compiled_experiment_rabi = self.session.compile(exp_rabi_length)
        
        self.rabi_length_results = self.session.run(compiled_experiment_rabi)
        
        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_rabi, start_time=0, length=20e-6, component=component)
            
    def plot_Rabi_length(self, is_fit = True):
        
        if self.is_single_shot:
        ### data processing ###############################################################

            averaged_nums = len(self.rabi_length_results.acquired_results['rabi_length'].axis[0])
            # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

            self.rabi_length_data = self.rabi_length_results.get_data("rabi_length") # (2^N, npts) array
            # time = self.rabi_length_results.acquired_results['rabi_length'].axis[1]

            time = np.linspace(0, self.exp_Rabi_length_dict["duration"], self.exp_Rabi_length_dict["npts"])

            averaged_data = np.mean(self.rabi_length_data, axis = 0)

            if self.which_data == "I":
                data = np.real(averaged_data)
                std_data = np.real(np.std(self.rabi_length_data, axis = 0)/np.sqrt(averaged_nums))
            else:
                data = np.imag(averaged_data)
                std_data = np.imag(np.std(self.rabi_length_data, axis = 0)/np.sqrt(averaged_nums))

        ### data plot ######################################################################
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.errorbar(time*1e6, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, 
                        ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
        
        else:
        ### data processing ###############################################################
            self.rabi_length_data = self.rabi_length_results.get_data("rabi_length")

            if self.which_data == "I":
                data = np.real(self.rabi_length_data)
            else:
                data = np.imag(self.rabi_length_data)

            time = np.linspace(0, self.exp_Rabi_length_dict["duration"], self.exp_Rabi_length_dict["npts"])

        ### data plot ######################################################################

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.plot(time*1e6, data, color="r", marker="o", linestyle = '--', markeredgecolor='black')

        if is_fit :
            sfit1 = sFit('ExpCos', time, data)
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            _,freq,decay_rate,_,_ = popt
            _,freq_err,decay_rate_err,_,_ = np.sqrt(np.diag(pcov))

            # Var(1/X) = Var(X)/X^4 => std(1/X) = std(X)/X^2

            ax.plot(time*1e6, sfit1.func(time, *popt))
            an = ax.annotate((f'decay time = {(1/decay_rate*1e6):.2f}±{(1/(decay_rate)**2*decay_rate_err*1e6):.2f}[us], freq = {(freq*1e-6):.3f}±{(freq_err*1e-6):.3f}[MHz]'), 
                                xy = (np.average(time), np.average(data)+(np.max(data)-np.min(data))*0.3),
                                size = 16)
            an.draggable()
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_title("Rabi_length measurement", fontsize=20)
            ax.set_xlabel("Time (us)", fontsize=20)
            ax.set_ylabel(f'{self.which_data} (a.u.)', fontsize=20)

# In[]

    def cr_calibration_amp(self, average_exponent = 12, npts = 12, control_qubit = 0, target_qubit = 1, is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        
        control_qubit_component = list(qubits_parameters.keys())[control_qubit]
        target_qubit_component = list(qubits_parameters.keys())[target_qubit]
        
        ## define pulses used for experiment
        target_qubit_readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[target_qubit_component]["readout_pulse_length"], 
            amplitude=qubits_parameters[target_qubit_component]["readout_amp"], 
        )
        
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        target_qubit_readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[target_qubit_component]["readout_integration_length"],
            amplitude=qubits_parameters[target_qubit_component]["readout_integration_amp"], 
        )
        
        
        target_qubit_drive_pulse = pulse_library.gaussian(uid="target_qubit_drive_pulse", 
                                length = qubits_parameters[target_qubit_component]['drive_pulse_length'], 
                                amplitude = qubits_parameters[target_qubit_component]["drive_amp"])
        
        control_qubit_drive_pulse = pulse_library.gaussian(uid="control_qubit_drive_pulse", 
                                length = qubits_parameters[control_qubit_component]['drive_pulse_length'], 
                                amplitude = qubits_parameters[control_qubit_component]["drive_amp"])
        
        amplitude_sweep = LinearSweepParameter(uid="amplitude_sweep", start=0, stop=1, count=npts)
        on_off_cases = LinearSweepParameter(uid="on_off_case", start=0, stop=1, count=2)

        phase = qubits_parameters[target_qubit_component]["readout_phase"]

        exp_cr = Experiment(
            uid="cr",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("drive_ef"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        def on_off_control_drive(on_off_cases: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(on_off_cases, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=on_off_cases):
                        for v in on_off_cases.values:
                            with exp.case(v):
                                f(v)

            return decorator
        
   
        with exp_cr.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - real time sweep of Rabi ampitudes
            with exp_cr.sweep(uid="rabi_sweep", parameter=amplitude_sweep):
                # play qubit excitation pulse - pulse amplitude is swept
                with exp_cr.sweep(uid="on_off_case", parameter=on_off_cases):
                    with exp_cr.section(
                        uid="control_qubit_excitation", alignment=SectionAlignment.RIGHT
                    ):
                        
                        @on_off_control_drive(on_off_cases, exp = exp_cr)
                        def play_control_drive(on_off_case):
                            with exp_cr.section():
                                if on_off_case == 0:
                                    pass
                                else:
                                    exp_cr.play(signal="drive", pulse=control_qubit_drive_pulse)

                    with exp_cr.section(
                        uid="target_qubit_drive", play_after = "control_qubit_excitation"
                    ):
                        exp_cr.play(
                            signal="drive_ef", pulse=target_qubit_drive_pulse, amplitude=amplitude_sweep
                        )
                    # readout pulse and data acquisition
                    with exp_cr.section(uid="readout_section", play_after="target_qubit_drive"):
                        # play readout pulse on measure line
                        exp_cr.play(signal="measure", pulse=target_qubit_readout_pulse, phase = phase)
                        # trigger signal data acquisition
                        exp_cr.acquire(
                            signal="acquire",
                            handle="cr_caliibration",
                            kernel=target_qubit_readout_weighting_function,
                        )
                    # relax time after readout - for qubit relaxation to groundstate and signal processing
                    with exp_cr.section(uid="reserve", length=qubits_parameters[target_qubit_component]["reset_delay_length"]):
                        exp_cr.reserve(signal="measure")
    
        signal_map = self.signal_map(control_qubit_component, which_qubit = "control")
        exp_cr.set_signal_map(signal_map)
        signal_map = self.signal_map(target_qubit_component, which_qubit= "target")
        exp_cr.set_signal_map(signal_map)
        
        compiled_experiment_cr_calib_amp = self.session.compile(exp_cr)
        
        self.cr_calibration_results_amp = self.session.run(compiled_experiment_cr_calib_amp)
        
        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_cr_calib_amp, start_time=0, length=20e-6)

    def plot_cr_calibration_amp(self, is_fit = True):

        ### data processing ###############################################################

        control_off_data = self.cr_calibration_results_amp.get_data("cr_caliibration")[:,0]
        control_on_data = self.cr_calibration_results_amp.get_data("cr_caliibration")[:,1]

        amp_sweep = self.cr_calibration_results_amp.acquired_results['cr_caliibration'].axis[0]

        if self.which_data == "I":
            control_off_data = np.real(control_off_data)
            control_on_data = np.real(control_on_data)
        else:
            control_off_data = np.imag(control_off_data)
            control_on_data = np.imag(control_on_data)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(amp_sweep, control_off_data, color="b", marker="o", label = "control qubit off")
        ax.plot(amp_sweep, control_on_data, color="r", marker="o", label = "control qubit on")
        ax.set_title("CR calibration", fontsize=20)
        ax.set_xlabel("Amplitude", fontsize=20)
        ax.set_ylabel(f'{self.which_data} (a.u.)', fontsize=20)
        ax.legend()



    def cr_calibration_length(self, average_exponent = 12, npts = 100, duration = 1e-6, target_qubit_amp = 0.5,
                            control_qubit = 0, target_qubit = 1, is_plot_simulation = False):
        
        self.exp_cr_calibration_length = {
            "duration": duration,
            "target_qubit_amp": target_qubit_amp,
            "npts": npts,
        }

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        
        control_qubit_component = list(qubits_parameters.keys())[control_qubit]
        target_qubit_component = list(qubits_parameters.keys())[target_qubit]
        
        ## define pulses used for experiment
        target_qubit_readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[target_qubit_component]["readout_pulse_length"], 
            amplitude=qubits_parameters[target_qubit_component]["readout_amp"], 
        )
        
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        target_qubit_readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[target_qubit_component]["readout_integration_length"],
            amplitude=qubits_parameters[target_qubit_component]["readout_integration_amp"], 
        )
        
        control_qubit_drive_pi_pulse = pulse_library.gaussian(uid="drive_pulse", 
                                length = qubits_parameters[control_qubit_component]['pi_length'], 
                                amplitude = qubits_parameters[control_qubit_component]["pi_amp"])
        
        target_qubit_pi2_pulse = pulse_library.gaussian(uid="target_qubit_pulse", 
                                length = qubits_parameters[target_qubit_component]['pi2_length'], 
                                amplitude = qubits_parameters[target_qubit_component]["pi2_amp"])
        
        target_drive_chunk = pulse_library.const(uid="target_drive_chunk", 
                                             length = duration/npts, 
                                             amplitude = target_qubit_amp)
        
        ramp_up = pulse_library.gaussian_rise(uid="ramp_up", 
                                        length=qubits_parameters[target_qubit_component]["ramp_length"], 
                                        amplitude=target_qubit_amp)
        ramp_down = pulse_library.gaussian_fall(uid="ramp_down", 
                                        length=qubits_parameters[target_qubit_component]["ramp_length"], 
                                        amplitude=target_qubit_amp)
        
        target_length_sweep = LinearSweepParameter(uid="pulses", start=0, stop=npts-1, count=npts)
        on_off_cases = LinearSweepParameter(uid="on_off_case", start=0, stop=1, count=2)
        x_y_z_cases = LinearSweepParameter(uid="x_y_z_case", start=0, stop=2, count=3)

        phase = qubits_parameters[target_qubit_component]["readout_phase"]

        exp_cr = Experiment(
            uid="cr",
            signals=[
                ExperimentSignal("control_drive"),
                ExperimentSignal("target_drive_1"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
                ExperimentSignal("drive"),
            ],
        )

        def repeat(count: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                if v == 0:
                                    exp.play(signal="target_drive_1", pulse=ramp_up)
                                    exp.play(signal="target_drive_1", pulse=ramp_down)
                                else:
                                    exp.play(signal="target_drive_1", pulse=ramp_up)
                                    for _ in range(int(v)):
                                        f()
                                    exp.play(signal="target_drive_1", pulse=ramp_down)
                else:
                    for _ in range(count):
                        f()

            return decorator


        def switch(cases: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(cases, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=cases):
                        for v in cases.values:
                            with exp.case(v):
                                f(v)

            return decorator
        

        with exp_cr.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - real time sweep of Rabi ampitudes
            with exp_cr.sweep(uid="rabi_sweep", parameter=target_length_sweep):
                # play qubit excitation pulse - pulse amplitude is swept
                with exp_cr.sweep(uid="on_off_case", parameter=on_off_cases):
                    with exp_cr.sweep(uid="x_y_z_case", parameter=x_y_z_cases):
                        with exp_cr.section(
                            uid="control_qubit_excitation", alignment=SectionAlignment.RIGHT
                        ):
                            
                            @switch(on_off_cases, exp = exp_cr)
                            def play_control_drive(on_off_case):
                                with exp_cr.section():
                                    if on_off_case == 0:
                                        pass
                                    else:
                                        exp_cr.play(signal="control_drive", pulse=control_qubit_drive_pi_pulse)

                        with exp_cr.section(
                            uid="target_qubit_drive", play_after = "control_qubit_excitation"
                        ):
                            @repeat(target_length_sweep, exp_cr)
                            def play_rabi():
                                exp_cr.play(signal = "target_drive_1", pulse = target_drive_chunk)

                        with exp_cr.section(uid = "x_y_z_measurement", play_after = "target_qubit_drive"):
                            @switch(x_y_z_cases, exp = exp_cr)
                            def play_x_y_z(x_y_z_case):
                                with exp_cr.section():
                                    if x_y_z_case == 0: #  X measure
                                        exp_cr.play(signal="drive", pulse=target_qubit_pi2_pulse, phase = 0)
                                    elif x_y_z_case == 1: # Y measure
                                        exp_cr.play(signal="drive", pulse=target_qubit_pi2_pulse, phase = np.pi/2)
                                    else: # Z measure
                                        pass
                        # readout pulse and data acquisition
                        with exp_cr.section(uid="readout_section", play_after="x_y_z_measurement"):
                            # play readout pulse on measure line
                            exp_cr.play(signal="measure", pulse=target_qubit_readout_pulse, phase = phase)
                            # trigger signal data acquisition
                            exp_cr.acquire(
                                signal="acquire",
                                handle="cr_caliibration",
                                kernel=target_qubit_readout_weighting_function,
                            )
                        # relax time after readout - for qubit relaxation to groundstate and signal processing
                        with exp_cr.section(uid="reserve", length=qubits_parameters[target_qubit_component]["reset_delay_length"]):
                            exp_cr.reserve(signal="measure")
    
        signal_map = self.signal_map(control_qubit_component, which_qubit = "control")
        exp_cr.set_signal_map(signal_map)
        signal_map = self.signal_map(target_qubit_component)
        exp_cr.set_signal_map(signal_map)
        
        compiled_experiment_cr_calib_length = self.session.compile(exp_cr)
        
        self.cr_calibration_results_length = self.session.run(compiled_experiment_cr_calib_length)
        
        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_cr_calib_length, start_time=0, length=20e-6)

    
    def plot_cr_calibration_length(self, is_fit = True):

        ### data processing ###############################################################

        control_off_X_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,0,0]
        control_on_X_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,1,0]

        control_off_Y_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,0,1]
        control_on_Y_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,1,1]

        control_off_Z_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,0,2]
        control_on_Z_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,1,2]

        time = np.linspace(0, self.exp_cr_calibration_length["duration"], self.exp_cr_calibration_length["npts"])

        if self.which_data == "I":
            control_off_X_data = np.real(control_off_X_data)
            control_on_X_data = np.real(control_on_X_data)
            control_off_Y_data = np.real(control_off_Y_data)
            control_on_Y_data = np.real(control_on_Y_data)
            control_off_Z_data = np.real(control_off_Z_data)
            control_on_Z_data = np.real(control_on_Z_data)
        else:
            control_off_X_data = np.imag(control_off_X_data)
            control_on_X_data = np.imag(control_on_X_data)
            control_off_Y_data = np.imag(control_off_Y_data)
            control_on_Y_data = np.imag(control_on_Y_data)
            control_off_Z_data = np.imag(control_off_Z_data)
            control_on_Z_data = np.imag(control_on_Z_data)

        fig, ax = plt.subplots(3, 1, figsize=(10, 20))

        ax[0].plot(time*1e6, control_off_X_data, color="b", marker="o", label = "control qubit off <X>")
        ax[0].plot(time*1e6, control_on_X_data, color="r", marker="o", label = "control qubit on <X>")
        ax[1].plot(time*1e6, control_off_Y_data, color="b", marker="o", label = "control qubit off <Y>")
        ax[1].plot(time*1e6, control_on_Y_data, color="r", marker="o", label = "control qubit on <Y>")
        ax[2].plot(time*1e6, control_off_Z_data, color="b", marker="o", label = "control qubit off <Z>")
        ax[2].plot(time*1e6, control_on_Z_data, color="r", marker="o", label = "control qubit on <Z>")

        fig.suptitle(f"CR calibration, target qubit amp : {self.exp_cr_calibration_length["target_qubit_amp"]}", fontsize=20)

        fig.text(0.5, 0.04, "Time [us]", ha="center", fontsize=20)
        fig.text(0.04, 0.5, f"{self.which_data} (a.u.)", va="center", rotation="vertical", fontsize=20)

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

# In[]

    def All_XY(self, average_exponent = 12, is_plot_simulation = False):

        self.AllXY_pulses = [["I", "I"],
                            ["X", "X"],
                            ["Y", "Y"],
                            ["X", "Y"],
                            ["Y", "X"],
                            ["X/2", "I"],
                            ["Y/2", "I"],
                            ["X/2", "Y/2"],
                            ["Y/2", "X/2"],
                            ["X/2", "Y"],
                            ["Y/2", "X"],
                            ["X", "Y/2"],
                            ["Y", "X/2"],
                            ["X/2", "X"],
                            ["X", "X/2"],
                            ["Y/2", "Y"],
                            ["Y", "Y/2"],
                            ["X", "I"],
                            ["Y", "I"],
                            ["X/2", "X/2"],
                            ["Y/2", "Y/2"]]
        
        AllXY_pulses = self.AllXY_pulses

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        pi2_pulse = pulse_library.drag(
            uid="pi2_pulse", 
            length = qubits_parameters[component]['pi2_length'], 
            amplitude = qubits_parameters[component]["pi2_amp"],
            beta = qubits_parameters[component]["pi2_beta"]
        )
        
        pi_pulse = pulse_library.drag(
            uid="pi_pulse", 
            length = qubits_parameters[component]['pi_length'], 
            amplitude = qubits_parameters[component]["pi_amp"],
            beta = qubits_parameters[component]["pi_beta"]
        )

        phase = qubits_parameters[component]["readout_phase"]

        def label_to_drag_play(name):
                if name == "I":
                    return False
                elif name == "X":
                    return pi_pulse, 0, {'beta': qubits_parameters[component]["pi_beta"]}
                elif name == "Y":
                    return pi_pulse, np.pi/2, {'beta': qubits_parameters[component]["pi_beta"]}
                elif name == "X/2":
                    return pi2_pulse, 0, {'beta': qubits_parameters[component]["pi2_beta"]}
                elif name == "Y/2":
                    return pi2_pulse, np.pi/2, {'beta': qubits_parameters[component]["pi2_beta"]}

        # Calculate the length of each experiment sections
        drive_section_length = (np.ceil((pi_pulse.length + pi2_pulse.length)/64e-9))*64e-9
        readout_section_length = (np.ceil(readout_pulse.length/64e-9))*64e-9
        reset_section_length = (np.ceil((qubits_parameters[component]["reset_delay_length"])/64e-9))*64e-9

        All_XY_sweep=LinearSweepParameter(uid="All_XY_sweep", start = 0, stop = len(AllXY_pulses)-1, count = len(AllXY_pulses))

        # define the sweep function
        def sweep_exp_cases(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for exp_case_num in sweep_case.values:
                            with exp.case(exp_case_num):
                                f(exp_case_num)
            return decorator
        

        exp_All_XY = Experiment(
            uid="All_XY",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_All_XY.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):

            with exp_All_XY.sweep(uid="sweep", parameter=All_XY_sweep, alignment=SectionAlignment.RIGHT):
                with exp_All_XY.section(uid="All_XY", length = drive_section_length, alignment=SectionAlignment.RIGHT):
                    @sweep_exp_cases(All_XY_sweep, exp = exp_All_XY)
                    def play_all_XY(exp_case_num):
                    
                        with exp_All_XY.section():
                            if not AllXY_pulses[round(exp_case_num)][0] == "I":
                                exp_All_XY.play(signal = "drive", 
                                                pulse = label_to_drag_play(AllXY_pulses[round(exp_case_num)][0])[0],
                                                phase = label_to_drag_play(AllXY_pulses[round(exp_case_num)][0])[1],
                                                pulse_parameters = label_to_drag_play(AllXY_pulses[round(exp_case_num)][0])[2],
                                                )
                            if not AllXY_pulses[round(exp_case_num)][1] == "I":
                                exp_All_XY.play(signal = "drive",
                                                pulse = label_to_drag_play(AllXY_pulses[round(exp_case_num)][1])[0],
                                                phase = label_to_drag_play(AllXY_pulses[round(exp_case_num)][1])[1],
                                                pulse_parameters = label_to_drag_play(AllXY_pulses[round(exp_case_num)][1])[2],
                                                )
                            if (AllXY_pulses[round(exp_case_num)][0] == "I") and (AllXY_pulses[round(exp_case_num)][1] == "I"):
                                exp_All_XY.reserve(signal = "drive")
                
                with exp_All_XY.section(uid="measure", play_after="All_XY", alignment=SectionAlignment.LEFT, length = readout_section_length):
                    exp_All_XY.play(signal="measure", pulse=readout_pulse, phase = phase)
                    exp_All_XY.acquire(
                        signal="acquire",
                        handle="All_XY",
                        kernel=readout_weighting_function,
                    )

                with exp_All_XY.section(uid="relax", length=reset_section_length):
                    exp_All_XY.reserve(signal="measure")

        signal_map = self.signal_map(component)

        exp_All_XY.set_signal_map(signal_map)

        compiled_experiment_All_XY = self.session.compile(exp_All_XY)

        self.All_XY_results = self.session.run(compiled_experiment_All_XY)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_All_XY, start_time=0, length=20e-6, component=component)
        
    def plot_All_XY(self):

        self.All_XY_data = self.All_XY_results.get_data("All_XY")
        
        if self.which_data == "I":
            data = np.real(self.All_XY_data)
        else:
            data = np.imag(self.All_XY_data)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        data_min = np.min(data)
        data_max = np.max(data)
        for grid_n in range(len(data)):
            ax.plot([np.arange(len(self.AllXY_pulses))[grid_n], np.arange(len(self.AllXY_pulses))[grid_n]], 
                    [data_min - 0.1*(data_max-data_min), data[grid_n]], linestyle='-', color='b', alpha=0.3)

        ax.plot(np.arange(len(self.AllXY_pulses)), data, label='I', marker='o', linestyle=':', color='k')
        
        # x axis labels: AllXY pulses names
        ax.set_xticks(np.arange(len(self.AllXY_pulses)))
        ax.set_xticklabels(self.AllXY_pulses)
        ax.set_title('AllXY data')
        ax.tick_params(axis='x', rotation=80, labelsize=10)

# In[]

    def drag_calibration(self, average_exponent = 12, beta_start = 0, beta_stop = 1, beta_count = 11, is_plot_simulation = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        
        pi2_pulse = pulse_library.drag(
            uid="pi2_pulse", 
            length = qubits_parameters[component]['pi2_length'], 
            amplitude = qubits_parameters[component]["pi2_amp"],
            beta = qubits_parameters[component]["pi2_beta"]
        )
        
        pi_pulse = pulse_library.drag(
            uid="pi_pulse", 
            length = qubits_parameters[component]['pi_length'], 
            amplitude = qubits_parameters[component]["pi_amp"],
            beta = qubits_parameters[component]["pi_beta"]
        )
        
        sweep_case = LinearSweepParameter(uid="sweep_case", start=0, stop=2, count=3)
        sweep_betas = LinearSweepParameter(uid="sweep_betas", start = beta_start, stop = beta_stop, count = beta_count)

        phase = qubits_parameters[component]["readout_phase"]
        # Calculate the length of each experiment sections
        drive_section_length = (np.ceil((pi_pulse.length + pi2_pulse.length)/64e-9))*64e-9
        readout_section_length = (np.ceil(readout_pulse.length/64e-9))*64e-9
        reset_section_length = (np.ceil((qubits_parameters[component]["reset_delay_length"])/64e-9))*64e-9

        def sweep_exp_cases(sweep_case: int | SweepParameter | LinearSweepParameter,
                            sweep_betas: int | SweepParameter | LinearSweepParameter,
                            exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for exp_case_num in sweep_case.values:
                            with exp.case(exp_case_num):
                                with exp.match(sweep_parameter=sweep_betas):
                                    for sweep_value in sweep_betas.values:
                                        with exp.case(sweep_value):
                                            f(exp_case_num, sweep_value)

            return decorator
        
        exp_drag_calibration = Experiment(
            uid="drag_calibration",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_drag_calibration.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_drag_calibration.sweep(uid="sweep", parameter= sweep_case):
                with exp_drag_calibration.sweep(uid="drag_sweep", parameter= sweep_betas):
                    with exp_drag_calibration.section(uid="drag_calibration", 
                                                      length = drive_section_length,
                                                      alignment=SectionAlignment.RIGHT):
                        @sweep_exp_cases(sweep_case, sweep_betas, exp = exp_drag_calibration)
                        def play_drag_calibration(exp_case_num, sweep_value):
                            
                            with exp_drag_calibration.section():
                                if exp_case_num == 0:
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi2_pulse,
                                        phase = 0, 
                                        pulse_parameters={"beta": sweep_value})
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi_pulse,
                                        phase = np.pi/2, 
                                        pulse_parameters={"beta": sweep_value})
                                elif exp_case_num == 1:
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi2_pulse,
                                        phase = 0, 
                                        pulse_parameters={"beta": sweep_value})
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi_pulse,
                                        phase = 0, 
                                        pulse_parameters={"beta": sweep_value})
                                elif exp_case_num == 2:
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi2_pulse,
                                        phase = 0, 
                                        pulse_parameters={"beta": sweep_value})
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi_pulse,
                                        phase = -np.pi/2, 
                                        pulse_parameters={"beta": sweep_value})
                
                    with exp_drag_calibration.section(uid="measure", play_after="drag_calibration"):
                        exp_drag_calibration.play(signal="measure", pulse=readout_pulse, phase = phase)
                        exp_drag_calibration.acquire(
                            signal="acquire",
                            handle="drag_calibration",
                            kernel=readout_weighting_function,
                        )
                    with exp_drag_calibration.section(uid="relax", length=reset_section_length):
                        exp_drag_calibration.reserve(signal="measure")
        
        signal_map = self.signal_map(component)

        exp_drag_calibration.set_signal_map(signal_map)

        compiled_experiment_drag_calibration = self.session.compile(exp_drag_calibration)

        self.drag_calibration_results = self.session.run(compiled_experiment_drag_calibration)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_drag_calibration, start_time=0, length=20e-6, component=component)

    def plot_drag_calibration(self):

        self.drag_calibration_data = self.drag_calibration_results.get_data("drag_calibration")

        if self.which_data == "I":
            data = np.real(self.drag_calibration_data)
        else:
            data = np.imag(self.drag_calibration_data)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        beta_sweep = self.drag_calibration_results.acquired_results['drag_calibration'].axis[1]

        ax.plot(beta_sweep, data[0], label='Y(+pi)', marker='o', linestyle=':', color='r')
        ax.plot(beta_sweep, data[1], label='X(+pi)', marker='o', linestyle=':', color='m')
        ax.plot(beta_sweep, data[2], label='Y(-pi)', marker='o', linestyle=':', color='y')
        ax.set_title('beta sweep - I data')
        ax.set_xlabel('beta')
        ax.set_ylabel('I')
        ax.legend()

        # calculate crossing point
        dist_I_data = (data[0]-data[1])**2 + (data[1]-data[2])**2 + (data[2]-data[0])**2
        cross_idx = np.argmin(dist_I_data)
        ax.scatter([beta_sweep[cross_idx]],[data[0][cross_idx]], color='k', marker='s', s=100)
        ax.plot([beta_sweep[cross_idx],beta_sweep[cross_idx]],[np.min(np.concatenate((data[0],data[1],data[2]))), np.max(np.concatenate((data[0],data[1],data[2])))], 'k')
        ax.text(beta_sweep[cross_idx]*0.9+np.max(beta_sweep)*0.1,np.min(np.concatenate((data[0],data[1],data[2])))*0.9+np.max(np.concatenate((data[0],data[1],data[2])))*0.1,s=f"best beta={beta_sweep[cross_idx]:.6f}")

# In[]
    def error_amplification(self, average_exponent=12, pulse_npts=12, amp_npts=10, start_amp=0, end_amp=1,
                            is_drag_beta_calibration=False, 
                            is_plot_simulation=False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        component = list(qubits_parameters.keys())[self.which_qubit]

        # Define pulses
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse",
            length=qubits_parameters[component]["readout_pulse_length"],
            amplitude=qubits_parameters[component]["readout_amp"]
        )

        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function",
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"]
        )

        pi_pulse = pulse_library.drag(
            uid="pi_pulse",
            length=qubits_parameters[component]["pi_length"],
            amplitude=qubits_parameters[component]["pi_amp"],
            beta=qubits_parameters[component]["pi_beta"]
        )

        phase = qubits_parameters[component]["readout_phase"]

        # Sweep parameters
        pulse_count = LinearSweepParameter(uid="pulses", start=0, stop=pulse_npts - 1, count=pulse_npts)

        amp_sweep = LinearSweepParameter(uid="amplitude", start=start_amp, stop=end_amp, count=amp_npts)

        # Repeat decorator for custom looping
        def repeat(count, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                for _ in range(int(v)):
                                    f()
                else:
                    for _ in range(count):
                        f()

            return decorator

        # Define the experiment
        error_amplification = Experiment(
            uid="error_amplification",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with error_amplification.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with error_amplification.sweep(uid="sweep1", parameter=pulse_count):
                with error_amplification.sweep(uid="sweep2", parameter=amp_sweep):

                    with error_amplification.section(uid="pi_section", alignment=SectionAlignment.RIGHT):
                        @repeat(pulse_count, error_amplification)
                        def play_pi():
                            if not is_drag_beta_calibration:
                                error_amplification.play(signal="drive", pulse=pi_pulse, 
                                                        amplitude=amp_sweep)
                                error_amplification.play(signal="drive", pulse=pi_pulse, 
                                                        amplitude=amp_sweep)
                            else:
                                error_amplification.play(signal="drive", pulse=pi_pulse,
                                                         phase = 0,
                                                        pulse_parameters={"beta": amp_sweep})
                                error_amplification.play(signal="drive", pulse=pi_pulse,
                                                         phase = np.pi,
                                                        pulse_parameters={"beta": amp_sweep})

                    with error_amplification.section(uid="measure", play_after="pi_section"):
                        error_amplification.play(signal="measure", pulse=readout_pulse, phase=phase)
                        error_amplification.acquire(
                            signal="acquire",
                            handle="ac_error_amplification",
                            kernel=readout_weighting_function
                        )

                    with error_amplification.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                        error_amplification.reserve(signal="measure")

        signal_map = self.signal_map(component)
        error_amplification.set_signal_map(signal_map)

        compiled_error_amplification = self.session.compile(error_amplification)
        self.error_amplification_results = self.session.run(compiled_error_amplification)

        if is_plot_simulation:
            self.simulation_plot(compiled_error_amplification, start_time=0, length=40e-6, component=component)
        
    def plot_error_amplification(self, is_drag_beta_calibration=True):
        qubits_parameters = self.qubits_parameters
        component = list(qubits_parameters.keys())[self.which_qubit]

        self.error_amplification_data = self.error_amplification_results.get_data("ac_error_amplification")  # (2^N, npts) array
        data = np.real(self.error_amplification_data)

        x_vals = np.round(self.error_amplification_results.get_axis("ac_error_amplification")[1], 3)

        plt.figure(figsize=(14, 10))
        plt.imshow(data, cmap='viridis', origin='lower', aspect='auto')
        plt.title("Error amplification")

        step = 3  
        tick_positions = np.arange(0, len(x_vals), step)
        tick_labels = [x_vals[i] for i in tick_positions]

        plt.xticks(tick_positions, tick_labels, rotation=60)

        if is_drag_beta_calibration:
            plt.xlabel("drag beta")
        else:
            plt.xlabel("drive amplitude")

        plt.ylabel("# of Rabi pulses(pi x 2)")
        plt.colorbar(label='I value')
        plt.tight_layout()
        plt.show()

# %% cavity crosskerr effect
    def cavity_T1(self, average_exponent=12, start = 1e-6, duration=100e-6, npts=101, is_plot_simulation=False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters

        component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[component]["readout_phase"]

        # Sweep parameters
        delay_sweep = LinearSweepParameter(uid="delay", start=start, stop=start+duration, count=npts)

        sweep_case_2 = LinearSweepParameter(uid="crosskerr_check", start=0, stop=1, count=2)

        def on_off_cond_pi_pulse(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)

            return decorator

        # Define the experiment
        exp_cavity_T1 = Experiment(
            uid="cavity_T1",
            signals=[
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_cavity_T1.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # with exp_cavity_T1.sweep(uid="crosskerr_check", parameter=sweep_case_2):
            with exp_cavity_T1.sweep(uid="sweep", parameter=delay_sweep, alignment=SectionAlignment.RIGHT):
                with exp_cavity_T1.section(uid="cavity_excitation", alignment=SectionAlignment.RIGHT):
                    exp_cavity_T1.play(signal="cavity_drive", pulse=cavity_drive_pulse)
                    exp_cavity_T1.delay(signal="cavity_drive", time=delay_sweep)

                # with exp_cavity_T1.section(uid="cond_pi_pulse", play_after="cavity_excitation"):
                #     @on_off_cond_pi_pulse(sweep_case_2, exp=exp_cavity_T1)
                #     def play_crosskerr_check(v):
                #         if v == 0:
                #             pass
                #         elif v == 1:
                #             exp_cavity_T1.play(signal="drive", pulse=cond_pi_pulse)

                with exp_cavity_T1.section(uid="measure", play_after="cavity_excitation"):
                    exp_cavity_T1.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_cavity_T1.acquire(
                        signal="acquire",
                        handle="cavity_T1",
                        kernel=readout_weighting_function,
                    )
                with exp_cavity_T1.section(uid="relax", 
                                        length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_cavity_T1.reserve(signal="measure")
        
        signal_map = {
            "measure": device_setup.logical_signal_groups[component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_cavity_T1.set_signal_map(signal_map)

        compiled_experiment_cavity_T1 = self.session.compile(exp_cavity_T1)

        self.cavity_T1_results = self.session.run(compiled_experiment_cavity_T1)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_cavity_T1, start_time=0, length=20e-6)

    def plot_cavity_T1(self, is_fit = True):

        ### data processing ###############################################################

        averaged_nums = len(self.cavity_T1_results.acquired_results['cavity_T1'].axis[0])
        # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

        self.cavity_T1_data = self.cavity_T1_results.get_data("cavity_T1") # (2^N, npts) array
        time = self.cavity_T1_results.acquired_results['cavity_T1'].axis[1]
        # time = self.cavity_T1_results.acquired_results['cavity_T1'].axis[2]

        if self.which_data == "I":
            
            # data_0 = np.real(np.mean(self.cavity_T1_data, axis = 0)[0]) # v ==0
            # data_1 = np.real(np.mean(self.cavity_T1_data, axis = 0)[1]) # v ==1
            # std_data_0 = np.real(np.std(self.cavity_T1_data, axis = 0)[0]/np.sqrt(averaged_nums))
            # std_data_1 = np.real(np.std(self.cavity_T1_data, axis = 0)[1]/np.sqrt(averaged_nums))

            data = np.real(np.mean(self.cavity_T1_data, axis = 0))
            std_data = np.real(np.std(self.cavity_T1_data, axis = 0)/np.sqrt(averaged_nums))
           
        else:
            # data_0 = np.imag(np.mean(self.cavity_T1_data, axis = 0)[0]) # v ==0
            # data_1 = np.imag(np.mean(self.cavity_T1_data, axis = 0)[1]) # v ==1
            # std_data_0 = np.imag(np.std(self.cavity_T1_data, axis = 0)[0]/np.sqrt(averaged_nums))
            # std_data_1 = np.imag(np.std(self.cavity_T1_data, axis = 0)[1]/np.sqrt(averaged_nums))

            data = np.imag(np.mean(self.cavity_T1_data, axis = 0))
            std_data = np.imag(np.std(self.cavity_T1_data, axis = 0)/np.sqrt(averaged_nums))


        ### data plot ######################################################################

        # fig1, ax1 = plt.subplots(1, 2, figsize=(10, 20))

        # ax1[0].errorbar(time*1e6, data_0, yerr = std_data_0, fmt = '--or', capsize = 5, markersize = 3, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
        # ax1[1].errorbar(time*1e6, data_1, yerr = std_data_1, fmt = '--ob', capsize = 5, markersize = 3, ecolor = 'k', mfc=(0,0,1,0.5), mec = (0,0,0,1))

        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
        ax2.errorbar(time*1e6, data, yerr = std_data, fmt = '--ok', capsize = 5, markersize = 3, ecolor = 'k', mfc=(0,0,0,0.5), mec = (0,0,0,1))
        # ax2.errorbar(time*1e6, data_1-data_0, yerr = std_data_0+std_data_1, fmt = '--ok', capsize = 5, markersize = 3, ecolor = 'k', mfc=(0,0,0,0.5), mec = (0,0,0,1))
        
        if is_fit :
            sfit1 = sFit('Exp', time, data)
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            _,decay_rate,_ = popt
            _,decay_rate_err,_ = np.sqrt(np.diag(pcov))

            ax2.plot(time*1e6, sfit1.func(time, *popt))
            an = ax2.annotate((f'T1 = {(1/decay_rate*1e6):.2f}±{(1/(decay_rate)**2*decay_rate_err*1e6):.2f}[us]'), 
                             xy = (np.average(time*1e6), np.average(data[0:10]) ),
                             size = 16)
            an.draggable()

        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='minor', labelsize=15)

        ax2.set_title("cavity T1 measurement", fontsize=20)
        ax2.set_xlabel("Time (us)", fontsize=20)
        ax2.set_ylabel(f"{self.which_data} (a.u.)", fontsize=20)

        plt.show()

# In[]
    def cavity_mode_spectroscopy(self,
                                freq_start,
                                freq_stop,
                                npts,
                                average_exponent=12, 
                                is_plot_simulation=False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters

        component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse",
            length=qubits_parameters[component]["readout_pulse_length"],
            amplitude=qubits_parameters[component]["readout_amp"]
        )

        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function",
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"]
        )

        cavity_drive_pulse = pulse_library.gaussian_square(
            uid="cavity_drive_pulse",
            length=cavity_parameters[cavity_component]["cavity_drive_length"],
            amplitude=cavity_parameters[cavity_component]["cavity_drive_amp"]
        )

        phase = qubits_parameters[component]["readout_phase"]

        # Sweep parameters
        freq_sweep = LinearSweepParameter(uid="mode_freq", start=freq_start, stop=freq_stop, count=npts)

        # Define the experiment
        exp_cavity_mode_freq = Experiment(
            uid="cavity_mode_freq",
            signals=[
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_cavity_mode_freq.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_cavity_mode_freq.sweep(uid="sweep", parameter=freq_sweep, alignment=SectionAlignment.RIGHT):
                with exp_cavity_mode_freq.section(uid="cavity_excitation", alignment=SectionAlignment.RIGHT):
                    exp_cavity_mode_freq.play(signal="cavity_drive", pulse=cavity_drive_pulse)

                with exp_cavity_mode_freq.section(uid="measure", play_after="cavity_excitation"):
                    exp_cavity_mode_freq.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_cavity_mode_freq.acquire(
                        signal="acquire",
                        handle="cavity_mode_freq",
                        kernel=readout_weighting_function,
                    )
                with exp_cavity_mode_freq.section(uid="relax", 
                                           length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_cavity_mode_freq.reserve(signal="measure")
        
        exp_calibration = Calibration()
        # sets the oscillator of the experimental measure signal
        # for spectroscopy, set the sweep parameter as frequency
        cavity_mode_oscillator = Oscillator(
            "cavity_drive_if_osc",
            frequency=freq_sweep,
        )
        exp_calibration["cavity_drive"] = SignalCalibration(
            oscillator=cavity_mode_oscillator
        )

        exp_cavity_mode_freq.set_calibration(exp_calibration)

        signal_map = {
            "measure": device_setup.logical_signal_groups[component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }
        
        exp_cavity_mode_freq.set_signal_map(signal_map)

        compiled_experiment_cavity_mode_freq = self.session.compile(exp_cavity_mode_freq)

        self.cavity_mode_freq_results = self.session.run(compiled_experiment_cavity_mode_freq)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_cavity_mode_freq, start_time=0, length=20e-6)

    def plot_cavity_mode_spectroscopy(self):

        ### data processing ###############################################################

        averaged_nums = len(self.cavity_mode_freq_results.acquired_results['cavity_mode_freq'].axis[0])
        # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

        self.cavity_mode_freq_data = self.cavity_mode_freq_results.get_data("cavity_mode_freq") # (2^N, npts) array
        freqs = self.cavity_mode_freq_results.acquired_results['cavity_mode_freq'].axis[1]

        if self.which_data == "I":
            data = np.real(np.mean(self.cavity_mode_freq_data, axis = 0))
            std_data = np.real(np.std(self.cavity_mode_freq_data, axis = 0)/np.sqrt(averaged_nums))

        else:
            data = np.imag(np.mean(self.cavity_mode_freq_data, axis = 0))
            std_data = np.imag(np.std(self.cavity_mode_freq_data, axis = 0)/np.sqrt(averaged_nums))

        ### data plot ######################################################################

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        ax.errorbar(freqs, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))

        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=15)

        ax.set_title("cavity freq spectroscopy", fontsize=20)
        ax.set_xlabel("Freq", fontsize=20)
        ax.set_ylabel(f"{self.which_data} (a.u.)", fontsize=20)

        plt.show()

# In[] cavity_pi_nopi

    def cavity_pi_nopi(self, average_exponent=12, 
                       freq_start = 0, # freq in Hz
                       freq_stop = 0,
                       freq_npts = 0,
                       is_qubit2 = False,
                       qubit2=None,
                       auto_chunking=True,
                       is_plot_simulation=False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters

        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        if is_qubit2:
            self.is_qubit2 = True
            qubit2_component = list(qubits_parameters.keys())[qubit2]
            q2_pi2_pulse, q2_pi_pulse, q2_cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        freq = cavity_parameters[cavity_component]["cond_disp_pulse_frequency"]

        sweep_case_1 = LinearSweepParameter(uid="on_off_case", start=0, stop=1, count=2)

        sweep_case_2 = LinearSweepParameter(uid="crosskerr_check", start=0, stop=1, count=2)

        sweep_freq_cases = LinearSweepParameter(uid="sweep_freq", start = freq + freq_start, stop = freq + freq_stop, count = freq_npts)

        def sweep_exp_cases(sweep_case: int | SweepParameter | LinearSweepParameter,
                            sweep_freq_cases: int | SweepParameter | LinearSweepParameter,
                            exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for exp_case_num in sweep_case.values:
                            with exp.case(exp_case_num):
                                with exp.match(sweep_parameter=sweep_freq_cases):
                                    for sweep_value in sweep_freq_cases.values:
                                        with exp.case(sweep_value):
                                            f(exp_case_num, sweep_value)
            return decorator
        
        def on_off_cond_pi_pulse(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)

            return decorator
        
        if is_qubit2:
            exp_cavity_pi_nopi = Experiment(
                uid="cavity_pi_nopi",
                signals=[
                    ExperimentSignal("cavity_drive"),
                    ExperimentSignal("drive"),
                    ExperimentSignal("drive2"),
                    ExperimentSignal("measure"),
                    ExperimentSignal("acquire"),
                ],
            )

        else :
            exp_cavity_pi_nopi = Experiment(
                uid="cavity_pi_nopi",
                signals=[
                    ExperimentSignal("cavity_drive"),
                    ExperimentSignal("drive"),
                    ExperimentSignal("measure"),
                    ExperimentSignal("acquire"),
                ],
            )


        with exp_cavity_pi_nopi.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_cavity_pi_nopi.sweep(uid="crosskerr_check", parameter=sweep_case_2):
                with exp_cavity_pi_nopi.sweep(uid="on_off_pi", parameter=sweep_case_1):
                    with exp_cavity_pi_nopi.sweep(uid="sweep_freq", parameter=sweep_freq_cases, auto_chunking= auto_chunking):

                        with exp_cavity_pi_nopi.section(uid="pi_nopi_cavity"):

                            @sweep_exp_cases(sweep_case_1, sweep_freq_cases, exp=exp_cavity_pi_nopi)
                            def play_cavity_pi_nopi(exp_case_num, sweep_value):

                                if exp_case_num == 0:  # nopi case
                                    with exp_cavity_pi_nopi.section():
                                        exp_cavity_pi_nopi.play(signal="cavity_drive", 
                                                                pulse=cond_disp_pulse, 
                                                                pulse_parameters={"frequency": sweep_value})
                                        exp_cavity_pi_nopi.reserve(signal="drive")
                                    
                                elif exp_case_num == 1: # pi case

                                    if is_qubit2:
                                        with exp_cavity_pi_nopi.section():
                                            exp_cavity_pi_nopi.play(signal="drive", pulse=pi_pulse)
                                            exp_cavity_pi_nopi.play(signal="drive2", pulse=q2_pi_pulse)
                                            exp_cavity_pi_nopi.reserve(signal="cavity_drive")
                                    else:
                                        with exp_cavity_pi_nopi.section():
                                            exp_cavity_pi_nopi.play(signal="drive", pulse=pi_pulse)
                                            exp_cavity_pi_nopi.reserve(signal="cavity_drive")

                                    with exp_cavity_pi_nopi.section():
                                        exp_cavity_pi_nopi.play(signal="cavity_drive", 
                                                                pulse=cond_disp_pulse, 
                                                                pulse_parameters={"frequency": sweep_value})
                                        if is_qubit2:
                                            exp_cavity_pi_nopi.reserve(signal="drive")
                                            exp_cavity_pi_nopi.reserve(signal="drive2")
                                        else:
                                            exp_cavity_pi_nopi.reserve(signal="drive")

                                    if is_qubit2:
                                        with exp_cavity_pi_nopi.section():
                                            exp_cavity_pi_nopi.play(signal="drive", pulse=pi_pulse)
                                            exp_cavity_pi_nopi.play(signal="drive2", pulse=q2_pi_pulse)
                                    else:
                                        with exp_cavity_pi_nopi.section():
                                            exp_cavity_pi_nopi.play(signal="drive", pulse=pi_pulse)

                            @on_off_cond_pi_pulse(sweep_case_2, exp=exp_cavity_pi_nopi)
                            def play_crosskerr_check(v):
                                if v == 0:
                                    pass
                                elif v == 1:
                                    exp_cavity_pi_nopi.play(signal="drive", pulse=cond_pi_pulse)
                    
                        with exp_cavity_pi_nopi.section(uid="measure", play_after="pi_nopi_cavity"):
                            exp_cavity_pi_nopi.play(signal="measure", pulse=readout_pulse, phase=phase)
                            exp_cavity_pi_nopi.acquire(
                                signal="acquire",
                                handle="cavity_pi_nopi",
                                kernel=readout_weighting_function,
                            )

                        with exp_cavity_pi_nopi.section(uid="relax", 
                                                        length=cavity_parameters[cavity_component]["reset_delay_length"]):
                            exp_cavity_pi_nopi.reserve(signal="measure")

        if is_qubit2:
            signal_map = {
                "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
                "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
                "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
                "drive2": device_setup.logical_signal_groups[qubit2_component].logical_signals["drive"],
                "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
            }
        else:
            signal_map = {
                "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
                "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
                "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
                "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
            }

        exp_cavity_pi_nopi.set_signal_map(signal_map)

        compiled_experiment_cavity_pi_nopi = self.session.compile(exp_cavity_pi_nopi)

        self.cavity_pi_nopi_results = self.session.run(compiled_experiment_cavity_pi_nopi)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_cavity_pi_nopi, start_time=0, length=20e-6)
            show_pulse_sheet("cavity_pinopi", compiled_experiment_cavity_pi_nopi)
                    
    def plot_cavity_pi_nopi(self):

        self.cavity_pi_nopi_data = self.cavity_pi_nopi_results.get_data("cavity_pi_nopi")

        if self.which_data == "I":
            data = np.real(self.cavity_pi_nopi_data)
        else:
            data = np.imag(self.cavity_pi_nopi_data)

        freq_sweep_list = self.cavity_pi_nopi_results.acquired_results['cavity_pi_nopi'].axis[2]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        freq = cavity_parameters[cavity_component]["cond_disp_pulse_frequency"]

        length = cavity_parameters[cavity_component]["cond_disp_pulse_length"]
        amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        detuning = cavity_parameters[cavity_component]["cond_disp_pulse_detuning"]
        sigma = cavity_parameters[cavity_component]["cond_disp_pulse_sigma"]

        ############# Plotting the data #####################

        fig, ax = plt.subplots(2, 1, figsize=(20, 16))

        fig.suptitle(f"length:{length}, amp:{amp}, detuning:{detuning/1e6}MHz, sigma:{sigma}", fontsize=18)

        ax[0].plot((freq_sweep_list-freq)/1e6, data[0][0], label='nopi', marker='o', linestyle=':', color='b')
        ax[0].plot((freq_sweep_list-freq)/1e6, data[0][1], label='pi', marker='o', linestyle=':', color='r')
        ax[0].set_title('Cavity pi-nopi, only cross Kerr effect')
        ax[0].set_xlabel('Frequency (MHz)')
        ax[0].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[0].legend()

        ax[1].plot((freq_sweep_list-freq)/1e6, data[1][0], label='nopi', marker='o', linestyle=':', color='b')
        ax[1].plot((freq_sweep_list-freq)/1e6, data[1][1], label='pi', marker='o', linestyle=':', color='r')
        ax[1].set_title('Cavity pi-nopi')
        ax[1].set_xlabel('Frequency (MHz)')
        ax[1].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[1].legend()

        fig2, ax2 = plt.subplots(1, 1, figsize=(20, 8))

        fig2.suptitle(f"length:{length}, amp:{amp}, detuning:{detuning/1e6}MHz, sigma:{sigma}", fontsize=18)

        ax2.plot((freq_sweep_list-freq)/1e6, data[1][0]-data[0][0], label='nopi', marker='o', linestyle=':', color='b')
        ax2.plot((freq_sweep_list-freq)/1e6, data[1][1]-data[0][1], label='pi', marker='o', linestyle=':', color='r')
        ax2.set_title('Cavity pi-nopi without cross Kerr effect')
        ax2.set_xlabel('Frequency (MHz)')
        ax2.set_ylabel(f'[{self.which_data}] (a.u.)')
        ax2.legend()

        plt.show()


# In[] CNOD_calibration

    def CNOD_calibration(self, average_exponent=12, amp_range=1, npts=11, qubit_phase = 0, is_displaced_state = False, is_plot_simulation=False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        self.qubit_phase = qubit_phase # real or imag of characteristic function
        self.is_displaced_state = is_displaced_state

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        self.CNOD_calibration_amp_range = amp_range
        amplitude_sweep = LinearSweepParameter(uid="delay", start= -amp_range, stop=amp_range, count=npts)

        sweep_case = LinearSweepParameter(uid="crosskerr_check", start=0, stop=1, count=2)

        def crosskerr_check(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)

            return decorator

        # Define the experiment
        exp_CNOD_calibration = Experiment(
            uid="CNOD_calibration",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_CNOD_calibration.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True,
        ):
            # with exp_CNOD_calibration.sweep(uid="crosskerr_check", parameter=sweep_case):
            with exp_CNOD_calibration.sweep(uid="sweep", parameter=amplitude_sweep, reset_oscillator_phase=True):
                
                if is_displaced_state:
                    with exp_CNOD_calibration.section(uid="disp_pulse"):
                        exp_CNOD_calibration.play(signal="cavity_drive", pulse=cavity_drive_pulse)

                    with exp_CNOD_calibration.section(uid="qubit_excitation_1", play_after="disp_pulse"):
                        exp_CNOD_calibration.play(signal="drive", pulse=pi2_pulse)

                    self.CNOD(exp = exp_CNOD_calibration, cond_disp_pulse = cond_disp_pulse,
                            pi_pulse = pi_pulse, amp = amplitude_sweep, 
                            prev_uid="qubit_excitation_1")

                else:
                    with exp_CNOD_calibration.section(uid="qubit_excitation_1"):
                        exp_CNOD_calibration.play(signal="drive", pulse=pi2_pulse)

                    self.CNOD(exp = exp_CNOD_calibration, cond_disp_pulse = cond_disp_pulse,
                            pi_pulse = pi_pulse, amp = amplitude_sweep, 
                            prev_uid="qubit_excitation_1")
                     
                # with exp_CNOD_calibration.section(uid="qubit_excitation_2", play_after="cond_disp_pulse_2"):
                #     @crosskerr_check(sweep_case, exp=exp_CNOD_calibration)
                #     def play_crosskerr_check(v):
                #                 if v == 0:
                #                     pass
                #                 elif v == 1:
                #                     exp_CNOD_calibration.play(signal="drive", pulse=pi2_pulse, phase = qubit_phase)
                with exp_CNOD_calibration.section(uid="qubit_excitation_2", play_after="cond_disp_pulse_2"):
                    exp_CNOD_calibration.play(signal="drive", pulse=pi2_pulse, phase = qubit_phase)
                    
                with exp_CNOD_calibration.section(uid="measure", play_after="qubit_excitation_2"):
                    exp_CNOD_calibration.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_CNOD_calibration.acquire(signal="acquire",
                                                handle="CNOD_calibration",
                                                kernel=readout_weighting_function)
                with exp_CNOD_calibration.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_CNOD_calibration.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_CNOD_calibration.set_signal_map(signal_map)

        compiled_exp_CNOD_calibration = self.session.compile(exp_CNOD_calibration)

        self.CNOD_calibration_results = self.session.run(compiled_exp_CNOD_calibration)

        if is_plot_simulation:
            self.simulation_plot(compiled_exp_CNOD_calibration, start_time=0, length=20e-6)
            show_pulse_sheet("CNOD_calibration", compiled_exp_CNOD_calibration)

    def plot_CNOD_calibration(self, scaling_factor=1):

        self.CNOD_calibration_data = self.CNOD_calibration_results.get_data("CNOD_calibration")

        if self.which_data == "I":
            data = np.real(self.CNOD_calibration_data)
        else:
            data = np.imag(self.CNOD_calibration_data)

        if np.isreal(self.CNOD_calibration_amp_range):
            # real axis sweep
            amplitude_sweep_list = np.real(self.CNOD_calibration_results.acquired_results['CNOD_calibration'].axis[0])
        else:
            # imaginary axis sweep
            amplitude_sweep_list = np.imag(self.CNOD_calibration_results.acquired_results['CNOD_calibration'].axis[0])

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        length = cavity_parameters[cavity_component]["cond_disp_pulse_length"]
        amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        detuning = cavity_parameters[cavity_component]["cond_disp_pulse_detuning"]
        sigma = cavity_parameters[cavity_component]["cond_disp_pulse_sigma"]
        # alpha_1_CNOD_amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"]

        if self.is_displaced_state:
            disp_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
            disp_length = cavity_parameters[cavity_component]["cavity_drive_length"]
            

        ############# Plotting the data #####################

        fig, ax = plt.subplots(1, 1, figsize=(20, 12))

        fig.suptitle(f"CNOD amp calibration", fontsize=18)
        
        fig.text(0.5, 0.91, f"length:{length}, amp:{amp}, detuning:{detuning/1e6}MHz, sigma:{sigma}", ha="center", fontsize=16, color="black")
        
        ax.plot(amp*amplitude_sweep_list/scaling_factor, data, marker='o', label = 'original', linestyle=':', color='k')
        

        if self.is_displaced_state:
            
            sfit1 = sFit('GaussianCos', amp*amplitude_sweep_list/scaling_factor, data)
            # Scale the amplitude into alpha (photon number) by using alpha_1_CNOD_amp
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            ax.plot(amp*amplitude_sweep_list/scaling_factor, 
                    sfit1.func(amp*amplitude_sweep_list/scaling_factor, *popt), label='fit', color='g')
            
            an1 = ax.annotate(rf'frequency = {popt[1]:.4f}±{np.sqrt(np.diag(pcov))[1]:.4f}'+'\n'
                              + rf'$\sigma$ = {popt[2]:.4f}±{np.sqrt(np.diag(pcov))[2]:.4f},'+ '\n'
                              + rf'$\beta$ = {popt[1]*np.pi:.4f}',
                xy=(np.average(amplitude_sweep_list), np.average(data)*0.95),
                size=12)
            an1.draggable()
            
            an2 = ax.annotate(f'Disp_amp = {disp_amp:.4f}, Disp_length = {np.round(disp_length*1e9)}ns',
                xy=(np.average(amplitude_sweep_list), np.average(data)*0.9),
                size=12)
            an2.draggable()
        else:
            sfit1 = sFit('Gaussian', amp*amplitude_sweep_list/scaling_factor, data)
            # Scale the amplitude into alpha (photon number) by using alpha_1_CNOD_amp
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            ax.plot(amp*amplitude_sweep_list/scaling_factor, 
                    sfit1.func(amp*amplitude_sweep_list/scaling_factor, *popt), label='fit', color='g')
            
            an1 = ax.annotate(rf'$\sigma$ = {popt[1]:.4f}±{np.sqrt(np.diag(pcov))[1]:.4f},',
                xy=(np.average(amplitude_sweep_list), np.average(data)*0.95),
                size=12)
            an1.draggable()
            
        
        if self.qubit_phase == 0:
            ax.set_title('real part')
        else:
            ax.set_title('imaginary part')
        ax.set_xlabel('Amplitude (a.u.)')
        ax.set_ylabel(f'[{self.which_data}] (a.u.)')
        ax.legend()
 
        plt.show()

# In[] acquired CNOD_geometric_phase

    # def CNOD_geophase_calibration(self, average_exponent=12, amp_sweep = 1, amp_npts = 11, is_plot_simulation=False):
        
    #     device_setup = self.device_setup
    #     qubits_parameters = self.qubits_parameters
    #     cavity_parameters = self.cavity_parameters
    #     qubits_component = list(qubits_parameters.keys())[self.which_qubit]
    #     cavity_component = list(cavity_parameters.keys())[self.which_mode]

    #     # Define pulses
    #     readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
    #                     qubits_component, cavity_component)

    #     pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
    #                     qubits_component, cavity_component)
        
    #     cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
    #                                                                qubits_component, cavity_component)

    #     phase = qubits_parameters[qubits_component]["readout_phase"]

    #     # Sweep parameters
    #     disp_amp_sweep = LinearSweepParameter(uid="disp_amp_sweep", start=-amp_sweep, stop=amp_sweep, count=amp_npts)

    #     # Define the experiment
    #     exp_disp_calibration = Experiment(
    #         uid="disp_calibration",
    #         signals=[
    #             ExperimentSignal("drive"),
    #             ExperimentSignal("cavity_drive"),
    #             ExperimentSignal("measure"),
    #             ExperimentSignal("acquire"),
    #         ],
    #     )
    #     with exp_disp_calibration.acquire_loop_rt(
    #         uid="shots",
    #         count=2**average_exponent,
    #         averaging_mode=AveragingMode.CYCLIC,
    #         acquisition_type=AcquisitionType.INTEGRATION,
    #         reset_oscillator_phase=True
    #     ):
    #         with exp_disp_calibration.sweep(uid="disp_amp_sweep", parameter=disp_amp_sweep, reset_oscillator_phase=True):
    #             with exp_disp_calibration.section(uid="qubit_excitation_1"):
    #                 exp_disp_calibration.play(signal="drive", pulse=pi2_pulse)
                
    #             self.CNOD(exp = exp_disp_calibration, cond_disp_pulse = cond_disp_pulse,
    #                       pi_pulse = pi_pulse, amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"], 
    #                       prev_uid="qubit_excitation_1", uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid= "pi_pulse_1")
                
    #             self.CNOD(exp = exp_disp_calibration, cond_disp_pulse = cond_disp_pulse,
    #                       pi_pulse = pi_pulse, amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"], 
    #                       prev_uid="cond_disp_pulse_2", uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid= "pi_pulse_2")
   
    #             with exp_disp_calibration.section(uid="qubit_excitation_2", play_after="disp_pulse_2"):
    #                 exp_disp_calibration.play(signal="drive", pulse=pi2_pulse, phase = 0)
                    
    #             with exp_disp_calibration.section(uid="measure", play_after="qubit_excitation_2"):
    #                 exp_disp_calibration.play(signal="measure", pulse=readout_pulse, phase=phase)
    #                 exp_disp_calibration.acquire(signal="acquire",
    #                                             handle="disp_calibration",
    #                                             kernel=readout_weighting_function)
    #             with exp_disp_calibration.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
    #                 exp_disp_calibration.reserve(signal="measure")
        
    #     signal_map = {
    #         "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
    #         "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
    #         "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
    #         "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
    #     }

    #     exp_disp_calibration.set_signal_map(signal_map)
    #     compiled_exp_disp_calibration = self.session.compile(exp_disp_calibration)
    #     self.disp_calibration_results = self.session.run(compiled_exp_disp_calibration)
    #     if is_plot_simulation:
    #         self.simulation_plot(compiled_exp_disp_calibration, start_time=0, length=20e-6)
    #         show_pulse_sheet("disp_calibration", compiled_exp_disp_calibration)

# In[] Char_func_displaced

    def Characteristic_function_2D(self, average_exponent=12, amp_range=1, npts=11, qubit_phase = 0, is_plot_simulation=False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        self.qubit_phase = qubit_phase # real or imag of characteristic function

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Create a 2D grid of complex amplitudes from (-amp_range-1j*amp_range) to (amp_range+1j*amp_range)
        real_vals = np.linspace(-0.5, 0.5, npts) # should not be over unity
        imag_vals = np.linspace(-0.5, 0.5, npts) # should not be over unity
        amplitude_grid = real_vals[:, None] + 1j * imag_vals[None, :]
        
        self.amp_npts = npts

        # Sweep parameters
        amplitude_sweep = SweepParameter(uid="amp_sweep", values=amplitude_grid.flatten())
        # Define the experiment
        exp_Characteristic_function_2D = Experiment(
            uid="Char_func_2D",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        def displaced_coherent_state():

            with exp_Characteristic_function_2D.section(uid="preparation", alignment=SectionAlignment.RIGHT):

                exp_Characteristic_function_2D.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude)
        
        def schrodinger_cat_state():
            with exp_Characteristic_function_2D.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_Characteristic_function_2D.section(uid="qubit_preparation"):
                    exp_Characteristic_function_2D.play(signal="drive", pulse=pi2_pulse)
                with exp_Characteristic_function_2D.section(uid="CNOD", play_after="qubit_preparation"):
                    self.CNOD(exp=exp_Characteristic_function_2D, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=1,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None)

        def schrodinger_cat_state_2():
            with exp_Characteristic_function_2D.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                
                with exp_Characteristic_function_2D.section(uid="alpha"):
                    exp_Characteristic_function_2D.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude)
                with exp_Characteristic_function_2D.section(uid="qubit", play_after="alpha"):
                    exp_Characteristic_function_2D.play(signal="drive", pulse=pi2_pulse)
                    exp_Characteristic_function_2D.delay(signal="drive", time=np.abs(1/(2*cavity_parameters[cavity_component]["cavity_mode_chi"])/2)) # delay for cross Kerr effect

        def cat_state():
            with exp_Characteristic_function_2D.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_Characteristic_function_2D.section(uid="qubit_preparation"):
                    exp_Characteristic_function_2D.play(signal="drive", pulse=pi2_pulse)
                with exp_Characteristic_function_2D.section(uid="CNOD_1", play_after="qubit_preparation"):
                    self.CNOD(exp=exp_Characteristic_function_2D, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=alpha,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid = "pi_pulse_1")
                with exp_Characteristic_function_2D.section(uid="qubit_1", play_after="CNOD_1"):
                    exp_Characteristic_function_2D.play(signal="drive", pulse=pi2_pulse)
                with exp_Characteristic_function_2D.section(uid="CNOD_2", play_after="qubit_1"):
                    self.CNOD(exp=exp_Characteristic_function_2D, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=1j*beta,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid = "pi_pulse_2")
                with exp_Characteristic_function_2D.section(uid="qubit_2", play_after="CNOD_2"):
                    exp_Characteristic_function_2D.play(signal="drive", pulse=pi2_pulse, phase=np.pi/2)




        with exp_Char_func_displaced.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True
        ):
            # with exp_CNOD_calibration.sweep(uid="crosskerr_check", parameter=sweep_case):
            with exp_Char_func_displaced.sweep(uid="sweep", parameter=amplitude_sweep, auto_chunking = True, reset_oscillator_phase=True):
                with exp_Char_func_displaced.section(uid="qubit_excitation_1"):
                    exp_Char_func_displaced.play(signal="drive", pulse=pi2_pulse)

                with exp_Char_func_displaced.section(uid="disp_pulse", play_after="qubit_excitation_1"):
                    exp_Char_func_displaced.play(signal="cavity_drive", pulse=cavity_drive_pulse)
                
                self.CNOD(exp = exp_Char_func_displaced, cond_disp_pulse = cond_disp_pulse, 
                          pi_pulse = pi_pulse, amp = amplitude_sweep, prev_uid="disp_pulse")
                
                with exp_Char_func_displaced.section(uid="qubit_excitation_2", play_after="cond_disp_pulse_2"):
                    exp_Char_func_displaced.play(signal="drive", pulse=pi2_pulse, phase = qubit_phase)
                    
                with exp_Char_func_displaced.section(uid="measure", play_after="qubit_excitation_2"):
                    exp_Char_func_displaced.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_Char_func_displaced.acquire(signal="acquire",
                                                handle="Char_func_displaced",
                                                kernel=readout_weighting_function)
                with exp_Char_func_displaced.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_Char_func_displaced.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_Char_func_displaced.set_signal_map(signal_map)

        compiled_exp_Char_func_displaced = self.session.compile(exp_Char_func_displaced)

        self.Char_func_displaced_results = self.session.run(compiled_exp_Char_func_displaced)

        if is_plot_simulation:
            self.simulation_plot(compiled_exp_Char_func_displaced, start_time=0, length=20e-6)
            show_pulse_sheet("CNOD_calibration", compiled_exp_Char_func_displaced)

    def plot_Characteristic_function_2D(self):

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        
        length = cavity_parameters[cavity_component]["cavity_drive_length"]
        amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        alpha_1_CNOD_amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"]
        cond_disp_pulse_amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]

        self.Char_func_displaced_data = self.Char_func_displaced_results.get_data("Char_func_displaced")

        if self.which_data == "I":
            data = np.real(self.Char_func_displaced_data)
        else:
            data = np.imag(self.Char_func_displaced_data)

        amplitude_sweep_list = self.Char_func_displaced_results.acquired_results['Char_func_displaced'].axis[0]

        Z = amplitude_sweep_list.reshape((self.amp_npts, self.amp_npts))
        x = np.real(Z) * cond_disp_pulse_amp/alpha_1_CNOD_amp # scaling to alpha (photon number)
        y = np.imag(Z) * cond_disp_pulse_amp/alpha_1_CNOD_amp   # scaling to alpha (photon number)
        data = data.reshape((self.amp_npts, self.amp_npts))

        ############# Plotting the data #####################

        plt.figure(figsize=(10, 10))
        pcm = plt.pcolormesh(x, y, data, shading='auto', cmap='RdBu_r')
        plt.colorbar(pcm, label=f'[{self.which_data}] (a.u.)')
        plt.title(f"Char func of displaced state, length:{length}, amp:{amp}")
        plt.xlabel(r'Re($\alpha$)')
        plt.ylabel(r'Im($\alpha$)')

        plt.show()

# In[] disp_pulse_calibration
    def disp_pulse_calibration_geophase(self, average_exponent=12, amp_sweep = 1, amp_npts = 11, is_plot_simulation=False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        disp_amp_sweep = LinearSweepParameter(uid="disp_amp_sweep", start=-amp_sweep, stop=amp_sweep, count=amp_npts)

        # Define the experiment
        exp_disp_calibration = Experiment(
            uid="disp_calibration",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        with exp_disp_calibration.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True
        ):
            with exp_disp_calibration.sweep(uid="disp_amp_sweep", parameter=disp_amp_sweep, reset_oscillator_phase=True):
                with exp_disp_calibration.section(uid="qubit_excitation_1"):
                    exp_disp_calibration.play(signal="drive", pulse=pi2_pulse)
                
                self.CNOD(exp = exp_disp_calibration, cond_disp_pulse = cond_disp_pulse,
                          pi_pulse = pi_pulse, amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"], 
                          prev_uid="qubit_excitation_1", uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid= "pi_pulse_1")

                with exp_disp_calibration.section(uid="disp_puls_1", play_after="cond_disp_pulse_2"):
                    exp_disp_calibration.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude = 1j*disp_amp_sweep)
                
                self.CNOD(exp = exp_disp_calibration, cond_disp_pulse = cond_disp_pulse,
                          pi_pulse = pi_pulse, amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"], 
                          prev_uid="disp_puls_1", uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid= "pi_pulse_2")

                with exp_disp_calibration.section(uid="disp_pulse_2", play_after="cond_disp_pulse_4"):
                    exp_disp_calibration.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude = 1j*disp_amp_sweep, phase = np.pi)
                
                with exp_disp_calibration.section(uid="qubit_excitation_2", play_after="disp_pulse_2"):
                    exp_disp_calibration.play(signal="drive", pulse=pi2_pulse, phase = 0)
                    
                with exp_disp_calibration.section(uid="measure", play_after="qubit_excitation_2"):
                    exp_disp_calibration.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_disp_calibration.acquire(signal="acquire",
                                                handle="disp_calibration",
                                                kernel=readout_weighting_function)
                with exp_disp_calibration.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_disp_calibration.reserve(signal="measure")
        
        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_disp_calibration.set_signal_map(signal_map)
        compiled_exp_disp_calibration = self.session.compile(exp_disp_calibration)
        self.disp_calibration_results = self.session.run(compiled_exp_disp_calibration)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_disp_calibration, start_time=0, length=20e-6)
            show_pulse_sheet("disp_calibration", compiled_exp_disp_calibration)
    
    def plot_disp_pulse_calibration_geophase(self, is_fit=True):

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        length = cavity_parameters[cavity_component]["cavity_drive_length"]
        amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        cond_pulse_length = cavity_parameters[cavity_component]["cond_disp_pulse_length"]
        cond_pulse_amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        detuning = cavity_parameters[cavity_component]["cond_disp_pulse_detuning"]
        sigma = cavity_parameters[cavity_component]["cond_disp_pulse_sigma"]

        self.disp_calibration_data = self.disp_calibration_results.get_data("disp_calibration")

        if self.which_data == "I":
            data = np.real(self.disp_calibration_data)
        else:
            data = np.imag(self.disp_calibration_data)

        disp_amp_sweep_list = self.disp_calibration_results.acquired_results['disp_calibration'].axis[0]

        ############# Plotting the data #####################

        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        fig.suptitle("Displacement Pulse Calibration", fontsize=18, weight='bold')
        fig.text(0.5, 0.94, f"Displacement pulse length: {length:.1e}", ha="center", fontsize=14)
        
        ax.plot(amp*disp_amp_sweep_list, data, marker='o', linestyle=':', color='k')

        if is_fit :
            sfit1 = sFit('Cos', amp*disp_amp_sweep_list, data)
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            ax.plot(amp*disp_amp_sweep_list, sfit1.func(amp*disp_amp_sweep_list, *popt), label='fit', color='g')
        
        freq = popt[1]
        freq_err = np.sqrt(np.diag(pcov))[1]
        annotation_text = f"freq = {freq:.4f} ± {freq_err:.4f}\nscaling factor = π x freq"

        an = ax.annotate(annotation_text,
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=10, ha='left', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        an.draggable()

        ax.set_xlabel('Driving Amplitude')
        ax.set_ylabel(f'[{self.which_data}] (a.u.)')

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Displacement ($\\beta$)')

        xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 6)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f"{x * np.pi * freq:.2f}" for x in xticks])

        ax.tick_params(axis='both', labelsize=10)
        ax2.tick_params(axis='x', labelsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


    def disp_pulse_calibration_parity(self, average_exponent=12, amp_start = 0, amp_stop=1, amp_npts = 11, is_plot_simulation=False):
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        disp_amp_sweep = LinearSweepParameter(uid="disp_amp_sweep", start= amp_start, stop= amp_stop, count=amp_npts)

        sweep_case = LinearSweepParameter(uid="correction", start=0, stop=1, count=2)

        def correction(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)

            return decorator

        # Define the experiment
        exp_disp_calibration_parity = Experiment(
            uid="disp_calibration_parity",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        with exp_disp_calibration_parity.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_disp_calibration_parity.sweep(uid="correction", parameter=sweep_case):
                with exp_disp_calibration_parity.sweep(uid="disp_amp_sweep", parameter=disp_amp_sweep):
                    with exp_disp_calibration_parity.section(uid="disp_amp_drive"):
                        exp_disp_calibration_parity.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude = disp_amp_sweep)
                    
                    with exp_disp_calibration_parity.section(uid="qubit_excitation_1", play_after="disp_amp_drive"):
                        exp_disp_calibration_parity.play(signal="drive", pulse=pi2_pulse)
                        exp_disp_calibration_parity.delay(signal="drive", time=np.abs(1/(2*cavity_parameters[cavity_component]["cavity_mode_chi"]))) # delay for cross Kerr effect

                    with exp_disp_calibration_parity.section(uid="qubit_excitation_2", play_after="qubit_excitation_1"):
                        @correction(sweep_case, exp=exp_disp_calibration_parity)
                        def play_correction(v):
                            if v == 0:
                                exp_disp_calibration_parity.play(signal="drive", pulse=pi2_pulse, phase = 0)
                            elif v == 1:
                                exp_disp_calibration_parity.play(signal="drive", pulse=pi2_pulse, phase = np.pi)
                    
                    with exp_disp_calibration_parity.section(uid="measure", play_after="qubit_excitation_2"):
                        exp_disp_calibration_parity.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_disp_calibration_parity.acquire(signal="acquire",
                                                handle="disp_calibration_parity",
                                                kernel=readout_weighting_function)
                    with exp_disp_calibration_parity.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_disp_calibration_parity.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_disp_calibration_parity.set_signal_map(signal_map)
        compiled_exp_disp_calibration_parity = self.session.compile(exp_disp_calibration_parity)
        self.disp_calibration_parity_results = self.session.run(compiled_exp_disp_calibration_parity)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_disp_calibration_parity, start_time=0, length=20e-6)
            show_pulse_sheet("disp_calibration_parity", compiled_exp_disp_calibration_parity)
                    
    def plot_disp_pulse_calibration_parity(self, is_fit=True, scaling_factor=1):

        self.disp_calibration_parity_data = self.disp_calibration_parity_results.get_data("disp_calibration_parity")

        if self.which_data == "I":
            data = np.real(self.disp_calibration_parity_data)
        else:
            data = np.imag(self.disp_calibration_parity_data)

        disp_amp_sweep_list = self.disp_calibration_parity_results.acquired_results['disp_calibration_parity'].axis[1]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        length = cavity_parameters[cavity_component]["cavity_drive_length"]
        amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        chi = cavity_parameters[cavity_component]["cavity_mode_chi"]

        ############# Plotting the data #####################

        fig, ax = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle(f"Displacement pulse calibration parity", fontsize=18)
        fig.text(0.5, 0.91, f"drive_length:{length}, chi:{chi}", ha="center", fontsize=16, color="black")
        ax[0].plot(amp*disp_amp_sweep_list/scaling_factor, data[0], marker='o', linestyle=':', color='k', label='uncorrected')
        ax[0].plot(amp*disp_amp_sweep_list/scaling_factor, data[1], marker='o', linestyle=':', color='r', label='correction')
        ax[1].plot(amp*disp_amp_sweep_list/scaling_factor, data[0]-data[1], marker='o', linestyle=':', color='k', label='corrected')

        if is_fit :

            sfit1 = sFit('Gaussian', amp*disp_amp_sweep_list/scaling_factor, data[0]-data[1])
            # Scale the amplitude into alpha (photon number) by using alpha_1_CNOD_amp
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            ax[1].plot(amp*disp_amp_sweep_list/scaling_factor, 
                    sfit1.func(amp*disp_amp_sweep_list/scaling_factor, *popt), label='fit', color='g')
        
            an = ax[1].annotate(f'sigma = {popt[1]:.4f}±{np.sqrt(np.diag(pcov))[1]:.4f}',
                            xy=(np.average(amp*disp_amp_sweep_list/scaling_factor), np.average(data[0]-data[1])))
            an.draggable()

        ax[0].set_xlabel('Drive Amplitude (a.u.)')
        ax[0].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[0].legend()
        ax[1].set_xlabel('Drive Amplitude (a.u.)')
        ax[1].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[1].legend()








# In[] out-and-back measurement

    def out_and_back_measurement(self, average_exponent=12, init_state = "g",
                                 phase_start=0, phase_stop=20, # deg
                                 cavity_drive_amp_start = 0, cavity_drive_amp_stop = 0.1,
                                 phase_npts=11, amp_npts=11,
                                 wait_time=1e-6,
                                 is_plot_simulation=False):
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        phase_sweep = LinearSweepParameter(uid="phase_sweep", start=phase_start, stop=phase_stop, count=phase_npts)

        drive_amp_sweep = LinearSweepParameter(uid="drive_amp_sweep", start=cavity_drive_amp_start, stop=cavity_drive_amp_stop, count=amp_npts)
        # Define the conditional pi pulse
        # Define the experiment
        exp_out_and_back = Experiment(
            uid="out_and_back",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        conditional_uid = None

        with exp_out_and_back.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_out_and_back.sweep(uid="drive_amp_sweep", parameter=drive_amp_sweep):
                with exp_out_and_back.sweep(uid="phase_sweep", parameter=phase_sweep):
                    if init_state == "e":
                        conditional_uid = "qubit_excitation"
                        with exp_out_and_back.section(uid=conditional_uid):
                            exp_out_and_back.play(signal="drive", pulse=pi_pulse)
                    
                    with exp_out_and_back.section(uid="displacement", play_after=conditional_uid):
                        exp_out_and_back.play(signal="cavity_drive", pulse=cavity_drive_pulse,
                                              amplitude=drive_amp_sweep)
                        exp_out_and_back.delay(signal="cavity_drive", time=wait_time)
                        exp_out_and_back.play(signal="cavity_drive", pulse=cavity_drive_pulse, 
                                            phase=np.pi/180*phase_sweep+np.pi,
                                            amplitude=drive_amp_sweep)
                    
                    with exp_out_and_back.section(uid="cond_disp_pi", play_after="displacement"):
                        exp_out_and_back.play(signal="drive", pulse=cond_pi_pulse)
                    
                    with exp_out_and_back.section(uid="measure", play_after="cond_disp_pi"):
                        exp_out_and_back.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_out_and_back.acquire(signal="acquire",
                                                    handle="out_and_back",
                                                    kernel=readout_weighting_function)
                    with exp_out_and_back.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_out_and_back.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_out_and_back.set_signal_map(signal_map)
        compiled_exp_out_and_back = self.session.compile(exp_out_and_back)
        self.out_and_back_results = self.session.run(compiled_exp_out_and_back)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_out_and_back, start_time=0, length=20e-6)
            show_pulse_sheet("out_and_back", compiled_exp_out_and_back)


    def plot_out_and_back_measurement(self, fitting=False, x_threshold=None,y_threshold=None, wait_time=1e-6, init_state = "g",):
        from sklearn.linear_model import LinearRegression

        self.out_and_back_data = self.out_and_back_results.get_data("out_and_back")

        if self.which_data == "I":
            data = np.real(self.out_and_back_data)
        else:
            data = np.imag(self.out_and_back_data)

        drive_amp_sweep_list = self.out_and_back_results.acquired_results['out_and_back'].axis[0]
        phase_sweep_list = self.out_and_back_results.acquired_results['out_and_back'].axis[1]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        max_drive_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        alpha_1_cavity_drive_amp = cavity_parameters[cavity_component]['alpha_1_cavity_drive_amp']

        x_axis = np.array(drive_amp_sweep_list*max_drive_amp) ** 2 / alpha_1_cavity_drive_amp**2
        y_axis = phase_sweep_list
        X_grid, Y_grid = np.meshgrid(x_axis, y_axis, indexing='ij')  # shape = (M, N)
        Z_data = data  # shape = (M, N)

        X_flat = X_grid.flatten()
        Y_flat = Y_grid.flatten()
        Z_flat = Z_data.flatten()

        self.nopi_value

        if self.nopi_value > self.pi_value: # pi nopi measurement 값을 비교
            if init_state == "g":
                z_max = np.max(Z_flat) # 나중에 pi nopi measurement에서 z_max, z_min을 이용해서
                # z_range를 구하고, 그 범위에 해당하는 데이터만을 선택하여 fitting을 수행.
                z_min = np.min(Z_flat)
                z_range = (1/3) * (z_max - z_min)
                mask = (Z_flat >= z_min) & (Z_flat <= z_min + z_range)
            else:
                z_max = np.max(Z_flat)
                z_min = np.min(Z_flat)
                z_range = (1/3) * (z_max - z_min)
                mask = (Z_flat <= z_max) & (Z_flat >= z_max - z_range)
        
        else:
            if init_state == "g":
                z_max = np.max(Z_flat)
                z_min = np.min(Z_flat)
                z_range = (1/3) * (z_max - z_min)
                mask = (Z_flat <= z_max) & (Z_flat >= z_max - z_range)
            else:
                z_max = np.max(Z_flat)
                z_min = np.min(Z_flat)
                z_range = (1/3) * (z_max - z_min)
                mask = (Z_flat >= z_min) & (Z_flat <= z_min + z_range)

        if x_threshold is not None:
            mask = mask & (X_flat >= x_threshold)

        if y_threshold is not None:
            mask = mask & (Y_flat <= y_threshold)

        X_vals = X_flat[mask].reshape(-1, 1)
        Y_vals = Y_flat[mask]

        # y축 라인별로 x값 평균 구하고 scatter plot
        X_avgs = []
        Y_avgs = []
        for i in range(len(x_axis)):
            y_mask = (X_vals.flatten() == x_axis[i])
            if np.any(y_mask):
                mean_value = np.mean(Y_vals[y_mask]) 
                X_avgs.append(x_axis[i])
                Y_avgs.append(mean_value)

        model = LinearRegression()
        model.fit(np.array(X_avgs).reshape(-1, 1), np.array(Y_avgs))
        fit_line = model.predict(np.array(X_avgs).reshape(-1, 1))
        slope = model.coef_[0]
        intercept = model.intercept_
        print(f" relative frequency(Hz) = {slope/ (wait_time  * 360):.3f} * Photon # + {intercept/ (wait_time  * 360):.3f}")
        print(f" K_c = 2Pi X {-1 * slope/ (2 * wait_time  * 360):.3f} Hz")

        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        fig2.suptitle("Out and back measurement - Scatter Plot", fontsize=18)
        plt.scatter(X_avgs, np.array(Y_avgs)/ (wait_time  * 360) , label=f'Phase = {mean_value:.2f} deg', alpha=1, facecolors='none', edgecolors='black')
        ax2.plot(X_avgs, fit_line/ (wait_time  * 360), color='dodgerblue', label='Linear Fit')
        ax2.set_xlabel('Photon number')
        ax2.set_ylabel('relative frequency (Hz)')
        ax2.text(0.5, 0.05, f'K_c = {slope/ (2 * wait_time  * 360):.3f} Hz', transform=ax2.transAxes, fontsize=12, verticalalignment='top')
        plt.show()
        
        # Create a 2D plot with the original data
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle("Out and back measurement", fontsize=18)
        pcm = ax.pcolormesh(x_axis, y_axis, Z_data.T, shading='auto')
        plt.colorbar(pcm, ax=ax, label=f'[{self.which_data}] (a.u.)')
        ax.plot(X_vals, model.predict(np.array(X_vals).reshape(-1,1)), color='dodgerblue', label='Linear Fit')
        ax.legend()

        ax.set_xlabel('Photon number')
        ax.set_ylabel('Phase (deg)')
        plt.show()

# In[] to get chi between memory and qubit, ramsey-like experiment
    def qubit_state_revival(self, average_exponent=12, 
                            wait_time=1e-6,
                            wait_npts = 11,
                            is_plot_simulation=False):
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        delay_sweep = LinearSweepParameter(uid="delay_sweep", start=0, stop=wait_time, count=wait_npts)

        # Define the experiment
        exp_qubit_state_revival = Experiment(
            uid="qubit_state_revival",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_qubit_state_revival.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_qubit_state_revival.sweep(uid="drive_amp_sweep", parameter=delay_sweep):
                with exp_qubit_state_revival.section(uid="displacement"):
                    exp_qubit_state_revival.play(signal="cavity_drive", pulse=cavity_drive_pulse)
                                                 
                with exp_qubit_state_revival.section(uid="qubit_excitation_1", play_after="displacement"):
                    exp_qubit_state_revival.play(signal="drive", pulse=pi2_pulse)
                    exp_qubit_state_revival.delay(signal="drive", time=delay_sweep)
                    exp_qubit_state_revival.play(signal="drive", pulse=pi2_pulse)
                
                with exp_qubit_state_revival.section(uid="measure", play_after="qubit_excitation_1"):
                    exp_qubit_state_revival.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_qubit_state_revival.acquire(signal="acquire",
                                                    handle="qubit_state_revival",
                                                    kernel=readout_weighting_function)
                with exp_qubit_state_revival.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_qubit_state_revival.reserve(signal="measure")
        
        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_qubit_state_revival.set_signal_map(signal_map)
        compiled_exp_qubit_state_revival = self.session.compile(exp_qubit_state_revival)
        self.qubit_state_revival_results = self.session.run(compiled_exp_qubit_state_revival)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_qubit_state_revival, start_time=0, length=20e-6)
            show_pulse_sheet("qubit_state_revival", compiled_exp_qubit_state_revival)
        
    def plot_qubit_state_revival(self, is_fitting=True):

        self.qubit_state_revival_data = self.qubit_state_revival_results.get_data("qubit_state_revival")

        if self.which_data == "I":
            data = np.real(self.qubit_state_revival_data)
        else:
            data = np.imag(self.qubit_state_revival_data)

        delay_sweep_list = self.qubit_state_revival_results.acquired_results['qubit_state_revival'].axis[0]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        cavity_drive_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        cavity_drive_length = cavity_parameters[cavity_component]["cavity_drive_length"]

        ############# Plotting the data #####################

        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        fig.suptitle(f"Qubit state revival using Ramsey interferometry", fontsize=18)
        
        ax.plot(delay_sweep_list, data, marker='o', linestyle=':', color='k')

        an = ax.annotate(f'cavity_drive_amp = {cavity_drive_amp:.4f}, cavity_drive_length = {cavity_drive_length*1e9:.0f}ns',
                        xy=(np.average(delay_sweep_list), np.average(data)),
                        size=12)
        an.draggable()

        ax.set_xlabel('Wait time (s)')
        ax.set_ylabel(f'[{self.which_data}] (a.u.)')

        plt.show()

# In[] storage mode characterization

    def storage_mode_characterization(self, average_exponent=12, 
                                    wait_time = 50e-6,
                                    wait_npts = 101,
                                    detuning = 0*1e6,
                                    init_state = "g",
                                    is_plot_simulation=False):
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        delay_sweep = LinearSweepParameter(uid="delay_sweep", start=0, stop=wait_time, count=wait_npts)

        sweep_case = LinearSweepParameter(uid="correction", start=0, stop=1, count=2)

        def correction(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)

            return decorator

        # Define the experiment
        exp_storage_mode_characterization = Experiment(
            uid="storage_mode_characterization",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_storage_mode_characterization.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True
        ):
            with exp_storage_mode_characterization.sweep(uid="correction", parameter=sweep_case, reset_oscillator_phase=True):
                with exp_storage_mode_characterization.sweep(uid="delay_sweep", parameter=delay_sweep, reset_oscillator_phase=True):
                    with exp_storage_mode_characterization.section(uid="displacement", alignment=SectionAlignment.RIGHT):
                        exp_storage_mode_characterization.play(signal="cavity_drive", pulse=cavity_drive_pulse)
                        if init_state == "e":
                            exp_storage_mode_characterization.play(signal="drive", pulse=pi_pulse)
                                  
                    with exp_storage_mode_characterization.section(uid="delay_drive", play_after="displacement"):
                        exp_storage_mode_characterization.delay(signal="cavity_drive", time=delay_sweep)
                        exp_storage_mode_characterization.delay(signal="drive", time=delay_sweep)
                        exp_storage_mode_characterization.play(signal="drive", pulse=pi2_pulse)
                    
                    self.CNOD(exp = exp_storage_mode_characterization, cond_disp_pulse = cond_disp_pulse,
                                pi_pulse = pi_pulse, amp = 1, # amp of asym pulse is defined at "cond_disp_pulse_amp" 
                                prev_uid="delay_drive", uid2 = "cond_disp_pulse_2")

                    with exp_storage_mode_characterization.section(uid="qubit_excitation_2", play_after="cond_disp_pulse_2"):
                        @correction(sweep_case, exp=exp_storage_mode_characterization)
                        def play_correction(v):
                            if v == 0:
                                exp_storage_mode_characterization.play(signal="drive", pulse=pi2_pulse, phase = 0)
                            elif v == 1:
                                exp_storage_mode_characterization.play(signal="drive", pulse=pi2_pulse, phase = np.pi)
                    
                    with exp_storage_mode_characterization.section(uid="measure", play_after="qubit_excitation_2"):
                        exp_storage_mode_characterization.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_storage_mode_characterization.acquire(signal="acquire",
                                                    handle="storage_mode_characterization",
                                                    kernel=readout_weighting_function)
                    with exp_storage_mode_characterization.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_storage_mode_characterization.reserve(signal="measure")
        
        exp_calibration = Calibration()
        
        cavity_mode_oscillator = Oscillator(
            "cavity_drive_if_osc",
            frequency=self.cavity_parameters[cavity_component][f"{cavity_component}_freq_IF"] + detuning,
        )
        exp_calibration["cavity_drive"] = SignalCalibration(
            oscillator=cavity_mode_oscillator
        )
        exp_storage_mode_characterization.set_calibration(exp_calibration)

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_storage_mode_characterization.set_signal_map(signal_map)
        compiled_exp_storage_mode_characterization = self.session.compile(exp_storage_mode_characterization)
        self.storage_mode_characterization_results = self.session.run(compiled_exp_storage_mode_characterization)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_storage_mode_characterization, start_time=0, length=20e-6)
            show_pulse_sheet("storage_mode_characterization", compiled_exp_storage_mode_characterization)

# init_guess = [amplitude,omega,T1,detuning,offset]
    def plot_storage_mode_characterization(self, is_fit=True, init_guess=None):
        self.storage_mode_characterization_data = self.storage_mode_characterization_results.get_data("storage_mode_characterization")

        if self.which_data == "I":
            data = np.real(self.storage_mode_characterization_data)
        else:
            data = np.imag(self.storage_mode_characterization_data)

        delay_sweep_list = self.storage_mode_characterization_results.acquired_results['storage_mode_characterization'].axis[1]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        cnod_amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        cavity_drive_length = cavity_parameters[cavity_component]["cavity_drive_length"]
        cavity_drive_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]

        ############# Plotting the data #####################

        fig, ax = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle(f"Storage mode characterization", fontsize=18)
        
        ax[0].plot(delay_sweep_list, data[0], marker='o', linestyle=':', color='k', label='uncorrected')
        ax[0].plot(delay_sweep_list, data[1], marker='o', linestyle=':', color='r', label='correction')
        ax[1].plot(delay_sweep_list, data[0]-data[1], marker='o', linestyle=':', color='k', label='corrected')


        an = ax[1].annotate(f'cond_disp_pulse_amp = {cnod_amp:.4f}, disp_pulse_amp = {cavity_drive_amp:.4f}',
                        xy=(np.average(delay_sweep_list), np.average(data[0]-data[1])),
                        size=12)
        an.draggable()

        if is_fit :

            sfit1 = sFit('Storage_Characterization', delay_sweep_list, data[0]-data[1], init_guess=init_guess)
            # Scale the amplitude into alpha (photon number) by using alpha_1_CNOD_amp
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            print("popt:", popt)
            print("pcov:", pcov)

            ax[1].plot(delay_sweep_list, 
                    sfit1.func(delay_sweep_list, *popt), label='fit', color='g')

            an = ax[1].annotate(f'freq = {popt[3]:.4f}±{np.sqrt(np.diag(pcov))[3]:.4f}Hz'+'\n'
                                + f'omega = {popt[1]:.4f}±{np.sqrt(np.diag(pcov))[1]:.4f}'+'\n'
                                + f'T1 = {popt[2]:.4f}±{np.sqrt(np.diag(pcov))[2]:.4f}s',
                            xy=(np.average(delay_sweep_list), np.average(data[0]-data[1])*0.95))
            an.draggable()

        ax[0].set_xlabel('Wait time (s)')
        ax[0].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[0].legend()
        ax[1].set_xlabel('Wait time (s)')
        ax[1].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[1].legend()

        plt.show()

# In[]

    def wigner_characteristic_function_2D(self, average_exponent=12, 
                                    npts_x = 21,
                                    npts_y = 21,
                                    amplitude = 0,
                                    is_wigner_function=False,
                                    is_coherent_state=False,
                                    is_schrodinger_cat_state=False,
                                    is_schrodinger_cat_state_2=False,
                                    is_cat_state=False,
                                    alpha = 1, # for cat state
                                    beta = 1, # for cat state
                                    qubit_phase = 0,
                                    is_correction = False,
                                    is_plot_simulation=False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        self.exp_wigner_characteristic_function_amplitude = amplitude
        self.is_wigner_function = is_wigner_function

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Create a 2D grid of complex amplitudes from (-amp_range-1j*amp_range) to (amp_range+1j*amp_range)
        real_vals = np.linspace(-0.5, 0.5, npts_x) # should not be over unity
        imag_vals = np.linspace(-0.5, 0.5, npts_y) # should not be over unity
        amplitude_grid = (real_vals[:, None] + 1j * imag_vals[None, :]).T
        
        self.amp_npts_x = npts_x
        self.amp_npts_y = npts_y

        # Sweep parameters
        amplitude_sweep = SweepParameter(uid="amp_sweep", values=amplitude_grid.flatten())

        if is_correction:
            sweep_case = LinearSweepParameter(uid="correction", start=0, stop=1, count=2)
        else:
            sweep_case = LinearSweepParameter(uid="correction", start=0, stop=0, count=1)
    
        # Define the experiment
        exp_wigner_characteristic_function = Experiment(
            uid="wigner_function",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        def displaced_coherent_state():

            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):

                exp_wigner_characteristic_function.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude)
        
        def schrodinger_cat_state():
            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_wigner_characteristic_function.section(uid="qubit_preparation"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                with exp_wigner_characteristic_function.section(uid="CNOD", play_after="qubit_preparation"):
                    self.CNOD(exp=exp_wigner_characteristic_function, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=1,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None)

        def schrodinger_cat_state_2():
            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):

                with exp_wigner_characteristic_function.section(uid="alpha"):
                    exp_wigner_characteristic_function.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude)
                with exp_wigner_characteristic_function.section(uid="qubit", play_after="alpha"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                    exp_wigner_characteristic_function.delay(signal="drive", time=np.abs(1/(2*cavity_parameters[cavity_component]["cavity_mode_chi"])/2)) # delay for cross Kerr effect

        def cat_state():
            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_wigner_characteristic_function.section(uid="qubit_preparation"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                with exp_wigner_characteristic_function.section(uid="CNOD_1", play_after="qubit_preparation"):
                    self.CNOD(exp=exp_wigner_characteristic_function, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=alpha,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid = "pi_pulse_1")
                with exp_wigner_characteristic_function.section(uid="qubit_1", play_after="CNOD_1"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                with exp_wigner_characteristic_function.section(uid="CNOD_2", play_after="qubit_1"):
                    self.CNOD(exp=exp_wigner_characteristic_function, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=1j*beta,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid = "pi_pulse_2")
                with exp_wigner_characteristic_function.section(uid="qubit_2", play_after="CNOD_2"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse, phase=np.pi/2)


        with exp_wigner_characteristic_function.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True,
        ):
            with exp_wigner_characteristic_function.sweep(uid="correction", parameter=sweep_case):
                with exp_wigner_characteristic_function.sweep(uid="amplitude_sweep", parameter=amplitude_sweep, auto_chunking = True, reset_oscillator_phase=True):

                    if is_coherent_state:
                        displaced_coherent_state()
                    elif is_schrodinger_cat_state:
                        schrodinger_cat_state()
                    elif is_schrodinger_cat_state_2:
                        schrodinger_cat_state_2()
                    elif is_cat_state:
                        cat_state()

                    #### wigner function measurement ####

                    if is_wigner_function:

                        self.wigner_function(exp=exp_wigner_characteristic_function, cavity_drive_pulse=cavity_drive_pulse, 
                                            pi2_pulse=pi2_pulse, amplitude_sweep=amplitude_sweep, sweep_case=sweep_case)
                    else:

                        self.characteristic_function(exp=exp_wigner_characteristic_function,
                                                    pi2_pulse=pi2_pulse, pi_pulse=pi_pulse, 
                                                    cond_disp_pulse=cond_disp_pulse,
                                                    amplitude_sweep=amplitude_sweep, qubit_phase = qubit_phase)

                    with exp_wigner_characteristic_function.section(uid="measure", play_after="qubit_excitation_2"):
                        exp_wigner_characteristic_function.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_wigner_characteristic_function.acquire(signal="acquire",
                                                handle="wigner_characteristic_function_measurement",
                                                kernel=readout_weighting_function)
                        
                    with exp_wigner_characteristic_function.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_wigner_characteristic_function.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_wigner_characteristic_function.set_signal_map(signal_map)
        compiled_exp_wigner_characteristic_function = self.session.compile(exp_wigner_characteristic_function)
        self.wigner_characteristic_function_results = self.session.run(compiled_exp_wigner_characteristic_function)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_wigner_characteristic_function, start_time=0, length=20e-6)
            show_pulse_sheet("wigner_characteristic_function", compiled_exp_wigner_characteristic_function)

    def plot_wigner_characteristic_function_measurement_2D(self, vmin, vmax):

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        
        alpha_1_cavity_drive_amp = cavity_parameters[cavity_component]["alpha_1_cavity_drive_amp"]
        cavity_drive_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        cond_disp_pulse_amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        alpha_1_CNOD_amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"]

        length = cavity_parameters[cavity_component]["cavity_drive_length"]
        # amp = self.exp_wigner_function_amplitude * cavity_drive_amp

        self.wigner_characteristic_function_data = self.wigner_characteristic_function_results.get_data("wigner_characteristic_function_measurement")
        

        if self.which_data == "I":
            data = np.real(self.wigner_characteristic_function_data)
        else:
            data = np.imag(self.wigner_characteristic_function_data)

        amplitude_sweep_list = self.wigner_characteristic_function_results.acquired_results['wigner_characteristic_function_measurement'].axis[1]


        if self.is_wigner_function:
            Z = amplitude_sweep_list.reshape((self.amp_npts_y, self.amp_npts_x))
            x = np.real(Z) * cavity_drive_amp/alpha_1_cavity_drive_amp # scaling to alpha (photon number)
            y = np.imag(Z) * cavity_drive_amp/alpha_1_cavity_drive_amp   # scaling to alpha (photon number)
            data = data.reshape((self.amp_npts_y, self.amp_npts_x))

        else:
            Z = amplitude_sweep_list.reshape((self.amp_npts_y, self.amp_npts_x))
            x = np.real(Z) * cond_disp_pulse_amp/alpha_1_CNOD_amp # scaling to alpha (photon number)
            y = np.imag(Z) * cond_disp_pulse_amp/alpha_1_CNOD_amp   # scaling to alpha (photon number)
            data = data.reshape((self.amp_npts_y, self.amp_npts_x))

        np.save(f"wigner_characteristic_function", self.wigner_characteristic_function_data)
        np.save(f"Z_value", Z)
        ############# Plotting the data #####################

        plt.figure(figsize=(10, 10))
        pcm = plt.pcolormesh(x, y, data, shading='auto', cmap='RdBu_r', vmin = vmin, vmax = vmax)
        plt.colorbar(pcm, label=f'[{self.which_data}] (a.u.)')
        if self.is_wigner_function:
            plt.title(f"Wigner function")
        else:
            plt.title(f"Characteristic function")
        plt.xlabel(r'Re($\alpha$)')
        plt.ylabel(r'Im($\alpha$)')

        plt.show()

############ sweep beta for calibration of cnod amplitude used for disentanglement of the cat
    def disentangling_power_sweep(self, average_exponent=12,
                                  alpha = 1,
                                  beta_sweep_start = -1,
                                  beta_sweep_stop = 1,
                                  beta_sweep_count = 21,
                                  is_xyz=False,
                                  is_plot_simulation=False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        self.disentangling_power_sweep_alpha = alpha

        beta_sweep = LinearSweepParameter(uid="beta_sweep", start=beta_sweep_start, stop=beta_sweep_stop, count=beta_sweep_count)

        if is_xyz:
            sweep_case = LinearSweepParameter(uid="correction", start=0, stop=2, count=3)
            self.is_xyz = True
        else:
            sweep_case = LinearSweepParameter(uid="correction", start=0, stop=0, count=1)
        
        def _xyz(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)
    
            return decorator

        # Define the experiment
        exp_disentangling_power_sweep = Experiment(
            uid="disentangling_power_sweep",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_disentangling_power_sweep.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True,
        ):
            with exp_disentangling_power_sweep.sweep(uid="xyz", parameter=sweep_case):
                with exp_disentangling_power_sweep.sweep(uid="amplitude_sweep", parameter=beta_sweep, auto_chunking = True, reset_oscillator_phase=True):

                    with exp_disentangling_power_sweep.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                        with exp_disentangling_power_sweep.section(uid="qubit_preparation"):
                            exp_disentangling_power_sweep.play(signal="drive", pulse=pi2_pulse)
                        with exp_disentangling_power_sweep.section(uid="CNOD_1", play_after="qubit_preparation"):
                            self.CNOD(exp=exp_disentangling_power_sweep, cond_disp_pulse=cond_disp_pulse,
                                    pi_pulse=pi_pulse, amp=alpha,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                                    prev_uid=None,
                                    uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid = "pi_pulse_1")
                        with exp_disentangling_power_sweep.section(uid="qubit_1", play_after="CNOD_1"):
                            exp_disentangling_power_sweep.play(signal="drive", pulse=pi2_pulse)
                        with exp_disentangling_power_sweep.section(uid="CNOD_2", play_after="qubit_1"):
                            self.CNOD(exp=exp_disentangling_power_sweep, cond_disp_pulse=cond_disp_pulse,
                                    pi_pulse=pi_pulse, amp=1j*beta_sweep,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                                    prev_uid=None,
                                    uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid = "pi_pulse_2")
                        with exp_disentangling_power_sweep.section(uid="qubit_2", play_after="CNOD_2"):
                            @_xyz(sweep_case, exp=exp_disentangling_power_sweep)
                            def play_drive(v):
                                if v == 0: # X
                                    exp_disentangling_power_sweep.play(signal="drive", pulse=pi2_pulse, phase = 0)
                                elif v == 1: # Y
                                    exp_disentangling_power_sweep.play(signal="drive", pulse=pi2_pulse, phase = np.pi/2)
                                elif v == 2: # Z
                                    pass
               
                    with exp_disentangling_power_sweep.section(uid="measure", play_after="preparation"):
                        exp_disentangling_power_sweep.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_disentangling_power_sweep.acquire(signal="acquire",
                                                handle="disentangling_power_sweep",
                                                kernel=readout_weighting_function)
                        
                    with exp_disentangling_power_sweep.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_disentangling_power_sweep.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_disentangling_power_sweep.set_signal_map(signal_map)
        compiled_exp_disentangling_power_sweep = self.session.compile(exp_disentangling_power_sweep)
        self.disentangling_power_sweep_results = self.session.run(compiled_exp_disentangling_power_sweep)
        if is_plot_simulation:
            self.plot_results(self.disentangling_power_sweep_results)
            self.simulation_plot(compiled_exp_disentangling_power_sweep, start_time=0, length=20e-6)
            show_pulse_sheet("disentangling_power_sweep", compiled_exp_disentangling_power_sweep)
    
    def plot_disentangling_power_sweep(self):

        self.disentangling_power_sweep_data = self.disentangling_power_sweep_results.get_data("disentangling_power_sweep")

        if self.which_data == "I":
            data = np.real(self.disentangling_power_sweep_data)
        else:
            data = np.imag(self.disentangling_power_sweep_data)

        beta_sweep_list = self.disentangling_power_sweep_results.acquired_results['disentangling_power_sweep'].axis[1]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        cond_disp_pulse_amp = cavity_parameters[cavity_component]['cond_disp_pulse_amp']
        alpha_1_CNOD_amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"]

        ############# Plotting the data #####################

        if self.is_xyz:
            fig, ax = plt.subplots(3, 1, figsize=(20, 12))
            fig.suptitle(f"disentangling power sweep, alpha = {cond_disp_pulse_amp/alpha_1_CNOD_amp*self.disentangling_power_sweep_alpha}", fontsize=18)

            ax[0].plot(cond_disp_pulse_amp/alpha_1_CNOD_amp*beta_sweep_list, data[0], marker='o', linestyle=':', color='k', label='<X>')
            ax[1].plot(cond_disp_pulse_amp/alpha_1_CNOD_amp*beta_sweep_list, data[1], marker='o', linestyle=':', color='k', label='<Y>')
            ax[2].plot(cond_disp_pulse_amp/alpha_1_CNOD_amp*beta_sweep_list, data[2], marker='o', linestyle=':', color='k', label='<Z>')

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
        
        else:
            fig, ax = plt.subplots(1, 1, figsize=(20, 12))
            fig.suptitle(f"disentangling power sweep, alpha = {cond_disp_pulse_amp/alpha_1_CNOD_amp*self.disentangling_power_sweep_alpha}", fontsize=18)

            ax.plot(cond_disp_pulse_amp/alpha_1_CNOD_amp*beta_sweep_list, data, marker='o', linestyle=':', color='k', label='<X>')

            ax.legend()

        fig.text(0.5, 0.04, 'Beta (a.u.)', ha='center', fontsize=16)


# In[]
