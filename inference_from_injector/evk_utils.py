# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import sys

try:
    from metavision_core.event_io import EventsIterator, RawReader, EventDatReader
    from metavision_ml.preprocessing.event_to_tensor import histo_quantized, histo, diff_quantized
    from metavision_core_ml.preprocessing.event_to_tensor_torch import event_image, event_cd_to_torch
    import dv_processing as dv
except ImportError:
    print("Need `metavision` library installed.", file=sys.stderr)
    exit(1)

import cv2
import numpy as np
import time
from threading import Thread

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.peripherals.dvs.transformation import Compose, EventVolume

class EventFrameDisplay(AbstractProcess):
    """
    Process that receives:
    (1) raw events / images
    (2) spike rates of selective DNF
    (3) spike rates of multi-peak DNF

    It sends these values through a multiprocessing pipe (rather than a Lava
    OutPort) to allow for plotting.

    Parameters
    ----------
    event_shape: (int, int)
        Shape of the frame to visualize.
    """
    def __init__(self,
                 event_frame_shape):
        super().__init__(event_frame_shape=event_frame_shape)

        self.event_frame_port = InPort(shape=event_frame_shape)

@implements(proc=EventFrameDisplay, protocol=LoihiProtocol)
@requires(CPU)
class EventFrameDisplayProcess(PyLoihiProcessModel):
    event_frame_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)

        # Create windows
        cv2.namedWindow("Event Frame", cv2.WINDOW_NORMAL)
        self.event_frame = np.ndarray(proc_params["event_frame_shape"])

    def post_guard(self):
        # Ensures that run_post_mgmt runs after run_spk at every
        # time step.
        return True

    def run_post_mgmt(self):
        self.event_frame = self.event_frame_port.recv()

    def _display(self, event_frame):
        """Visualize images on OpenCV windows

        Takes a NumPy image array formatted as RGBA and sends to OpenCV for
        visualization.

        Parameters
        ----------
        dvs_image           [np.Array]: NumPy array of DVS Image
        """

        cv2.imshow("Event Frame", event_frame)
        cv2.waitKey(1)

    def run_spk(self):
        self._display(self.event_frame)


class PropheseeCamera(AbstractProcess):
    """
    Process that receives events from Prophesee device and sends them out as a
    histogram.

    Parameters
    ----------
    filename: str
        String to filename if reading from a RAW/DAT file or empty string for
        using a camera.
    biases: dict
        Dictionary of biases for the DVS Camera.
    filters: list
        List containing metavision filters.
    mode: str
        String to specify to load events by numbers of events, timeslice or the first
        met criterion. (Choice of "mixed", "delta_t", or "n_events"). Default is "mixed".
    n_events: int
        Number of events in the timeslice.
    delta_t: int
        Duration of served event slice in us.
    transformations: Compose
        Transformations to be applied to the events before sending them out.
    num_output_time_bins: int
        The number of output time bins to use for the ToFrame transformation.
    num_steps: int
        Number of time steps used to run (used in latency collection)
    """
    def __init__(
        self,
        sensor_shape: tuple,
        filename: str = "",
        biases: dict = None,
        filters: list = [],
        mode: str = "mixed",
        n_events: int = 1000,
        delta_t: int = 1000,
        transformations: Compose = None,
        num_output_time_bins: int = 1,
        out_shape: tuple = None,
        num_steps: int = 1,
    ):
        if not isinstance(n_events, int) or n_events < 0:
            raise ValueError(
                "n_events must be a positive integer value."
            )

        if (
            not isinstance(num_output_time_bins, int)
            or num_output_time_bins < 0
        ):
            raise ValueError(
                "num_output_time_bins must be a positive integer value."
            )

        if biases is not None and not filename == "":
            raise ValueError("Cant set biases if reading from file.")

        self.filename = filename
        self.biases = biases
        self.mode = mode
        self.n_events = n_events
        self.delta_t = delta_t

        self.filters = filters
        self.transformations = transformations
        self.num_output_time_bins = num_output_time_bins

        height, width = sensor_shape

        if out_shape is not None:
            self.shape = out_shape
        # Automatically determine out_shape
        else:
            event_shape = EventVolume(height=height, width=width, polarities=2)
            if transformations is not None:
                event_shape = self.transformations.determine_output_shape(
                    event_shape
                )
            self.shape = (
                num_output_time_bins,
                event_shape.polarities,
                event_shape.height,
                event_shape.width,
            )

        # Check whether provided transformation is valid
        if self.transformations is not None:
            try:
                # Generate some artificial data
                n_random_spikes = 1000
                test_data = np.zeros(
                    n_random_spikes,
                    dtype=np.dtype(
                        [("y", int), ("x", int), ("p", int), ("t", int)]
                    ),
                )
                test_data["x"] = np.random.rand(n_random_spikes) * width
                test_data["y"] = np.random.rand(n_random_spikes) * height
                test_data["p"] = np.random.rand(n_random_spikes) * 2
                test_data["t"] = np.sort(np.random.rand(n_random_spikes) * 1e6)

                # Transform data
                self.transformations(test_data)
                if len(test_data) > 0:
                    volume = np.zeros(self.shape, dtype=np.uint8)
                    histo_quantized(test_data, volume, np.max(test_data["t"]))
                    # frames = event_image(
                    #     event_cd_to_torch(test_data),
                    #     1,
                    #     height,
                    #     width)

            except Exception:
                raise Exception(
                    "Your transformation is not compatible with the provided \
                    data."
                )

        self.s_out = OutPort(shape=self.shape)

        self.dvs_start_time = Var(shape=((num_steps),), init=0)
        self.before_processing = Var(shape=((num_steps),), init=0)
        self.after_processing = Var(shape=((num_steps),), init=0)
        self.end_time = Var(shape=((num_steps),), init=0)

        super().__init__(
            shape=self.shape,
            biases=self.biases,
            filename=self.filename,
            filters=self.filters,
            mode = self.mode,
            n_events=self.n_events,
            delta_t=self.delta_t,
            transformations=self.transformations,
            num_output_time_bins=self.num_output_time_bins,
        )


class EventsIteratorWrapper():
    """
    PropheseeEventsIterator class for PropheseeCamera which will create a
    thread in the background to always grab events within a time window and
    put them in a buffer.

    delta_t: Control the size of the time window for collecting events,
    and generate frames for the events in this time window.Increase delta_t,
    a frame can capture more events, but when you move fast, there will be
    too many events, at this time, you should consider reducing delta_t.
    If delta_t is too large and greater than time interval between two steps
    of processmodel, the thread will recv the old events data from the
    previous time step.

    Parameters
    ----------
    device: str
        String to filename if reading from a RAW/DAT file or empty string for
        using a camera.
    sensor_shape: (int, int)
        Shape of the camera sensor or file recording.
    mode: str
        String to specify to load events by numbers of events, timeslice or the first
        met criterion. (Choice of "mixed", "delta_t", or "n_events"). Default is "mixed".
    n_events: int
        Number of events in the timeslice.
    delta_t: int
        Duration of served event slice in us.
    biases: list
        Bias settings for camera.
    """
    def __init__(self,
                 device: str,
                 sensor_shape: tuple,
                 mode: str = "mixed",
                 n_events: int = 10000,
                 delta_t: int = 10000,
                 biases: dict = None,):
        self.true_height, self.true_width = sensor_shape

        self.mv_iterator = EventsIterator(input_path=device, mode=mode,
                                          n_events=n_events, delta_t=delta_t)

        if biases is not None:
            # Setting Biases for DVS camera
            device_biases = self.mv_iterator.reader.device.get_i_ll_biases()
            for k, v in biases.items():
                device_biases.set(k, v)

        # afk = self.mv_iterator.reader.device.get_i_antiflicker_module()
        # afk.set_frequency_band(min_freq=50, max_freq=520)
        # afk.enable(b=True)
        
        # NOTE: DigitalCrop currently not working...
        # alternatively use get_i_roi()
        roi = self.mv_iterator.reader.device.get_i_digital_crop()
        roi.set_window_region([0, 0, 344, 260], False)
        roi.enable(True)

        print(roi.get_window_region())
        
        self.thread = Thread(target=self.recv_from_dvs)
        self.stop = False
        self.res = np.zeros(((self.true_width, self.true_height) + (1,) + (1,)),
                            dtype=np.dtype([("y", int), ("x", int),
                                            ("p", int), ("t", int)]))

    def start(self):
        self.thread.start()

    def join(self):
        self.stop = True
        self.thread.join()

    def get_events(self):
        return self.res

    def recv_from_dvs(self):
        for evs in self.mv_iterator:
            self.res = evs

            if self.stop:
                return


@implements(proc=PropheseeCamera, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt", 'event_it')
class PyPropheseeCameraEventsIteratorModel(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    dvs_start_time: np.ndarray = LavaPyType(np.ndarray, float)
    before_processing: np.ndarray = LavaPyType(np.ndarray, float)
    after_processing: np.ndarray = LavaPyType(np.ndarray, float)
    end_time: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params["shape"]
        (
            self.num_output_time_bins,
            self.polarities,
            self.height,
            self.width,
        ) = self.shape
        self.filename = proc_params["filename"]
        self.filters = proc_params['filters']
        self.mode = proc_params['mode']
        self.n_events = proc_params['n_events']
        self.delta_t = proc_params['delta_t']
        self.biases = proc_params['biases'] # None
        self.transformations = proc_params['transformations']
        self.sensor_shape = (self.height,
                             self.width)

        self.reader = EventsIteratorWrapper(
            device=self.filename,
            sensor_shape=self.sensor_shape,
            mode=self.mode,
            n_events=self.n_events,
            delta_t=self.delta_t,
            biases=self.biases)
        self.reader.start()

        self.volume = np.zeros(
            (
                self.num_output_time_bins,
                self.polarities,
                self.height,
                self.width,
            ),
            dtype=float,
        )

    def run_spk(self):
        """
        Load events from DVS, apply filters and transformations and send
        spikes as frame
        """
        # self.dvs_start_time[self.time_step - 1] = time.time_ns()
        events = self.reader.get_events()
        # self.before_processing[self.time_step - 1] = time.time_ns()

        # Apply filters to events
        for filter in self.filters:
            events_out = filter.get_empty_output_buffer()
            filter.process_events(events, events_out)
            events = events_out

        if len(self.filters) > 0:
            events = events.numpy()

        # Transform events
        if self.transformations is not None and len(events) > 0:
            self.transformations(events)

        # Transform to frame
        if len(events) > 0:
            histo_quantized(events, self.volume, 1000, reset=True, normalization=True)
            frames = self.volume
            # frames = event_image(
            #     event_cd_to_torch(events),
            #     1,
            #     self.s_out.shape[2],
            #     self.s_out.shape[3])
            # frames = np.expand_dims(frames.cpu().numpy().astype(np.int8), axis=0)
            # frames[frames != 0] = 1
        else:
            frames = np.zeros(self.s_out.shape, dtype=float)

        # self.after_processing[self.time_step - 1] = time.time_ns()
        self.s_out.send(frames)
        # self.end_time[self.time_step - 1] = time.time_ns()

    def _pause(self):
        """Pause was called by the runtime"""
        super()._pause()
        self.t_pause = time.time_ns()

    def _stop(self):
        """
        Stop was called by the runtime.
        Helper thread for DVS is also stopped.
        """
        self.reader.join()
        super()._stop()


@implements(proc=PropheseeCamera, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt", 'raw_reader')
class PyPropheseeCameraRawReaderModel(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
    dvs_start_time: np.ndarray = LavaPyType(np.ndarray, float)
    before_processing: np.ndarray = LavaPyType(np.ndarray, float)
    after_processing: np.ndarray = LavaPyType(np.ndarray, float)
    end_time: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params["shape"]
        (
            self.num_output_time_bins,
            self.polarities,
            self.height,
            self.width,
        ) = self.shape
        self.filename = proc_params["filename"]
        self.filters = proc_params["filters"]
        self.n_events = proc_params["n_events"]
        self.biases = proc_params["biases"]
        self.transformations = proc_params["transformations"]

        if self.filename.split('.')[-1] == 'dat':
            self.reader = EventDatReader(self.filename)
        else:
            self.reader = RawReader(self.filename,
                                    max_events=self.n_events)

        if self.biases is not None:
            # Setting Biases for DVS camera
            device_biases = self.reader.device.get_i_ll_biases()
            for k, v in self.biases.items():
                device_biases.set(k, v)

        self.volume = np.zeros(
            (
                self.num_output_time_bins,
                self.polarities,
                self.height,
                self.width,
            ),
            dtype=np.uint8,
        )
        self.t_pause = time.time_ns()
        self.t_last_iteration = time.time_ns()

    def run_spk(self):
        """Load events from DVS, apply filters and transformations and send
        spikes as frame"""
        # self.dvs_start_time[self.time_step - 1] = time.time_ns()self.time_
        # Time passed since last iteration
        t_now = time.time_ns()

        # Load new events since last iteration
        if self.t_pause > self.t_last_iteration:
            # Runtime was paused in the meantime
            delta_t = np.max(
                [10000, (self.t_pause - self.t_last_iteration) // 1000]
            )
            delta_t_drop = np.max([10000, (t_now - self.t_pause) // 1000])

            events = self.reader.load_delta_t(delta_t)
            _ = self.reader.load_delta_t(delta_t_drop)
        else:
            # Runtime was not paused in the meantime
            delta_t = np.max([10000, (t_now - self.t_last_iteration) // 1000])
            events = self.reader.load_delta_t(delta_t)

        # self.before_processing[self.time_step - 1] = time.time_ns()

        # Apply filters to events
        for filter in self.filters:
            events_out = filter.get_empty_output_buffer()
            filter.process_events(events, events_out)
            events = events_out

        if len(self.filters) > 0:
            events = events.numpy()

        # Transform events
        if self.transformations is not None and len(events) > 0:
            self.transformations(events)

        # Transform to frame
        if len(events) > 0:
            # ORIGINAL: using histo_quantized
            histo_quantized(events, self.volume, delta_t, reset=True, normalization=False)
            frames = self.volume
            # NEW: using event_image
            # frames = event_image(
            #     event_cd_to_torch(events),
            #     1,
            #     self.s_out.shape[2],
            #     self.s_out.shape[3])
            # frames = np.expand_dims(frames.cpu().numpy().astype(np.int8), axis=0)
            # frames[frames != 0] = 1
        else:
            frames = np.zeros(self.s_out.shape) #, dtype=float)

        # self.after_processing[self.time_step - 1] = time.time_ns()
        # Send
        # frames = frames.astype(float) / frames.max()
        self.s_out.send(frames)
        self.t_last_iteration = t_now
        # self.end_time[self.time_step - 1] = time.time_ns()

    def _pause(self):
        """Pause was called by the runtime"""
        super()._pause()
        self.t_pause = time.time_ns()

class Inivation(AbstractProcess):
    """
    Process that receives events from Inivation device and sends them out as a
    histogram.

    Parameters
    ----------
    sensor_shape: (int, int)
        Shape of the sensor.
    filename: str
        String to filename if reading from a recording file or empty string for
        using a camera.
    transformations: Compose
        Tonic transformations to be applied to the events before sending them out.
    num_output_time_bins: int
        The number of output time bins to use for the ToFrame transformation.
    out_shape: (int, int)
        Output shape of the process.
    """

    def __init__(self,
                 sensor_shape: tuple,
                 filename: str,
                 transformations: Compose = None,
                 num_output_time_bins: int = 1,
                 out_shape: tuple = None,
                 ):
        if (not isinstance(num_output_time_bins, int)
                or num_output_time_bins < 0
                or num_output_time_bins > 1):
            raise ValueError("Not Implemented: num_output_time_bins must be 1.")

        self.filename = filename
        self.transformations = transformations
        self.num_output_time_bins = num_output_time_bins

        height, width = sensor_shape

        if out_shape is not None:
            self.shape = out_shape
        # Automatically determine out_shape
        else:
            event_shape = EventVolume(height=height, width=width, polarities=2)
            if transformations is not None:
                event_shape = self.transformations.determine_output_shape(event_shape)
            self.shape = (num_output_time_bins,
                          event_shape.polarities,
                          event_shape.height,
                          event_shape.width)
        # Check whether provided transformation is valid
        if self.transformations is not None:
            try:
                # Generate some artificial data
                n_random_spikes = 1000
                test_data = np.zeros(n_random_spikes,
                                     dtype=np.dtype([("y", int), ("x", int),
                                                     ("p", int), ("t", int)]))
                test_data["x"] = np.random.rand(n_random_spikes) * width
                test_data["y"] = np.random.rand(n_random_spikes) * height
                test_data["p"] = np.random.rand(n_random_spikes) * 2
                test_data["t"] = np.sort(np.random.rand(n_random_spikes) * 1e6)

                # Transform data
                self.transformations(test_data)
                if len(test_data) > 0:
                    volume = np.zeros(self.shape, dtype=np.uint8)
                    histo_quantized(test_data, volume, np.max(test_data['t']))
            except Exception:
                raise Exception("Your transformation is not compatible with \
                                the provided data.")

        self.s_out = OutPort(shape=self.shape)

        super().__init__(shape=self.shape,
                         filename=self.filename,
                         transformations=self.transformations,
                         num_output_time_bins=num_output_time_bins)


@implements(proc=Inivation, protocol=LoihiProtocol)
@requires(CPU)
class DensePyInivationPM(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params['shape']
        self.num_output_time_bins, self.polarities, self.height, self.width = self.shape
        self.filename = proc_params['filename']
        self.transformations = proc_params['transformations']

        if self.filename == "":
            self.capture = dv.io.CameraCapture()
        else:
            self.capture = dv.io.MonoCameraRecording(self.filename)

        self.volume = np.zeros((self.num_output_time_bins, self.polarities,
                                self.height, self.width),
                               dtype=np.uint8)
        self.t_pause = time.time_ns()
        self.t_last_iteration = time.time_ns()

    def run_spk(self):
        """
        Load events from DVS, apply filters and transformations and send
        spikes as frame
        """

        # Time passed since last iteration
        t_now = time.time_ns()

        # Load new events since last iteration
        if self.t_pause > self.t_last_iteration:
            raise NotImplementedError("Pausing it not implemented yet.")
        else:
            events = self.capture.getNextEventBatch()

        if events is not None:
            events_np = events.numpy()
            events = np.zeros(len(events_np),
                              dtype=np.dtype([("y", int), ("x", int),
                                              ("p", int), ("t", int)]))
            events['t'] = events_np['timestamp']
            events['p'] = events_np['polarity']
            events['x'] = events_np['x']
            events['y'] = events_np['y']
        else:
            events = np.array([])

        # Transform events
        if self.transformations is not None and len(events) > 0:
            self.transformations(events)

        # Transform to frame
        if len(events) > 0:
            histo_quantized(events,
                            self.volume, events['t'][-1] - events['t'][0],
                            reset=True)
            frames = self.volume
            frames[frames != 0] = 1
        else:
            frames = np.zeros(self.s_out.shape)

        # Send
        self.s_out.send(frames)
        self.t_last_iteration = t_now

    def _pause(self):
        """Pause was called by the runtime"""
        super()._pause()
        self.t_pause = time.time_ns()