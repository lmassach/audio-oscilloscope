#!/usr/bin/env python3
"""Simple oscilloscope using the microphone input."""
import argparse
from collections import namedtuple
import datetime
import queue
import sys
import threading
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

FALLING_EDGE = '\u21a7'
RISING_EDGE = '\u21a5'


class Reporter:
    """Behaves like print(), but only writes once every `interval` seconds."""
    def __init__(self, interval=0.5):
        interval = float(interval)
        if not interval > 0:
            raise ValueError("Interval must be positive")
        self._interval = datetime.timedelta(seconds=interval)
        self._last = datetime.datetime.now()

    def __call__(self, *args, **kwargs):
        now = datetime.datetime.now()
        if now > self._last + self._interval:
            self._last = now
            print(*args, **kwargs)


class NoneContext:
    """A dummy context manager that just returns None."""
    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        pass


def safe_open(fn, *args, **kwargs):
    """Like open(), but returns a NoneContext if the filename is None."""
    if fn is not None:
        return open(fn, *args, **kwargs)
    return NoneContext()


# A class for keeping processed data
TriggeredData = namedtuple("TriggeredData", "count time data")
# count: the trigger number
# time: the trigger time, in seconds since the acquisition began
# data: a 1D array of samples (half before the trigger, half after)


class Oscilloscope:
    """A simple oscilloscope based on matplotlib and sounddevice."""
    def __init__(self, device=None, channel=0, window=0.01, interval=0.03,
                 sampling_rate=None, downsample=1, trigger_level=None,
                 trigger_edge_falling=True, output_file=None,
                 print_debug_info=False):
        # Configuration input
        self._device_name = device
        self._channel = int(channel)
        if not self._channel >= 0:
            raise ValueError("The channel number must be >= 0")
        self._window = float(window)  # In seconds
        self._interval = float(interval)  # In seconds
        if not self._interval > 0:
            raise ValueError("The plot refresh interval must be > 0")
        self._rate = (  # In Hz
            float(sampling_rate) if sampling_rate is not None else
            sd.query_devices(self._device_name, 'input')['default_samplerate'])
        self._downsample = int(downsample)
        if not self._downsample > 0:
            raise ValueError("The downsample must be >= 1")
        self._level = None if trigger_level is None else float(trigger_level)
        if self._level is not None and not -1 < self._level < 1:
            raise ValueError("Trigger level must be in (-1, 1)")
        self._edge = bool(trigger_edge_falling)
        self._debug = bool(print_debug_info)
        self._out_fn = output_file

        # Computed constants
        self._window_len = max(256, (int(self._window * self._rate) // 2) * 2)
        self._plot_time = np.linspace(  # In ms
            -self._window_len // 2 / self._rate * 1000,
            self._window_len // 2 / self._rate * 1000,
            self._window_len)

        # Variables
        self._q_data = queue.Queue()
        self._stop = True  # Exit flag
        self._trg_count = 0  # Trigger count
        self._q_trg = queue.Queue()
        self._fig_n = None  # Figure number
        self._plot_h = None  # Plot handle

    def _audio_cb(self, data, frames, time, status):
        """Called from a separate thread for each new input data block."""
        if status:
            print("Oscilloscope: input device:", status, file=sys.stderr)
        # In run(), the blocksize parameter of sd.InputStream is used to
        # force this method to receive <= self._window_len samples each time
        self._q_data.put(data[::self._downsample, self._channel].copy())

    def _data_proc(self):
        count, start, report = 0, datetime.datetime.now(), Reporter()
        buf = np.zeros(self._window_len * 2)
        trg_idx = 0  # Position of the last trigger in the buffer
        while True:
            if self._stop:
                break
            # Get new chunk of data
            try:
                new_data = self._q_data.get(timeout=0.05)
            except queue.Empty:
                continue  # Allows to check if self._stop was set every 50 ms
            l = len(new_data)  # l is always <= self._window_len (see blocksize)
            count += l
            if self._debug:
                rate = count / (datetime.datetime.now() - start).total_seconds()
                report(f"{count} samples ({rate:.0f} Hz)", end='\r')
            # Put new data in the buffer
            buf[:-l] = buf[l:]
            buf[-l:] = new_data
            trg_idx -= l
            # Look for triggers
            if self._level is None:  # Auto mode: trigger every self._window_len samples
                while trg_idx < self._window_len:
                    self._trg_count += 1
                    self._q_trg.put(TriggeredData(
                        self._trg_count, (count - len(buf) + trg_idx) / self._rate,
                        buf[trg_idx:trg_idx+self._window_len].copy()))
                    trg_idx += self._window_len
            else:
                # Avoid triggering more than once every self._window_len
                first = max(self._window_len // 2, trg_idx + self._window_len)
                last = 3 * self._window_len // 2
                if last <= first:
                    continue
                # Look for triggers
                if self._edge:  # Falling edge
                    t = (buf[first:last-1] >= self._level) & (buf[first+1:last] < self._level)
                else:  # Rising edge
                    t = (buf[first:last-1] <= self._level) & (buf[first+1:last] > self._level)
                # Only consider triggers that have self._window_len//2 samples to
                # their left and right, and have not been already processed
                try:
                    trg_idx = int(np.argwhere(t)[0]) + first
                except IndexError:
                    continue  # No trigger
                self._trg_count += 1
                self._q_trg.put(TriggeredData(
                    self._trg_count, (count - len(buf) + trg_idx) / self._rate,
                    buf[trg_idx-self._window_len//2:trg_idx+self._window_len//2].copy()))

    def run(self):
        """Run the oscilloscope. This will activate PyPlot's interactive.

        The caller thread will be used to run the plot main loop.
        Two more threads will be spawned: one to acquire data from the
        audio device, another to process it; both will be terminated
        before exiting this function."""
        # Prepare input processing
        stream = sd.InputStream(
            device=self._device_name, channels=self._channel+1,
            blocksize=min(self._window_len * self._downsample, 4096),
            samplerate=self._rate, callback=self._audio_cb)
        proc_th = threading.Thread(target=self._data_proc)
        # Prepare plotting
        plt.ion()
        if not plt.fignum_exists(self._fig_n):
            fig = plt.figure()
            self._fig_n = fig.number
            self._plot_h, = plt.plot(self._plot_time, np.zeros_like(self._plot_time))
            if self._level is not None:
                plt.axhline(self._level, c='tab:red', ls='--')
            plt.grid()
            plt.ylim(-1, 1)
            plt.xlabel("Time [ms]")
            plt.ylabel("Amplitude [a.u.]")
            plt.title("Waiting for trigger")
            dev_info = (f"{sd.query_devices(self._device_name)['name']}"
                        f" @ {stream.samplerate/1e3:.3g} kHz,"
                        f" channel {self._channel+1}")
            trg_info = ("AUTO" if self._level is None else
                        f"{FALLING_EDGE if self._edge else RISING_EDGE} {self._level}")
            plt.suptitle(dev_info)
        # Run main loop
        self._stop = False
        proc_th.start()
        try:
            with stream, safe_open(self._out_fn, 'a') as out_f:
                while proc_th.is_alive() and plt.fignum_exists(self._fig_n):
                    # Get the latest trigger
                    data = None
                    try:
                        # Limiting the number of calls to get_nowait avoids a deadlock
                        n = self._q_trg.qsize()
                        for _ in range(n):
                            data = self._q_trg.get_nowait()
                            if out_f is not None:
                                print(f"{data.count} {data.time}", file=out_f)
                    except queue.Empty:
                        pass
                    if data is not None:
                        # Do the plotting
                        plt.title(f"Trigger #{data.count} ({trg_info}) @ +{datetime.timedelta(seconds=data.time)}")
                        self._plot_h.set_ydata(data.data)
                        plt.draw()
                    plt.pause(self._interval)
        finally:
            self._stop = True
            proc_th.join()


def int_or_str(x):
    """Helper for argument parsing."""
    try:
        return int(x)
    except ValueError:
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-l", "--list", action="store_true",
                        help="Show a list of audio devices and exit.")
    parser.add_argument("-t", "--trigger", type=float,
                        help="Trigger level (default no trigger / auto mode).")
    parser.add_argument("-e", "--edge", choices=['r', 'f'], default='f',
                        help="Trigger edge: falling (default) or rising.")
    parser.add_argument("-c", "--channel", type=lambda x: int(x) - 1, default=1,
                        help="Input channel number (default 1, the first).")
    parser.add_argument('-d', '--device', type=int_or_str,
                        help='Input device (number or substring), see -l.')
    parser.add_argument('-w', '--window', type=float, default=10, metavar='WIDTH',
                        help='Visible time slot in ms (default: %(default)s ms).')
    parser.add_argument('-i', '--interval', type=float, default=30,
                        help='Minimum time between plot updates in ms (default: %(default)s ms).')
    parser.add_argument('-r', '--samplerate', type=float,
                        help='Sampling rate of audio device in Hz.')
    parser.add_argument('-n', '--downsample', type=int, default=1, metavar='N',
                        help='Display every Nth sample (default: %(default)s).')
    parser.add_argument('-o', '--output',
                        help="Save trigger times to this output file.")
    parser.add_argument('--debug', action='store_true', help='Print debug messages.')
    args = parser.parse_args()

    if args.list:
        print(sd.query_devices())
        parser.exit(0)

    Oscilloscope(
        args.device, args.channel, args.window / 1e3, args.interval / 1e3,
        args.samplerate, args.downsample, args.trigger, args.edge == 'f',
        args.output, args.debug).run()
