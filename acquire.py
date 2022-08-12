#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys
import signal
import datetime
import threading
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        '-c', '--channel', type=int, default=1,
        help='input channel to plot (default: the first)')
    parser.add_argument(
        '-t', '--trigger', type=float, default=0.0,
        help="Trigger level (triggers on falling edge)")
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring), use -l to list devices')
    parser.add_argument(
        '-w', '--window', type=float, default=10, metavar='DURATION',
        help='visible time slot in ms (default: %(default)s ms)')
    parser.add_argument(
        '-i', '--interval', type=float, default=30,
        help='minimum time between plot updates (default: %(default)s ms)')
    parser.add_argument(
        '-b', '--blocksize', type=int, help='block size (in samples)')
    parser.add_argument(
        '-r', '--samplerate', type=float, help='sampling rate of audio device')
    parser.add_argument(
        '-n', '--downsample', type=int, default=1, metavar='N',
        help='display every Nth sample (default: %(default)s)')
    parser.add_argument("-z", "--zoom", action="store_true", help="Automatic zoom")
    args = parser.parse_args(remaining)
    if args.channel < 1:
        parser.error('argument CHANNEL: must be >= 1')
    mapping = args.channel - 1  # Channel numbers start with 1

    q = queue.Queue()


    def audio_callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        q.put(indata[::args.downsample, mapping])


    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    stream = sd.InputStream(
        device=args.device, channels=args.channel, blocksize=args.blocksize,
        samplerate=args.samplerate, callback=audio_callback)

    window_length = int(args.window * args.samplerate / 1000)
    print("window", args.window, "ms, rate", args.samplerate, " Hz, length", window_length)
    plot_time = np.linspace(
        -window_length//2 / args.samplerate * 1000,
        window_length//2 / args.samplerate * 1000,
        2* (window_length//2))
    trigger_count = 0

    quit = False


    def sh(_a, _b):
        """Signal handler for graceful exit"""
        global quit
        quit = True


    signal.signal(signal.SIGINT, sh)


    def plot(data):
        """Convenience function for unified plot style"""
        global trigger_count
        plt.clf()
        plt.plot(plot_time, data)
        plt.xlabel("Time [ms]")
        plt.ylabel("Signal [a.u.]")
        # plt.xlim(-window_length//2 / args.samplerate * 1000, window_length//2 / args.samplerate * 1000)
        if not args.zoom:
            plt.ylim(-1, 1)
        plt.grid()
        plt.axhline(args.trigger, c='tab:red', ls='--')
        plt.title(f"Trigger #{trigger_count} at {datetime.datetime.now().isoformat()}")


    data = np.zeros(max(4096, 3 * window_length, 3 * (args.blocksize or 1024)))
    count, start_time = 0, datetime.datetime.now()
    q_plot = queue.Queue()


    def data_process():
        """This runs in yet another thread and exhaust the audio data queue."""
        global data, count, trigger_count
        last_report = datetime.datetime.now()
        it = -1
        while True:
            try:
                new_data = q.get(timeout=0.05)
                count += len(new_data)
            except queue.Empty:
                break
            l = len(new_data)
            data[:-l] = data[l:]  # Shift old data
            data[-l:] = new_data  # Put in new data
            it -= l  # Shift the last trigger index too

            if datetime.datetime.now() > datetime.timedelta(seconds=0.5) + last_report:
                last_report = datetime.datetime.now()
                rate = count / (datetime.datetime.now() - start_time).total_seconds()
                print(f"{count} samples ({rate:.0f} Hz)", end='\r')

            m = (data[:-1] >= args.trigger) & (data[1:] < args.trigger)
            for idx in np.argwhere(m[window_length//2:window_length]) + window_length//2:
                if idx <= it:
                    continue
                idx = int(idx)
                it = idx
                trigger_count += 1
                q_plot.put(data[idx-window_length//2:idx+window_length//2])


    print("Stop with CTRL+C")
    plt.ion()
    fig = plt.figure()
    fig_n = fig.number
    plot(np.zeros(2*window_length//2))

    with stream:
        th = threading.Thread(target=data_process)
        th.start()
        while True:
            plt.pause(args.interval / 1000)
            if quit or not plt.fignum_exists(fig_n) or not th.is_alive():
                break

            d = None
            try:
                while True:
                    d = q_plot.get_nowait()
            except queue.Empty:
                pass
            if d is None:
                continue
            plot(d)
