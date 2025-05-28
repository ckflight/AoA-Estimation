import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import zmq
import sys

# ==== CONFIGURATION ====
mode = 'live'           # 'file' or 'live'
smooth_plot = 1         # 1: smooth, 0: histogram
start_percent = 0       # For file mode: start reading at N% of file
plot_interval = 1       # Update plot every N chunks
chunk_size = 300_000    # Samples per channel (two channels interleaved)
# =======================

# Constants
d = 0.10                # Distance between antennas (meters)
c = 3e8                 # Speed of light (m/s)

# Frequencies for modes
freq_file = 800e6       # Hz for file mode
freq_live = 800e6      # Hz for live mode

wavelength_file = c / freq_file
wavelength_live = c / freq_live

# Histogram bins
bins_phase = np.linspace(-180, 180, 100)
bins_argument = np.linspace(-1, 1, 100)
bins_aoa = np.linspace(-90, 90, 100)
centers_phase = (bins_phase[:-1] + bins_phase[1:]) / 2
centers_argument = (bins_argument[:-1] + bins_argument[1:]) / 2
centers_aoa = (bins_aoa[:-1] + bins_aoa[1:]) / 2

plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

def plot_histograms(h_phase, h_argument, h_aoa, chunk_idx, total_chunks=None, title_suffix=""):
    axes[0].cla()
    axes[1].cla()
    axes[2].cla()

    if smooth_plot:
        axes[0].plot(centers_phase, h_phase, color='gray')
    else:
        axes[0].bar(bins_phase[:-1], h_phase, width=np.diff(bins_phase)[0], color='gray')
    axes[0].set_title("Phase Diff (deg)")
    axes[0].set_xlim(-180, 180)
    axes[0].grid(True)

    max_mid_arg = max(h_argument[1:-1]) if len(h_argument) > 2 else 1
    if smooth_plot:
        axes[1].plot(centers_argument, h_argument, color='orange')
    else:
        axes[1].bar(bins_argument[:-1], h_argument, width=np.diff(bins_argument)[0], color='orange')
    axes[1].set_title("Arcsin Argument")
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(0, max_mid_arg * 1.1)
    axes[1].grid(True)

    max_mid_aoa = max(h_aoa[1:-1]) if len(h_aoa) > 2 else 1
    if smooth_plot:
        axes[2].plot(centers_aoa, h_aoa, color='blue')
    else:
        axes[2].bar(bins_aoa[:-1], h_aoa, width=np.diff(bins_aoa)[0], color='blue')
    axes[2].set_title("Estimated AoA")
    axes[2].set_xlim(-90, 90)
    axes[2].set_ylim(0, max_mid_aoa * 1.1)
    axes[2].grid(True)

    if total_chunks:
        plt.suptitle(f"Chunk {chunk_idx+1}/{total_chunks} {title_suffix}")
    else:
        plt.suptitle(f"Chunk {chunk_idx+1} {title_suffix}")
    plt.pause(0.001)

def process_samples(raw, wavelength):
    rx1 = raw[0::2]
    rx2 = raw[1::2]
    cross = rx1 * np.conj(rx2)
    phase_diff = np.angle(cross)
    phase_diff_deg = np.degrees(phase_diff)
    argument = np.clip(phase_diff * wavelength / (2 * np.pi * d), -1, 1)
    aoa_deg = np.degrees(np.arcsin(argument))

    h_phase, _ = np.histogram(phase_diff_deg, bins=bins_phase)
    h_argument, _ = np.histogram(argument, bins=bins_argument)
    h_aoa, _ = np.histogram(aoa_deg, bins=bins_aoa)

    if smooth_plot:
        h_phase = gaussian_filter1d(h_phase, sigma=2)
        h_argument = gaussian_filter1d(h_argument, sigma=2)
        h_aoa = gaussian_filter1d(h_aoa, sigma=2)

    return h_phase, h_argument, h_aoa

if mode == 'file':
    filename = "/home/ubuntu/Desktop/aoa_estimation/aoa_log.dat"
    sample_bytes = 8  # 8 bytes = 2x complex64 (each 4 bytes)
    wavelength = wavelength_file

    file_size = os.path.getsize(filename)
    total_samples = file_size // sample_bytes
    total_chunks = total_samples // (chunk_size * 2)

    start_sample = int(total_samples * start_percent / 100)
    start_sample -= start_sample % (chunk_size * 2)
    remaining_samples = total_samples - start_sample
    remaining_chunks = remaining_samples // (chunk_size * 2)

    print(f"Starting at {start_percent}% ({start_sample} samples)")
    print(f"Remaining chunks: {remaining_chunks}")

    try:
        with open(filename, "rb") as f:
            f.seek(start_sample * sample_bytes)

            for chunk_idx in tqdm(range(remaining_chunks), desc="Chunks", unit="chunk"):
                raw = np.fromfile(f, dtype=np.complex64, count=chunk_size * 2)
                if len(raw) < chunk_size * 2:
                    break

                h_phase, h_argument, h_aoa = process_samples(raw, wavelength)

                if chunk_idx % plot_interval == 0:
                    plot_histograms(h_phase, h_argument, h_aoa, chunk_idx, remaining_chunks, "(File)")

                if not plt.fignum_exists(fig.number):
                    print("Figure closed by user.")
                    break

    except Exception as e:
        print("Error:", e)

elif mode == 'live':
    wavelength = wavelength_live
    samples_per_chunk = chunk_size * 2
    bytes_per_chunk = samples_per_chunk * 8

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt(zmq.SUBSCRIBE, b"")

    buffer = bytearray()
    chunk_idx = 0

    try:
        while True:
            msg = socket.recv()
            buffer.extend(msg)

            while len(buffer) >= bytes_per_chunk:
                chunk_bytes = buffer[:bytes_per_chunk]
                buffer = buffer[bytes_per_chunk:]

                raw = np.frombuffer(chunk_bytes, dtype=np.complex64)
                h_phase, h_argument, h_aoa = process_samples(raw, wavelength)

                if chunk_idx % plot_interval == 0:
                    plot_histograms(h_phase, h_argument, h_aoa, chunk_idx, title_suffix="(Live)")

                chunk_idx += 1

                if not plt.fignum_exists(fig.number):
                    print("Figure closed by user.")
                    break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        socket.close()
        context.term()

else:
    print(f"Unknown mode '{mode}'. Please select 'file' or 'live'.")

plt.ioff()
plt.show()
