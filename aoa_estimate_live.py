import zmq
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==== CONFIGURATION ====
smooth_plot = 1          # 1: smooth, 0: histogram
plot_interval = 1        # Update plot every N chunks
chunk_size = 300_000     # Samples per channel (two channels interleaved)
# =======================

# Constants
d = 0.10                 # Distance between antennas (meters)
c = 3e8                  # Speed of light (m/s)
f = 800e6               # Signal frequency (Hz)
wavelength = c / f       # Signal wavelength

samples_per_chunk = chunk_size * 2   # total samples (2 channels interleaved)
bytes_per_chunk = samples_per_chunk * 8  # 8 bytes per complex64

# Histogram bins for plotting
bins_phase = np.linspace(-180, 180, 100)
bins_argument = np.linspace(-1, 1, 100)
bins_aoa = np.linspace(-90, 90, 100)
centers_phase = (bins_phase[:-1] + bins_phase[1:]) / 2
centers_argument = (bins_argument[:-1] + bins_argument[1:]) / 2
centers_aoa = (bins_aoa[:-1] + bins_aoa[1:]) / 2

# Prepare live plot
plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# ZeroMQ subscriber setup
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")  # Update to your GNURadio zmq PUB address
socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all topics

buffer = bytearray()
chunk_idx = 0

try:
    while True:
        msg = socket.recv()
        buffer.extend(msg)

        # Process all full chunks currently in buffer
        while len(buffer) >= bytes_per_chunk:
            chunk_bytes = buffer[:bytes_per_chunk]
            buffer = buffer[bytes_per_chunk:]

            raw = np.frombuffer(chunk_bytes, dtype=np.complex64)

            # Separate interleaved channels
            rx1 = raw[0::2]
            rx2 = raw[1::2]

            # Cross correlation to find phase difference
            cross = rx1 * np.conj(rx2)
            phase_diff = np.angle(cross)
            phase_diff_deg = np.degrees(phase_diff)

            # Calculate arcsin argument and clip to [-1,1]
            argument = np.clip(phase_diff * wavelength / (2 * np.pi * d), -1, 1)
            aoa_deg = np.degrees(np.arcsin(argument))

            # Histograms
            h_phase, _ = np.histogram(phase_diff_deg, bins=bins_phase)
            h_argument, _ = np.histogram(argument, bins=bins_argument)
            h_aoa, _ = np.histogram(aoa_deg, bins=bins_aoa)

            if smooth_plot:
                h_phase = gaussian_filter1d(h_phase, sigma=2)
                h_argument = gaussian_filter1d(h_argument, sigma=2)
                h_aoa = gaussian_filter1d(h_aoa, sigma=2)

            # Plot every N chunks
            if chunk_idx % plot_interval == 0:
                axes[0].cla()
                axes[1].cla()
                axes[2].cla()

                # Phase Difference Histogram
                if smooth_plot:
                    axes[0].plot(centers_phase, h_phase, color='gray')
                else:
                    axes[0].bar(bins_phase[:-1], h_phase, width=np.diff(bins_phase)[0], color='gray')
                axes[0].set_title("Phase Diff (deg)")
                axes[0].set_xlim(-180, 180)
                axes[0].grid(True)

                # Argument Histogram
                max_mid_arg = max(h_argument[1:-1]) if len(h_argument) > 2 else 1
                if smooth_plot:
                    axes[1].plot(centers_argument, h_argument, color='orange')
                else:
                    axes[1].bar(bins_argument[:-1], h_argument, width=np.diff(bins_argument)[0], color='orange')
                axes[1].set_title("Arcsin Argument")
                axes[1].set_xlim(-1, 1)
                axes[1].set_ylim(0, max_mid_arg * 1.1)
                axes[1].grid(True)

                # AoA Histogram
                max_mid_aoa = max(h_aoa[1:-1]) if len(h_aoa) > 2 else 1
                if smooth_plot:
                    axes[2].plot(centers_aoa, h_aoa, color='blue')
                else:
                    axes[2].bar(bins_aoa[:-1], h_aoa, width=np.diff(bins_aoa)[0], color='blue')
                axes[2].set_title("Estimated AoA")
                axes[2].set_xlim(-90, 90)
                axes[2].set_ylim(0, max_mid_aoa * 1.1)
                axes[2].grid(True)

                plt.suptitle(f"Chunk {chunk_idx + 1} (Live Data)")
                plt.pause(0.001)

            chunk_idx += 1

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    plt.ioff()
    plt.show()
    socket.close()
    context.term()
