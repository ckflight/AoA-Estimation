import numpy as np
import matplotlib.pyplot as plt
import zmq
import struct
from scipy.ndimage import gaussian_filter1d

# ==== CONFIGURATION ====
mode = 'realtime'           # Operating mode: 'realtime' receives live data from sockets.
                            # 'file' mode could be added for reading from files.
smooth_plot = 1             # If 1, apply smoothing filter to histograms for nicer visualization.
plot_interval = 1           # Update the plot every N chunks of received data.
chunk_size = 500_000        # Number of complex samples per channel per chunk.
initial_freq = 800e6        # Initial center frequency in Hz (e.g., 800 MHz).
# =======================

# CONSTANTS
d = 0.10                   # Distance between antennas in meters.
c = 3e8                    # Speed of light in meters per second.

# Starting frequency and wavelength
current_freq = initial_freq
current_wavelength = c / current_freq   # wavelength = speed_of_light / frequency

# Define histogram bins for:
# Phase difference (degrees), arcsin argument (dimensionless), and angle of arrival (degrees)
bins_phase = np.linspace(-180, 180, 100)
bins_argument = np.linspace(-1, 1, 100)
bins_aoa = np.linspace(-90, 90, 100)

# Calculate bin centers for plotting purposes
centers_phase = (bins_phase[:-1] + bins_phase[1:]) / 2
centers_argument = (bins_argument[:-1] + bins_argument[1:]) / 2
centers_aoa = (bins_aoa[:-1] + bins_aoa[1:]) / 2

# Turn interactive mode on for matplotlib (enables dynamic updating)
plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(14, 5))  # Create figure with 3 subplots side by side

def plot_histograms(h_phase, h_argument, h_aoa, chunk_idx, title_suffix=""):
    # Clear previous plot data
    axes[0].cla()
    axes[1].cla()
    axes[2].cla()

    # --- Phase difference plot ---
    if smooth_plot:
        axes[0].plot(centers_phase, h_phase, color='gray')
    else:
        axes[0].bar(bins_phase[:-1], h_phase, width=np.diff(bins_phase)[0], color='gray')
    axes[0].set_title("Phase Diff (deg)")
    axes[0].set_xlim(-180, 180)
    axes[0].grid(True)

    # Vertical lines every 60 degrees on phase plot
    phase_lines = np.arange(-180, 181, 60)
    for x in phase_lines:
        axes[0].axvline(x=x, color='lightgray', linestyle='--', linewidth=0.8)
        # Add text labels just above the top of y-axis ticks
        ymax = axes[0].get_ylim()[1]
        axes[0].text(x, ymax * 0.95, f"{x}", color='gray', fontsize=8, ha='center', va='top')

    # --- Arcsin argument plot ---
    max_mid_arg = max(h_argument[1:-1]) if len(h_argument) > 2 else 1
    if smooth_plot:
        axes[1].plot(centers_argument, h_argument, color='orange')
    else:
        axes[1].bar(bins_argument[:-1], h_argument, width=np.diff(bins_argument)[0], color='orange')
    axes[1].set_title("Arcsin Argument")
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(0, max_mid_arg * 1.1)
    axes[1].grid(True)

    # No vertical lines or labels here to keep clean (optional: can add few if you want)

    # --- Angle of Arrival plot ---
    max_mid_aoa = max(h_aoa[1:-1]) if len(h_aoa) > 2 else 1
    if smooth_plot:
        axes[2].plot(centers_aoa, h_aoa, color='blue')
    else:
        axes[2].bar(bins_aoa[:-1], h_aoa, width=np.diff(bins_aoa)[0], color='blue')
    axes[2].set_title("Estimated AoA")
    axes[2].set_xlim(-90, 90)
    axes[2].set_ylim(0, max_mid_aoa * 1.1)
    axes[2].grid(True)

    # Vertical lines every 30 degrees on AoA plot
    aoa_lines = np.arange(-90, 91, 30)
    for x in aoa_lines:
        axes[2].axvline(x=x, color='lightgray', linestyle='--', linewidth=0.8)
        ymax = axes[2].get_ylim()[1]
        axes[2].text(x, ymax * 0.95, f"{x}", color='gray', fontsize=8, ha='center', va='top')

    # Main title
    freq_mhz = current_freq / 1e6
    plt.suptitle(f"Chunk {chunk_idx+1} {title_suffix} — Freq: {freq_mhz:.3f} MHz")
    plt.pause(0.001)



def process_samples(raw, wavelength):

    # Separate samples for the two antennas (assumed interleaved)
    rx1 = raw[0::2]  # Even samples = antenna 1
    rx2 = raw[1::2]  # Odd samples = antenna 2

    # Cross correlation: element-wise multiplication of rx1 by conjugate of rx2
    cross = rx1 * np.conj(rx2)

    # Extract phase difference (angle) in radians and convert to degrees
    phase_diff = np.angle(cross)
    phase_diff_deg = np.degrees(phase_diff)

    # Calculate arcsin argument used for AoA estimation:
    # argument = phase_diff * wavelength / (2π * antenna_spacing)
    # Clip values to [-1,1] for valid arcsin input
    argument = np.clip(phase_diff * wavelength / (2 * np.pi * d), -1, 1)

    # Calculate AoA in degrees
    aoa_deg = np.degrees(np.arcsin(argument))

    # Compute histograms for visualization
    h_phase, _ = np.histogram(phase_diff_deg, bins=bins_phase)
    h_argument, _ = np.histogram(argument, bins=bins_argument)
    h_aoa, _ = np.histogram(aoa_deg, bins=bins_aoa)

    # Optionally smooth histograms using Gaussian filter for nicer plots
    if smooth_plot:
        h_phase = gaussian_filter1d(h_phase, sigma=2)
        h_argument = gaussian_filter1d(h_argument, sigma=2)
        h_aoa = gaussian_filter1d(h_aoa, sigma=2)

    return h_phase, h_argument, h_aoa

if mode == 'realtime':
    # Calculate number of samples and bytes per chunk
    # Each chunk has chunk_size complex samples per channel
    samples_per_chunk = chunk_size * 2  # 2 antennas interleaved = total samples
    bytes_per_chunk = samples_per_chunk * 8  # complex64 = 8 bytes per complex sample

    context = zmq.Context()

    # ZMQ subscriber socket to receive complex sample chunks (port 5555)
    socket_samples = context.socket(zmq.SUB)
    socket_samples.connect("tcp://localhost:5555")
    socket_samples.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages

    # ZMQ subscriber socket to receive frequency updates (port 5556)
    socket_freq = context.socket(zmq.SUB)
    socket_freq.connect("tcp://localhost:5556")
    socket_freq.setsockopt(zmq.SUBSCRIBE, b"")

    # Poller to check both sockets without blocking indefinitely
    poller = zmq.Poller()
    poller.register(socket_samples, zmq.POLLIN)
    poller.register(socket_freq, zmq.POLLIN)

    buffer = bytearray()  # Buffer to accumulate incoming sample bytes
    chunk_idx = 0         # Chunk counter

    try:
        while True:
            # Poll sockets with 100 ms timeout
            socks = dict(poller.poll(100))

            # Check if frequency update message is available
            if socket_freq in socks and socks[socket_freq] == zmq.POLLIN:
                freq_msg_raw = socket_freq.recv()

                # Example message bytes (hex):
                # 07 02 00 0A 66 72 65 71 5F 72 61 6E 67 65 04 41 B9 CF 0E 40 00 00 00
                #                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                #                      The ASCII string "freq_range" followed by:
                #                      0x04 - PMT type tag (not part of the frequency value)
                #                      41 B9 CF 0E 40 00 00 00 - 8 bytes representing a big-endian
                # - For instance, 0x41B9CF0E40000000 corresponds to 433,000,000 Hz (433 MHz) IEEE 754 double float.

                print("Raw freq bytes:", ' '.join(f'{b:02X}' for b in freq_msg_raw))

                try:
                    # Search for the byte pattern corresponding to key 'freq_range' in the message
                    freq_range_bytes = b'freq_range'
                    start_index = freq_msg_raw.find(freq_range_bytes)
                    if start_index == -1:
                        print("freq_range key not found in frequency message")
                        continue

                    # The frequency value is stored as an 8-byte double immediately after 'freq_range' key + 1 byte (PMT type)
                    double_start = start_index + len(freq_range_bytes) + 1
                    double_bytes = freq_msg_raw[double_start: double_start + 8]

                    # Unpack the 8 bytes as big-endian double precision float ('>d')
                    new_freq = struct.unpack('>d', double_bytes)[0]

                    # If frequency changed and positive, update current frequency and wavelength
                    if new_freq != current_freq and new_freq > 0:
                        current_freq = new_freq
                        current_wavelength = c / current_freq
                        print(f"Frequency updated: {current_freq / 1e6:.3f} MHz")

                except Exception as e:
                    print(f"Frequency decode error: {e}")

            # Check if sample data message is available
            if socket_samples in socks and socks[socket_samples] == zmq.POLLIN:
                msg = socket_samples.recv()
                buffer.extend(msg)  # Append received bytes to buffer

                # While buffer contains at least one full chunk, process it
                while len(buffer) >= bytes_per_chunk:
                    chunk_bytes = buffer[:bytes_per_chunk]  # Extract one chunk of bytes
                    buffer = buffer[bytes_per_chunk:]       # Remove processed chunk from buffer

                    # Convert bytes to numpy array of complex64 samples (interleaved)
                    raw = np.frombuffer(chunk_bytes, dtype=np.complex64)

                    # Process samples to get histograms
                    h_phase, h_argument, h_aoa = process_samples(raw, current_wavelength)

                    # Update plot at configured intervals
                    if chunk_idx % plot_interval == 0:
                        plot_histograms(h_phase, h_argument, h_aoa, chunk_idx, title_suffix="(Realtime)")

                    chunk_idx += 1

                    # Exit if user closed the figure window
                    if not plt.fignum_exists(fig.number):
                        print("Figure closed by user.")
                        raise KeyboardInterrupt()

    except KeyboardInterrupt:
        print("User interrupted")

    finally:
        # Clean up sockets and context
        socket_samples.close()
        socket_freq.close()
        context.term()
        print("Exited cleanly")
