import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# ==== CONFIGURATION ====
smooth_plot = 1          # 1: smooth, 0: histogram
start_percent = 0        # Start from N% of file
plot_interval = 1        # Update plot every N chunks
chunk_size = 300_000     # Larger = fewer reads, faster
# =======================

# Constants
d = 0.10                 # Distance between antennas (meters)
c = 3e8                  # Speed of light (m/s)
f = 1000e6               # Signal frequency (Hz)
wavelength = c / f       # Signal wavelength

filename = "/home/ubuntu/Desktop/aoa_estimation/aoa_log.dat"
sample_bytes = 8         # 8 bytes = 2x complex64 (each 4 bytes)

# File info
file_size = os.path.getsize(filename)
total_samples = file_size // sample_bytes
total_chunks = total_samples // (chunk_size * 2)

start_sample = int(total_samples * start_percent / 100)
start_sample -= start_sample % (chunk_size * 2)
remaining_samples = total_samples - start_sample
remaining_chunks = remaining_samples // (chunk_size * 2)

print(f"Starting at {start_percent}% ({start_sample} samples)")
print(f"Remaining chunks: {remaining_chunks}")

# Histogram bins for plotting
bins_phase = np.linspace(-180, 180, 100)
bins_argument = np.linspace(-1, 1, 100)
bins_aoa = np.linspace(-90, 90, 100)
centers_phase = (bins_phase[:-1] + bins_phase[1:]) / 2
centers_argument = (bins_argument[:-1] + bins_argument[1:]) / 2
centers_aoa = (bins_aoa[:-1] + bins_aoa[1:]) / 2

# Prepare plot
plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

try:
    with open(filename, "rb") as f:
        f.seek(start_sample * sample_bytes)

        for chunk_idx in tqdm(range(remaining_chunks), desc="Chunks", unit="chunk"):
            # Read interleaved complex samples: rx1_0, rx2_0, rx1_1, rx2_1, ...
            raw = np.fromfile(f, dtype=np.complex64, count=chunk_size * 2)
            if len(raw) < chunk_size * 2:
                break

            # Separate interleaved channels
            rx1 = raw[0::2]  # Take every 2nd sample starting from 0 -> channel 1
            rx2 = raw[1::2]  # Take every 2nd sample starting from 1 -> channel 2

            # Compute complex cross-product to get relative phase between antennas
            # np.conj(rx2): complex conjugate flips imaginary part -> (a + bj) -> (a - bj)
            # This isolates the phase difference: angle(rx1 * conj(rx2)) = angle(rx1) - angle(rx2)
            cross = rx1 * np.conj(rx2)

            # Extract phase difference in radians and convert to degrees
            phase_diff = np.angle(cross)               # Range: [-π, π]
            phase_diff_deg = np.degrees(phase_diff)    # Convert to degrees for plotting

            # Normalize phase difference to arcsin argument range
            # wavelength / (2πd) scales to sin(θ), clip ensures arcsin won't error
            argument = np.clip(phase_diff * wavelength / (2 * np.pi * d), -1, 1)

            # Estimate Angle of Arrival (AoA) in degrees
            aoa_deg = np.degrees(np.arcsin(argument))

            # Histograms for visualization
            h_phase, _ = np.histogram(phase_diff_deg, bins=bins_phase)
            h_argument, _ = np.histogram(argument, bins=bins_argument)
            h_aoa, _ = np.histogram(aoa_deg, bins=bins_aoa)

            # Optional smoothing for nicer plots
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

                # Argument Histogram (used in arcsin step)
                max_mid_arg = max(h_argument[1:-1]) if len(h_argument) > 2 else 1
                if smooth_plot:
                    axes[1].plot(centers_argument, h_argument, color='orange')
                else:
                    axes[1].bar(bins_argument[:-1], h_argument, width=np.diff(bins_argument)[0], color='orange')
                axes[1].set_title("Arcsin Argument")
                axes[1].set_xlim(-1, 1)
                axes[1].set_ylim(0, max_mid_arg * 1.1)
                axes[1].grid(True)

                # Estimated AoA Histogram
                max_mid_aoa = max(h_aoa[1:-1]) if len(h_aoa) > 2 else 1
                if smooth_plot:
                    axes[2].plot(centers_aoa, h_aoa, color='blue')
                else:
                    axes[2].bar(bins_aoa[:-1], h_aoa, width=np.diff(bins_aoa)[0], color='blue')
                axes[2].set_title("Estimated AoA")
                axes[2].set_xlim(-90, 90)
                axes[2].set_ylim(0, max_mid_aoa * 1.1)
                axes[2].grid(True)

                # Update plot
                plt.suptitle(f"Chunk {chunk_idx+1}/{remaining_chunks} (Start at {start_percent}%)")
                plt.pause(0.001)

                # Exit if plot is closed
                if not plt.fignum_exists(fig.number):
                    print("Figure closed by user.")
                    break

except Exception as e:
    print("Error:", e)

# Finalize plot
plt.ioff()
plt.show()
