# AoA Estimation with bladeRF 2.0 Micro 2RX Channel and HackRF Signal Source

This project implements **Angle of Arrival (AoA) estimation** using two receiving channels of a bladeRF 2.0 Micro SDR device. The signal source is generated by a HackRF device, and the processing and visualization are handled in Python.

---

## Overview

- **Hardware Setup:**
  - **Receiver:** bladeRF 2.0 Micro (2RX channels)
  - **Transmitter:** HackRF SDR
- **Signal Frequency:** ~800 MHz
- **Purpose:** Estimate the AoA based on phase difference between two antennas spaced 10 cm apart.
- **Data Handling:**
  - Real-time data streaming and plotting via ZeroMQ (ZMQ) sockets from GNU Radio.
  - Offline analysis from recorded `.dat` files.
- **Python Scripts:**
  - `live_zmq_plot.py`: Subscribes to ZMQ data feed and plots live histograms of phase difference, arcsin argument, and AoA estimate.
  - (Another script) to plot AoA from recorded `.dat` files.

---

## Theory

The Angle of Arrival (AoA) estimation is based on measuring the phase difference of a received radio frequency (RF) signal at two spatially separated antennas. When a plane wave arrives from a direction $\theta$ (relative to the antenna array normal), the path difference between the two antennas causes a phase shift in the received signals.

### Signal Model

Assuming two antennas spaced by distance $d$, the time difference of arrival $\Delta t$ between antennas is:

$\Delta t = \frac{d \sin \theta}{c}$

where:
- $c$ is the speed of light (approximately 3 × 10^8 m/s),
- $\theta$ is the AoA to be estimated.

The phase difference $\Delta \phi$ between the signals at the two antennas, assuming a carrier frequency $f$, is related to $\Delta t$ as:

$\Delta \phi = 2 \pi f \Delta t = \frac{2 \pi d \sin \theta}{\lambda}$

where:
- $\lambda = \frac{c}{f}$ is the signal wavelength.

Rearranging to solve for $\theta$, we get the AoA estimate:

$\theta = \arcsin \left( \frac{\Delta \phi \cdot \lambda}{2 \pi d} \right)$

### Practical Considerations

- **Ambiguity:** The maximum unambiguous AoA range is limited by the antenna spacing $d$ relative to the wavelength $\lambda$. To avoid spatial aliasing, $d$ is typically less than or equal to $\frac{\lambda}{2}$. In this project, the spacing is 0.10 m, suitable for 800 MHz signals.
- **Noise and Multipath:** Real signals are corrupted by noise and multipath reflections. To improve robustness, this implementation uses histograms and smoothing of phase difference estimates over many samples.
- **Complex IQ Samples:** The system uses complex baseband IQ samples captured simultaneously on two receiver channels (bladeRF 2.0 Micro 2RX). The phase difference is calculated via cross-correlation of these channels.

---

This theoretical framework enables estimation of the direction from which the RF signal arrives using just two antennas and phase measurements.


## Example Plots

![Image](https://github.com/user-attachments/assets/fd282ac3-22f1-44dd-8fe6-54b0a07b7e29)

![Image](https://github.com/user-attachments/assets/a1754ede-5587-4edc-9164-fd345fc38b9d)

![Image](https://github.com/user-attachments/assets/39d65447-a955-458d-b119-f349925caf8b)

![Image](https://github.com/user-attachments/assets/2e9c227f-f705-4f8d-91b7-59e650093004)

