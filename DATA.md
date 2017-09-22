# Data description

Each CSV file in the `train` directory contains a single 5-second recording.
Each recording has 20 time intervals (duration is 0.25 seconds) and 6 one-hot-encoded activity labels at the end.
Time interval = `10 (DFT bins) * 2 (magnitude and phase) * 3 (X, Y and Z axes) * 2 (accelerometer and gyroscope)`.

```text
acc_?, gyro_?
```
