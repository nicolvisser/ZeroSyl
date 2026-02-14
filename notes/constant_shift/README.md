# Constant shifts in the boundary evaluation

As in Sylber's evaluation, we allow for a constant time shift in the predicted boundaries of each system to account for potential latency in the representations;
these shifts are tuned using R-value on development data.

Based on the images in this directory we selected the shifts as:

| System             | Shift (ms) |
|--------------------|------------|
| SyllableLM 5.0 Hz  | -10        |
| SyllableLM 6.25 Hz | -15        |
| SyllableLM 8.33 Hz | -20        |
| Syber              | -35        |
| ZeroSyl            | -5         |