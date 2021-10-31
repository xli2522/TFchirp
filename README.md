[![PyPI version](https://badge.fury.io/py/TFchirp.svg)](https://badge.fury.io/py/TFchirp)

## Time Frequency Transform for Chirp Signals

Step 1: Quadratic chirp signal

```Python
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Generate a quadratic chirp signal
dt = 0.0001
rate = int(1/dt)
ts = np.linspace(0, 5, int(1/dt))
data = scipy.signal.chirp(ts, 10, 5, 300, method='quadratic')
```

Step 2: S Transform Spectrogram

```Python
import TFchirp

# Compute S Transform Spectrogram
spectrogram = TFchirp.sTransform(data, sample_rate=rate)
plt.imshow(abs(spectrogram), origin='lower', aspect='auto')
plt.title('Original Spectrogram')
plt.show()
```

![Original Spectrogram](https://raw.githubusercontent.com/xli2522/TFchirp/blob/master/img/original_spectrogram.png)

Step 3: Quick Inverse S Transform

```Python
# Quick Inverse S Transform
inverse_ts = TFchirp.inverse_S(spectrogram)
plt.plot(inverse_ts-data)
plt.title('Time Series Reconstruction Error')
plt.show()
```

![Reconstruction Error](https://raw.githubusercontent.com/xli2522/TFchirp/blob/master/img/reconstruction_error.png)

Recovered spectrogram:

```Python
# Compute S Transform Spectrogram on the recovered time series
inverseSpectrogram = TFchirp.sTransform(inverse_ts, sample_rate=rate)
plt.imshow(abs(inverseSpectrogram), origin='lower', aspect='auto')
plt.title('Recovered Specctrogram')
plt.show()
```

![Recovered](https://raw.githubusercontent.com/xli2522/TFchirp/blob/master/img/recovered_spectrogram.png)

