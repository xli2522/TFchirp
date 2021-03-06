[![PyPI version](https://badge.fury.io/py/TFchirp.svg)](https://badge.fury.io/py/TFchirp) [![Downloads](https://pepy.tech/badge/tfchirp)](https://pepy.tech/project/tfchirp) [![Downloads](https://pepy.tech/badge/tfchirp/month)](https://pepy.tech/project/tfchirp)

## Time Frequency Transform for Chirp Signals

Step 1: Quadratic chirp signal

Generate a quadratic chirp signal from 10 Hz to 120 Hz in 1 second with 10,000 sampling points.

```Python
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Generate a quadratic chirp signal
dt = 0.0001
rate = int(1/dt)
ts = np.linspace(0, 1, int(1/dt))
data = scipy.signal.chirp(ts, 10, 1, 120, method='quadratic')
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

![Original Spectrogram](https://github.com/xli2522/S-Transform/blob/main/img/original_spectrogram.png?raw=true)

Step 3: Quick recovery of full ts from S transform * 0 frequency row*

(This recovered ts is computed based on the fact that the 0 frequency row always contain the full FFT result of the ts in this program by design.)

```Python
# Quick Recovery of ts from S Transform 0 frequency row
recovered_ts = TFchirp.recoverS(spectrogram)
plt.plot(recovered_ts-data)
plt.title('Time Series Reconstruction Error')
plt.show()
```

![Reconstruction Error](https://github.com/xli2522/S-Transform/blob/main/img/reconstruction_error.png?raw=true)

Step 4: Recovered spectrogram:

```Python
# Compute S Transform Spectrogram on the recovered time series
recoveredSpectrogram = TFchirp.sTransform(recovered_ts, sample_rate=rate, frange=[0,500])
plt.imshow(abs(recoveredSpectrogram), origin='lower', aspect='auto')
plt.title('Recovered Specctrogram')
plt.show()
```

![Recovered](https://github.com/xli2522/S-Transform/blob/main/img/recovered_spectrogram.png?raw=true)

Step 5: The real inverse S transform

```python
# Quick Inverse of ts from S Transform
inverse_ts, inverse_tsFFT = TFchirp.inverseS(spectrogram)
plt.plot(inverse_ts)
plt.plot(inverse_ts-data)
plt.title('Time Series Reconstruction Error')
plt.legend(['Recovered ts', 'Error'])
plt.show()
```

![Recovered ts and Error](https://github.com/xli2522/S-Transform/blob/main/img/recovered_ts_error.png?raw=true)

Step 6: Recovered spectrogram on the *real* inverse S transform ts

```python
# Compute S Transform Spectrogram on the recovered time series
inverseSpectrogram = TFchirp.sTransform(inverse_ts, sample_rate=rate, frange=[0,500])
plt.imshow(abs(inverseSpectrogram), origin='lower', aspect='auto')
plt.title('Recovered Specctrogram')
plt.show()
```

![Recovered Spectrogram](https://github.com/xli2522/S-Transform/blob/main/img/real_recovered_spectrogram.png?raw=true)

