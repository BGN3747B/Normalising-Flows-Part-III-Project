import torch
from nflows.distributions.base import Distribution
from nflows.distributions.uniform import BoxUniform
import numpy as np

#None of this works
class NflowsBoxUniformBase(Distribution):
    def __init__(self, low, high):
        super().__init__()

        low = torch.as_tensor(low, dtype=torch.float32)
        high = torch.as_tensor(high, dtype=torch.float32)

        if low.shape != high.shape:
            raise ValueError("low and high must have the same shape.")

        self._shape = torch.Size(low.shape)

        self.register_buffer("low", low)
        self.register_buffer("high", high)

        log_volume = torch.sum(torch.log(high - low))
        self.register_buffer("log_volume", log_volume)

    def _log_prob(self, inputs, context=None):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                f"Expected input shape [batch, {self._shape}], got {inputs.shape}"
            )

        inside = ((inputs >= self.low) & (inputs <= self.high)).all(dim=1)

        logp_inside = -self.log_volume.expand(inputs.shape[0])
        logp_outside = torch.full_like(logp_inside, -torch.inf)

        return torch.where(inside, logp_inside, logp_outside)

    def _sample(self, num_samples, context=None):
        if context is not None:
            raise NotImplementedError("Context not implemented for this uniform base.")

        u = torch.rand(
            num_samples,
            *self._shape,
            device=self.low.device,
            dtype=self.low.dtype
        )

        return self.low + (self.high - self.low) * u

    def _mean(self, context=None):
        mean = 0.5 * (self.low + self.high)

        if context is not None:
            return mean.expand(context.shape[0], *self._shape)

        return mean

def bounded_to_unit_interval(X, bounds):
    X = np.asarray(X, dtype=np.float32)
    X_scaled = np.zeros_like(X)

    for k in range(X.shape[1]):
        low = bounds[2*k]
        high = bounds[2*k + 1]
        X_scaled[:, k] = 2.0 * (X[:, k] - low) / (high - low) - 1.0

    return X_scaled


def unit_interval_to_bounded(X_scaled, bounds):
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    X = np.zeros_like(X_scaled)

    for k in range(X_scaled.shape[1]):
        low = bounds[2*k]
        high = bounds[2*k + 1]
        X[:, k] = 0.5 * (X_scaled[:, k] + 1.0) * (high - low) + low

    return X