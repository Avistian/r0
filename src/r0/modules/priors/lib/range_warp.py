"""Log-domain Kumaraswamy range warp with an explicit source domain restriction."""

import math

import torch


def _log_difference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    difference = left - right
    result = difference.log()
    overflow = ~torch.isfinite(difference)
    if torch.any(overflow):
        result = torch.where(overflow, (left / 2 - right / 2).log() + math.log(2), result)
    return result


def _cdf(log_ratio: torch.Tensor, a: float, b: float) -> torch.Tensor:
    # log_ratio=log((1-u)/u). Compute log(-a*log(u)) without rounding u to
    # zero/one or losing a small positive product before a later exponent.
    log_softplus = torch.logaddexp(torch.zeros_like(log_ratio), log_ratio).log()
    log_softplus = torch.where(log_ratio < -36, log_ratio, log_softplus)
    log_t = math.log(a) + log_softplus
    t = log_t.exp()
    log_q = torch.empty_like(t)  # log(1-exp(-t))
    small = t <= math.log(2)
    log_q[small] = (-torch.expm1(-t[small])).log()
    log_q[~small] = torch.log1p(-torch.exp(-t[~small]))
    # Below this scale the correction to log(t) is below binary64 precision;
    # retaining log_t also avoids quantizing a positive subnormal t.
    log_q = torch.where(t < 1e-16, log_t, log_q)
    transformed = -torch.expm1(b * log_q)
    far = t > 36
    if torch.any(far):
        # log(1-exp(-t))=-exp(-t) to binary64 relative precision here.
        # Combine log(b)-t before exponentiation to preserve rescued tails.
        transformed[far] = -torch.expm1(-torch.exp(math.log(b) - t[far]))
    return transformed


def domain_masks(
    value: torch.Tensor, minimum: tuple[float, ...], maximum: tuple[float, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    low, high = value.new_tensor(minimum), value.new_tensor(maximum)
    return value < low, (value > high) & ((value - high) > 1e-30)


def kumaraswamy_warp(
    value: torch.Tensor,
    minimum: tuple[float, ...],
    maximum: tuple[float, ...],
    a: float,
    b: float,
) -> torch.Tensor:
    """Evaluate source min-max+epsilon and CDF without materializing fragile u.

    Binary64 arithmetic is stabilized across stages: the mathematical epsilon
    remains effective even when adding it directly to the range would round away.
    """
    low, high = value.new_tensor(minimum), value.new_tensor(maximum)
    lower = value <= low
    upper = (value > high) & ((value - high) >= 1e-30)
    inside = ~lower & ~upper
    result = torch.where(upper, torch.ones_like(value), torch.zeros_like(value))
    if not torch.any(inside):
        return result
    x = value[inside]
    lo, hi = low.expand_as(value)[inside], high.expand_as(value)[inside]
    log_num = _log_difference(x, lo)
    nonnegative = hi >= x
    log_complement = torch.empty_like(x)
    log_complement[nonnegative] = torch.logaddexp(
        _log_difference(hi[nonnegative], x[nonnegative]),
        x.new_full((int(nonnegative.sum()),), math.log(1e-30)),
    )
    log_complement[~nonnegative] = (1e-30 - (x[~nonnegative] - hi[~nonnegative])).log()
    result[inside] = _cdf(log_complement - log_num, a, b)
    return result
