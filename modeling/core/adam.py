"""
Generic Adam optimiser that works with an arbitrary collection of named
parameter arrays and a user-supplied loss+gradient function.

Usage
-----
>>> params = {"u": u_array, "v": v_array}
>>> result = adam_optimize(params, loss_and_grad_fn, steps=2000, lr=0.03)
>>> final_params, final_loss, steps_used = result
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# The callable must return (loss: float, grads: dict[str, ndarray])
LossGradFn = Callable[[Dict[str, np.ndarray]], Tuple[float, Dict[str, np.ndarray]]]


def adam_optimize(
    params: Dict[str, np.ndarray],
    loss_and_grad_fn: LossGradFn,
    steps: int,
    lr: float,
    tol: float = 1e-8,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    frozen: Optional[set[str]] = None,
    desc: str = "Adam",
    show_progress: bool = True,
) -> Tuple[Dict[str, np.ndarray], float, int]:
    """Run Adam on *params* in-place and return ``(params, final_loss, steps_used)``."""
    frozen = frozen or set()

    # Moments
    m: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}
    v: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}

    prev_loss: Optional[float] = None
    last_loss = float("inf")
    total_steps = max(1, int(steps))

    use_tqdm = _HAS_TQDM and show_progress
    iterator = tqdm(range(1, total_steps + 1), desc=desc, unit="step", dynamic_ncols=True) if use_tqdm else range(1, total_steps + 1)

    for step in iterator:
        loss, grads = loss_and_grad_fn(params)
        last_loss = loss

        if use_tqdm and (step == 1 or step % 10 == 0 or step == total_steps):
            iterator.set_postfix(loss=f"{loss:.6f}")

        if prev_loss is not None and abs(prev_loss - loss) < tol:
            return params, last_loss, step
        prev_loss = loss

        for k in params:
            g = grads[k]
            if k in frozen:
                g = np.zeros_like(g)

            m[k] = beta1 * m[k] + (1.0 - beta1) * g
            v[k] = beta2 * v[k] + (1.0 - beta2) * (g * g)

            m_hat = m[k] / (1.0 - beta1 ** step)
            v_hat = v[k] / (1.0 - beta2 ** step)

            params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return params, last_loss, total_steps
