"""Fail-fast numerical diagnostics for long DiffusionDrive training runs."""

import os
from typing import Any, Iterable, List, Tuple

import torch


def finite_trace_enabled() -> bool:
    return os.environ.get("DIFFUSIONDRIVE_FINITE_TRACE", "0").lower() in {"1", "true", "yes", "on"}


def _tensor_summary(tensor: torch.Tensor) -> str:
    value = tensor.detach()
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    total_count = value.numel()
    if finite_count:
        finite_values = value[finite].float()
        minimum = float(finite_values.min().item())
        maximum = float(finite_values.max().item())
        abs_max = float(finite_values.abs().max().item())
    else:
        minimum = float("nan")
        maximum = float("nan")
        abs_max = float("nan")

    if value.ndim == 0:
        invalid_batches: List[int] = [0] if finite_count == 0 else []
    elif value.shape[0] == 0:
        invalid_batches = []
    else:
        per_batch = finite.reshape(value.shape[0], -1).all(dim=1)
        invalid_batches = torch.nonzero(~per_batch, as_tuple=False).flatten().cpu().tolist()

    return (
        f"shape={tuple(value.shape)} dtype={value.dtype} finite={finite_count}/{total_count} "
        f"min={minimum:.7g} max={maximum:.7g} abs_max={abs_max:.7g} "
        f"invalid_batches={invalid_batches}"
    )


def assert_finite(stage: str, *values: Any) -> None:
    """Raise at the first non-finite tensor and print a compact numeric summary."""

    if not finite_trace_enabled():
        return
    for index, value in enumerate(values):
        if not torch.is_tensor(value) or not (value.is_floating_point() or value.is_complex()):
            continue
        if not bool(torch.isfinite(value.detach()).all().item()):
            print(
                f"[DiffusionDrive][finite_trace][FORWARD] stage={stage}[{index}] "
                + _tensor_summary(value),
                flush=True,
            )
            raise FloatingPointError(f"Non-finite tensor first observed at {stage}[{index}]")


def iter_tensor_leaves(prefix: str, value: Any) -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_tensor_leaves(f"{prefix}.{key}", item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_tensor_leaves(f"{prefix}[{index}]", item)


def assert_tree_finite(stage: str, value: Any) -> None:
    if not finite_trace_enabled():
        return
    for name, tensor in iter_tensor_leaves(stage, value):
        assert_finite(name, tensor)


def find_nonfinite_named(named_tensors: Iterable[Tuple[str, torch.Tensor]]) -> List[str]:
    """Return summaries for non-finite parameters or gradients."""

    failures: List[str] = []
    for name, tensor in named_tensors:
        if tensor is None or not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            failures.append(f"{name}: {_tensor_summary(tensor)}")
    return failures


def finite_gradient_norm(named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    """Compute a diagnostic global norm without modifying gradients."""

    norms = []
    for _, parameter in named_parameters:
        if parameter.grad is not None:
            norms.append(torch.linalg.vector_norm(parameter.grad.detach().float()))
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms))
