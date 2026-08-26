"""Offline adaptive anchor-bank construction for DiffusionDrive."""

from navsim.agents.diffusiondrive.anchors.base_expander import (
    AnchorBuilderConfig,
    BaseExpansionResult,
    expand_base_anchors,
)
from navsim.agents.diffusiondrive.anchors.composer import CompositionResult, compose_anchor_bank
from navsim.agents.diffusiondrive.anchors.command_anchor_bank import (
    CommandAnchorBankConfig,
    CommandAnchorBankResult,
    InitialCommandAnchorBankConfig,
    InitialCommandAnchorBankResult,
    build_command_anchor_bank,
    build_initial_command_anchor_bank,
    command_conditioned_assignment,
)
from navsim.agents.diffusiondrive.anchors.anchor_dedup import (
    AnchorDedupResult,
    deduplicate_command_anchor_bank,
)
from navsim.agents.diffusiondrive.anchors.anchor_metrics import (
    command_anchor_coverage_metrics,
    command_anchor_diversity_metrics,
)
from navsim.agents.diffusiondrive.anchors.residual_codebook import (
    ResidualCodebookResult,
    learn_residual_codebooks,
)

__all__ = [
    "AnchorBuilderConfig",
    "AnchorDedupResult",
    "BaseExpansionResult",
    "CompositionResult",
    "CommandAnchorBankConfig",
    "CommandAnchorBankResult",
    "InitialCommandAnchorBankConfig",
    "InitialCommandAnchorBankResult",
    "ResidualCodebookResult",
    "compose_anchor_bank",
    "build_command_anchor_bank",
    "build_initial_command_anchor_bank",
    "command_conditioned_assignment",
    "command_anchor_coverage_metrics",
    "command_anchor_diversity_metrics",
    "deduplicate_command_anchor_bank",
    "expand_base_anchors",
    "learn_residual_codebooks",
]
