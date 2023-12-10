from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np

from nonmin_pose.constraints import constraints as cnt
from nonmin_pose.constraints.constraints import Parameter

ConstraintConfig = Union[Dict[str, Optional[List[int]]], Dict[str, None]]


class ConstraintManager:
    """Manager of the metadata of constraints (blocks, values, indexes, etc.)."""

    CONSTRAINT_CLASSES = {
        "manif_def_left": cnt.ManifDefLeft,
        "manif_def_right": cnt.ManifDefRight,
        "norm_t": cnt.NormT,
        "norm_q": cnt.NormQ,
        "e_def_left": cnt.EDefLeft,
        "e_def_right": cnt.EDefRight,
        "e_def_left_right": cnt.EDefLeftRight,
        "homogenization": cnt.Homogenization,
        "adjoint": cnt.Adjoint,
        "norm_e": cnt.NormE,
        "right_null_space": cnt.RightNullSpace,
        "left_null_space": cnt.LeftNullSpace,
        "cheirality_translation": cnt.CheiralityTranslation,
        "cheirality_translation_v2": cnt.CheiralityTranslationV2,
        "cheirality_rotation": cnt.CheiralityRotation,
        "cheirality_rotation_q": cnt.CheiralityRotationQ,
        "cheirality_midpoint": cnt.CheiralityMidpoint,
        "orthogonality": cnt.Orthogonality,
        "determinant_r": cnt.DeterminantR,
        "t_q_definition": cnt.TQDefinition,
        "skew_t_q_definition": cnt.SkewTQDefinition,
        "convex_hull_so3": cnt.ConvexHullSO3,
    }

    DYNAMIC_CONSTRAINTS = {
        "cheirality_translation",
        "cheirality_translation_v2",
        "cheirality_rotation",
        "cheirality_rotation_q",
        "cheirality_midpoint",
    }

    def __init__(
        self,
        parameters: List[Parameter],
        constraints: ConstraintConfig,
        use_top_k: Optional[int] = None,
    ) -> None:
        """Constraint Manager

        Args:
            parameters: List of parameters.
            constraints: Dictionary of constraints.
                Keys are the names of the constraints and values are lists of
                indexes of the equations to be dropped in each constraint (a constraint
                can be composed of multiple equations). If None, all equations are used.
            use_top_k: If not None, only the top_k correspondences with highest weights
                are used to compute the dynamic coefficients. If None, all corresponden-
                ces are used. Default: None.
        """
        if use_top_k is not None and use_top_k <= 0:
            raise ValueError(f"use_top_k must be positive. Got {use_top_k}")
        self.use_top_k = use_top_k

        # check parameters and get block indexes with their respective sizes.
        self.block_sizes = check_params_and_get_blocks(parameters)
        params = {p.name: p for p in parameters}

        # constraints with coefficients determined at runtime.
        self.dynamic_constraints = []

        idx_first_el, idx_first_eq = 0, 0
        idx_, coeffs_, values_, blocks_, rows_, cols_ = [], [], [], [], [], []

        for name, drop_eqs in constraints.items():
            if name not in self.CONSTRAINT_CLASSES:
                raise ValueError(f"Unknown constraint: {name}")

            class_ = self.CONSTRAINT_CLASSES[name]
            constraint = class_(params, idx_first_el, idx_first_eq, drop_eqs)

            if name in self.DYNAMIC_CONSTRAINTS:
                self.dynamic_constraints.append(constraint)

            idx_.extend(constraint.constraint_idx)
            coeffs_.extend(constraint.coeffs)
            values_.extend(constraint.values)
            blocks_.extend(constraint.blocks)
            rows_.extend(constraint.rows)
            cols_.extend(constraint.cols)

            idx_first_el = constraint.idx_last_el
            idx_first_eq += len(constraint.values)

        self.constraint_idx = np.array(idx_)
        self.coeffs = np.array(coeffs_)
        self.values = np.array(values_)
        self.blocks = np.array(blocks_)
        self.rows = np.array(rows_)
        self.cols = np.array(cols_)

        self.n_blocks = len(self.block_sizes)
        self.n_constraints = len(self.values)

        self.has_dynamic_constraints = len(self.dynamic_constraints) > 0

    def compute_dynamic_coeffs(
        self, f0: np.ndarray, f1: np.ndarray, inliers_conf: Optional[np.ndarray] = None
    ):
        if not self.has_dynamic_constraints:
            return

        if self.use_top_k is not None:
            assert inliers_conf is not None, "weights must be provided when using top_k"
            if self.use_top_k < len(inliers_conf):
                idx = np.argpartition(inliers_conf, -self.use_top_k)[-self.use_top_k :]
                f0, f1 = f0[:, idx], f1[:, idx]

        for constraint in self.dynamic_constraints:
            constraint.compute_coeffs(self.coeffs, f0, f1)

    @property
    def available_constraints(self) -> List[str]:
        return list(self.CONSTRAINT_CLASSES)


def check_params_and_get_blocks(params: List[Parameter]) -> Dict[int, int]:
    checked_blocks = defaultdict(list)
    for p in params:
        checked_blocks[p.block].extend(p.block_ids)

    for block, ids in checked_blocks.items():
        if block < 1:
            raise ValueError(f"block must be positive. Got block: {block}.")

        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Repeated block_ids for different parameters in block {block}"
            )

        if list(sorted(ids)) != list(range(1, len(ids) + 1)):
            raise ValueError(
                "block_ids must be consecutive integers starting from 1. "
                f"Got {ids} for block {block}"
            )

    blocks = dict(sorted({b: len(ids) for b, ids in checked_blocks.items()}.items()))
    return blocks
