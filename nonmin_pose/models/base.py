from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from nonmin_pose.constraints.constraint_manager import (
    ConstraintConfig,
    ConstraintManager,
)
from nonmin_pose.constraints.constraints import Parameter
from nonmin_pose.sdpa import SDPA
from nonmin_pose.utils import compute_data_matrix_C, decompose_essmat


class NonMinRelPoseBase(ABC):
    """Non-minimal Essential matrix estimation using SDPA solver."""

    DEFAULT_CFG = {
        # PARAMETER_STABLE_BUT_SLOW, PARAMETER_DEFAULT, PARAMETER_UNSTABLE_BUT_FAST
        "sdpa_param_type": SDPA.PARAMETER_DEFAULT,
        "th_rank_optimality": 1e-5,
        "th_pure_rot_post": 1 - 1e-8,  # for Zhao's and Garcia-Salguero's methods.
        "th_pure_rot_sdp": 1e-3,  # for C2P
        "th_pure_rot_noisefree_sdp": 1e-4,  # for C2P
        # for computing the constraint coefficients that are determined at runtime.
        "use_top_k": None,
    }

    SDP_COMPUTES_POSE: bool

    def __init__(
        self,
        parameters: Optional[List[Parameter]] = None,
        constraints: Optional[ConstraintConfig] = None,
        cfg: Optional[Dict] = None,
    ):
        if cfg is None:
            cfg = {}
        self.cfg = {**self.DEFAULT_CFG, **cfg}
        self.params, self.cnt_man = self._get_params_and_manager(
            parameters, constraints
        )
        # self.cnt_man = ConstraintManager(parameters, constraints)
        # self.params = {p.name: p for p in parameters}
        self._init_constant_costmat_params()
        self._init_solver()

        assert isinstance(
            self.SDP_COMPUTES_POSE, bool
        ), "SDP_COMPUTES_POSE class attribute must be defined and be a boolean."

    def __call__(
        self,
        f0: np.ndarray,
        f1: np.ndarray,
        w: Optional[np.ndarray] = None,
        already_unitary: bool = False,
        do_disambiguation: bool = True,
        inliers_conf: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, bool]]:
        """Non-minimal relative pose estimation.

        Args:
            f0: (3, n) bearing vectors in camera 0.
            f1: (3, n) bearing vectors in camera 1.
            w: (n,) array of weights for each residual. (default: None).
            already_unitary: True if the input coordinates are already unitary
                (default: False).
            do_disambiguation: True to decompose the essential matrix into relative
                rotation and translation. This argument is ignored when using C2P and is
                only used for the essential matrix solvers of Zhao and Garcia-Salguero.
                (default: True).

        Returns:
            sol: dict of solution parameters, with keys and values:
                - E01: (3, 3) essential matrix.
                - R01: (3, 3) rotation matrix. It is not returned for the methods of
                    Zhao and G.Salguero et al. if do_disambiguation is False.
                - t01: (3, 1) translation vector. It is not returned for the methods of
                    Zhao and G.Salguero et al. if do_disambiguation is False.
                - is_optimal: True if the solution is optimal.
                - is_pure_rot: True if a (near-)pure rotation is detected. It is not
                    returned for the methods of Zhao and G.Salguero et al. if
                    do_disambiguation is False.
        """
        sh0, sh1 = f0.shape, f1.shape
        assert sh0 == sh1 and len(sh0) == 2 and sh0[0] == 3 and sh0[1] >= 5

        if not already_unitary:
            f0 = f0 / np.linalg.norm(f0, axis=0)
            f1 = f1 / np.linalg.norm(f1, axis=0)

        C = compute_data_matrix_C(f0, f1, w)
        self.solveSDP(C, f0, f1, inliers_conf)
        # retrieve estimated model, optimality and related measures.
        sol = self.retrieve_solution()

        if not self.SDP_COMPUTES_POSE:
            # Ensure E01 belongs to the normalized essential manifold.
            U, _, Vt = np.linalg.svd(sol["E01"])
            sol["E01"] = U[:, :2] @ Vt[:2]  # the same as U @ diag(1, 1, 0) @ Vt

            if do_disambiguation:
                (
                    sol["R01"],
                    sol["t01"],
                    sol["is_pure_rot"],
                ) = decompose_essmat(U, Vt, f0, f1, self.cfg["th_pure_rot_post"])

        return sol

    def solveSDP(
        self,
        C: np.ndarray,
        f0: np.ndarray,
        f1: np.ndarray,
        inliers_conf: Optional[np.ndarray] = None,
    ) -> None:
        p = self.npt_problem
        cnt = self.cnt_man

        p.initializeUpperTriangleSpace()
        # constraint values.
        p.inputAllCVec(cnt.values)
        # nonzero elements of the data matrix C0.
        p.inputAllElements(
            self.C_constraints_idx,
            self.C_block_idx,
            self.C_row_idx_1based,
            self.C_col_idx_1based,
            # SDPA does max optimization, so we change the sign of C.
            -C[self.C_row_idx, self.C_col_idx],
        )
        # compute dynamic coefficients (if any) and set all coeffs for each constraint.
        cnt.compute_dynamic_coeffs(f0, f1, inliers_conf)
        p.inputAllElements(
            cnt.constraint_idx,
            cnt.blocks,
            cnt.rows,
            cnt.cols,
            cnt.coeffs,
        )
        p.initializeUpperTriangle(False)
        # solve SDP.
        p.initializeSolve()
        p.solve()

    def _init_solver(self):
        """Initialize SDPA solver."""
        self.npt_problem = p = SDPA()
        p.setParameterType(self.cfg["sdpa_param_type"])
        p.inputConstraintNumber(self.cnt_man.n_constraints)
        p.inputBlockNumber(self.cnt_man.n_blocks)
        for bi, size in self.cnt_man.block_sizes.items():
            p.inputBlockSize(bi, size)
            # p.inputBlockType(bi, SDPA.SDP)
        for bi in self.cnt_man.block_sizes:
            p.inputBlockType(bi, SDPA.SDP)

    def _init_constant_costmat_params(self):
        """ "Set constant parameters related to the data matrix C.

        NOTE: We must define upper-triangular elements' info only.
        """
        self.C_constraints_idx = np.zeros((45,), dtype=np.int32)
        self.C_block_idx = np.ones((45,), dtype=np.int32)
        self.C_row_idx, self.C_col_idx = np.triu_indices(9)
        self.C_row_idx_1based = self.C_row_idx + 1
        self.C_col_idx_1based = self.C_col_idx + 1

    @abstractmethod
    def _get_params_and_manager(
        self, p: Optional[List[Parameter]], c: Optional[ConstraintConfig]
    ) -> Tuple[Dict[str, Parameter], ConstraintManager]:
        """Get parameters and constraint manager."""
        pass

    @abstractmethod
    def retrieve_solution(self) -> Dict[str, Union[np.ndarray, bool]]:
        """Retrieve essential matrix, optimality and related measures."""
        pass
