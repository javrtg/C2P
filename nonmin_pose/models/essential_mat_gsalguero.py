from typing import Dict, Union

import numpy as np

from nonmin_pose.constraints.constraint_manager import ConstraintManager
from nonmin_pose.constraints.constraints import Parameter
from nonmin_pose.models.base import NonMinRelPoseBase


class EssentialGSalguero(NonMinRelPoseBase):
    """Essential matrix solver using García-Salguero's ADJ (adjugate) method [1].

    [1] A Tighter Relaxation for the Relative Pose Problem Between Cameras,
        M. Garcia-Salguero, J. Briales and J. Gonzalez-Jimenez.
    """

    SDP_COMPUTES_POSE = False
    PARAMS = {
        "E": Parameter("E", 1, list(range(1, 10))),
        "t": Parameter("t", 2, list(range(1, 4))),
        "q": Parameter("q", 2, list(range(4, 7))),
    }
    CONSTRAINTS = {
        "manif_def_left": [0],
        "manif_def_right": [0],
        "norm_t": None,
        "norm_q": None,
        "adjoint": None,
        "norm_e": None,
    }

    def _get_params_and_manager(self, *args, **kwargs):
        params_list = list(self.PARAMS.values())
        manager = ConstraintManager(params_list, self.CONSTRAINTS)
        return self.PARAMS, manager

    def retrieve_solution(self) -> Dict[str, Union[np.ndarray, bool]]:
        E, t, q = self.params["E"], self.params["t"], self.params["q"]
        Eb, tb, qb = E.block, t.block, q.block
        assert Eb != tb == qb

        # get X = xx^\top.
        Xe = self.npt_problem.getResultYMat(Eb)
        Xtq = self.npt_problem.getResultYMat(tb)

        # check optimality.
        eps = self.cfg["th_rank_optimality"]
        _, svals_e, Vt_e = np.linalg.svd(Xe)
        svals_tq = np.linalg.svd(Xtq, compute_uv=False)
        is_optimal = (svals_e > eps).sum() == 1 == (svals_tq > eps).sum()

        E01 = Vt_e[0].reshape(3, 3)
        return {"E01": E01, "is_optimal": is_optimal}
