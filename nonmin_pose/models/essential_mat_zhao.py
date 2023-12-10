from typing import Dict, Union

import numpy as np

from nonmin_pose.constraints.constraint_manager import ConstraintManager
from nonmin_pose.constraints.constraints import Parameter
from nonmin_pose.models.base import NonMinRelPoseBase


class EssentialZhao(NonMinRelPoseBase):
    """Essential matrix estimation using Zhao's method [1].

    [1] An efficient solution to non-minimal case essential matrix estimation, J. Zhao.
    """

    SDP_COMPUTES_POSE = False
    PARAMS = {
        "E": Parameter("E", 1, list(range(1, 10))),
        "t": Parameter("t", 1, list(range(10, 13))),
    }
    CONSTRAINTS = {"manif_def_left": None, "norm_t": None}

    def _get_params_and_manager(self, *args, **kwargs):
        params_list = list(self.PARAMS.values())
        manager = ConstraintManager(params_list, self.CONSTRAINTS)
        return self.PARAMS, manager

    def retrieve_solution(self) -> Dict[str, Union[np.ndarray, bool]]:
        """Get the essential matrix and check its optimality."""
        E, t = self.params["E"], self.params["t"]
        Eb, tb = E.block, t.block
        assert Eb == tb

        # get X = xx^\top.
        X = self.npt_problem.getResultYMat(Eb)
        Xe, Xt = X[:9, :9], X[9:, 9:]

        # check optimality.
        eps = self.cfg["th_rank_optimality"]
        _, svals_e, Vt_e = np.linalg.svd(Xe)
        _, svals_t, _ = np.linalg.svd(Xt)
        is_optimal = (svals_e > eps).sum() == 1 == (svals_t > eps).sum()

        E01 = Vt_e[0].reshape(3, 3)
        return {"E01": E01, "is_optimal": is_optimal}
