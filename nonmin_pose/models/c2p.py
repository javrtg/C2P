from typing import Dict, Union

import numpy as np

from nonmin_pose.constraints.constraint_manager import ConstraintManager
from nonmin_pose.constraints.constraints import Parameter
from nonmin_pose.models.base import NonMinRelPoseBase
from nonmin_pose.utils import rot_given_Etq


class C2P(NonMinRelPoseBase):
    SDP_COMPUTES_POSE = True
    DEFAULT_PARAMS = {
        "E": Parameter("E", 1, list(range(1, 10))),
        "t": Parameter("t", 1, list(range(10, 13))),
        "q": Parameter("q", 1, list(range(13, 16))),
        "h": Parameter("h", 1, [16]),
        "sct": Parameter("sct", 2, [1]),
        "scr": Parameter("scr", 3, [1]),
        # "scr2": Parameter("scr2", 4, [1]),
    }
    DEFAULT_CONSTRAINTS = {
        "manif_def_left": [0],
        "manif_def_right": [0],
        "norm_t": None,
        "norm_q": None,
        "homogenization": None,
        "adjoint": None,
        "norm_e": None,
        "cheirality_translation_v2": None,
        "cheirality_rotation": None,
        # "right_null_space": None,
        # "left_null_space": None,
        # "cheirality_rotation_q": None,
    }

    def _get_params_and_manager(self, params=None, constraints=None):
        params = self.DEFAULT_PARAMS if params is None else {p.name: p for p in params}
        constraints = self.DEFAULT_CONSTRAINTS if constraints is None else constraints

        params_list = list(params.values())
        manager = ConstraintManager(params_list, constraints, self.cfg["use_top_k"])
        return params, manager

    def retrieve_solution(self) -> Dict[str, Union[np.ndarray, bool]]:
        X = self.npt_problem.getResultYMat(self.params["h"].block)

        _, svals, Vt = np.linalg.svd(X)
        idx = 0
        E01 = Vt[idx, :9].reshape(3, 3)
        t01 = Vt[idx, 9:12].reshape(3, 1)
        q = Vt[idx, 12:15].reshape(3, 1)
        h = Vt[idx, 15]
        E01, t01, q = (E01, t01, q) if h > 0 else (-E01, -t01, -q)

        sct = self.npt_problem.getResultYMat(2)[0, 0]

        is_pure_rot = sct < self.cfg["th_pure_rot_sdp"]
        if sct < self.cfg["th_pure_rot_noisefree_sdp"]:
            # improve numerical conditioning.
            _, _, Vt = np.linalg.svd(X[:15, :15])
            E01_ = Vt[idx, :9].reshape(3, 3)
            t01_ = Vt[idx, 9:12].reshape(3, 1)
            q_ = Vt[idx, 12:15].reshape(3, 1)
            # correct sign.
            id_mx = np.abs(t01).argmax()
            E01, t01, q = (
                (E01_, t01_, q_)
                if t01[id_mx, 0] * t01_[id_mx, 0] > 0
                else (-E01_, -t01_, -q_)
            )

        # manifold projections.
        Ue, _, Vte = np.linalg.svd(E01)
        E01 = Ue[:, :2] @ Vte[:2]
        t01 = t01 / np.linalg.norm(t01)
        q = q / np.linalg.norm(q)

        R01 = rot_given_Etq(E01, t01, q)

        # check optimality.
        eps = self.cfg["th_rank_optimality"]
        is_optimal = (svals > eps).sum() <= 3
        return {
            "R01": R01,
            "t01": t01,
            "E01": E01,
            "is_optimal": is_optimal,
            "is_pure_rot": is_pure_rot,
        }


class C2PFast(NonMinRelPoseBase):
    SDP_COMPUTES_POSE = True
    PARAMS = {
        "E": Parameter("E", 1, list(range(1, 10))),
        "t": Parameter("t", 1, list(range(10, 13))),
        "q": Parameter("q", 1, list(range(13, 16))),
        "h": Parameter("h", 1, [16]),
        "sct": Parameter("sct", 2, [1]),
        "scr": Parameter("scr", 3, [1]),
    }
    CONSTRAINTS = {
        "norm_t": None,
        "norm_q": None,
        "homogenization": None,
        "adjoint": None,
        "norm_e": None,
        "cheirality_translation_v2": None,
        "cheirality_rotation": None,
    }

    def _get_params_and_manager(self, *args, **kwargs):
        params_list = list(self.PARAMS.values())
        manager = ConstraintManager(params_list, self.CONSTRAINTS)
        return self.PARAMS, manager

    def retrieve_solution(self) -> Dict[str, Union[np.ndarray, bool]]:
        X = self.npt_problem.getResultYMat(self.params["h"].block)

        _, svals, Vt = np.linalg.svd(X)
        idx = 0
        E01 = Vt[idx, :9].reshape(3, 3)
        t01 = Vt[idx, 9:12].reshape(3, 1)
        q = Vt[idx, 12:15].reshape(3, 1)
        h = Vt[idx, 15]
        E01, t01, q = (E01, t01, q) if h > 0 else (-E01, -t01, -q)

        sct = self.npt_problem.getResultYMat(2)[0, 0]

        is_pure_rot = sct < self.cfg["th_pure_rot_sdp"]
        if sct < self.cfg["th_pure_rot_noisefree_sdp"]:
            # improve numerical conditioning.
            _, _, Vt = np.linalg.svd(X[:15, :15])
            E01_ = Vt[idx, :9].reshape(3, 3)
            t01_ = Vt[idx, 9:12].reshape(3, 1)
            q_ = Vt[idx, 12:15].reshape(3, 1)
            # correct sign.
            id_mx = np.abs(t01).argmax()
            E01, t01, q = (
                (E01_, t01_, q_)
                if t01[id_mx, 0] * t01_[id_mx, 0] > 0
                else (-E01_, -t01_, -q_)
            )

        # manifold projections.
        Ue, _, Vte = np.linalg.svd(E01)
        E01 = Ue[:, :2] @ Vte[:2]
        t01 = t01 / np.linalg.norm(t01)
        q = q / np.linalg.norm(q)

        R01 = rot_given_Etq(E01, t01, q)

        # check optimality.
        eps = self.cfg["th_rank_optimality"]
        is_optimal = (svals > eps).sum() <= 3
        return {
            "R01": R01,
            "t01": t01,
            "E01": E01,
            "is_optimal": is_optimal,
            "is_pure_rot": is_pure_rot,
        }
