""" Module for instantiating constraints.

    The general definitions and conventions for the parameters are:
        a) The rotation matrix R01 and the translation vector are those that transform
           coordinate points pi from the reference frame {1} to the {0}:
               p0 = R01 p1 + t01
        b) q = R01^T t01
        c) E01 = [t01]_x R01 = R01 [q]_x
        d) E := E01; R := R01; t := t01
        e) The order of the parameters considered within this module is:
            1) E:    9 param.
            2) t:    3 param.
            3) q:    3 param.
            4) h:    1 param.
            5) R:    9 param.
            6) sct:  1 param.  -> slack variable (s.v.) for disambiguating t (its sign).
            7) scr:  1 param.  -> s.v. for disambiguating R.
            8) scm1: 1 param.  -> s.v. for disambiguating t and R with midpoint eqs.
            9) scm2: 1 param.  -> s.v. for disambiguating t and R with midpoint eqs.
            10) Zc: 16 params. -> for SO(3) convex hull's linear matrix inequality (LMI)
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


def assert_smaller_idxes(param1i, param2i):
    """Assert that the block indexes of one parameter are smaller than the other.

    When we use the SDPA's solver, we need to specify upper-diagonal values of the
    symmetric matrices corresponding to the quadratic constraints and the cost matrix.
    This function helps to ensure this in parameters that are located in the same block.

    Args:
        param1i: block-indexes of the first parameter
        param2i: block-indexes of the second parameter

    Raises:
        AssertionError: if the block-indexes of param1i are not smaller than param2i
    """
    assert all(
        all(param1i_ < param2i_ for param2i_ in param2i) for param1i_ in param1i
    ), "param1 block-indexes must be < param2i block-indexes."


class Parameter:
    """Class for defining a parameter.

    Attributes:
        name: e.g. E, R, t, etc. This MUST match the name being used on the constraints.
        block: 1-based index of the block.
        block_ids: 1-based index of each parameter element in the block.
    """

    __slots__ = ("name", "block", "block_ids")

    def __init__(self, name: str, block: int, block_ids: List[int]):
        assert block > 0, "block must be positive"
        assert all(idx > 0 for idx in block_ids), "block_id must be positive"

        self.name = name
        self.block = block
        self.block_ids = block_ids


class Constraint(ABC):
    CONSTRAINT_IDX_PER_EQ: List[List[int]]
    COEFFS_PER_EQ: List[List[float]]
    CONSTRAINT_VALUES: List[float]

    __slots__ = (
        "constraint_idx",
        "drop_eqs",
        "coeffs",
        "values",
        "blocks",
        "rows",
        "cols",
        "idx_first_el",
        "idx_last_el",
    )

    def __init__(
        self,
        params: dict,
        idx_first_el: int,
        idx_first_eq: int = 0,
        drop_eqs: Optional[List[int]] = None,
    ):
        blocks, rows, cols = self.get_eqs_info(params)
        self.flatten_eqs_info(idx_first_eq, blocks, rows, cols, drop_eqs)

        # index of the first and last element of the  constraint in the global arrays.
        self.idx_first_el = idx_first_el
        self.idx_last_el = idx_first_el + len(self.constraint_idx)

        self.drop_eqs = drop_eqs

    def flatten_eqs_info(self, idx_first_eq, blocks, rows, cols, drop_eqs):
        idx_, coeffs_, values_, blocks_, rows_, cols_ = [], [], [], [], [], []
        added_eq_id = 1
        for eq_id, (c, v, b, r, col) in enumerate(
            zip(self.COEFFS_PER_EQ, self.CONSTRAINT_VALUES, blocks, rows, cols)
        ):
            if drop_eqs is None or eq_id not in drop_eqs:
                idx_.extend([idx_first_eq + added_eq_id] * len(c))
                coeffs_.extend(c)
                values_.append(v)
                blocks_.extend(b)
                rows_.extend(r)
                cols_.extend(col)
                added_eq_id += 1

        assert len(idx_) == len(coeffs_) == len(blocks_) == len(rows_) == len(cols_)
        self.constraint_idx = np.array(idx_)
        self.coeffs = np.array(coeffs_)
        self.values = np.array(values_)
        self.blocks = np.array(blocks_)
        self.rows = np.array(rows_)
        self.cols = np.array(cols_)

    @abstractmethod
    def get_eqs_info(self, params):
        raise NotImplementedError


class Adjoint(Constraint):
    """adj(E) = cofactors(E).T = qt^T."""

    EQUATION = "adj(E) = qt^T"
    COEFFS_PER_EQ = [
        [1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
    ]
    CONSTRAINT_VALUES = [0.0] * 9

    __slots__ = ()

    def get_eqs_info(self, params):
        E, t, q = params["E"], params["t"], params["q"]
        Eb, tb, qb = E.block, t.block, q.block
        assert tb == qb
        blocks = [[Eb, Eb, tb]] * 9
        Ei, ti, qi = E.block_ids, t.block_ids, q.block_ids
        assert_smaller_idxes(ti, qi)
        rows = [
            [Ei[4], Ei[5], ti[0]],
            [Ei[5], Ei[3], ti[0]],
            [Ei[3], Ei[4], ti[0]],
            [Ei[2], Ei[1], ti[1]],
            [Ei[0], Ei[2], ti[1]],
            [Ei[1], Ei[0], ti[1]],
            [Ei[1], Ei[2], ti[2]],
            [Ei[2], Ei[0], ti[2]],
            [Ei[0], Ei[1], ti[2]],
        ]
        cols = [
            [Ei[8], Ei[7], qi[0]],
            [Ei[6], Ei[8], qi[1]],
            [Ei[7], Ei[6], qi[2]],
            [Ei[7], Ei[8], qi[0]],
            [Ei[8], Ei[6], qi[1]],
            [Ei[6], Ei[7], qi[2]],
            [Ei[5], Ei[4], qi[0]],
            [Ei[3], Ei[5], qi[1]],
            [Ei[4], Ei[3], qi[2]],
        ]
        return blocks, rows, cols


class NormT(Constraint):
    """||t||^2 = 1"""

    EQUATION = "||t||^2 = 1"
    COEFFS_PER_EQ = [[1.0, 1.0, 1.0]]
    CONSTRAINT_VALUES = [1.0]

    __slots__ = ()

    def get_eqs_info(self, params):
        t = params["t"]
        tb = t.block
        blocks = [[tb, tb, tb]]
        ti = t.block_ids
        rows = [[ti[0], ti[1], ti[2]]]
        cols = [[ti[0], ti[1], ti[2]]]
        return blocks, rows, cols


class NormQ(Constraint):
    """||q||^2 = 1"""

    EQUATION = "||q||^2 = 1"
    COEFFS_PER_EQ = [[1.0, 1.0, 1.0]]
    CONSTRAINT_VALUES = [1.0]

    __slots__ = ()

    def get_eqs_info(self, params):
        q = params["q"]
        qb = q.block
        blocks = [[qb, qb, qb]]
        qi = q.block_ids
        rows = [[qi[0], qi[1], qi[2]]]
        cols = [[qi[0], qi[1], qi[2]]]
        return blocks, rows, cols


class NormE(Constraint):
    """norm(E) = tr(E^T E) = 2

    e0^2 + e1^2 + e2^2 + e3^2 + e4^2 + e5^2 + e6^2 + e7^2 + e8^2 = 2
    """

    EQUATION = "norm(E) = 2"
    COEFFS_PER_EQ = [[1.0] * 9]
    CONSTRAINT_VALUES = [2.0]

    __slots__ = ()

    def get_eqs_info(self, params):
        E = params["E"]
        Eb = E.block
        blocks = [[Eb] * 9]
        Ei = E.block_ids
        rows = [[Ei[0], Ei[1], Ei[2], Ei[3], Ei[4], Ei[5], Ei[6], Ei[7], Ei[8]]]
        cols = [[Ei[0], Ei[1], Ei[2], Ei[3], Ei[4], Ei[5], Ei[6], Ei[7], Ei[8]]]
        return blocks, rows, cols


class Homogenization(Constraint):
    """h^2 = 1"""

    EQUATION = "h^2 = 1"
    COEFFS_PER_EQ = [[1.0]]
    CONSTRAINT_VALUES = [1.0]

    __slots__ = ()

    def get_eqs_info(self, params):
        h = params["h"]
        hb = h.block
        blocks = [[hb]]
        hi = h.block_ids
        rows = [[hi[0]]]
        cols = [[hi[0]]]
        return blocks, rows, cols


class CheiralityTranslationV2(Constraint):
    """f0^T t01 - t01^T R01 f1 >= 0 => f0^T t01 - q^T f1 - sct^2 = 0

    * sct^2 represents a (positive) slack variable.
    * The corresponding expanded equation with the quadratic term is:
    f0_0*h*t0 + f0_1*h*t1 + f0_2*h*t2 - f1_0*h*q0 - f1_1*h*q1 - f1_2*h*q2 - sct^2 = 0
    """

    EQUATION = "f0^T t01 - q^T f1 - sct^2 = 0"
    # Note: these coefficients will be modified based on the input data (bearings).
    COEFFS_PER_EQ = [[1.0] * 6 + [-1.0]]
    CONSTRAINT_VALUES = [0.0]
    CONSTRAINT_VALUES = [0.0]

    __slots__ = ()

    def get_eqs_info(self, params):
        h, t, q, sct = params["h"], params["t"], params["q"], params["sct"]
        hb, tb, qb, sctb = h.block, t.block, q.block, sct.block
        assert hb == tb == qb
        blocks = [[hb] * 6 + [sctb]]
        hi, ti, qi, scti = h.block_ids, t.block_ids, q.block_ids, sct.block_ids
        assert_smaller_idxes(ti, hi)
        assert_smaller_idxes(qi, hi)
        rows = [[ti[0], ti[1], ti[2], qi[0], qi[1], qi[2], scti[0]]]
        cols = [[hi[0], hi[0], hi[0], hi[0], hi[0], hi[0], scti[0]]]
        return blocks, rows, cols

    def compute_coeffs(self, coeffs: np.ndarray, f0: np.ndarray, f1: np.ndarray):
        """Compute the data-dependent coefficients of the constraint."""
        assert f0.ndim == f1.ndim == 2
        # f0, f1 = f0.sum(1), f1.sum(1)
        f0, f1 = f0.mean(1), f1.mean(1)
        # TODO: solve edge case where the means are close to a zero vector.

        # view of the coefficients to modify.
        coeffs = coeffs[self.idx_first_el : self.idx_last_el]
        coeffs[:3] = f0
        coeffs[3:6] = -f1


class CheiralityRotation(Constraint):
    """(t01 x R01 f1).(t01 x f0) >= 0 => f1^T E01^T [t01] f0 - scr^2 =0

    * scr^2 represents a (positive) slack variable.
    * The corresponding expanded equation with the quadratic term is:
          f0_0*f1_0*e3*t2 + f0_0*f1_1*e4*t2 + f0_0*f1_2*e5*t2 + f0_1*f1_0*e6*t0
        + f0_1*f1_1*e7*t0 + f0_1*f1_2*e8*t0 + f0_2*f1_0*e0*t1 + f0_2*f1_1*e1*t1
        + f0_2*f1_2*e2*t1 - f0_0*f1_0*e6*t1 - f0_0*f1_1*e7*t1 - f0_0*f1_2*e8*t1
        - f0_1*f1_0*e0*t2 - f0_1*f1_1*e1*t2 - f0_1*f1_2*e2*t2 - f0_2*f1_0*e3*t0
        - f0_2*f1_1*e4*t0 - f0_2*f1_2*e5*t0 - scr^2 = 0
    """

    EQUATION = "f1^T E01^T [t01] f0 - scr^2 =0"
    # Note: these coefficients will be modified based on the input data (bearings).
    COEFFS_PER_EQ = [[1.0] * 18 + [-1.0]]
    CONSTRAINT_VALUES = [0.0]

    __slots__ = ()

    def get_eqs_info(self, params):
        E, t, scr = params["E"], params["t"], params["scr"]
        Eb, tb, scrb = E.block, t.block, scr.block
        assert Eb == tb
        blocks = [[tb] * 18 + [scrb]]
        Ei, ti, scri = E.block_ids, t.block_ids, scr.block_ids
        assert_smaller_idxes(Ei, ti)
        # fmt: off
        rows   = [
            [Ei[3], Ei[4], Ei[5], Ei[6],
             Ei[7], Ei[8], Ei[0], Ei[1],
             Ei[2], Ei[6], Ei[7], Ei[8],
             Ei[0], Ei[1], Ei[2], Ei[3],
             Ei[4], Ei[5], scri[0]]
        ]
        cols = [
            [ti[2], ti[2], ti[2], ti[0],
             ti[0], ti[0], ti[1], ti[1],
             ti[1], ti[1], ti[1], ti[1],
             ti[2], ti[2], ti[2], ti[0],
             ti[0], ti[0], scri[0]]
        ]
        # fmt: on
        return blocks, rows, cols

    def compute_coeffs(self, coeffs: np.ndarray, f0: np.ndarray, f1: np.ndarray):
        """Compute the data-dependent coefficients of the constraint."""
        f0_outer_f1 = self.aggregate_data(f0, f1)
        # view of the coefficients to modify.
        coeffs_ = coeffs[self.idx_first_el : self.idx_last_el]
        coeffs_[:9] = f0_outer_f1
        coeffs_[9:18] = -f0_outer_f1

    @staticmethod
    def aggregate_data(f0, f1):
        """Sum of outer products of each f0-f1 pair."""
        n = f0.shape[1]
        assert f0.shape == f1.shape == (3, n)
        f0_outer_f1 = np.einsum("in, jn -> ij", f0, f1).ravel()  # (9,)
        # TODO: solve edge case where f0_outer_f1 is close to a zero vector.
        return f0_outer_f1 / n


class ManifDefLeft(Constraint):
    """E E^T - [t][t]^T = 0_{3x3}"""

    EQUATION = "E E^T = [t][t]^T"
    COEFFS_PER_EQ = [
        [1.0, 1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    CONSTRAINT_VALUES = [0.0] * 6

    __slots__ = ()

    def get_eqs_info(self, params):
        E, t = params["E"], params["t"]
        Eb, tb = E.block, t.block
        blocks = [[Eb, Eb, Eb, tb, tb]] * 3 + [[Eb, Eb, Eb, tb]] * 3
        Ei, ti = E.block_ids, t.block_ids
        rows = [
            [Ei[0], Ei[1], Ei[2], ti[1], ti[2]],
            [Ei[3], Ei[4], Ei[5], ti[0], ti[2]],
            [Ei[6], Ei[7], Ei[8], ti[0], ti[1]],
            [Ei[0], Ei[1], Ei[2], ti[0]],
            [Ei[0], Ei[1], Ei[2], ti[0]],
            [Ei[3], Ei[4], Ei[5], ti[1]],
        ]
        cols = [
            [Ei[0], Ei[1], Ei[2], ti[1], ti[2]],
            [Ei[3], Ei[4], Ei[5], ti[0], ti[2]],
            [Ei[6], Ei[7], Ei[8], ti[0], ti[1]],
            [Ei[3], Ei[4], Ei[5], ti[1]],
            [Ei[6], Ei[7], Ei[8], ti[2]],
            [Ei[6], Ei[7], Ei[8], ti[2]],
        ]
        return blocks, rows, cols


class ManifDefRight(Constraint):
    """E^T E = [q][q]^T"""

    EQUATION = "E^T E = [q][q]^T"
    COEFFS_PER_EQ = [
        [1.0, 1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    CONSTRAINT_VALUES = [0.0] * 6

    __slots__ = ()

    def get_eqs_info(self, params):
        E, q = params["E"], params["q"]
        Eb, qb = E.block, q.block
        blocks = [[Eb, Eb, Eb, qb, qb]] * 3 + [[Eb, Eb, Eb, qb]] * 3
        Ei, qi = E.block_ids, q.block_ids
        rows = [
            [Ei[0], Ei[3], Ei[6], qi[1], qi[2]],
            [Ei[1], Ei[4], Ei[7], qi[0], qi[2]],
            [Ei[2], Ei[5], Ei[8], qi[0], qi[1]],
            [Ei[0], Ei[3], Ei[6], qi[0]],
            [Ei[0], Ei[3], Ei[6], qi[0]],
            [Ei[1], Ei[4], Ei[7], qi[1]],
        ]
        cols = [
            [Ei[0], Ei[3], Ei[6], qi[1], qi[2]],
            [Ei[1], Ei[4], Ei[7], qi[0], qi[2]],
            [Ei[2], Ei[5], Ei[8], qi[0], qi[1]],
            [Ei[1], Ei[4], Ei[7], qi[1]],
            [Ei[2], Ei[5], Ei[8], qi[2]],
            [Ei[2], Ei[5], Ei[8], qi[2]],
        ]
        return blocks, rows, cols


class EDefLeft(Constraint):
    """hE = [t]R. h is the homogenization variable."""

    EQUATION = "hE = [t]R"
    COEFFS_PER_EQ = [
        [1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
    ]
    CONSTRAINT_VALUES = [0.0] * 9

    __slots__ = ()

    def get_eqs_info(self, params):
        E, h, t, R = params["E"], params["h"], params["t"], params["R"]
        Eb, hb, tb, Rb = E.block, h.block, t.block, R.block
        assert Eb == hb and Rb == tb
        blocks = [[Eb, tb, tb]] * 9
        Ei, hi, ti, Ri = E.block_ids, h.block_ids, t.block_ids, R.block_ids
        assert_smaller_idxes(Ei, hi)
        assert_smaller_idxes(ti, Ri)
        rows = [
            [Ei[0], ti[2], ti[1]],
            [Ei[1], ti[2], ti[1]],
            [Ei[2], ti[2], ti[1]],
            [Ei[3], ti[2], ti[0]],
            [Ei[4], ti[2], ti[0]],
            [Ei[5], ti[2], ti[0]],
            [Ei[6], ti[1], ti[0]],
            [Ei[7], ti[1], ti[0]],
            [Ei[8], ti[1], ti[0]],
        ]
        cols = [
            [hi[0], Ri[3], Ri[6]],
            [hi[0], Ri[4], Ri[7]],
            [hi[0], Ri[5], Ri[8]],
            [hi[0], Ri[0], Ri[6]],
            [hi[0], Ri[1], Ri[7]],
            [hi[0], Ri[2], Ri[8]],
            [hi[0], Ri[0], Ri[3]],
            [hi[0], Ri[1], Ri[4]],
            [hi[0], Ri[2], Ri[5]],
        ]
        return blocks, rows, cols


class EDefRight(Constraint):
    """hE = R[q]. h is the homogenization variable."""

    EQUATION = "hE = R[q]"
    COEFFS_PER_EQ = [
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
    ]
    CONSTRAINT_VALUES = [0.0] * 9

    __slots__ = ()

    def get_eqs_info(self, params):
        E, h, R, q = params["E"], params["h"], params["R"], params["q"]
        Eb, hb, Rb, qb = E.block, h.block, R.block, q.block
        assert Eb == hb and Rb == qb
        blocks = [[Eb, qb, qb]] * 9
        Ei, hi, Ri, qi = E.block_ids, h.block_ids, R.block_ids, q.block_ids
        assert_smaller_idxes(Ei, hi)
        assert_smaller_idxes(qi, Ri)
        rows = [
            [Ei[0], qi[2], qi[1]],
            [Ei[1], qi[2], qi[0]],
            [Ei[2], qi[1], qi[0]],
            [Ei[3], qi[2], qi[1]],
            [Ei[4], qi[2], qi[0]],
            [Ei[5], qi[1], qi[0]],
            [Ei[6], qi[2], qi[1]],
            [Ei[7], qi[2], qi[0]],
            [Ei[8], qi[1], qi[0]],
        ]
        cols = [
            [hi[0], Ri[1], Ri[2]],
            [hi[0], Ri[0], Ri[2]],
            [hi[0], Ri[0], Ri[1]],
            [hi[0], Ri[4], Ri[5]],
            [hi[0], Ri[3], Ri[5]],
            [hi[0], Ri[3], Ri[4]],
            [hi[0], Ri[7], Ri[8]],
            [hi[0], Ri[6], Ri[8]],
            [hi[0], Ri[6], Ri[7]],
        ]
        return blocks, rows, cols


class EDefLeftRight(Constraint):
    """[t]R - R[q] = 0_{3x3}

     t1*r6 - t2*r3 + q1*r2 - q2*r1 = 0
     t1*r7 - t2*r4 - q0*r2 + q2*r0 = 0
     t1*r8 - t2*r5 + q0*r1 - q1*r0 = 0
    -t0*r6 + t2*r0 + q1*r5 - q2*r4 = 0
    -t0*r7 + t2*r1 - q0*r5 + q2*r3 = 0
    -t0*r8 + t2*r2 + q0*r4 - q1*r3 = 0
     t0*r3 - t1*r0 + q1*r8 - q2*r7 = 0
     t0*r4 - t1*r1 - q0*r8 + q2*r6 = 0
     t0*r5 - t1*r2 + q0*r7 - q1*r6 = 0
    """

    EQUATION = "[t]R = R[q] = 0"
    COEFFS_PER_EQ = [
        [1.0, -1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
    ]
    CONSTRAINT_VALUES = [0.0] * 9

    __slots__ = ()

    def get_eqs_info(self, params):
        t, q, R = params["t"], params["q"], params["R"]
        tb, qb, Rb = t.block, q.block, R.block
        assert tb == qb == Rb
        blocks = [[tb, tb, tb, tb]] * 9
        ti, qi, Ri = t.block_ids, q.block_ids, R.block_ids
        assert_smaller_idxes(ti, qi)
        assert_smaller_idxes(qi, Ri)
        rows = [
            [ti[1], ti[2], qi[1], qi[2]],
            [ti[1], ti[2], qi[0], qi[2]],
            [ti[1], ti[2], qi[0], qi[1]],
            [ti[0], ti[2], qi[1], qi[2]],
            [ti[0], ti[2], qi[0], qi[2]],
            [ti[0], ti[2], qi[0], qi[1]],
            [ti[0], ti[1], qi[1], qi[2]],
            [ti[0], ti[1], qi[0], qi[2]],
            [ti[0], ti[1], qi[0], qi[1]],
        ]
        cols = [
            [Ri[6], Ri[3], Ri[2], Ri[1]],
            [Ri[7], Ri[4], Ri[2], Ri[0]],
            [Ri[8], Ri[5], Ri[1], Ri[0]],
            [Ri[6], Ri[0], Ri[5], Ri[4]],
            [Ri[7], Ri[1], Ri[5], Ri[3]],
            [Ri[8], Ri[2], Ri[4], Ri[3]],
            [Ri[3], Ri[0], Ri[8], Ri[7]],
            [Ri[4], Ri[1], Ri[8], Ri[6]],
            [Ri[5], Ri[2], Ri[7], Ri[6]],
        ]
        return blocks, rows, cols


class RightNullSpace(Constraint):
    """E q = 0

    e0*q0 + e1*q1 + e2*q2 = 0,
    e3*q0 + e4*q1 + e5*q2 = 0
    e6*q0 + e7*q1 + e8*q2 = 0
    """

    EQUATION = "E q = 0"
    COEFFS_PER_EQ = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    CONSTRAINT_VALUES = [0.0] * 3

    __slots__ = ()

    def get_eqs_info(self, params):
        E, q = params["E"], params["q"]
        Eb, qb = E.block, q.block
        assert Eb == qb
        blocks = [[Eb, Eb, Eb]] * 3
        Ei, qi = E.block_ids, q.block_ids
        assert_smaller_idxes(Ei, qi)
        rows = [
            [Ei[0], Ei[1], Ei[2]],
            [Ei[3], Ei[4], Ei[5]],
            [Ei[6], Ei[7], Ei[8]],
        ]
        cols = [
            [qi[0], qi[1], qi[2]],
            [qi[0], qi[1], qi[2]],
            [qi[0], qi[1], qi[2]],
        ]
        return blocks, rows, cols


class LeftNullSpace(Constraint):
    """t^T E = 0

    e0*t0 + e3*t1 + e6*t2 = 0
    e1*t0 + e4*t1 + e7*t2 = 0
    e2*t0 + e5*t1 + e8*t2 = 0
    """

    EQUATION = "E^T t = 0"
    COEFFS_PER_EQ = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    CONSTRAINT_VALUES = [0.0] * 3

    __slots__ = ()

    def get_eqs_info(self, params):
        E, t = params["E"], params["t"]
        Eb, tb = E.block, t.block
        assert Eb == tb
        blocks = [[Eb, Eb, Eb]] * 3
        Ei, ti = E.block_ids, t.block_ids
        assert_smaller_idxes(Ei, ti)
        rows = [
            [Ei[0], Ei[3], Ei[6]],
            [Ei[1], Ei[4], Ei[7]],
            [Ei[2], Ei[5], Ei[8]],
        ]
        cols = [
            [ti[0], ti[1], ti[2]],
            [ti[0], ti[1], ti[2]],
            [ti[0], ti[1], ti[2]],
        ]
        return blocks, rows, cols


class CheiralityTranslation(Constraint):
    """f0^T t01 - t01^T R01 f1 >= 0 => f0^T R01 q - t01^T R01 f1 - sct^2 = 0

    * sct^2 represents a (positive) slack variable.
    * The corresponding expanded equation with the quadratic term is:
          f0_0*q0*r0 + f0_0*q1*r1 + f0_0*q2*r2 + f0_1*q0*r3 + f0_1*q1*r4
        + f0_1*q2*r5 + f0_2*q0*r6 + f0_2*q1*r7 + f0_2*q2*r8 - f1_0*r0*t0
        - f1_0*r3*t1 - f1_0*r6*t2 - f1_1*r1*t0 - f1_1*r4*t1 - f1_1*r7*t2
        - f1_2*r2*t0 - f1_2*r5*t1 - f1_2*r8*t2 - sct^2 = 0
    """

    EQUATION = "f0^T R01 q - t01^T R01 f1 - s1^2 = 0"
    # Note: these coefficients will be modified based on the input data (bearings).
    COEFFS_PER_EQ = [[1.0] * 18 + [-1.0]]
    CONSTRAINT_VALUES = [0.0]

    __slots__ = ()

    def get_eqs_info(self, params):
        t, q, R, sct = params["t"], params["q"], params["R"], params["sct"]
        tb, qb, Rb, sctb = t.block, q.block, R.block, sct.block
        assert tb == qb == Rb
        blocks = [[tb] * 18 + [sctb]]
        ti, qi, Ri, scti = t.block_ids, q.block_ids, R.block_ids, sct.block_ids
        assert_smaller_idxes(ti, qi)
        assert_smaller_idxes(qi, Ri)
        # fmt: off
        rows = [
            [qi[0], qi[1], qi[2], qi[0], qi[1],
             qi[2], qi[0], qi[1], qi[2], ti[0],
             ti[1], ti[2], ti[0], ti[1], ti[2],
             ti[0], ti[1], ti[2], scti[0]]
        ]
        cols = [
            [Ri[0], Ri[1], Ri[2], Ri[3], Ri[4],
             Ri[5], Ri[6], Ri[7], Ri[8], Ri[0],
             Ri[3], Ri[6], Ri[1], Ri[4], Ri[7],
             Ri[2], Ri[5], Ri[8], scti[0]]
        ]
        # fmt: on
        return blocks, rows, cols

    def compute_coeffs(self, coeffs: np.ndarray, f0: np.ndarray, f1: np.ndarray):
        """Compute the data-dependent coefficients of the constraint."""
        assert f0.ndim == f1.ndim == 2
        f0, f1 = f0.sum(1), f1.sum(1)

        # view of the coefficients to modify.
        coeffs = coeffs[self.idx_first_el : self.idx_last_el]
        coeffs[:3] = f0[0]
        coeffs[3:6] = f0[1]
        coeffs[6:9] = f0[2]
        coeffs[9:12] = -f1[0]
        coeffs[12:15] = -f1[1]
        coeffs[15:18] = -f1[2]


class CheiralityRotationQ(Constraint):
    """(q x R01^T f0).(q x f1) <= 0 => f0^T E01 [q] f1 + scr^2 = 0

    * scr^2 represents a (positive) slack variable.
    * The corresponding expanded equation with the quadratic term is:
          f0_0*f1_0*e1*q2 + f0_0*f1_1*e2*q0 + f0_0*f1_2*e0*q1 + f0_1*f1_0*e4*q2
        + f0_1*f1_1*e5*q0 + f0_1*f1_2*e3*q1 + f0_2*f1_0*e7*q2 + f0_2*f1_1*e8*q0
        + f0_2*f1_2*e6*q1 - f0_0*f1_0*e2*q1 - f0_0*f1_1*e0*q2 - f0_0*f1_2*e1*q0
        - f0_1*f1_0*e5*q1 - f0_1*f1_1*e3*q2 - f0_1*f1_2*e4*q0 - f0_2*f1_0*e8*q1
        - f0_2*f1_1*e6*q2 - f0_2*f1_2*e7*q0 - scr^2 = 0
    """

    EQUATION = "f0^T E01 [q] f1 + scr^2 = 0"
    # Note: these coefficients will be modified based on the input data (bearings).
    COEFFS_PER_EQ = [[1.0] * 19]
    CONSTRAINT_VALUES = [0.0]

    __slots__ = ()

    def get_eqs_info(self, params):
        E, q, scr = params["E"], params["q"], params["scr2"]
        Eb, qb, scrb = E.block, q.block, scr.block
        assert Eb == qb
        blocks = [[qb] * 18 + [scrb]]
        Ei, qi, scri = E.block_ids, q.block_ids, scr.block_ids
        assert_smaller_idxes(Ei, qi)
        # fmt: off
        rows   = [
            [Ei[1], Ei[2], Ei[0], Ei[4],
             Ei[5], Ei[3], Ei[7], Ei[8],
             Ei[6], Ei[2], Ei[0], Ei[1],
             Ei[5], Ei[3], Ei[4], Ei[8],
             Ei[6], Ei[7], scri[0]]
        ]
        cols = [
            [qi[2], qi[0], qi[1], qi[2],
             qi[0], qi[1], qi[2], qi[0],
             qi[1], qi[1], qi[2], qi[0],
             qi[1], qi[2], qi[0], qi[1],
             qi[2], qi[0], scri[0]]
        ]
        # fmt: on
        return blocks, rows, cols

    def compute_coeffs(self, coeffs: np.ndarray, f0: np.ndarray, f1: np.ndarray):
        """Compute the data-dependent coefficients of the constraint."""
        f0_outer_f1 = self.aggregate_data(f0, f1)
        # view of the coefficients to modify.
        coeffs_ = coeffs[self.idx_first_el : self.idx_last_el]
        coeffs_[:9] = f0_outer_f1
        coeffs_[9:18] = -f0_outer_f1

    @staticmethod
    def aggregate_data(f0, f1):
        """Sum of outer products of each f0-f1 pair."""
        n = f0.shape[1]
        assert f0.shape == f1.shape == (3, n)
        f0_outer_f1 = np.einsum("in, jn -> ij", f0, f1).ravel()  # (9,)
        return f0_outer_f1


class CheiralityMidpoint(Constraint):
    """Inequalities stemming from the midpoint method (positive lambdas):

    The constraints are:
        1) (-t01.T @ R01 @ f1) + (f0.T @ R @ f1) * (f0.T @ t) > 0
        2) (f0.T @ R01 @ q) - (f1.T @ q) * (f0.T @ R01 @ f1) > 0
    The corresponding expanded equations with the quadratic terms are:
    1)
      (f0_0**2*f1_0-f1_0)*r0*t0 + (f0_0**2*f1_1-f1_1)*r1*t0 + (f0_0**2*f1_2-f1_2)*r2*t0
    + (f0_1**2*f1_0-f1_0)*r3*t1 + (f0_1**2*f1_1-f1_1)*r4*t1 + (f0_1**2*f1_2-f1_2)*r5*t1
    + (f0_2**2*f1_0-f1_0)*r6*t2 + (f0_2**2*f1_1-f1_1)*r7*t2 + (f0_2**2*f1_2-f1_2)*r8*t2
    + f0_0*f0_1*f1_0*r0*t1      + f0_0*f0_1*f1_1*r1*t1      + f0_0*f0_1*f1_2*r2*t1
    + f0_0*f0_2*f1_0*r0*t2      + f0_0*f0_2*f1_1*r1*t2      + f0_0*f0_2*f1_2*r2*t2
    + f0_1*f0_2*f1_0*r3*t2      + f0_1*f0_2*f1_1*r4*t2      + f0_1*f0_2*f1_2*r5*t2
    + f0_0*f0_1*f1_0*r3*t0      + f0_0*f0_1*f1_1*r4*t0      + f0_0*f0_1*f1_2*r5*t0
    + f0_0*f0_2*f1_0*r6*t0      + f0_0*f0_2*f1_1*r7*t0      + f0_0*f0_2*f1_2*r8*t0
    + f0_1*f0_2*f1_0*r6*t1      + f0_1*f0_2*f1_1*r7*t1      + f0_1*f0_2*f1_2*r8*t1
    - scm1^2 = 0
    2)
      (f0_0-f1_0**2*f0_0)*r0*q0 + (f0_1-f1_0**2*f0_1)*r3*q0 + (f0_2-f1_0**2*f0_2)*r6*q0
    + (f0_0-f1_1**2*f0_0)*r1*q1 + (f0_1-f1_1**2*f0_1)*r4*q1 + (f0_2-f1_1**2*f0_2)*r7*q1
    + (f0_0-f1_2**2*f0_0)*r2*q2 + (f0_1-f1_2**2*f0_1)*r5*q2 + (f0_2-f1_2**2*f0_2)*r8*q2
    - f1_0*f1_1*f0_0*r0*q1      - f1_0*f1_1*f0_1*r3*q1      - f1_0*f1_1*f0_2*r6*q1
    - f1_0*f1_2*f0_0*r0*q2      - f1_0*f1_2*f0_1*r3*q2      - f1_0*f1_2*f0_2*r6*q2
    - f1_1*f1_2*f0_0*r1*q2      - f1_1*f1_2*f0_1*r4*q2      - f1_1*f1_2*f0_2*r7*q2
    - f1_0*f1_1*f0_0*r1*q0      - f1_0*f1_1*f0_1*r4*q0      - f1_0*f1_1*f0_2*r7*q0
    - f1_0*f1_2*f0_0*r2*q0      - f1_0*f1_2*f0_1*r5*q0      - f1_0*f1_2*f0_2*r8*q0
    - f1_1*f1_2*f0_0*r2*q1      - f1_1*f1_2*f0_1*r5*q1      - f1_1*f1_2*f0_2*r8*q1
    - scm2^2 = 0
    """

    EQUATION = "f0^T R f1 - t^T R f1 - scm1^2 = 0, f0^T R q - f1^T q - scm2^2 = 0"
    # Note: these coefficients will be modified based on the input data (bearings).
    COEFFS_PER_EQ = [[1.0] * 27 + [-1.0], [1.0] * 27 + [-1.0]]
    CONSTRAINT_VALUES = [0.0, 0.0]

    __slots__ = ()

    def get_eqs_info(self, params):
        t, q, R = params["t"], params["q"], params["R"]
        scm1, scm2 = params["scm1"], params["scm2"]

        tb, qb, Rb, scm1b, scm2b = t.block, q.block, R.block, scm1.block, scm2.block
        assert tb == qb == Rb
        blocks = [[tb] * 27 + [scm1b], [tb] * 27 + [scm2b]]

        ti, qi, Ri, scm1i, scm2i = (
            t.block_ids,
            q.block_ids,
            R.block_ids,
            scm1.block_ids,
            scm2.block_ids,
        )
        assert_smaller_idxes(ti, qi)
        assert_smaller_idxes(qi, Ri)

        # fmt: off
        rows = [
            [
             ti[0], ti[0], ti[0],
             ti[1], ti[1], ti[1],
             ti[2], ti[2], ti[2],
             ti[1], ti[1], ti[1],
             ti[2], ti[2], ti[2],
             ti[2], ti[2], ti[2],
             ti[0], ti[0], ti[0],
             ti[0], ti[0], ti[0],
             ti[1], ti[1], ti[1],
             scm1i[0]],
            [qi[0], qi[0], qi[0],
             qi[1], qi[1], qi[1],
             qi[2], qi[2], qi[2],
             qi[1], qi[1], qi[1],
             qi[2], qi[2], qi[2],
             qi[2], qi[2], qi[2],
             qi[0], qi[0], qi[0],
             qi[0], qi[0], qi[0],
             qi[1], qi[1], qi[1],
             scm2i[0]],
        ]
        cols = [
            [
             Ri[0], Ri[1], Ri[2],
             Ri[3], Ri[4], Ri[5],
             Ri[6], Ri[7], Ri[8],
             Ri[0], Ri[1], Ri[2],
             Ri[0], Ri[1], Ri[2],
             Ri[3], Ri[4], Ri[5],
             Ri[3], Ri[4], Ri[5],
             Ri[6], Ri[7], Ri[8],
             Ri[6], Ri[7], Ri[8],
             scm1i[0]],
            [Ri[0], Ri[3], Ri[6],
             Ri[1], Ri[4], Ri[7],
             Ri[2], Ri[5], Ri[8],
             Ri[0], Ri[3], Ri[6],
             Ri[0], Ri[3], Ri[6],
             Ri[1], Ri[4], Ri[7],
             Ri[1], Ri[4], Ri[7],
             Ri[2], Ri[5], Ri[8],
             Ri[2], Ri[5], Ri[8],
             scm2i[0]],
        ]
        # fmt: on
        return blocks, rows, cols

    def compute_coeffs(self, coeffs: np.ndarray, f0: np.ndarray, f1: np.ndarray):
        """Compute the data-dependent coefficients of the constraint."""
        f0_sq_outer_f1_f1sum, f0_ct_outer_f1 = self.aggregate_data_1st_ineq(f0, f1)
        f0sum_f1_sq_outer_f0, f1_ct_outer_f0 = self.aggregate_data_2nd_ineq(f0, f1)

        # view of the coefficients to modify.
        coeffs_ = coeffs[self.idx_first_el : self.idx_last_el]
        # 1st inequality.
        coeffs_[:9] = f0_sq_outer_f1_f1sum
        coeffs_[9:18] = f0_ct_outer_f1
        coeffs_[18:27] = f0_ct_outer_f1
        # 2nd inequality.
        coeffs_[28:37] = f0sum_f1_sq_outer_f0
        coeffs_[37:46] = f1_ct_outer_f0
        coeffs_[46:55] = f1_ct_outer_f0

    @staticmethod
    def aggregate_data_1st_ineq(f0: np.ndarray, f1: np.ndarray):
        n = f0.shape[1]
        assert f0.shape == f1.shape == (3, n)

        f0_sq = f0**2
        f0_sq_outer_f1 = np.einsum("in, jn -> ij", f0_sq, f1)
        f0_sq_outer_f1_f1sum = (f0_sq_outer_f1 - f1.sum(1)[None]).ravel()  # (9,)

        f0_ct = f0 * np.concatenate((f0[1:], f0[:1]))
        f0_ct[1:] = f0_ct[:0:-1]
        f0_ct_outer_f1 = np.einsum("in, jn -> ij", f0_ct, f1).ravel()  # (9,)

        return f0_sq_outer_f1_f1sum, f0_ct_outer_f1

    @staticmethod
    def aggregate_data_2nd_ineq(f0: np.ndarray, f1: np.ndarray):
        n = f0.shape[1]
        assert f0.shape == f1.shape == (3, n)

        f1_sq = f1**2
        f1_sq_outer_f0 = np.einsum("in, jn -> ij", f1_sq, f0)
        f0sum_f1_sq_outer_f0 = (f0.sum(1)[None] - f1_sq_outer_f0).ravel()  # (9,)

        f1_ct = f1 * np.concatenate((f1[1:], f1[:1]))
        f1_ct[1:] = f1_ct[:0:-1]
        f1_ct_outer_f0 = np.einsum("in, jn -> ij", f1_ct, f0).ravel()  # (9,)

        return f0sum_f1_sq_outer_f0, -f1_ct_outer_f0


class Orthogonality(Constraint):
    """R R.T = I and R.T R = I

    r0**2 + r1**2 + r2**2 = 1
    r3**2 + r4**2 + r5**2 = 1
    r6**2 + r7**2 + r8**2 = 1
    r0*r3 + r1*r4 + r2*r5 = 0
    r0*r6 + r1*r7 + r2*r8 = 0
    r3*r6 + r4*r7 + r5*r8 = 0

    r0**2 + r3**2 + r6**2 = 1 -> removed (linear dependence)
    r1**2 + r4**2 + r7**2 = 1
    r2**2 + r5**2 + r8**2 = 1
    r0*r1 + r3*r4 + r6*r7 = 0
    r0*r2 + r3*r5 + r6*r8 = 0
    r1*r2 + r4*r5 + r7*r8 = 0
    """

    EQUATION = "R R.T = I, R.T R = I"
    COEFFS_PER_EQ = [[1.0, 1.0, 1.0]] * 11
    CONSTRAINT_VALUES = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]

    __slots__ = ()

    def get_eqs_info(self, params):
        R = params["R"]
        Rb = R.block
        blocks = [[Rb, Rb, Rb]] * 11
        Ri = R.block_ids
        rows = [
            [Ri[0], Ri[1], Ri[2]],
            [Ri[3], Ri[4], Ri[5]],
            [Ri[6], Ri[7], Ri[8]],
            [Ri[0], Ri[1], Ri[2]],
            [Ri[0], Ri[1], Ri[2]],
            [Ri[3], Ri[4], Ri[5]],
            # [Ri[0], Ri[3], Ri[6]],
            [Ri[1], Ri[4], Ri[7]],
            [Ri[2], Ri[5], Ri[8]],
            [Ri[0], Ri[3], Ri[6]],
            [Ri[0], Ri[3], Ri[6]],
            [Ri[1], Ri[4], Ri[7]],
        ]
        cols = [
            [Ri[0], Ri[1], Ri[2]],
            [Ri[3], Ri[4], Ri[5]],
            [Ri[6], Ri[7], Ri[8]],
            [Ri[3], Ri[4], Ri[5]],
            [Ri[6], Ri[7], Ri[8]],
            [Ri[6], Ri[7], Ri[8]],
            # [Ri[0], Ri[3], Ri[6]],
            [Ri[1], Ri[4], Ri[7]],
            [Ri[2], Ri[5], Ri[8]],
            [Ri[1], Ri[4], Ri[7]],
            [Ri[2], Ri[5], Ri[8]],
            [Ri[2], Ri[5], Ri[8]],
        ]
        return blocks, rows, cols


class DeterminantR(Constraint):
    """det(R)=+1

    This cubic constraint can be equivalently formulated as 9 quadratic constraints:
        h*ri - rj x rk = 0, (i, j, k) = {(0, 1, 2), (2, 3, 1), (3, 1, 2)}
    This can also be derived from the adjoint matrix of R:
        R^T = R^{-1} = (1 / det(R)) adj(R) = (1 / det(R)) cofactor(R)^T,
        i.e. R = cofactor(R);
    thus, if det(R)=+1, the quadratic constraints h * R = cofactor(R) need to hold.

    The corresponding expanded equations with the quadratic terms are:
    h*r0 - r4*r8 + r5*r7 = 0
    h*r1 - r5*r6 + r3*r8 = 0
    h*r2 - r3*r7 + r4*r6 = 0
    h*r3 - r2*r7 + r1*r8 = 0
    h*r4 - r0*r8 + r2*r6 = 0
    h*r5 - r1*r6 + r0*r7 = 0
    h*r6 - r1*r5 + r2*r4 = 0
    h*r7 - r2*r3 + r0*r5 = 0
    h*r8 - r0*r4 + r1*r3 = 0
    """

    EQUATION = "hR = cofactor(R)"
    COEFFS_PER_EQ = [[1.0, -1.0, 1.0]] * 9
    CONSTRAINT_VALUES = [0.0] * 9

    __slots__ = ()

    def get_eqs_info(self, params):
        h, R = params["h"], params["R"]
        hb, Rb = h.block, R.block
        assert hb == Rb
        blocks = [[hb, hb, hb]] * 9
        hi, Ri = h.block_ids, R.block_ids
        assert_smaller_idxes(hi, Ri)
        rows = [
            [hi[0], Ri[4], Ri[5]],
            [hi[0], Ri[5], Ri[3]],
            [hi[0], Ri[3], Ri[4]],
            [hi[0], Ri[2], Ri[1]],
            [hi[0], Ri[0], Ri[2]],
            [hi[0], Ri[1], Ri[0]],
            [hi[0], Ri[1], Ri[2]],
            [hi[0], Ri[2], Ri[0]],
            [hi[0], Ri[0], Ri[1]],
        ]
        cols = [
            [Ri[0], Ri[8], Ri[7]],
            [Ri[1], Ri[6], Ri[8]],
            [Ri[2], Ri[7], Ri[6]],
            [Ri[3], Ri[7], Ri[8]],
            [Ri[4], Ri[8], Ri[6]],
            [Ri[5], Ri[6], Ri[7]],
            [Ri[6], Ri[5], Ri[4]],
            [Ri[7], Ri[3], Ri[5]],
            [Ri[8], Ri[4], Ri[3]],
        ]
        return blocks, rows, cols


class TQDefinition(Constraint):
    """ht - Rq = 0, and hq - R^Tt = 0

    h*t0 - q0*r0 - q1*r1 - q2*r2 = 0
    h*t1 - q0*r3 - q1*r4 - q2*r5 = 0
    h*t2 - q0*r6 - q1*r7 - q2*r8 = 0
    h*q0 - t0*r0 - t1*r3 - t2*r6 = 0
    h*q1 - t0*r1 - t1*r4 - t2*r7 = 0
    h*q2 - t0*r2 - t1*r5 - t2*r8 = 0
    """

    EQUATION = "ht - Rq = 0; hq - R^Tt = 0"
    COEFFS_PER_EQ = [[1.0, -1.0, -1.0, -1.0]] * 6
    CONSTRAINT_VALUES = [0.0] * 6

    __slots__ = ()

    def get_eqs_info(self, params):
        h, t, q, R = params["h"], params["t"], params["q"], params["R"]
        hb, tb, qb, Rb = h.block, t.block, q.block, R.block
        assert hb == tb == qb == Rb
        blocks = [[hb, hb, hb, hb]] * 6
        hi, ti, qi, Ri = h.block_ids, t.block_ids, q.block_ids, R.block_ids
        assert_smaller_idxes(ti, qi)
        assert_smaller_idxes(qi, hi)
        assert_smaller_idxes(hi, Ri)
        rows = [
            [ti[0], qi[0], qi[1], qi[2]],
            [ti[1], qi[0], qi[1], qi[2]],
            [ti[2], qi[0], qi[1], qi[2]],
            [qi[0], ti[0], ti[1], ti[2]],
            [qi[1], ti[0], ti[1], ti[2]],
            [qi[2], ti[0], ti[1], ti[2]],
        ]
        cols = [
            [hi[0], Ri[0], Ri[1], Ri[2]],
            [hi[0], Ri[3], Ri[4], Ri[5]],
            [hi[0], Ri[6], Ri[7], Ri[8]],
            [hi[0], Ri[0], Ri[3], Ri[6]],
            [hi[0], Ri[1], Ri[4], Ri[7]],
            [hi[0], Ri[2], Ri[5], Ri[8]],
        ]
        return blocks, rows, cols


class SkewTQDefinition(Constraint):
    """h[t] - ER^T, and h[q] - R^T E

           - e0*r0 - e1*r1 - e2*r2 = 0
    - h*t2 - e0*r3 - e1*r4 - e2*r5 = 0
    + h*t1 - e0*r6 - e1*r7 - e2*r8 = 0
    + h*t2 - e3*r0 - e4*r1 - e5*r2 = 0
           - e3*r3 - e4*r4 - e5*r5 = 0
    - h*t0 - e3*r6 - e4*r7 - e5*r8 = 0
    - h*t1 - e6*r0 - e7*r1 - e8*r2 = 0
    + h*t0 - e6*r3 - e7*r4 - e8*r5 = 0
           - e6*r6 - e7*r7 - e8*r8 = 0

           - e0*r0 - e3*r3 - e6*r6 = 0
    - h*q2 - e1*r0 - e4*r3 - e7*r6 = 0
    + h*q1 - e2*r0 - e5*r3 - e8*r6 = 0
    + h*q2 - e0*r1 - e3*r4 - e6*r7 = 0
           - e1*r1 - e4*r4 - e7*r7 = 0
    - h*q0 - e2*r1 - e5*r4 - e8*r7 = 0
    - h*q1 - e0*r2 - e3*r5 - e6*r8 = 0
    + h*q0 - e1*r2 - e4*r5 - e7*r8 = 0
           - e2*r2 - e5*r5 - e8*r8 = 0
    """

    EQUATION = "h[t] - ER^T, h[q] - R^T E"
    COEFFS_PER_EQ = [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
    ]
    CONSTRAINT_VALUES = [0.0] * 18

    __slots__ = ()

    def get_eqs_info(self, params):
        h, t, q, R, E = params["h"], params["t"], params["q"], params["R"], params["E"]
        hb, tb, qb, Rb, Eb = h.block, t.block, q.block, R.block, E.block
        assert hb == tb == qb and Rb == Eb
        blocks = [
            [Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [Eb, Eb, Eb],
            [Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [hb, Eb, Eb, Eb],
            [Eb, Eb, Eb],
        ]
        hi, ti, qi, Ri, Ei = (
            h.block_ids,
            t.block_ids,
            q.block_ids,
            R.block_ids,
            E.block_ids,
        )
        assert_smaller_idxes(ti, qi)
        assert_smaller_idxes(qi, hi)
        assert_smaller_idxes(Ei, Ri)
        rows = [
            [Ei[0], Ei[1], Ei[2]],
            [ti[2], Ei[0], Ei[1], Ei[2]],
            [ti[1], Ei[0], Ei[1], Ei[2]],
            [ti[2], Ei[3], Ei[4], Ei[5]],
            [Ei[3], Ei[4], Ei[5]],
            [ti[0], Ei[3], Ei[4], Ei[5]],
            [ti[1], Ei[6], Ei[7], Ei[8]],
            [ti[0], Ei[6], Ei[7], Ei[8]],
            [Ei[6], Ei[7], Ei[8]],
            [Ei[0], Ei[3], Ei[6]],
            [qi[2], Ei[1], Ei[4], Ei[7]],
            [qi[1], Ei[2], Ei[5], Ei[8]],
            [qi[2], Ei[0], Ei[3], Ei[6]],
            [Ei[1], Ei[4], Ei[7]],
            [qi[0], Ei[2], Ei[5], Ei[8]],
            [qi[1], Ei[0], Ei[3], Ei[6]],
            [qi[0], Ei[1], Ei[4], Ei[7]],
            [Ei[2], Ei[5], Ei[8]],
        ]
        cols = [
            [Ri[0], Ri[1], Ri[2]],
            [hi[0], Ri[3], Ri[4], Ri[5]],
            [hi[0], Ri[6], Ri[7], Ri[8]],
            [hi[0], Ri[0], Ri[1], Ri[2]],
            [Ri[3], Ri[4], Ri[5]],
            [hi[0], Ri[6], Ri[7], Ri[8]],
            [hi[0], Ri[0], Ri[1], Ri[2]],
            [hi[0], Ri[3], Ri[4], Ri[5]],
            [Ri[6], Ri[7], Ri[8]],
            [Ri[0], Ri[3], Ri[6]],
            [hi[0], Ri[0], Ri[3], Ri[6]],
            [hi[0], Ri[0], Ri[3], Ri[6]],
            [hi[0], Ri[1], Ri[4], Ri[7]],
            [Ri[1], Ri[4], Ri[7]],
            [hi[0], Ri[1], Ri[4], Ri[7]],
            [hi[0], Ri[2], Ri[5], Ri[8]],
            [hi[0], Ri[2], Ri[5], Ri[8]],
            [Ri[2], Ri[5], Ri[8]],
        ]
        return blocks, rows, cols


class ConvexHullSO3(Constraint):
    """Following [1, Eq. 4.1], [2, Eq. 10], the convex hull of SO(3) is defined by
    the positive semidefinite (PSD) matrices, formed with elements rij of R in R^{3,3}:

    [1 + r00 + r11 + r22, r21 - r12,           r02 - r20,           r10 - r01          ]
    [r21 - r12,           1 + r00 - r11 - r22, r10 + r01,           r02 + r20          ]
    [r02 - r20,           r10 + r01,           1 - r00 + r11 - r22, r21 + r12          ]
    [r10 - r01,           r02 + r20,           r21 + r12,           1 - r00 - r11 + r22]

    We formulate this linear matrix inequality (LMI) in SDPA format as follows: we
    augment the objective matrix X with an additional block diagonal matrix Zc:
        X_augmented = [X  0]
                      [0 Zc]
    we thus ensure that Zc is PSD. Then, we constraint the values of Zc following the
    definition above. Since these constraints are linear, we use an homogenization
    variable "h" to make them quadratic:
        h*z0  - h*r0 - h*r4 - h*r8 = 1
        h*z1  - h*r7 + h*r5        = 0
        h*z2  - h*r2 + h*r6        = 0
        h*z3  - h*r3 + h*r1        = 0
        h*z5  - h*r0 + h*r4 + h*r8 = 1
        h*z6  - h*r3 - h*r1        = 0
        h*z7  - h*r2 - h*r6        = 0
        h*z10 + h*r0 - h*r4 + h*r8 = 1
        h*z11 - h*r7 - h*r5        = 0
        h*z15 + h*r0 + h*r4 - h*r8 = 1

    TODO: use equivalent formulations from [1, Eq. 4.2] or [2, Sec. 1.2]
    [1] Orbitopes, R. Sanyal, F. Sottile and B. Sturmfels, 2011.
    [2] Semidefinite descriptions of the convex hull of rotation matrices,
    J. Saunderson, P. A. Parrilo, A. S. Willsky, 2014.
    """

    EQUATION = "conv SO(3)"
    COEFFS_PER_EQ = [
        [1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, -1.0],
    ]
    CONSTRAINT_VALUES = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]

    __slots__ = ()

    def get_eqs_info(self, params):
        h, R, Zc = params["h"], params["R"], params["Zc"]
        hb, Rb, Zcb = h.block, R.block, Zc.block
        assert hb == Rb == Zcb
        blocks = [
            [hb, hb, hb, hb],
            [hb, hb, hb],
            [hb, hb, hb],
            [hb, hb, hb],
            [hb, hb, hb, hb],
            [hb, hb, hb],
            [hb, hb, hb],
            [hb, hb, hb, hb],
            [hb, hb, hb],
            [hb, hb, hb, hb],
        ]
        hi, Ri, Zci = h.block_ids, R.block_ids, Zc.block_ids
        assert_smaller_idxes(hi, Ri)
        assert_smaller_idxes(Ri, Zci)
        rows = [
            [hi[0], hi[0], hi[0], hi[0]],
            [hi[0], hi[0], hi[0]],
            [hi[0], hi[0], hi[0]],
            [hi[0], hi[0], hi[0]],
            [hi[0], hi[0], hi[0], hi[0]],
            [hi[0], hi[0], hi[0]],
            [hi[0], hi[0], hi[0]],
            [hi[0], hi[0], hi[0], hi[0]],
            [hi[0], hi[0], hi[0]],
            [hi[0], hi[0], hi[0], hi[0]],
        ]
        cols = [
            [Zci[0], Ri[0], Ri[4], Ri[8]],
            [Zci[1], Ri[7], Ri[5]],
            [Zci[2], Ri[2], Ri[6]],
            [Zci[3], Ri[3], Ri[1]],
            [Zci[5], Ri[0], Ri[4], Ri[8]],
            [Zci[6], Ri[3], Ri[1]],
            [Zci[7], Ri[2], Ri[6]],
            [Zci[10], Ri[0], Ri[4], Ri[8]],
            [Zci[11], Ri[7], Ri[5]],
            [Zci[15], Ri[0], Ri[4], Ri[8]],
        ]
        return blocks, rows, cols
