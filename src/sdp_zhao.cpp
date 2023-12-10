#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sdpa_call.h>

namespace py = pybind11;

py::array_t<double> solve(py::array_t<double> C)
{
    py::buffer_info buf = C.request();
    if (buf.ndim != 2 || buf.shape[0] != 9 || buf.shape[1] != 9)
    {
        throw std::runtime_error("Expected a numpy ndarray of shape (9, 9)");
    }
    double *ptr = static_cast<double *>(buf.ptr);

    SDPA npt_problem;
    npt_problem.setParameterType(SDPA::PARAMETER_DEFAULT);

    // number of constraints and blocks
    int mDIM = 7;
    int nBlock = 1;
    npt_problem.inputConstraintNumber(mDIM);
    npt_problem.inputBlockNumber(nBlock);
    npt_problem.inputBlockSize(1, 12);
    npt_problem.inputBlockType(1, SDPA::SDP);

    npt_problem.initializeUpperTriangleSpace();

    for (int i = 1; i <= 6; i++)
    {
        npt_problem.inputCVec(i, 0);
    }
    npt_problem.inputCVec(7, -1);

    // Input F0
    for (int i = 0; i < 9; i++)
    {
        for (int j = i; j < 9; j++)
        {
            npt_problem.inputElement(0, 1, i + 1, j + 1, -ptr[i * 9 + j]);
        }
    }

    // Input F_1 -- F_7
    int e11 = 1;
    int e21 = 2;
    int e31 = 3;
    int e12 = 4;
    int e22 = 5;
    int e32 = 6;
    int e13 = 7;
    int e23 = 8;
    int e33 = 9;
    int t1 = 10;
    int t2 = 11;
    int t3 = 12;

    npt_problem.inputElement(1, 1, e11, e11, -1);
    npt_problem.inputElement(1, 1, e12, e12, -1);
    npt_problem.inputElement(1, 1, e13, e13, -1);
    npt_problem.inputElement(1, 1, t3, t3, 1);
    npt_problem.inputElement(1, 1, t2, t2, 1);

    npt_problem.inputElement(2, 1, e21, e21, -1);
    npt_problem.inputElement(2, 1, e22, e22, -1);
    npt_problem.inputElement(2, 1, e23, e23, -1);
    npt_problem.inputElement(2, 1, t1, t1, 1);
    npt_problem.inputElement(2, 1, t3, t3, 1);

    npt_problem.inputElement(3, 1, e31, e31, -1);
    npt_problem.inputElement(3, 1, e32, e32, -1);
    npt_problem.inputElement(3, 1, e33, e33, -1);
    npt_problem.inputElement(3, 1, t1, t1, 1);
    npt_problem.inputElement(3, 1, t2, t2, 1);

    npt_problem.inputElement(4, 1, e11, e21, -1);
    npt_problem.inputElement(4, 1, e12, e22, -1);
    npt_problem.inputElement(4, 1, e13, e23, -1);
    npt_problem.inputElement(4, 1, t1, t2, -1);

    npt_problem.inputElement(5, 1, e11, e31, -1);
    npt_problem.inputElement(5, 1, e12, e32, -1);
    npt_problem.inputElement(5, 1, e13, e33, -1);
    npt_problem.inputElement(5, 1, t1, t3, -1);

    npt_problem.inputElement(6, 1, e21, e31, -1);
    npt_problem.inputElement(6, 1, e22, e32, -1);
    npt_problem.inputElement(6, 1, e23, e33, -1);
    npt_problem.inputElement(6, 1, t2, t3, -1);

    npt_problem.inputElement(7, 1, t1, t1, -1);
    npt_problem.inputElement(7, 1, t2, t2, -1);
    npt_problem.inputElement(7, 1, t3, t3, -1);

    npt_problem.initializeUpperTriangle();

    // perform optimization
    npt_problem.initializeSolve();
    npt_problem.solve();

    // extract solution
    double *X = npt_problem.getResultYMat(1);

    // corresponding ndarray.
    std::vector<ssize_t> shape = {12, 12};
    return py::array_t<double>(shape, X);
};

PYBIND11_MODULE(sdp_zhao, m1)
{
    m1.doc() = "Zhao's non-minimal essential matrix solver";

    m1.def(
        "solve", &solve,
        "Solve the SDP using Zhao's characterization.\n\n"
        "Args:\n"
        "    C: (9, 9) data cost matrix.\n\n"
        "Returns:\n"
        "    X: (12, 12) SDP's positive-semidefinite matrix solution.");
};
