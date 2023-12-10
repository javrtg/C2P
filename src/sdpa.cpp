#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sdpa_call.h>

namespace py = pybind11;

void processVec(
    SDPA &self,
    const py::array_t<double> &vec,
    void (SDPA::*inputVec)(int, double))
{
    py::buffer_info buf = vec.request();
    const double *ptr = static_cast<double *>(buf.ptr);

    if (buf.ndim != 1)
    {
        throw std::runtime_error("cvec should be a 1-dimensional array.");
    }

    const int mDIM = self.getConstraintNumber();
    if (buf.shape[0] != mDIM)
    {
        throw std::runtime_error("cvec should have the same length as the number of constraints.");
    }

    for (int i = 1; i <= mDIM; i++)
    {
        (self.*inputVec)(i, ptr[i - 1]);
    }
};

void processInitMat(
    SDPA &self,
    const py::array_t<int> &arr_l,
    const py::array_t<int> &arr_i,
    const py::array_t<int> &arr_j,
    const py::array_t<double> &arr_values,
    void (SDPA::*inputInitMat)(int, int, int, double))
{
    py::buffer_info buf_l = arr_l.request();
    py::buffer_info buf_i = arr_i.request();
    py::buffer_info buf_j = arr_j.request();
    py::buffer_info buf_values = arr_values.request();

    if (buf_l.ndim != 1 || buf_i.ndim != 1 || buf_j.ndim != 1 || buf_values.ndim != 1)
    {
        throw std::runtime_error("All arrays should be 1-dimensional.");
    }

    const int len_l = buf_l.shape[0];
    const int len_i = buf_i.shape[0];
    const int len_j = buf_j.shape[0];
    const int len_values = buf_values.shape[0];

    if (len_l != len_i || len_l != len_j || len_l != len_values)
    {
        throw std::runtime_error("All arrays should have the same length.");
    }

    const int *ptr_l = static_cast<int *>(buf_l.ptr);
    const int *ptr_i = static_cast<int *>(buf_i.ptr);
    const int *ptr_j = static_cast<int *>(buf_j.ptr);
    const double *ptr_values = static_cast<double *>(buf_values.ptr);

    for (int idx = 0; idx < len_l; idx++)
    {
        (self.*inputInitMat)(ptr_l[idx], ptr_i[idx], ptr_j[idx], ptr_values[idx]);
    }
};

PYBIND11_MODULE(sdpa, m)
{
    py::class_<SDPA> sdpa_(m, "SDPA");

    // see http://www.gnu-darwin.org/distfiles/sdpa/sdpa.6.2.0.manual.pdf
    sdpa_.def(py::init<>())
        .def("setParameterType", &SDPA::setParameterType)
        .def("inputConstraintNumber", &SDPA::inputConstraintNumber)
        .def("inputBlockNumber", &SDPA::inputBlockNumber)
        .def("inputBlockSize", &SDPA::inputBlockSize)
        .def("inputBlockType", &SDPA::inputBlockType)
        .def("inputCVec", &SDPA::inputCVec)
        .def("inputElement", &SDPA::inputElement)
        .def(
            "inputAllCVec",
            [](SDPA &self, const py::array_t<double> &cvec)
            {
                processVec(self, cvec, &SDPA::inputCVec);
            },
            py::arg("cvec"))
        .def(
            "inputAllElements",
            [](SDPA &self,
               const py::array_t<int> &arr_k,
               const py::array_t<int> &arr_l,
               const py::array_t<int> &arr_i,
               const py::array_t<int> &arr_j,
               const py::array_t<double> &arr_values)
            {
                py::buffer_info buf_k = arr_k.request();
                py::buffer_info buf_l = arr_l.request();
                py::buffer_info buf_i = arr_i.request();
                py::buffer_info buf_j = arr_j.request();
                py::buffer_info buf_values = arr_values.request();

                if (buf_k.ndim != 1 || buf_l.ndim != 1 || buf_i.ndim != 1 || buf_j.ndim != 1 || buf_values.ndim != 1)
                {
                    throw std::runtime_error("All arrays should be 1-dimensional.");
                }

                const int len_k = buf_k.shape[0];
                const int len_l = buf_l.shape[0];
                const int len_i = buf_i.shape[0];
                const int len_j = buf_j.shape[0];
                const int len_values = buf_values.shape[0];

                if (len_k != len_l || len_k != len_i || len_k != len_j || len_k != len_values)
                {
                    throw std::runtime_error("All arrays should have the same length.");
                }

                const int *ptr_k = static_cast<int *>(buf_k.ptr);
                const int *ptr_l = static_cast<int *>(buf_l.ptr);
                const int *ptr_i = static_cast<int *>(buf_i.ptr);
                const int *ptr_j = static_cast<int *>(buf_j.ptr);
                const double *ptr_values = static_cast<double *>(buf_values.ptr);

                for (int idx = 0; idx < len_k; idx++)
                {
                    self.inputElement(ptr_k[idx], ptr_l[idx], ptr_i[idx], ptr_j[idx], ptr_values[idx]);
                }
            },
            py::arg("constraint_indices"), py::arg("block_indices"),
            py::arg("row_indices"), py::arg("col_indices"), py::arg("values"))
        .def("initializeUpperTriangleSpace", &SDPA::initializeUpperTriangleSpace)
        .def("initializeUpperTriangle", &SDPA::initializeUpperTriangle)
        .def("initializeSolve", &SDPA::initializeSolve)
        .def("solve", &SDPA::solve)
        // Section 10.3
        .def(
            "getResultXVec",
            [](SDPA &self) -> py::array_t<double>
            {
                // raw pointer to the primal vector.
                double *primal_vec_els = self.getResultXVec();
                // its shape is determined by the number of constraints.
                const int mDIM = self.getConstraintNumber();
                const std::vector<ssize_t> shape = {mDIM};
                // corresponding ndarray without copying.
                return py::array_t<double>(shape, primal_vec_els);
            })
        .def(
            "getResultXMat",
            [](SDPA &self, const int l) -> py::array_t<double>
            {
                // raw pointer to the l-th block of the primal matrix.
                double *primal_mat_els = self.getResultXMat(l);
                const int blocksize = self.getBlockSize(l);
                const std::vector<ssize_t> shape = {blocksize, blocksize};
                return py::array_t<double>(shape, primal_mat_els);
            },
            py::arg("block"))
        .def(
            "getResultYMat",
            [](SDPA &self, const int l) -> py::array_t<double>
            {
                // raw pointer to the l-th block of the dual matrix.
                double *dual_mat_els = self.getResultYMat(l);
                const int blocksize = self.getBlockSize(l);
                const std::vector<ssize_t> shape = {blocksize, blocksize};
                return py::array_t<double>(shape, dual_mat_els);
            },
            py::arg("block"))

        .def("getPrimalObj", &SDPA::getPrimalObj)
        .def("getDualObj", &SDPA::getDualObj)
        .def("getPrimalError", &SDPA::getPrimalError)
        .def("getDualError", &SDPA::getDualError)
        .def("getIteration", &SDPA::getIteration)
        .def("getDualityGap", &SDPA::getDualityGap)

        .def("getConstraintNumber", &SDPA::getConstraintNumber)
        .def("getBlockNumber", &SDPA::getBlockNumber)
        .def("getBlockSize", &SDPA::getBlockSize, py::arg("block"))

        // Section 10.4
        .def("inputInitXVec", &SDPA::inputInitXVec)
        .def("inputInitXMat", &SDPA::inputInitXMat)
        .def("inputInitYMat", &SDPA::inputInitYMat)
        .def(
            "inputInitAllXVec",
            [](SDPA &self, const py::array_t<double> &xvec)
            {
                processVec(self, xvec, &SDPA::inputInitXVec);
            },
            py::arg("xvec"))
        .def(
            "inputInitAllXMat",
            [](SDPA &self,
               const py::array_t<int> &arr_l,
               const py::array_t<int> &arr_i,
               const py::array_t<int> &arr_j,
               const py::array_t<double> &arr_values)
            {
                processInitMat(self, arr_l, arr_i, arr_j, arr_values, &SDPA::inputInitXMat);
            },
            py::arg("block_indices"), py::arg("row_indices"), py::arg("col_indices"), py::arg("values"))
        .def(
            "inputInitAllYMat",
            [](SDPA &self,
               const py::array_t<int> &arr_l,
               const py::array_t<int> &arr_i,
               const py::array_t<int> &arr_j,
               const py::array_t<double> &arr_values)
            {
                processInitMat(self, arr_l, arr_i, arr_j, arr_values, &SDPA::inputInitYMat);
            },
            py::arg("block_indices"), py::arg("row_indices"), py::arg("col_indices"), py::arg("values"))
        .def("terminate", &SDPA::terminate);

    py::enum_<SDPA::ParameterType>(sdpa_, "ParameterType")
        .value("PARAMETER_DEFAULT", SDPA::ParameterType::PARAMETER_DEFAULT)
        .value("PARAMETER_UNSTABLE_BUT_FAST", SDPA::ParameterType::PARAMETER_UNSTABLE_BUT_FAST)
        .value("PARAMETER_STABLE_BUT_SLOW", SDPA::ParameterType::PARAMETER_STABLE_BUT_SLOW)
        .export_values();

    py::enum_<SDPA::ConeType>(sdpa_, "ConeType")
        .value("SDP", SDPA::ConeType::SDP)
        .value("SOCP", SDPA::ConeType::SOCP)
        .value("LP", SDPA::ConeType::LP)
        .export_values();
};