#include <pybind11/pybind11.h>
#include "DemoClass.hpp"

namespace py = pybind11;

PYBIND11_MODULE(demoClass, m)
{
    py::class_<DemoClass>(m, "DemoClass")
        .def(py::init<int, int>())
        .def("print", &DemoClass::print)
        .def("add", &DemoClass::add)
        .def("multiply", &DemoClass::multiply)
        .def("subtract", &DemoClass::subtract)
        .def("divide", &DemoClass::divide)
        .def("getA", &DemoClass::getA)
        .def("getB", &DemoClass::getB)
        .def("setA", &DemoClass::setA)
        .def("setB", &DemoClass::setB)
        .def("isEven", &DemoClass::isEven)
        .def("calcFibAB", &DemoClass::calcFibAB)
        .def("getMap", &DemoClass::getMap)
        .def("dounut", &DemoClass::doughnut);
}