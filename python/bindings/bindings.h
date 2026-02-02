#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ChromaPrint3D::pybind {

void BindCommon(pybind11::module_& m);
void BindColorDB(pybind11::module_& m);
void BindImgProc(pybind11::module_& m);
void BindMatch(pybind11::module_& m);
void BindGeo(pybind11::module_& m);
void BindCalib(pybind11::module_& m);

} // namespace ChromaPrint3D::pybind
