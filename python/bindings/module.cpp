#include <pybind11/pybind11.h>

#include "bindings.h"

PYBIND11_MODULE(ChromaPrint3D, m) {
    m.doc() = "ChromaPrint3D core bindings";

    ChromaPrint3D::pybind::BindCommon(m);
    ChromaPrint3D::pybind::BindColorDB(m);
    ChromaPrint3D::pybind::BindImgProc(m);
    ChromaPrint3D::pybind::BindMatch(m);
    ChromaPrint3D::pybind::BindGeo(m);
    ChromaPrint3D::pybind::BindCalib(m);
}
