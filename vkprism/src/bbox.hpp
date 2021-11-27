#pragma once

#include <numeric>

#include <glm/vec2.hpp>

namespace prism {

template <typename T>
struct BBox2
{
    BBox2() : pmin(std::numeric_limits<T>::min()), pmax(std::numeric_limits<T>::max()) {}
    BBox2(const glm::tvec2<T>& p0, const glm::tvec2<T>& p1) : pmin(glm::min(p0, p1)), pmax(glm::max(p0, p1)) {}

    glm::tvec2<T> diagonal() const { return pmax - pmin; }

    glm::tvec2<T> pmin, pmax;
};

using BBox2d = BBox2<double>;
using BBox2f = BBox2<float>;
using BBox2i = BBox2<int>;
using BBox2u = BBox2<unsigned>;

} // namespace prism