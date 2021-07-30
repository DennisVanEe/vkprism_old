#pragma once

#include <tuple>

#include <glm/gtx/quaternion.hpp>
#include <glm/vec3.hpp>

namespace prism {

// Returns -1 if v < 0, returns 1 if v > 0, and returns 0 if v = 0
template <typename T>
int sign(T v)
{
    return (static_cast<T>(0) < v) - (v < static_cast<T>(0));
}

// Performs a decomposition into a T * R * S (where R is a quaternion).
std::tuple<glm::vec3, glm::quat, glm::vec3> decompose(const glm::mat4& mat);

} // namespace prism