#include "math.hpp"

namespace prism {
std::tuple<glm::vec3, glm::quat, glm::vec3> decompose(const glm::mat4& mat)
{
    // GLM is column major, so we have:
    const auto translation = glm::vec3(mat[3]);
    const auto scale = glm::vec3(glm::length(glm::vec3(mat[0])), glm::length(glm::vec3(mat[1])),
                                 sign(glm::determinant(mat)) * glm::length(glm::vec3(mat[2])));
    const auto rotation = glm::quat_cast(glm::mat3(mat[0] / scale[0], mat[1] / scale[1], mat[2] / scale[2]));

    return std::make_tuple(translation, rotation, scale);
}
} // namespace prism