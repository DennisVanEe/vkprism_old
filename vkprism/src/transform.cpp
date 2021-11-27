#include "transform.hpp"

#include <algorithm>
#include <array>

#include <glm/gtc/matrix_access.hpp>

namespace prism {

Transform::operator vk::TransformMatrixKHR() const
{
    // GLM stores their matrices column wise while (for some reason) Vulkan's transform matrix is row-wise. So, we
    // first perform a transpose.
    const auto row0 = row(m_matrix, 0);
    const auto row1 = row(m_matrix, 1);
    const auto row2 = row(m_matrix, 2);

    return vk::TransformMatrixKHR{
        .matrix = std::to_array({
            std::to_array({row0[0], row0[1], row0[2], row0[3]}),
            std::to_array({row1[0], row1[1], row1[2], row1[3]}),
            std::to_array({row2[0], row2[1], row2[2], row2[3]}),
        }),
    };
}

} // namespace prism