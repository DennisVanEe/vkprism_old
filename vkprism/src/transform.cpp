#include "transform.hpp"

#include <algorithm>
#include <array>

namespace prism {

Transform::operator vk::TransformMatrixKHR() const
{
    // GLM stores their matrices column wise while (for some reason) Vulkan's transform matrix is row-wise. So, we
    // first perform a transpose.
    const auto transpose = glm::transpose(m_matrix);
    return vk::TransformMatrixKHR{
        .matrix = std::to_array({
            std::to_array({
                transpose[0][0],
                transpose[0][1],
                transpose[0][2],
                transpose[0][3],
            }),
            std::to_array({
                transpose[1][0],
                transpose[1][1],
                transpose[1][2],
                transpose[1][3],
            }),
            std::to_array({
                transpose[2][0],
                transpose[2][1],
                transpose[2][2],
                transpose[2][3],
            }),
        }),
    };
}

} // namespace prism