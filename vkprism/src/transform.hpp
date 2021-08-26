#pragma once

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

namespace prism {

class Transform
{
  public:
    // TODO: check that the matrix can be decomposed into TRS:
    Transform(const glm::mat4& mat) : m_matrix(mat) {}

    explicit operator vk::TransformMatrixKHR() const;

  private:
    glm::mat4 m_matrix;
};

} // namespace prism