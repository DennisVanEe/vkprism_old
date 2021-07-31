#pragma once

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

namespace prism {

class Transform
{
  public:
    bool isIdentity() const { return m_isIdentity; }

    explicit operator vk::TransformMatrixKHR() const;

  private:
    glm::mat4 m_matrix;
    bool      m_isIdentity;
};

} // namespace prism