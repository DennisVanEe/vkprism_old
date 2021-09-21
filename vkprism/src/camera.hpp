#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include <context.hpp>

namespace prism {

class Camera
{
  public:
    virtual void updateDescriptorSet(const Context& context, const vk::DescriptorSet& set) const = 0;

  protected:
};

class PerspectiveCamera : public Camera
{
  public:
    PerspectiveCamera(const glm::mat4& cameraToWorld, float fov);

    void updateDescriptorSet(const Context& context, const vk::DescriptorSet& set) const override;

  private:
    glm::mat4 cameraToWorld;
};

} // namespace prism