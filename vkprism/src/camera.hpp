#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <allocator.hpp>
#include <bbox.hpp>
#include <context.hpp>
#include <descriptor.hpp>
#include <film.hpp>

namespace prism {

class Camera
{
  public:
    // Returns the uniform data to be sent to the GPU:
    virtual std::vector<std::byte> getCameraShaderData() const = 0;
    // Returns the shader SPIR-V file for the camera:
    virtual std::string getCameraSPVPath() const = 0;

  protected:
};

struct PerspectiveCameraParam
{
    glm::mat4 cameraToWorld;
    float     lensRadius;
    float     focalDistance;
    float     fov;
    BBox2f    screenWindow;
};

class PerspectiveCamera : public Camera
{
  public:
    PerspectiveCamera(const PerspectiveCameraParam& param, const Film& film);

    std::vector<std::byte> getCameraShaderData() const override;
    std::string_view       getCameraSPVPath() const override;

  private:
    float m_lensRadius;
    float m_focalDistance;

    glm::mat4 m_cameraToWorld;
    glm::mat4 m_cameraToScreen;
    glm::mat4 m_screenToRaster; // Goes from screen to raster space
    glm::mat4 m_rasterToScreen;
    glm::mat4 m_rasterToCamera;
};

} // namespace prism