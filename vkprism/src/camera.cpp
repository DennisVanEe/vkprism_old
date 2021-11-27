#include "camera.hpp"

// Contains the shared structs that we have to pass along:
#include <shaders/cameras/perspective.hpp>

#include <glm/gtx/transform.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec4.hpp>

#include <array>

namespace prism {

// Instead of using glm's perspective matrix, we will use pbrt's (note that FOV is in degrees):
glm::mat4 createPerspectiveMat(float fov, float near, float far)
{
    // clang-format off

    // Easier to understand if written out as rows, so we have to perform a tranpose:

    const auto perspDiv = glm::transpose(glm::mat4(
        glm::vec4(1.f, 0.f, 0.f, 0.f),
        glm::vec4(0.f, 1.f, 0.f, 0.f),
        glm::vec4(0.f, 0.f, far / (far - near), -far * near / (far - near)),
        glm::vec4(0.f, 0.f, 1.f, 0.f)));

    // clang-format on

    const auto invTan = 1.f / std::tan(glm::radians(fov) / 2.f);
    return glm::scale(glm::vec3(invTan, invTan, 1.f)) * perspDiv;
}

prism::PerspectiveCamera::PerspectiveCamera(const PerspectiveCameraParam& param, const Film& film) :
    m_lensRadius(param.lensRadius),
    m_focalDistance(param.focalDistance),
    m_cameraToWorld(param.cameraToWorld),
    m_cameraToScreen(createPerspectiveMat(param.fov, 1e-2f, 1000.f))
{
    m_screenToRaster = [&]() {
        const auto screenWindowDiag = param.screenWindow.diagonal();
        const auto res              = film.getResolution();

        // First we construct the matrix that goes from screen space to raster space:
        return glm::scale(glm::vec3(res.x, res.y, 1.f)) *
               glm::scale(glm::vec3(1.f / screenWindowDiag.x, 1.f / screenWindowDiag.y, 1.f)) *
               glm::translate(glm::vec3(-param.screenWindow.pmin.x, -param.screenWindow.pmax.y, 0.f));
    }();

    m_rasterToScreen = glm::inverse(m_screenToRaster);
    m_rasterToCamera = glm::inverse(m_cameraToScreen) * m_rasterToScreen;
}

std::vector<std::byte> PerspectiveCamera::getCameraShaderData() const
{
    const shader::PerspectiveCamera shaderData{
        .rasterToCamera = m_rasterToCamera,
        .cameraToWorld  = m_cameraToWorld,
        .lensRadius     = m_lensRadius,
        .focalDistance  = m_focalDistance,
    };

    std::vector<std::byte> shaderDataBytes(sizeof(shader::PerspectiveCamera));
    std::memcpy(shaderDataBytes.data(), &shaderData, sizeof(shader::PerspectiveCamera));
    return shaderDataBytes;
}

std::string_view PerspectiveCamera::getCameraSPVPath() const {}

} // namespace prism