// clang-format off

#pragma once

#ifdef __cplusplus
#define GLM glm::
#else
#define GLM
#endif

#ifdef __cplusplus

#include <glm/mat4x4.hpp>

namespace prism { 
namespace shader {
#endif

struct PerspectiveCamera
{
    GLM mat4 rasterToCamera;
    GLM mat4 cameraToWorld;
    float    lensRadius;
    float    focalDistance;
};

#ifdef __cplusplus
}
}
#endif

// clang-format on