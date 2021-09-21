// This file contains all of the data structures that are shared between C++ and GLSL.

#ifndef GLSL_INCLUDE
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#endif

#ifdef GLSL_INCLUDE
#define GLM
#else
#define GLM glm::
#endif

#ifndef GLSL_INCLUDE
namespace prism {
#endif

// The perspective camera data:
struct PerspectiveCamera
{
    GLM mat4 cameraToWorld;
    GLM mat4 rasterToCamera;
    // TODO: add support for lens and whatnot...
};

#ifndef GLSL_INCLUDE
} // namespace prism
#endif