#pragma once

#include <algorithm>
#include <array>

#include <context.hpp>

namespace prism {

// For convenience, all shaders names are defined here:
static constexpr const char* SHADER_FILE_RAYGEN      = "raygen.glsl.comp";
static constexpr const char* SHADER_FILE_MISS        = "raymiss.glsl.comp";
static constexpr const char* SHADER_FILE_CLOSEST_HIT = "rayclosesthit.glsl.comp";

vk::UniqueShaderModule loadShaderUnique(const Context& context, std::string_view shaderName);
vk::ShaderModule       loadShader(const Context& context, std::string_view shader);

} // namespace prism