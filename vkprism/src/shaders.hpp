#pragma once

#include <algorithm>
#include <array>

#include <context.hpp>

namespace prism {

// All shaders will have the following entry function name:
constexpr const char* SHADER_ENTRY = "main";

vk::UniqueShaderModule loadShaderUnique(const Context& context, std::string_view shaderName);
vk::ShaderModule loadShader(const Context& context, std::string_view shader);

} // namespace prism