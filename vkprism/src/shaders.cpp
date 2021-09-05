#include "shaders.hpp"

#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

namespace prism {

std::vector<char> loadShaderData(const std::string_view shaderName)
{
    // TODO: use format:
    const auto path = "/shaders/" + std::string(shaderName) + ".spv";

    std::ifstream shaderFile(path, std::ios::binary);
    if (!shaderFile.is_open()) {
        throw std::runtime_error("Could not find the spv file at: " + path);
    }

    return std::vector<char>((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
}

vk::UniqueShaderModule loadShaderUnique(const Context& context, const std::string_view shaderName)
{
    const auto data = loadShaderData(shaderName);
    return context.device().createShaderModuleUnique(vk::ShaderModuleCreateInfo{
        .codeSize = data.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(data.data()),
    });
}

vk::ShaderModule loadShader(const Context& context, const std::string_view shaderName)
{
    const auto data = loadShaderData(shaderName);
    return context.device().createShaderModule(vk::ShaderModuleCreateInfo{
        .codeSize = data.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(data.data()),
    });
}

} // namespace prism