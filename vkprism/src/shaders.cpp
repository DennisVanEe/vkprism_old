#include "shaders.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace prism {

Shaders::Shaders(const Context& context)
{
    const std::string SHADER_ROOT_PATH = "/shaders/";

    for (size_t i = 0; i < m_shaderNames.size(); ++i) {
        const auto    path = SHADER_ROOT_PATH + std::string(m_shaderNames[i]) + ".spv";
        std::ifstream shaderFile(path, std::ios::binary);

        if (!shaderFile.is_open()) {
            throw std::runtime_error("Could not find the spv file at: " + path);
        }

        const std::vector<char> shaderFileData((std::istreambuf_iterator<char>(shaderFile)),
                                               std::istreambuf_iterator<char>());

        m_shaderModules[i] = context.device().createShaderModuleUnique(vk::ShaderModuleCreateInfo{
            .codeSize = shaderFileData.size(),
            .pCode    = reinterpret_cast<const uint32_t*>(shaderFileData.data()),
        });
    }
}

} // namespace prism