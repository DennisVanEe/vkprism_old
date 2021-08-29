#pragma once

#include <array>
#include <algorithm>

#include <context.hpp>

namespace prism {

class Shaders
{
  public:
    static constexpr std::string_view RAYGEN      = "raygen.glsl.comp";
    static constexpr std::string_view MISS        = "raymiss.glsl.comp";
    static constexpr std::string_view CLOSEST_HIT = "rayclosesthit.glsl.comp";

  public:
    Shaders(const Context& context);

    const vk::ShaderModule& getModule(std::string_view shaderName) const
    {
        const auto idx = getShaderIndex(shaderName);
        return *m_shaderModules[idx];
    }

  private:
    static constexpr size_t getShaderIndex(std::string_view shaderName)
    {
        const auto itr = std::ranges::find(m_shaderNames, shaderName);
        assert(itr != m_shaderNames.end() && "Invalid shader name specified");
        return std::distance(m_shaderNames.begin(), itr);
    }
    static constexpr std::array<std::string_view, 3> m_shaderNames = {RAYGEN, MISS, CLOSEST_HIT};

  private:
    std::array<vk::UniqueShaderModule, m_shaderNames.size()> m_shaderModules;
};

} // namespace prism