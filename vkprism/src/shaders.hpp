#pragma once

#include <algorithm>
#include <array>

#include <context.hpp>

namespace prism {

class ShaderFileInfo
{
  public:
    explicit constexpr ShaderFileInfo(uint32_t index, const char* file) : m_file(file), m_index(index) {}

    const char* file() const { return m_file; }

    operator uint32_t() const { return m_index; }

  private:
    const char* m_file;
    uint32_t    m_index;
};

static constexpr auto sRAYGEN      = ShaderFileInfo(0, "raytrace.rgen");
static constexpr auto sMISS        = ShaderFileInfo(1, "raytrace.rmiss");
static constexpr auto sCLOSEST_HIT = ShaderFileInfo(2, "raytrace.rchit");

// All shaders must have an entry function with the following name:
static constexpr auto SHADER_ENTRY = "main";

static constexpr size_t TOTAL_NUM_SHADERS = 3;

vk::UniqueShaderModule loadShaderUnique(const Context& context, std::string_view shaderName);
inline vk::UniqueShaderModule loadShaderUnique(const Context& context, const ShaderFileInfo& shaderInfo)
{
    return loadShaderUnique(context, shaderInfo.file());
}

vk::ShaderModule loadShader(const Context& context, std::string_view shader);
inline vk::ShaderModule loadShader(const Context& context, const ShaderFileInfo& shaderInfo)
{
    return loadShader(context, shaderInfo.file());
}

} // namespace prism