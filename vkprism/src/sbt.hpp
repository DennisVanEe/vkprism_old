#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include <allocator.hpp>
#include <context.hpp>

namespace prism {

enum class ShaderGroup : size_t
{
    Raygen,
    Miss,
    Hit,
    Callable,
};

class ShaderBindingTable;

class ShaderBindingTableBuilder
{
  public:
    ShaderBindingTableBuilder() = default;

    // You can embed data into the SBT:
    template <typename T>
    void addData(ShaderGroup shaderGroup, uint32_t groupIndex, const T& data);

  private:
    friend class ShaderBindingTable;
    using SBTData = std::vector<std::byte>;

    std::array<std::unordered_map<uint32_t, SBTData>, 4> m_datas;
};

class ShaderBindingTable
{
  public:
    ShaderBindingTable(const Context& context, const GpuAllocator& gpuAllocator, const vk::Pipeline& pipeline,
                       const vk::RayTracingPipelineCreateInfoKHR& createInfo, const ShaderBindingTableBuilder& builder);

  public:
};

} // namespace prism