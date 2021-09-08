#include "sbt.hpp"

namespace prism {

template <typename T>
void ShaderBindingTableBuilder::addData(ShaderGroup shaderGroup, uint32_t groupIndex, const T& data)
{
    const auto shaderGroupIdx = static_cast<size_t>(shaderGroup);

    // First, allocate space for the data:
    const auto* dataPtr                 = reinterpret_cast<const std::byte*>(&data);
    m_datas[shaderGroupIdx][groupIndex] = SBTData(dataPtr, dataPtr + sizeof(T));
}

ShaderBindingTable::ShaderBindingTable(const Context& context, const GpuAllocator& gpuAllocator,
                                       const vk::Pipeline&                        pipeline,
                                       const vk::RayTracingPipelineCreateInfoKHR& createInfo,
                                       const ShaderBindingTableBuilder&           builder)
{
    const auto collectIndicesFn = [&](const vk::ShaderStageFlags& flag) {
        std::vector<uint32_t> indices;
        for (uint32_t i = 0; i < createInfo.groupCount; ++i) {
            if (createInfo.pGroups[i].stage == flag) {
                indices.emplace_back(i);
            }
        }
        return indices;
    };

    const auto groupIndices = std::to_array({collectIndicesFn(vk::RayTracingShaderGroupTypeKHR::e)
        })

    // First, we have to group the shader groups together into their respective array indices:
    const auto groupIndices = [&]() {
        
        const auto collectIndicesFn = [&](const vk::ShaderStageFlags& flag) {
            std::vector<uint32_t i> 
            for (uint32_t i = 0; i < createInfo.groupCount; ++i) {
            
            }
        };


    }();
}

} // namespace prism