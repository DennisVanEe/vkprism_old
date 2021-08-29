#include "pipeline.hpp"

#include <algorithm>
#include <array>
#include <ranges>
#include <vector>

namespace prism {

Pipeline::Buffers Pipeline::createBuffers(const PipelineParam& param, const GpuAllocator& allocator)
{
    return Buffers{.output = allocator.allocateBuffer(param.outputWidth * param.outputHeight * sizeof(glm::vec3),
                                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                                          vk::BufferUsageFlagBits::eTransferSrc,
                                                      VMA_MEMORY_USAGE_GPU_ONLY)};
}

template <size_t NumSets>
Pipeline::Descriptor<NumSets>::Descriptor(const Context&                                        context,
                                          const std::span<const vk::DescriptorSetLayoutBinding> bindings)
{
    setLayout = context.device().createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()});

    const auto poolSizes = [&]() {
        std::vector<vk::DescriptorPoolSize> poolSizes;

        for (const auto& binding : bindings) {
            auto itr = std::ranges::find_if(
                poolSizes, [&](const auto& poolSize) { return poolSize.type == binding.descriptorType; });
            if (itr == poolSizes.end()) {
                poolSizes.emplace_back(binding.descriptorType, binding.descriptorCount * NumSets);
            } else {
                itr->descriptorCount += binding.descriptorCount * NumSets;
            }
        }

        return poolSizes;
    }();

    pool = context.device().createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo{
        .maxSets = NumSets, .poolSizeCount = static_cast<uint32_t>(poolSizes.size()), .pPoolSizes = poolSizes.data()});

    // Allocate the number of descriptor sets we need:
    std::array<vk::DescriptorSetLayout, NumSets> setLayouts;
    setLayouts.fill(*setLayout);

    auto descriptorSetVec = context.device().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
        .descriptorPool = *pool, .descriptorSetCount = NumSets, .pSetLayouts = setLayouts.data()});

    std::ranges::copy(descriptorSetVec, set.begin());
}

Pipeline::Descriptors Pipeline::createDescriptors(const Context& context, const Scene& scene, const Buffers& buffers)
{
    auto raygen = [&]() {
        Descriptor<1> descriptor(
            context, std::to_array({
                         // TLAS:
                         vk::DescriptorSetLayoutBinding{.binding        = 0,
                                                        .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
                                                        .descriptorCount = 1,
                                                        .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},
                         // Output Buffer:
                         vk::DescriptorSetLayoutBinding{.binding         = 1,
                                                        .descriptorType  = vk::DescriptorType::eStorageBuffer,
                                                        .descriptorCount = 1,
                                                        .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},
                     }));

        const vk::StructureChain<vk::WriteDescriptorSet, vk::WriteDescriptorSetAccelerationStructureKHR> tlasWrite{
            vk::WriteDescriptorSet{.dstSet          = descriptor.set[0],
                                   .dstBinding      = 0,
                                   .dstArrayElement = 0,
                                   .descriptorCount = 1,
                                   .descriptorType  = vk::DescriptorType::eAccelerationStructureKHR},
            vk::WriteDescriptorSetAccelerationStructureKHR{
                .accelerationStructureCount = 1,
                .pAccelerationStructures    = &scene.tlas(),
            },
        };

        const vk::DescriptorBufferInfo outputBufferInfo{.buffer = *buffers.output, .range = VK_WHOLE_SIZE};
        const vk::WriteDescriptorSet   outputWrite{
            .dstSet          = descriptor.set[0],
            .dstBinding      = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo     = &outputBufferInfo,
        };

        context.device().updateDescriptorSets({tlasWrite.get<vk::WriteDescriptorSet>(), outputWrite}, {});

        return descriptor;
    }();

    return Descriptors{
        .raygen = std::move(raygen),
    };
}

Pipeline::Pipeline(const PipelineParam& param, const Context& context, const Shaders& shaders,
                   const GpuAllocator& allocator, const Scene& scene) :
    m_buffers(createBuffers(param, allocator)),
    m_descriptors(createDescriptors(context, scene, m_buffers)),
    m_raytracingPipeline(createRaytracingPipeline(context, m_descriptors, shaders))
{}

Pipeline::RaytracingPipeline Pipeline::createRaytracingPipeline(const Context& context, const Descriptors& descriptors,
                                                                const Shaders& shaders)
{
    enum ShaderStage : size_t
    {
        Raygen,
        Miss,
        ClosestHit,
        COUNT,
    };

    const auto k = shaders.getModule(Shaders::RAYGEN);

    //std::array<vk::PipelineShaderStageCreateInfo, ShaderStage::COUNT> shaderStages;
    //shaderStages[ShaderStage::Raygen] =
    //    vk::PipelineShaderStageCreateInfo{.stage  = vk::ShaderStageFlagBits::eRaygenKHR,
    //                                      .module = shaders.getModule<Shaders::RAYGEN>(),
    //                                      .pName  = "main"};

    return RaytracingPipeline();
}

} // namespace prism