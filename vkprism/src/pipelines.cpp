#include "pipelines.hpp"

#include <algorithm>
#include <array>
#include <ranges>
#include <vector>

#include <util.hpp>

namespace prism {

Pipelines::Buffers Pipelines::createBuffers(const PipelineParam& param, const GpuAllocator& allocator)
{
    return Buffers{.output = allocator.allocateBuffer(param.outputWidth * param.outputHeight * sizeof(glm::vec3),
                                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                                          vk::BufferUsageFlagBits::eTransferSrc,
                                                      VMA_MEMORY_USAGE_GPU_ONLY)};
}

template <size_t NumSets>
Pipelines::Descriptor<NumSets>::Descriptor(const Context&                                        context,
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

Pipelines::Descriptors Pipelines::createDescriptors(const Context& context, const Scene& scene, const Buffers& buffers)
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

Pipelines::Pipelines(const PipelineParam& param, const Context& context, const GpuAllocator& allocator,
                     const Scene& scene) :
    m_buffers(createBuffers(param, allocator)),
    m_descriptors(createDescriptors(context, scene, m_buffers)),
    m_rtPipeline(context, m_descriptors)
{}

Pipelines::RTPipeline::RTPipeline(const Context& context, const Descriptors& descriptors)
{
    //
    // Create the pipeline layout:

    // For now, each shader will get the same data from the push constant, will fine tune in the future:
    const vk::PushConstantRange pushConstRange{.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR |
                                                             vk::ShaderStageFlagBits::eClosestHitKHR |
                                                             vk::ShaderStageFlagBits::eMissKHR,
                                               .offset = 0,
                                               .size   = sizeof(PushConst)};

    m_layout = context.device().createPipelineLayoutUnique(
        vk::PipelineLayoutCreateInfo{.setLayoutCount         = 1,
                                     .pSetLayouts            = &*descriptors.raygen.setLayout,
                                     .pushConstantRangeCount = 1,
                                     .pPushConstantRanges    = &pushConstRange});

    //
    // Set the shader stages and the groups up:

    // Load all of the shaders first:

    const auto shaderStages = [&]() {
        std::array<vk::PipelineShaderStageCreateInfo, TOTAL_NUM_SHADERS> shaderStages;
        shaderStages[sRAYGEN] = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eRaygenKHR, .module = loadShader(context, sRAYGEN), .pName = "main"};
        shaderStages[sMISS] = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eMissKHR, .module = loadShader(context, sMISS), .pName = "main"};
        shaderStages[sCLOSEST_HIT] = vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eClosestHitKHR,
                                                                       .module = loadShader(context, sCLOSEST_HIT),
                                                                       .pName  = "main"};
        return shaderStages;
    }();
    // Make sure to delete them at the end:
    Defer shaderStageCleanup([&]() {
        for (const auto& stage : shaderStages) {
            context.device().destroyShaderModule(stage.module);
        }
    });

    // Now, specify all of the shader groups:
    enum ShaderGroup : uint32_t
    {
        RaygenIdx,
        MissIdx,
        HitIdx,
        CallableIdx,
    };

    const auto [shaderGroups, numRaygenGroups, numMissGroups, numHitGroups, numCallableGroups] = [&]() {
        const auto raygenGroups = std::to_array({vk::RayTracingShaderGroupCreateInfoKHR{
            .type               = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader      = sRAYGEN,
            .closestHitShader   = VK_SHADER_UNUSED_KHR,
            .anyHitShader       = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        }});
        const auto missGroups   = std::to_array({vk::RayTracingShaderGroupCreateInfoKHR{
            .type               = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader      = sMISS,
            .closestHitShader   = VK_SHADER_UNUSED_KHR,
            .anyHitShader       = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        }});
        const auto hitGroups    = std::to_array({vk::RayTracingShaderGroupCreateInfoKHR{
            .type               = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
            .generalShader      = VK_SHADER_UNUSED_KHR,
            .closestHitShader   = sCLOSEST_HIT,
            .anyHitShader       = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        }});
        const std::array<vk::RayTracingShaderGroupCreateInfoKHR, 0> callableGroups;

        //
        // Kind of messy, but we essentially combine the shader groups:

        std::array<vk::RayTracingShaderGroupCreateInfoKHR,
                   raygenGroups.size() + missGroups.size() + hitGroups.size() + callableGroups.size()>
            shaderGroups;

        auto shaderGroupsItr = shaderGroups.begin();
        std::ranges::copy(raygenGroups, shaderGroupsItr);
        shaderGroupsItr += raygenGroups.size();
        std::ranges::copy(missGroups, shaderGroupsItr);
        shaderGroupsItr += missGroups.size();
        std::ranges::copy(hitGroups, shaderGroupsItr);
        shaderGroupsItr += hitGroups.size();
        std::ranges::copy(callableGroups, shaderGroupsItr);

        return std::make_tuple(shaderGroups, raygenGroups.size(), missGroups.size(), hitGroups.size(),
                               callableGroups.size());
    }();

    const vk::RayTracingPipelineCreateInfoKHR pipelineCreateInfo{
        .stageCount                   = static_cast<uint32_t>(shaderStages.size()),
        .pStages                      = shaderStages.data(),
        .groupCount                   = static_cast<uint32_t>(shaderGroups.size()),
        .pGroups                      = shaderGroups.data(),
        .maxPipelineRayRecursionDepth = 1, // No recursion will be used, instead queues and whatnot...
        .layout                       = *m_layout,
    };

    m_pipeline = [&]() {
        auto result = context.device().createRayTracingPipelinesKHRUnique({}, {}, pipelineCreateInfo);
        switch (result.result) {
        case vk::Result::eSuccess:
        case vk::Result::eOperationDeferredKHR:
        case vk::Result::eOperationNotDeferredKHR:
        case vk::Result::ePipelineCompileRequiredEXT:
            return std::move(result.value[0]);
        }
        vkCall(result.result);
    }();

    //
    // Create the shader binding table:

    const auto& rtPipelineProps = context.properties().get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    const auto  alignedGroupSize =
        alignUp(rtPipelineProps.shaderGroupHandleSize, rtPipelineProps.shaderGroupBaseAlignment);
    const auto sbtSize = static_cast<uint32_t>(shaderGroups.size()) * alignedGroupSize;
}

} // namespace prism