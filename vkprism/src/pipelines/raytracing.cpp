#include "raytracing.hpp"

#include <shaders.hpp>

namespace prism {
namespace pipeline {

RayTracing::RayTracing(const Context& context, const Param& param, const GPUAllocator& gpuAllocator) :
    m_sceneInfoDesc(createSceneInfoDesc(context, param)),
    m_outputBuffersDesc(createOuputBufferDesc(context, param)),
    m_descriptorSets(std::to_array({m_sceneInfoDesc.set, m_outputBuffersDesc.set})),
    m_pipelineLayout([&]() {
        // Create the pipeline layout:

        const auto descriptorSetLayouts = std::to_array({*m_sceneInfoDesc.setLayout, *m_outputBuffersDesc.setLayout});

        return context.device().createPipelineLayoutUnique(
            vk::PipelineLayoutCreateInfo{.setLayoutCount         = descriptorSetLayouts.size(),
                                         .pSetLayouts            = descriptorSetLayouts.data(),
                                         .pushConstantRangeCount = 0, // no push constants for now
                                         .pPushConstantRanges    = nullptr});
    }())
{

    //
    // Set the shader stages and the groups up:
    //

    // Load all of the shaders first:

    const auto shaderStages =
        std::to_array({vk::PipelineShaderStageCreateInfo{.stage  = vk::ShaderStageFlagBits::eRaygenKHR,
                                                         .module = loadShader(context, param.scene.cameraSPVPath()),
                                                         .pName  = SHADER_ENTRY},
                       vk::PipelineShaderStageCreateInfo{.stage  = vk::ShaderStageFlagBits::eMissKHR,
                                                         .module = loadShader(context, "raytrace.rmiss"),
                                                         .pName  = SHADER_ENTRY},
                       vk::PipelineShaderStageCreateInfo{.stage  = vk::ShaderStageFlagBits::eClosestHitKHR,
                                                         .module = loadShader(context, "raytrace.rchit"),
                                                         .pName  = SHADER_ENTRY}});

    // Make sure to delete the shader stages when we leave the function:
    Defer shaderStageCleanup([&]() {
        for (const auto& stage : shaderStages) {
            context.device().destroyShaderModule(stage.module);
        }
    });

    const auto [shaderGroups, numMissGroups, numHitGroups, numCallableGroups] = [&]() {
        const auto raygenGroup = vk::RayTracingShaderGroupCreateInfoKHR{
            .type               = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader      = 0,
            .closestHitShader   = VK_SHADER_UNUSED_KHR,
            .anyHitShader       = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };

        const auto missGroups = std::to_array({vk::RayTracingShaderGroupCreateInfoKHR{
            .type               = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader      = 1,
            .closestHitShader   = VK_SHADER_UNUSED_KHR,
            .anyHitShader       = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        }});
        const auto hitGroups  = std::to_array({vk::RayTracingShaderGroupCreateInfoKHR{
            .type               = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
            .generalShader      = VK_SHADER_UNUSED_KHR,
            .closestHitShader   = 2,
            .anyHitShader       = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        }});
        const std::array<vk::RayTracingShaderGroupCreateInfoKHR, 0> callableGroups;

        //
        // Kind of messy, but we essentially combine the shader groups:
        const uint32_t numMissGroups     = missGroups.size();
        const uint32_t numHitGroups      = hitGroups.size();
        const uint32_t numCallableGroups = callableGroups.size();

        std::array<vk::RayTracingShaderGroupCreateInfoKHR, 1 + numMissGroups + numHitGroups + numCallableGroups>
            shaderGroups;

        shaderGroups[0] = raygenGroup;

        auto shaderGroupsItr = shaderGroups.begin() + 1;
        std::ranges::copy(missGroups, shaderGroupsItr);
        shaderGroupsItr += missGroups.size();
        std::ranges::copy(hitGroups, shaderGroupsItr);
        shaderGroupsItr += hitGroups.size();
        std::ranges::copy(callableGroups, shaderGroupsItr);

        return std::make_tuple(shaderGroups, numMissGroups, numHitGroups, numCallableGroups);
    }();

    auto pipeline = [&]() {
        const vk::RayTracingPipelineCreateInfoKHR pipelineCreateInfo{
            .stageCount                   = static_cast<uint32_t>(shaderStages.size()),
            .pStages                      = shaderStages.data(),
            .groupCount                   = static_cast<uint32_t>(shaderGroups.size()),
            .pGroups                      = shaderGroups.data(),
            .maxPipelineRayRecursionDepth = 1, // No recursion will be used, instead queues and whatnot...
            .layout                       = *m_pipelineLayout,
        };

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
    //

    const auto& rtPipelineProps = context.properties().get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    // The aligned handle size (note this may change for each class of shader groups if we embed our own data):
    const auto handleSize        = rtPipelineProps.shaderGroupHandleSize;
    const auto handleSizeAligned = alignUp(handleSize, rtPipelineProps.shaderGroupHandleAlignment);
    const auto raygenHandleSize =
        alignUp(handleSizeAligned, rtPipelineProps.shaderGroupBaseAlignment); // only 1 raygen shader...

    //
    // Set all of the shader ranges. Note that the first handle of each collection of handles (raygen, miss,
    // hit, etc.) has to be aligned to shaderGroupBaseAlignment, while each handle itself has to be aligned to
    // shaderGroupHandleAlignment.

    m_raygenAddrRegion = vk::StridedDeviceAddressRegionKHR{.stride = raygenHandleSize, .size = raygenHandleSize};
    m_missAddrRegion   = vk::StridedDeviceAddressRegionKHR{
        .stride = handleSizeAligned,
        .size   = alignUp(handleSizeAligned * numMissGroups, rtPipelineProps.shaderGroupBaseAlignment),
    };
    m_hitAddrRegion = vk::StridedDeviceAddressRegionKHR{
        .stride = handleSizeAligned,
        .size   = alignUp(handleSizeAligned * numHitGroups, rtPipelineProps.shaderGroupBaseAlignment),
    };
    m_callableAddrRegion = vk::StridedDeviceAddressRegionKHR{
        .stride = handleSizeAligned,
        .size   = alignUp(handleSizeAligned * numCallableGroups, rtPipelineProps.shaderGroupBaseAlignment),
    };

    //
    // Allocate memory for the SBT:

    const auto sbtSize =
        m_raygenAddrRegion.size + m_missAddrRegion.size + m_hitAddrRegion.size + m_callableAddrRegion.size;

    m_sbtBuffer = gpuAllocator.allocateBuffer(sbtSize,
                                              vk::BufferUsageFlagBits::eTransferDst |
                                                  vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                                  vk::BufferUsageFlagBits::eShaderBindingTableKHR,
                                              VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Assign the device addresses:
    const auto sbtAddress = m_sbtBuffer.deviceAddress(context.device());

    m_raygenAddrRegion.deviceAddress   = sbtAddress;
    m_missAddrRegion.deviceAddress     = sbtAddress + m_raygenAddrRegion.size;
    m_hitAddrRegion.deviceAddress      = sbtAddress + m_raygenAddrRegion.size + m_hitAddrRegion.size;
    m_callableAddrRegion.deviceAddress = 0; // no callables yet...

    //
    // Copy the the vulkan handles to our SBT:
    {
        const auto shaderHandles = context.device().getRayTracingShaderGroupHandlesKHR<std::byte>(
            *pipeline, 0, shaderGroups.size(), shaderGroups.size() * handleSize);
        const auto shaderHandlesSpan = std::span(shaderHandles);

        auto*      sbtBufferMapped = m_sbtBuffer.map<std::byte>();
        const auto copySBTHandleFn = [&](size_t sbtStartAddr, size_t sbtStride, size_t handleStartIndex,
                                         size_t handleCount) {
            for (size_t i = 0; i < handleCount; ++i) {
                std::ranges::copy(shaderHandlesSpan.subspan(handleStartIndex + i * handleSize, handleSize),
                                  sbtBufferMapped + sbtStartAddr + (i * sbtStride));
            }
        };

        size_t sbtStartAddr     = 0;
        size_t handleStartIndex = 0;

        copySBTHandleFn(sbtStartAddr, m_raygenAddrRegion.stride, handleStartIndex, 1); // raygen

        sbtStartAddr += m_raygenAddrRegion.size;
        handleStartIndex += 1;
        copySBTHandleFn(sbtStartAddr, m_missAddrRegion.stride, handleStartIndex, numMissGroups); // miss

        sbtStartAddr += m_missAddrRegion.size;
        handleStartIndex += numMissGroups;
        copySBTHandleFn(sbtStartAddr, m_hitAddrRegion.stride, handleStartIndex, numHitGroups); // hit

        sbtStartAddr += m_hitAddrRegion.size;
        handleStartIndex += numHitGroups;
        copySBTHandleFn(sbtStartAddr, m_callableAddrRegion.stride, handleStartIndex,
                        numCallableGroups); // callable

        m_sbtBuffer.unmap();
    }
}

Descriptor RayTracing::createSceneInfoDesc(const Context& context, const Param& param)
{
    Descriptor descriptor(
        context,
        std::to_array({// TLAS:
                       vk::DescriptorSetLayoutBinding{.binding         = 0,
                                                      .descriptorType  = vk::DescriptorType::eAccelerationStructureKHR,
                                                      .descriptorCount = 1,
                                                      .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR}}));

    // Write the descriptor set:

    const vk::StructureChain<vk::WriteDescriptorSet, vk::WriteDescriptorSetAccelerationStructureKHR> tlasWrite{
        vk::WriteDescriptorSet{.dstSet          = descriptor.set,
                               .dstBinding      = 0,
                               .dstArrayElement = 0,
                               .descriptorCount = 1,
                               .descriptorType  = vk::DescriptorType::eAccelerationStructureKHR},
        vk::WriteDescriptorSetAccelerationStructureKHR{
            .accelerationStructureCount = 1,
            .pAccelerationStructures    = &param.scene.tlas(),
        },
    };

    context.device().updateDescriptorSets(tlasWrite.get<vk::WriteDescriptorSet>(), {});

    return descriptor;
}

Descriptor RayTracing::createOuputBufferDesc(const Context& context, const Param& param)
{
    Descriptor descriptor(
        context,
        std::to_array({// Beauty output buffer:
                       vk::DescriptorSetLayoutBinding{.binding         = 0,
                                                      .descriptorType  = vk::DescriptorType::eStorageBuffer,
                                                      .descriptorCount = 1,
                                                      .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR}}));

    const vk::DescriptorBufferInfo beautyOutputBufferInfo{.buffer = param.beautyOutputBuffer, .range = VK_WHOLE_SIZE};

    const vk::WriteDescriptorSet beautyOutputWrite{
        .dstSet          = descriptor.set,
        .dstBinding      = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType  = vk::DescriptorType::eStorageBuffer,
        .pBufferInfo     = &beautyOutputBufferInfo,
    };

    context.device().updateDescriptorSets(beautyOutputWrite, {});

    return descriptor;
}

} // namespace pipeline
} // namespace prism