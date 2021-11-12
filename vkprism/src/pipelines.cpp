#include "pipelines.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <vector>

#include <util.hpp>

namespace prism {

Pipelines::Buffers Pipelines::createBuffers(const PipelineParam& param, const GPUAllocator& gpuAllocator)
{
    return Buffers{
        .beautyOutput = gpuAllocator.allocateBuffer(
            param.outputWidth * param.outputHeight * sizeof(glm::vec3),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY)
    };
}

Pipelines::Descriptors Pipelines::createDescriptors(const Context& context, const Scene& scene, const Buffers& buffers)
{
    // This contains the output buffers (final color image, other AOVs, etc.):
    auto outputBuffers = [&]() {
        Descriptor descriptor(
            context,
            std::to_array({
                // Beauty output buffer:
                vk::DescriptorSetLayoutBinding{
                    .binding         = 0,
                    .descriptorType  = vk::DescriptorType::eStorageBuffer,
                    .descriptorCount = 1,
                    .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR}}));

        const vk::DescriptorBufferInfo beautyOutputBufferInfo{
            .buffer = *buffers.beautyOutput,
            .range = VK_WHOLE_SIZE
        };

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
    }();

    // The scene info descriptor contains the TLAS, scene geometry, and scene material properties (to be added later):
    auto sceneInfo = [&]() {
        Descriptor descriptor(context,
                              std::to_array({// TLAS:
                                             vk::DescriptorSetLayoutBinding{
                                                 .binding         = 0,
                                                 .descriptorType  = vk::DescriptorType::eAccelerationStructureKHR,
                                                 .descriptorCount = 1,
                                                 .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR}}));

        const vk::StructureChain<vk::WriteDescriptorSet, vk::WriteDescriptorSetAccelerationStructureKHR> tlasWrite{
            vk::WriteDescriptorSet{.dstSet          = descriptor.set,
                                   .dstBinding      = 0,
                                   .dstArrayElement = 0,
                                   .descriptorCount = 1,
                                   .descriptorType  = vk::DescriptorType::eAccelerationStructureKHR},
            vk::WriteDescriptorSetAccelerationStructureKHR{
                .accelerationStructureCount = 1,
                .pAccelerationStructures    = &scene.tlas(),
            },
        };

        context.device().updateDescriptorSets(tlasWrite.get<vk::WriteDescriptorSet>(), {});

        return descriptor;
    }();

    return Descriptors{
        .outputBuffers = std::move(outputBuffers),
        .sceneInfo     = std::move(sceneInfo),
    };
}

Pipelines::RTPipeline Pipelines::createRTPipeline(
    const Context&      context,
    const GPUAllocator& gpuAllocator,
    const Descriptors&  descriptors)
{
    //
    // Create the pipeline layout:
    //

    // For now, the RT pipeline will have both the TLAS and the output buffer.
    // In reality it would be the TLAS and any queues.
    const auto descriptorSets = std::to_array({descriptors.sceneInfo.set, descriptors.outputBuffers.set});

    auto pipelineLayout = [&]() {
        const auto descriptorSetLayouts =
            std::to_array({
                *descriptors.sceneInfo.setLayout,
                *descriptors.outputBuffers.setLayout
            });

        return context.device().createPipelineLayoutUnique(
            vk::PipelineLayoutCreateInfo{
                .setLayoutCount         = descriptorSetLayouts.size(),
                .pSetLayouts            = descriptorSetLayouts.data(),
                .pushConstantRangeCount = 0, // no push constants for now
                .pPushConstantRanges    = nullptr
            });
    }();

    //
    // Set the shader stages and the groups up:
    //

    // Load all of the shaders first:

    const auto shaderStages = [&]() {
        std::array<vk::PipelineShaderStageCreateInfo, TOTAL_NUM_SHADERS> shaderStages;
        shaderStages[sRAYGEN] = vk::PipelineShaderStageCreateInfo{
            .stage  = vk::ShaderStageFlagBits::eRaygenKHR,
            .module = loadShader(context, sRAYGEN),
            .pName  = SHADER_ENTRY
        };
        shaderStages[sMISS] = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eMissKHR,
            .module = loadShader(context, sMISS),
            .pName = SHADER_ENTRY
        };
        shaderStages[sCLOSEST_HIT] = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eClosestHitKHR,
            .module = loadShader(context, sCLOSEST_HIT),
            .pName  = SHADER_ENTRY
        };
        return shaderStages;
    }();
    // Make sure to delete the shader stages when we leave the function:
    Defer shaderStageCleanup([&]() {
        for (const auto& stage : shaderStages) {
            context.device().destroyShaderModule(stage.module);
        }
    });

    const auto [shaderGroups, numMissGroups, numHitGroups, numCallableGroups] = [&]() {
        const auto raygenGroup = vk::RayTracingShaderGroupCreateInfoKHR{
            .type               = vk::RayTracingShaderGroupTypeKHR::eGeneral,
            .generalShader      = sRAYGEN,
            .closestHitShader   = VK_SHADER_UNUSED_KHR,
            .anyHitShader       = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };

        const auto missGroups = std::to_array({
            vk::RayTracingShaderGroupCreateInfoKHR{
                .type               = vk::RayTracingShaderGroupTypeKHR::eGeneral,
                .generalShader      = sMISS,
                .closestHitShader   = VK_SHADER_UNUSED_KHR,
                .anyHitShader       = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
            }
        });
        const auto hitGroups  = std::to_array({
            vk::RayTracingShaderGroupCreateInfoKHR{
                .type               = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                .generalShader      = VK_SHADER_UNUSED_KHR,
                .closestHitShader   = sCLOSEST_HIT,
                .anyHitShader       = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
            }
        });
        const std::array<vk::RayTracingShaderGroupCreateInfoKHR, 0> callableGroups;

        //
        // Kind of messy, but we essentially combine the shader groups:
        const uint32_t numMissGroups     = missGroups.size();
        const uint32_t numHitGroups      = hitGroups.size();
        const uint32_t numCallableGroups = callableGroups.size();

        std::array<vk::RayTracingShaderGroupCreateInfoKHR, 1 + numMissGroups + numHitGroups + numCallableGroups> shaderGroups;

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
            .layout                       = *pipelineLayout,
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
    // Set all of the shader ranges. Note that the first handle of each collection of handles (raygen, miss, hit,
    // etc.) has to be aligned to shaderGroupBaseAlignment, while each handle itself has to be aligned to
    // shaderGroupHandleAlignment.

    vk::StridedDeviceAddressRegionKHR raygenAddrRegion{
        .stride = raygenHandleSize,
        .size   = raygenHandleSize
    };
    vk::StridedDeviceAddressRegionKHR missAddrRegion{
        .stride = handleSizeAligned,
        .size   = alignUp(handleSizeAligned * numMissGroups, rtPipelineProps.shaderGroupBaseAlignment),
    };
    vk::StridedDeviceAddressRegionKHR hitAddrRegion{
        .stride = handleSizeAligned,
        .size   = alignUp(handleSizeAligned * numHitGroups, rtPipelineProps.shaderGroupBaseAlignment),
    };
    vk::StridedDeviceAddressRegionKHR callableAddrRegion{
        .stride = handleSizeAligned,
        .size   = alignUp(handleSizeAligned * numCallableGroups, rtPipelineProps.shaderGroupBaseAlignment),
    };

    //
    // Allocate memory for the SBT:

    const auto sbtSize = raygenAddrRegion.size + missAddrRegion.size + hitAddrRegion.size + callableAddrRegion.size;

    auto sbtBuffer = gpuAllocator.allocateBuffer(
        sbtSize,
        vk::BufferUsageFlagBits::eTransferDst             |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eShaderBindingTableKHR,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Assign the device addresses:
    const auto sbtAddress = sbtBuffer.deviceAddress(context.device());

    raygenAddrRegion.deviceAddress   = sbtAddress;
    missAddrRegion.deviceAddress     = sbtAddress + raygenAddrRegion.size;
    hitAddrRegion.deviceAddress      = sbtAddress + raygenAddrRegion.size + hitAddrRegion.size;
    callableAddrRegion.deviceAddress = 0; // no callables yet...

    //
    // Copy the the vulkan handles to our SBT:
    {
        const auto shaderHandles = context.device().getRayTracingShaderGroupHandlesKHR<std::byte>(
            *pipeline, 0, shaderGroups.size(), shaderGroups.size() * handleSize);
        const auto shaderHandlesSpan = std::span(shaderHandles);

        auto*      sbtBufferMapped = sbtBuffer.map<std::byte>();
        const auto copySBTHandleFn = [&](size_t sbtStartAddr, size_t sbtStride, size_t handleStartIndex, size_t handleCount) {
            for (size_t i = 0; i < handleCount; ++i) {
                std::ranges::copy(shaderHandlesSpan.subspan(handleStartIndex + i * handleSize, handleSize),
                                  sbtBufferMapped + sbtStartAddr + (i * sbtStride));
            }
        };

        size_t sbtStartAddr     = 0;
        size_t handleStartIndex = 0;

        copySBTHandleFn(sbtStartAddr, raygenAddrRegion.stride, handleStartIndex, 1); // raygen

        sbtStartAddr += raygenAddrRegion.size;
        handleStartIndex += 1;
        copySBTHandleFn(sbtStartAddr, missAddrRegion.stride, handleStartIndex, numMissGroups); // miss

        sbtStartAddr += missAddrRegion.size;
        handleStartIndex += numMissGroups;
        copySBTHandleFn(sbtStartAddr, hitAddrRegion.stride, handleStartIndex, numHitGroups); // hit

        sbtStartAddr += hitAddrRegion.size;
        handleStartIndex += numHitGroups;
        copySBTHandleFn(sbtStartAddr, callableAddrRegion.stride, handleStartIndex, numCallableGroups); // callable

        sbtBuffer.unmap();
    }

    return RTPipeline{
        .pipelineLayout     = std::move(pipelineLayout),
        .pipeline           = std::move(pipeline),
        .raygenAddrRegion   = raygenAddrRegion,
        .missAddrRegion     = missAddrRegion,
        .hitAddrRegion      = hitAddrRegion,
        .callableAddrRegion = callableAddrRegion,
        .sbtBuffer          = std::move(sbtBuffer),
        .descriptorSets     = descriptorSets,
    };
}

Pipelines::Pipelines(const PipelineParam& param, const Context& context, const GPUAllocator& gpuAllocator,
                     const Scene& scene) :
    m_buffers(createBuffers(param, gpuAllocator)),
    m_descriptors(createDescriptors(context, scene, m_buffers)),
    m_rtPipeline(createRTPipeline(context, gpuAllocator, m_descriptors))
{}

void prism::Pipelines::addBindRTPipelineCmd(const vk::CommandBuffer& commandBuffer, const RTPipelineParam& param) const
{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *m_rtPipeline.pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *m_rtPipeline.pipelineLayout, 0,
                                     m_rtPipeline.descriptorSets, {});
    // commandBuffer.pushConstants(*m_layout, ...) <- add when appropriate
    commandBuffer.traceRaysKHR(m_rtPipeline.raygenAddrRegion, m_rtPipeline.missAddrRegion, m_rtPipeline.hitAddrRegion,
                               m_rtPipeline.callableAddrRegion, param.width, param.height, 1);
}

} // namespace prism