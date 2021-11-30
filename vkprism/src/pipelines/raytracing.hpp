#pragma once

#include <array>

#include <vulkan/vulkan.hpp>

#include <context.hpp>
#include <descriptor.hpp>
#include <scene.hpp>

namespace prism {
namespace pipeline {

class RayTracing
{
  public:
    struct Param
    {
        const Scene& scene;

        // Any buffers the ray-tracing pipeline may need:
        vk::Buffer beautyOutputBuffer;
    };

    RayTracing(const Context& context, const Param& param, const GPUAllocator& gpuAllocator);

  private:
    static Descriptor createSceneInfoDesc(const Context& context, const Param& param);
    static Descriptor createOuputBufferDesc(const Context& context, const Param& param);

  private:
    // Descriptors:
    Descriptor m_sceneInfoDesc;
    Descriptor m_outputBuffersDesc;

    // A vector of all of the descriptors that the pipeline will be using:
    std::array<vk::DescriptorSet, 2> descriptorSets;

    // The pipeline itself:
    vk::UniquePipelineLayout m_pipelineLayout;
    vk::UniquePipeline       m_pipeline;

    // The SBT and the respective regions:
    vk::StridedDeviceAddressRegionKHR m_raygenAddrRegion;
    vk::StridedDeviceAddressRegionKHR m_missAddrRegion;
    vk::StridedDeviceAddressRegionKHR m_hitAddrRegion;
    vk::StridedDeviceAddressRegionKHR m_callableAddrRegion;
    UniqueBuffer                      m_sbtBuffer;

    // A vector of all of the descriptors that the pipeline will be using:
    std::array<vk::DescriptorSet, 2> m_descriptorSets;
};

} // namespace pipeline
} // namespace prism