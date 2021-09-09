#pragma once

#include <array>

#include <allocator.hpp>
#include <context.hpp>
#include <scene.hpp>
#include <shaders.hpp>

#include <vulkan/vulkan.hpp>

namespace prism {

// Not sure where this is defined, but here we will place it:
constexpr size_t MIN_PUSH_CONST_SIZE = 128;

struct PipelineParam
{
    uint32_t outputWidth;
    uint32_t outputHeight;
};

class Pipelines
{
  public:
    Pipelines(const PipelineParam& param, const Context& context, const GPUAllocator& gpuAllocator, const Scene& scene);

  private:
    // All of the buffers the pipeline will use (output buffer, ray queues, etc.):
    struct Buffers
    {
        UniqueBuffer output;
    };

    // For our use case, we only need one descriptor set. If any more are needed, add them here as appropriate.
    template <size_t NumSets>
    struct Descriptor
    {
        Descriptor(const Context& context, std::span<const vk::DescriptorSetLayoutBinding> bindings);

        vk::UniqueDescriptorSetLayout          setLayout;
        vk::UniqueDescriptorPool               pool;
        std::array<vk::DescriptorSet, NumSets> set;
    };

    struct Descriptors
    {
        Descriptor<1> raygen;
    };

    //
    // Any data for the RT pipeline:

    class RTPipeline
    {
      public:
        RTPipeline(const Context& context, const GPUAllocator& gpuAllocator, const Descriptors& descriptors);

      private:
        struct PushConst
        {};

      private:
        vk::UniquePipelineLayout m_layout;
        vk::UniquePipeline       m_pipeline;

        vk::StridedDeviceAddressRegionKHR m_raygenAddrRegion;
        vk::StridedDeviceAddressRegionKHR m_missAddrRegion;
        vk::StridedDeviceAddressRegionKHR m_hitAddrRegion;
        vk::StridedDeviceAddressRegionKHR m_callableAddrRegion;

        UniqueBuffer m_sbt;
    };

  private:
    static Buffers     createBuffers(const PipelineParam& param, const GPUAllocator& gpuAllocator);
    static Descriptors createDescriptors(const Context& context, const Scene& scene, const Buffers& buffers);

  private:
    Buffers     m_buffers;
    Descriptors m_descriptors;

    RTPipeline m_rtPipeline;
};

} // namespace prism