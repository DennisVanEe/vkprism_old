// This manages all of the pipelines that may be used in the renderer.
// TODO: I feel as if there's a better way to organize this, figure something better out when the time comes.

#pragma once

#include <array>

#include <allocator.hpp>
#include <context.hpp>
#include <scene.hpp>
#include <shaders.hpp>

#include <vulkan/vulkan.hpp>

namespace prism {

struct PipelineParam
{
    uint32_t outputWidth;
    uint32_t outputHeight;
};

// Function that checks if a pushconstant is valid:
template <typename T>
constexpr bool isValidPushConstSize()
{
    constexpr size_t size = sizeof(T);
    return (size <= 128) && (size % 4 == 0);
}

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

        void addToCommandBuffer(const vk::CommandBuffer& commandBuffer, const PipelineParam& param) const;

      private:
        struct PushConst
        {
            std::array<std::byte, 4> data; // placeholder
        };
        static_assert(isValidPushConstSize<PushConst>(), "PushConst is not a valid size.");

      private:
        // The pipeline itself:
        vk::UniquePipelineLayout m_layout;
        vk::UniquePipeline       m_pipeline;

        // The SBT and the respective regions:
        vk::StridedDeviceAddressRegionKHR m_raygenAddrRegion;
        vk::StridedDeviceAddressRegionKHR m_missAddrRegion;
        vk::StridedDeviceAddressRegionKHR m_hitAddrRegion;
        vk::StridedDeviceAddressRegionKHR m_callableAddrRegion;
        UniqueBuffer                      m_sbt;

        // A vector of all of the descriptors that the pipeline will be using:
        std::vector<vk::DescriptorSet> m_descriptorSets;
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