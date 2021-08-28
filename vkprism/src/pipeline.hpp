#pragma once

#include "pipeline.hpp"

#include <allocator.hpp>
#include <array>
#include <context.hpp>
#include <scene.hpp>

#include <vulkan/vulkan.hpp>

namespace prism {

struct PipelineParam
{
    uint32_t outputWidth;
    uint32_t outputHeight;
};

class Pipeline
{
  public:
    Pipeline(const PipelineParam& param, const Context& context, const GpuAllocator& allocator, const Scene& scene);

  private:
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

    // All of the buffers the pipeline will use (output buffer, ray queues, etc.):
    struct Buffers
    {
        UniqueBuffer output;
    };

  private:
    static Buffers     createBuffers(const PipelineParam& param, const GpuAllocator& allocator);
    static Descriptors createDescriptors(const Context& context, const Scene& scene, const Buffers& buffers);

  private:
    Buffers     m_buffers;
    Descriptors m_descriptors;
};

} // namespace prism