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
    Pipelines(const PipelineParam& param, const Context& context, const GpuAllocator& allocator,
              const Scene& scene);

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

    struct Pipeline
    {
        vk::UniquePipelineLayout layout;
        vk::UniquePipeline       pipeline;
    };

    //
    // Any push constants are defined here:

    struct RTPushConst
    {};

  private:
    static Buffers     createBuffers(const PipelineParam& param, const GpuAllocator& allocator);
    static Descriptors createDescriptors(const Context& context, const Scene& scene, const Buffers& buffers);
    static Pipeline    createRTPipeline(const Context& context, const Descriptors& descriptors);

  private:
    Buffers     m_buffers;
    Descriptors m_descriptors;

    Pipeline m_rtPipeline;
};

} // namespace prism