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

// Any parameters when binding the RTPipeline:
struct RTPipelineParam
{
    uint32_t width;
    uint32_t height;
};

// Function that checks if a pushconstant is valid:
template <typename T>
constexpr bool isValidPushConstSize()
{
    constexpr size_t size = sizeof(T);
    return (size <= 128) && (size % 4 == 0);
}

// The pipelines class stores all of the different pipelines that are going to be needed, including any descriptors that
// they may need.
class Pipelines
{
  public:
    Pipelines(const PipelineParam& param, const Context& context, const GPUAllocator& gpuAllocator, const Scene& scene);

    // Binds the ray-tracing pipeline (when performing ray-tracing operations):
    void addBindRTPipelineCmd(const vk::CommandBuffer& commandBuffer, const RTPipelineParam& param) const;

  private:
    // Defines a descriptor when given a set of bindings. Note that we only support 1 descriptor set at this moment.
    // Until more are needed, this just keeps things simple.
    struct Descriptor
    {
        Descriptor(const Context& context, vk::ArrayProxy<const vk::DescriptorSetLayoutBinding> bindings)
        {
            setLayout = context.device().createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()});

            // Return the pool sizes, essentially combining descriptors of the same type:
            const auto poolSizes = [&]() {
                std::vector<vk::DescriptorPoolSize> poolSizes;

                for (const auto& binding : bindings) {
                    auto itr = std::ranges::find_if(
                        poolSizes, [&](const auto& poolSize) { return poolSize.type == binding.descriptorType; });
                    if (itr == poolSizes.end()) {
                        poolSizes.emplace_back(binding.descriptorType, binding.descriptorCount);
                    } else {
                        itr->descriptorCount += binding.descriptorCount;
                    }
                }

                return poolSizes;
            }();

            pool = context.device().createDescriptorPoolUnique(
                vk::DescriptorPoolCreateInfo{.maxSets       = 1,
                                             .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                             .pPoolSizes    = poolSizes.data()});

            set = context.device().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
                .descriptorPool = *pool, .descriptorSetCount = 1, .pSetLayouts = &*setLayout})[0];
        }

        vk::UniqueDescriptorSetLayout setLayout;
        vk::UniqueDescriptorPool      pool;
        vk::DescriptorSet             set;
    };

    // All of the buffers the pipelines will use:
    struct Buffers
    {
        UniqueBuffer beautyOutput;
    };

    // All of the descriptors the pipelines will use:
    struct Descriptors
    {
        Descriptor outputBuffers;
        Descriptor sceneInfo;
        // Descriptor cameraInfo;
        // Descriptor queues; // contains all of the different queues and whatnot...
    };

    //
    // Any data for the RT pipeline:

    struct RTPipeline
    {
        struct PushConst
        {
            std::array<std::byte, 4> data; // placeholder
        };
        static_assert(isValidPushConstSize<PushConst>(), "PushConst is not a valid size.");

        // The pipeline itself:
        vk::UniquePipelineLayout pipelineLayout;
        vk::UniquePipeline       pipeline;

        // The SBT and the respective regions:
        vk::StridedDeviceAddressRegionKHR raygenAddrRegion;
        vk::StridedDeviceAddressRegionKHR missAddrRegion;
        vk::StridedDeviceAddressRegionKHR hitAddrRegion;
        vk::StridedDeviceAddressRegionKHR callableAddrRegion;
        UniqueBuffer                      sbtBuffer;

        // A vector of all of the descriptors that the pipeline will be using:
        std::array<vk::DescriptorSet, 2> descriptorSets;
    };

  private:
    static Buffers     createBuffers(const PipelineParam& param, const GPUAllocator& gpuAllocator);
    static Descriptors createDescriptors(const Context& context, const Scene& scene, const Buffers& buffers);

    static RTPipeline createRTPipeline(const Context& context, const GPUAllocator& gpuAllocator,
                                       const Descriptors& descriptors);

  private:
    Buffers     m_buffers;
    Descriptors m_descriptors;

    RTPipeline m_rtPipeline;
};

} // namespace prism