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
    template <size_t NumSets>
    struct Descriptor
    {
        Descriptor(const Context& context, std::span<const vk::DescriptorSetLayoutBinding> bindings)
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

            pool = context.device().createDescriptorPoolUnique(
                vk::DescriptorPoolCreateInfo{.maxSets       = NumSets,
                                             .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                             .pPoolSizes    = poolSizes.data()});

            // Allocate the number of descriptor sets we need:
            std::array<vk::DescriptorSetLayout, NumSets> setLayouts;
            setLayouts.fill(*setLayout);

            auto descriptorSetVec = context.device().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
                .descriptorPool = *pool, .descriptorSetCount = NumSets, .pSetLayouts = setLayouts.data()});

            std::ranges::copy(descriptorSetVec, sets.begin());
        }

        vk::UniqueDescriptorSetLayout          setLayout;
        vk::UniqueDescriptorPool               pool;
        std::array<vk::DescriptorSet, NumSets> sets;
    };

    // All of the buffers the pipeline will use (color output buffer, ray queues, etc.):
    struct Buffers
    {
        UniqueBuffer beautyOutput;
    };

    struct Descriptors
    {
        Descriptor<1> outputBuffers;
        Descriptor<1> sceneInfo;
        // Descriptor<1> cameraInfo;
        // Descriptor<1> queues; // contains all of the different queues and whatnot...
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
        std::array<vk::DescriptorSet, 1> descriptorSets;
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