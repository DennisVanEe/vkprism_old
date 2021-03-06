#pragma once

#include <functional>
#include <optional>
#include <span>
#include <vector>

#include <vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

#include <util.hpp>

namespace prism {

#define DEVICE_FEATURES_STRUCTURE                                                                                      \
    vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features,               \
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR

#define DEVICE_PROPERTIES_STRUCTURE                                                                                    \
    vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan11Properties, vk::PhysicalDeviceVulkan12Properties,         \
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR

using PhysicalDeviceFeatures   = vk::StructureChain<DEVICE_FEATURES_STRUCTURE>;
using PhysicalDeviceProperties = vk::StructureChain<DEVICE_PROPERTIES_STRUCTURE>;

struct ContextParam
{
    bool enableValidation = false;
    bool enableCallback   = false;

    bool enableRobustBufferAccess = true;
};

// The context has to outlive anything that requires vulkan functions as it maintains a handle to the vulkan library.
class Context
{
  public:
    explicit Context(const ContextParam& param);
    Context(const Context&) = delete;
    Context(Context&&)      = delete;

    const vk::Instance&       instance() const { return *m_instance; }
    const vk::PhysicalDevice& physicalDevice() const { return m_physDevInfo.physicalDevice; }
    const vk::Device&         device() const { return *m_device; }
    const uint32_t            queueFamilyIndex() const { return m_queueInfo.familyIndex; }
    const vk::Queue&          queue() const { return m_queueInfo.queue; }

    const PhysicalDeviceFeatures&   features() const { return m_physDevInfo.features; }
    const PhysicalDeviceProperties& properties() const { return m_physDevInfo.properties; }

  private:
    struct PhysicalDeviceInfo
    {
        PhysicalDeviceInfo(const vk::PhysicalDevice& physicalDevice, const ContextParam& param);

        vk::PhysicalDevice                     physicalDevice;
        PhysicalDeviceFeatures                 features;
        PhysicalDeviceProperties               properties;
        std::vector<vk::QueueFamilyProperties> queueFamilyProps;
    };

    struct QueueInfo
    {
        uint32_t  familyIndex;
        vk::Queue queue;
    };

  private:
    static std::vector<const char*> getRequiredDeviceExtensions(const ContextParam& param);
    static std::vector<const char*> getRequiredInstanceExtensions(const ContextParam& param);
    static std::vector<const char*> getRequiredInstanceLayers(const ContextParam& param);

    static vk::UniqueInstance createInstance(const ContextParam& param, const vk::DynamicLoader& dynamicLoader,
                                             std::span<const char* const> reqInstanceExts,
                                             std::span<const char* const> reqInstanceLayers);
    static PhysicalDeviceInfo createPhysicalDeviceInfo(const vk::Instance& instance, const ContextParam& param,
                                                       std::span<const char* const> reqDeviceExts);
    static vk::UniqueDevice   createDevice(const vk::Instance& instance, const PhysicalDeviceInfo& physDevInfo,
                                           std::span<const char* const> reqDeviceExts);
    static QueueInfo          createQueueInfo(const vk::Device& device, const PhysicalDeviceInfo& physDevInfo);

  private:
    vk::DynamicLoader m_dynamicLoader;

    std::vector<const char*> m_reqDeviceExtensions;
    std::vector<const char*> m_reqInstanceExtensions;
    std::vector<const char*> m_reqInstanceLayers;

    vk::UniqueInstance               m_instance;
    vk::UniqueDebugUtilsMessengerEXT m_debugUtilsMessenger;

    PhysicalDeviceInfo m_physDevInfo;
    vk::UniqueDevice   m_device;

    QueueInfo m_queueInfo;
};

// The default fence timeout of 1 minute (not sure how long this should be...)
constexpr uint64_t FENCE_TIMEOUT = 6e+10;

void submitAndWait(const Context& context, vk::ArrayProxy<const vk::CommandBuffer> commandBuffers,
                   std::string_view description = {}, uint64_t timeout = FENCE_TIMEOUT);

} // namespace prism
