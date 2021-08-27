#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

#include <util.hpp>

namespace prism {

#define DEVICE_FEATURES_STRUCTURE                                                                                      \
    vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features,               \
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR

#define DEVICE_PROPERTIES_STRUCTURE                                                                                    \
    vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan11Properties, vk::PhysicalDeviceVulkan12Properties

struct QueueInfo
{
    vk::Queue queue;
    uint32_t  familyIndex;
    uint32_t  queueIndex;
};

struct ContextParam
{
    bool enableValidation = false;
    bool enableCallback   = false;

    bool enableRobustBufferAccess = true;
};

struct PhysicalDeviceInfo
{
    PhysicalDeviceInfo(vk::PhysicalDevice physicalDevice, const ContextParam& param);
    PhysicalDeviceInfo(PhysicalDeviceInfo const&) = delete;
    PhysicalDeviceInfo(PhysicalDeviceInfo&&)      = delete;

    vk::PhysicalDevice                              physicalDevice;
    vk::StructureChain<DEVICE_FEATURES_STRUCTURE>   features;
    vk::StructureChain<DEVICE_PROPERTIES_STRUCTURE> properties;
    std::vector<vk::QueueFamilyProperties>          queueFamilyProps;
};

// The context has to outlive anything that requires vulkan functions as it maintains a handle to the vulkan library.
class Context
{
  public:
    explicit Context(const ContextParam& param);
    Context(const Context&) = delete;
    Context(Context&&)      = delete;

    const vk::Instance&       instance() const { return *m_instance; }
    const PhysicalDeviceInfo& physDevInfo() const { return m_physDevInfo; }
    const vk::Device&         device() const { return *m_device; }
    const uint32_t            queueFamilyIdx() const { return m_queueFamilyIdx; }
    const vk::Queue&          queue() const { return m_queue; }

  private:
    vk::DynamicLoader m_dynamicLoader;

    std::vector<const char*> m_reqDeviceExtensions;
    std::vector<const char*> m_reqInstanceExtensions;
    std::vector<const char*> m_reqInstanceLayers;

    vk::UniqueInstance               m_instance;
    vk::UniqueDebugUtilsMessengerEXT m_debugUtilsMessenger;

    PhysicalDeviceInfo m_physDevInfo;
    vk::UniqueDevice   m_device;

    uint32_t  m_queueFamilyIdx;
    vk::Queue m_queue;
};

} // namespace prism
