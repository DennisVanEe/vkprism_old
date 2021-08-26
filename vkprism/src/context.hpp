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

struct Context
{
    explicit Context(const ContextParam& param);
    Context(const Context&) = delete;
    Context(Context&&)      = delete;

    ContextParam param;

    std::vector<const char*> reqDeviceExtensions;
    std::vector<const char*> reqInstanceExtensions;
    std::vector<const char*> reqInstanceLayers;

    vk::UniqueInstance               instance;
    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;

    PhysicalDeviceInfo physDevInfo;
    vk::UniqueDevice   device;

    vk::Queue queue;
    uint32_t  queueFamilyIdx;
};

} // namespace prism
