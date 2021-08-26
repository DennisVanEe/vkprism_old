#include "context.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <configure.hpp>
#include <util.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace prism {

static void glfwCallback(const int error, const char const* desc) { spdlog::error("GLFW Error {}: {}", desc, error); }

static VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(const VkDebugUtilsMessageSeverityFlagBitsEXT      msgSeverity,
                                                         const VkDebugUtilsMessageTypeFlagsEXT             msgType,
                                                         const VkDebugUtilsMessengerCallbackDataEXT* const callbackData,
                                                         [[maybe_unused]] void* const                      userData)
{
    const auto msgTypeStr = [&]() {
        switch (msgType) {
        case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
            return "[General]";
        case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
            return "[Performance]";
        case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
            return "[Validation]";
        default:
            return "[Unknown]";
        }
    }();

    switch (msgSeverity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        spdlog::warn("{}: {}", msgTypeStr, callbackData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        spdlog::error("{}: {}", msgTypeStr, callbackData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
    default:
        spdlog::info("{}: {}", msgTypeStr, callbackData->pMessage);
        break;
    }

    return VK_FALSE;
}

constexpr vk::DebugUtilsMessengerCreateInfoEXT DEBUG_UTILS_MSGR_CREATE_INFO{
    .messageSeverity =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo,
    .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                   vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
    .pfnUserCallback = debugUtilsCallback,
};

static std::vector<const char*> getRequiredDeviceExtensions(const ContextParam& param)
{
    // All of the device extensions needed for ray-tracing acceleration:
    // return {};
    return {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME};
}

static std::vector<const char*> getRequiredInstanceExtensions(const ContextParam& param)
{
    if (param.enableCallback) {
        return {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    }
    return {};
}

static std::vector<const char*> getRequiredInstanceLayers(const ContextParam& param)
{
    if (param.enableValidation) {
        return {"VK_LAYER_KHRONOS_validation"};
    }
    return {};
}

static vk::UniqueInstance createInstance(const ContextParam& param, std::span<const char* const> reqInstanceExts,
                                         std::span<const char* const> reqInstanceLayers)
{
    // Setup the dynamic loader:
    vk::DynamicLoader dynamicLoader;
    const auto vkGetInstanceProcAddr = dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    const vk::ApplicationInfo applicationInfo{
        .pApplicationName   = PROJECT_NAME,
        .applicationVersion = VK_MAKE_VERSION(PROJECT_VER_MAJOR, PROJECT_VER_MINOR, PROJECT_VER_PATCH),
        .pEngineName        = ENGINE_NAME,
        .engineVersion      = VK_MAKE_VERSION(PROJECT_VER_MAJOR, PROJECT_VER_MINOR, PROJECT_VER_PATCH),
        .apiVersion         = VK_API_VERSION_1_2,
    };

    const vk::InstanceCreateInfo instanceCreateInfo{
        .pNext                   = param.enableCallback ? &DEBUG_UTILS_MSGR_CREATE_INFO : nullptr,
        .pApplicationInfo        = &applicationInfo,
        .enabledLayerCount       = static_cast<uint32_t>(reqInstanceLayers.size()),
        .ppEnabledLayerNames     = reqInstanceLayers.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(reqInstanceExts.size()),
        .ppEnabledExtensionNames = reqInstanceExts.data(),
    };

    auto instance = vk::createInstanceUnique(instanceCreateInfo);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

    return instance;
}

static PhysicalDeviceInfo pickPhysicalDevice(const vk::Instance& instance, const ContextParam& param,
                                             std::span<const char* const> reqDeviceExts)
{
    const auto physicalDevices = instance.enumeratePhysicalDevices();
    const auto itr =
        std::find_if(physicalDevices.begin(), physicalDevices.end(), [&](const vk::PhysicalDevice& physicalDevice) {
            const auto availableDeviceExts = physicalDevice.enumerateDeviceExtensionProperties();
            return std::all_of(reqDeviceExts.begin(), reqDeviceExts.end(), [&](const char* const reqExt) {
                return std::any_of(availableDeviceExts.begin(), availableDeviceExts.end(),
                                   [&](const vk::ExtensionProperties& ext) {
                                       return std::strcmp(ext.extensionName.data(), reqExt) == 0;
                                   });
            });
        });
    if (itr == physicalDevices.end()) {
        throw std::runtime_error("No compatible physical device found.");
    }
    return PhysicalDeviceInfo(*itr, param);
}

static vk::UniqueDevice createDevice(const vk::Instance& instance, const PhysicalDeviceInfo& physDevInfo,
                                     std::span<const char* const> reqDeviceExts)
{
    uint32_t                               maxQueueCount = 0;
    std::vector<vk::DeviceQueueCreateInfo> devQueueCreateInfos;
    devQueueCreateInfos.reserve(physDevInfo.queueFamilyProps.size());
    for (const auto& prop : physDevInfo.queueFamilyProps) {
        devQueueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = static_cast<uint32_t>(&prop - &physDevInfo.queueFamilyProps[0]),
            .queueCount       = prop.queueCount,
        });
        maxQueueCount = std::max(maxQueueCount, prop.queueCount);
    }

    // Allocate a priority buffer and assign it:
    const std::vector<float> queuePriorities(maxQueueCount, 1.f);
    for (auto& createInfo : devQueueCreateInfos) {
        createInfo.pQueuePriorities = queuePriorities.data();
    }

    const vk::DeviceCreateInfo devCreateInfo{
        .pNext                   = &physDevInfo.features.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount    = static_cast<uint32_t>(devQueueCreateInfos.size()),
        .pQueueCreateInfos       = devQueueCreateInfos.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(reqDeviceExts.size()),
        .ppEnabledExtensionNames = reqDeviceExts.data(),
        //.pEnabledFeatures        = &physDevInfo.features.get<vk::PhysicalDeviceFeatures2>().features,
    };

    auto device = physDevInfo.physicalDevice.createDeviceUnique(devCreateInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
    return device;
}

PhysicalDeviceInfo::PhysicalDeviceInfo(const vk::PhysicalDevice physicalDevice, const ContextParam& param) :
    physicalDevice(physicalDevice),
    features(physicalDevice.getFeatures2<DEVICE_FEATURES_STRUCTURE>()),
    properties(physicalDevice.getProperties2<DEVICE_PROPERTIES_STRUCTURE>()),
    queueFamilyProps(physicalDevice.getQueueFamilyProperties())
{
    //
    // Now we enable or disable any features according to the parameters:

    auto& robustBufferAccess = features.get<vk::PhysicalDeviceFeatures2>().features.robustBufferAccess;
    if (param.enableRobustBufferAccess) {
        if (!robustBufferAccess) {
            spdlog::warn("The chosen physical device doesn't support robust buffer access.");
        }
    } else {
        robustBufferAccess = VK_FALSE;
    }

    auto& bufferDeviceAddress = features.get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress;
    if (bufferDeviceAddress != VK_TRUE) {
        throw std::runtime_error("bufferDeviceAddress isn't supported by the chosen physcial device.");
    }
}

Context::Context(const ContextParam& param) :
    param(param),
    reqDeviceExtensions(getRequiredDeviceExtensions(param)),
    reqInstanceExtensions(getRequiredInstanceExtensions(param)),
    reqInstanceLayers(getRequiredInstanceLayers(param)),
    instance(createInstance(param, reqInstanceExtensions, reqInstanceLayers)),
    debugUtilsMessenger(instance->createDebugUtilsMessengerEXTUnique(DEBUG_UTILS_MSGR_CREATE_INFO)),
    physDevInfo(pickPhysicalDevice(*instance, param, reqDeviceExtensions)),
    device(createDevice(*instance, physDevInfo, reqDeviceExtensions))
{
    const auto itr = std::ranges::find_if(physDevInfo.queueFamilyProps, [&](const auto& prop) {
        return ((prop.queueFlags & vk::QueueFlagBits::eGraphics) == vk::QueueFlagBits::eGraphics) &&
               ((prop.queueFlags & vk::QueueFlagBits::eCompute) == vk::QueueFlagBits::eCompute) &&
               ((prop.queueFlags & vk::QueueFlagBits::eTransfer) == vk::QueueFlagBits::eTransfer);
    });

    if (itr == physDevInfo.queueFamilyProps.end()) {
        throw std::runtime_error(
            "Could not find a queue family that supports graphics, compute, and transfer operations.");
    }

    queueFamilyIdx = std::distance(itr, physDevInfo.queueFamilyProps.begin());
    queue          = device->getQueue(queueFamilyIdx, 0);
}

} // namespace prism
