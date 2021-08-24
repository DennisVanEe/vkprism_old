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

#include <GLFW/glfw3.h>
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
    auto reqInstExts = [&]() {
        uint32_t   glfwInstLayerCnt = 0;
        const auto glfwInstLayers   = glfwGetRequiredInstanceExtensions(&glfwInstLayerCnt);
        return std::vector<const char*>(glfwInstLayers, glfwInstLayers + glfwInstLayerCnt);
    }();

    if (param.enableCallback) {
        reqInstExts.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return reqInstExts;
}

static std::vector<const char*> getRequiredInstanceLayers(const ContextParam& param)
{
    if (param.enableValidation) {
        return {"VK_LAYER_KHRONOS_validation"};
    }
    return {};
}

static UniqueGLFWWindow createWindowAndInitGLFW(const ContextParam& param)
{
    // First we prepare glfw and check that Vulkan is even present:
    glfwSetErrorCallback(glfwCallback);
    if (!glfwInit()) {
        throw std::runtime_error("Could not create Context."); // Details specified in callback
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    const auto window = glfwCreateWindow(param.windowWidth, param.windowHeight, PROJECT_NAME, nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("Could not create Context."); // Details specified in callback
    }

    if (!glfwVulkanSupported()) {
        throw std::runtime_error("GLFW: Vulkan is not supported or found. Could not create Context.");
    }

    return UniqueGLFWWindow(window);
}

static vk::UniqueInstance createInstance(const ContextParam& param, std::span<const char* const> reqInstanceExts,
                                         std::span<const char* const> reqInstanceLayers)
{
    // Setup the dynamic loader using GLFW as a dynamic loader:
    VULKAN_HPP_DEFAULT_DISPATCHER.init(glfwGetInstanceProcAddress);

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

static QueueInfo createQueue(const vk::Device& device, const PhysicalDeviceInfo& physDevInfo)
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

    const uint32_t familyIndex = std::distance(itr, physDevInfo.queueFamilyProps.begin());
    return QueueInfo{
        .queue       = device.getQueue(familyIndex, 0),
        .familyIndex = familyIndex,
        .queueIndex  = 0,
    };
}

static UniqueVmaAllocator createVmaAllocator(const vk::Instance& instance, const vk::Device& device,
                                             const PhysicalDeviceInfo& physDevInfo)
{
    // We want to use the functions loaded from the dynamic dispatcher. I'm not a big fan of this implementation, I need
    // to look for a way to automate this process...
    const VmaVulkanFunctions vkFunctions
    {
        .vkGetPhysicalDeviceProperties       = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties,
        .vkGetPhysicalDeviceMemoryProperties = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties,
        .vkAllocateMemory                    = VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory,
        .vkFreeMemory                        = VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory,
        .vkMapMemory                         = VULKAN_HPP_DEFAULT_DISPATCHER.vkMapMemory,
        .vkUnmapMemory                       = VULKAN_HPP_DEFAULT_DISPATCHER.vkUnmapMemory,
        .vkFlushMappedMemoryRanges           = VULKAN_HPP_DEFAULT_DISPATCHER.vkFlushMappedMemoryRanges,
        .vkInvalidateMappedMemoryRanges      = VULKAN_HPP_DEFAULT_DISPATCHER.vkInvalidateMappedMemoryRanges,
        .vkBindBufferMemory                  = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory,
        .vkBindImageMemory                   = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory,
        .vkGetBufferMemoryRequirements       = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements,
        .vkGetImageMemoryRequirements        = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements,
        .vkCreateBuffer                      = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer,
        .vkDestroyBuffer                     = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer,
        .vkCreateImage                       = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage,
        .vkDestroyImage                      = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage,
        .vkCmdCopyBuffer                     = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer,
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
        .vkGetBufferMemoryRequirements2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2,
        .vkGetImageMemoryRequirements2KHR  = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2,
#endif
#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
        .vkBindBufferMemory2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory2,
        .vkBindImageMemory2KHR  = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory2,
#endif
#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
        .vkGetPhysicalDeviceMemoryProperties2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties2,
#endif
    };

    const VmaAllocatorCreateInfo createInfo{
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = physDevInfo.physicalDevice,
        .device           = device,
        .pVulkanFunctions = &vkFunctions,
        .instance         = instance,
        .vulkanApiVersion = VK_API_VERSION_1_2,
    };

    VmaAllocator allocator;
    vkCall(vmaCreateAllocator(&createInfo, &allocator));

    return UniqueVmaAllocator(allocator);
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
    window(createWindowAndInitGLFW(param)),
    instance(createInstance(param, reqInstanceExtensions, reqInstanceLayers)),
    debugUtilsMessenger(instance->createDebugUtilsMessengerEXTUnique(DEBUG_UTILS_MSGR_CREATE_INFO)),
    physDevInfo(pickPhysicalDevice(*instance, param, reqDeviceExtensions)),
    device(createDevice(*instance, physDevInfo, reqDeviceExtensions)),
    queueInfo(createQueue(*device, physDevInfo)),
    vmaAllocator(createVmaAllocator(*instance, *device, physDevInfo))
{}

UniqueBuffer Context::allocateBuffer(const vk::BufferCreateInfo&    bufferCreateInfo,
                                     const VmaAllocationCreateInfo& allocCreateInfo) const
{
    const VkBufferCreateInfo& convBufferCreateInfo = bufferCreateInfo;

    VkBuffer      buffer;
    VmaAllocation allocation;
    vmaCreateBuffer(vmaAllocator.get(), &convBufferCreateInfo, &allocCreateInfo, &buffer, &allocation, nullptr);

    return UniqueBuffer(buffer, allocation, vmaAllocator.get());
}

UniqueBuffer Context::allocateBuffer(size_t size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage) const
{
    return allocateBuffer(
        vk::BufferCreateInfo{
            .size  = size,
            .usage = bufferUsage,
        },
        VmaAllocationCreateInfo{
            .usage = memoryUsage,
        });
}

} // namespace prism
