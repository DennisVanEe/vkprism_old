﻿#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <GLFW/glfw3.h>
#include <vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

#include <util.hpp>

namespace prism {

struct ContextParam
{
    bool enableValidation = false;
    bool enableCallback   = false;

    bool enableRobustBufferAccess = true;

    int windowWidth  = 1280;
    int windowHeight = 720;

    // TODO: I will support this, just not yet (focus on getting it to work first)
    bool compactAccelStruct = false;
};

#define DEVICE_FEATURES_STRUCTURE                                                                                      \
    vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features,               \
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,           \
        vk::PhysicalDeviceBufferDeviceAddressFeaturesEXT

#define DEVICE_PROPERTIES_STRUCTURE                                                                                    \
    vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan11Properties, vk::PhysicalDeviceVulkan12Properties

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

struct QueueInfo
{
    vk::Queue queue;
    uint32_t  familyIndex;
    uint32_t  queueIndex;
};

struct Queues
{
    QueueInfo                general;
    std::optional<QueueInfo> compute;
    std::optional<QueueInfo> transfer;
};

struct CommandPools
{
    vk::UniqueCommandPool general;
    vk::UniqueCommandPool compute;
    vk::UniqueCommandPool transfer;
};

using UniqueGLFWWindow   = CustomUniquePtr<GLFWwindow, glfwDestroyWindow>;
using UniqueVmaAllocator = CustomUniquePtr<std::remove_pointer_t<VmaAllocator>, vmaDestroyAllocator>;

// A buffer that is allocated by Context and is automatically destructed if it goes out of scope. This does mean that
// every buffer has to also store a pointer to the allocator, but until this becomes a problem, it'll just have to be.
// NOTE: the lifetime of Context should outlive everything, so m_allocator won't be destroyed before a UniqueBuffer
// is...
class UniqueBuffer
{
  public:
    UniqueBuffer()                    = default;
    UniqueBuffer(const UniqueBuffer&) = delete;
    UniqueBuffer(UniqueBuffer&&)      = default;
    ~UniqueBuffer() { vmaDestroyBuffer(m_allocator, m_buffer, m_allocation); }

    UniqueBuffer& operator=(UniqueBuffer&&) = default;
    vk::Buffer    operator*() const { return get(); }
                  operator bool() const { return m_buffer; }

    vk::Buffer        get() const { return m_buffer; }
    vk::DeviceAddress deviceAddress(const vk::Device& device) const
    {
        return device.getBufferAddress(vk::BufferDeviceAddressInfo{
            .buffer = m_buffer,
        });
    }

    template <typename T>
    T* map() const
    {
        void* mappedMem = nullptr;
        vkCall(vmaMapMemory(m_allocator, m_allocation, &mappedMem));
        return static_cast<T*>(mappedMem);
    }

    void unmap() const { vmaUnmapMemory(m_allocator, m_allocation); }

  private:
    friend class Context;
    UniqueBuffer(VkBuffer buffer, VmaAllocation allocation, VmaAllocator allocator) :
        m_buffer(buffer), m_allocation(allocation), m_allocator(allocator)
    {}

    vk::Buffer    m_buffer;
    VmaAllocation m_allocation = nullptr;
    VmaAllocator  m_allocator  = nullptr;
};

struct Context
{
    explicit Context(const ContextParam& param);
    Context(const Context&) = delete;
    Context(Context&&)      = delete;

    ContextParam param;

    UniqueBuffer allocateBuffer(const vk::BufferCreateInfo&    bufferCreateInfo,
                                const VmaAllocationCreateInfo& allocCreateInfo) const;
    UniqueBuffer allocateBuffer(size_t size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage) const;

    std::vector<const char*> reqDeviceExtensions;
    std::vector<const char*> reqInstanceExtensions;
    std::vector<const char*> reqInstanceLayers;

    // CustomUniquePtr<GLFWwindow, glfwDestroyWindow> window;
    UniqueGLFWWindow window;

    vk::UniqueInstance               instance;
    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;

    PhysicalDeviceInfo physDevInfo;
    vk::UniqueDevice   device;

    Queues       queues;
    CommandPools commandPools;

    QueueInfo transferQueue; // specializes in transfers, should be faster than normal queue (at least with NVidia)

    UniqueVmaAllocator vmaAllocator;
};

} // namespace prism
