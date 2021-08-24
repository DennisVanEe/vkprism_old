#pragma once

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
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR

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

using UniqueGLFWWindow   = CustomUniquePtr<GLFWwindow, glfwDestroyWindow>;
using UniqueVmaAllocator = CustomUniquePtr<std::remove_pointer_t<VmaAllocator>, vmaDestroyAllocator>;

// A buffer that is allocated by Context and is automatically destructed if it goes out of scope. This does mean that
// every buffer has to also store a pointer to the allocator, but until this becomes a problem, it'll just have to be.
// NOTE: the lifetime of Context should outlive everything, so m_allocator won't be destroyed before a UniqueBuffer
// is...
class UniqueBuffer
{
  public:
    UniqueBuffer() = default;

    UniqueBuffer(UniqueBuffer&& other) :
        m_buffer(other.m_buffer), m_allocation(other.m_allocation), m_allocator(other.m_allocator)
    {
        // After moving, the object should be in a valid state (due to destructors...).
        // In most cases, the compiler should be able to optimize this out (when returning a UniqueBuffer from a
        // function).
        other.m_buffer = VK_NULL_HANDLE;
    }

    UniqueBuffer& operator=(UniqueBuffer&& other)
    {
        if (this == &other) {
            return *this;
        }

        m_buffer       = other.m_buffer;
        m_allocation   = other.m_allocation;
        m_allocator    = other.m_allocator;
        other.m_buffer = VK_NULL_HANDLE;
        return *this;
    }

    UniqueBuffer(const UniqueBuffer&) = delete;
    UniqueBuffer& operator=(const UniqueBuffer&) = delete;

    ~UniqueBuffer()
    {
        if (m_buffer) {
            vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        }
    }

    vk::Buffer operator*() const { return get(); }
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

    vk::Queue queue;
    uint32_t  queueFamilyIdx;

    UniqueVmaAllocator vmaAllocator;
};

} // namespace prism
