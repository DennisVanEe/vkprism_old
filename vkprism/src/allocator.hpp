#pragma once

#include <ranges>
#include <span>

#include <vulkan/vulkan.hpp>

#include <context.hpp>

namespace prism {

// A buffer that is allocated by Context and is automatically destructed if it goes out of scope. This does mean that
// every buffer has to also store a pointer to the allocator, but until this becomes a problem, it'll just have to be.
// NOTE: the lifetime of Context should outlive everything, so m_allocator won't be destroyed before a UniqueBuffer
// is...
class UniqueBuffer
{
  public:
    UniqueBuffer() = default;

    UniqueBuffer(UniqueBuffer&& other) noexcept :
        m_buffer(other.m_buffer), m_allocation(other.m_allocation), m_allocator(other.m_allocator)
    {
        // After moving, the object should be in a valid state (due to destructors...).
        // In most cases, the compiler should be able to optimize this out (when returning a UniqueBuffer from a
        // function).
        other.m_buffer = VK_NULL_HANDLE;
    }

    UniqueBuffer& operator=(UniqueBuffer&& other) noexcept
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
    friend class GPUAllocator;
    UniqueBuffer(VkBuffer buffer, VmaAllocation allocation, VmaAllocator allocator) :
        m_buffer(buffer), m_allocation(allocation), m_allocator(allocator)
    {}

    vk::Buffer    m_buffer;
    VmaAllocation m_allocation = nullptr;
    VmaAllocator  m_allocator  = nullptr;
};

class GPUAllocator
{
  public:
    GPUAllocator(const Context& context);
    GPUAllocator(const GPUAllocator&) = delete;
    GPUAllocator(GPUAllocator&&)      = delete;

    UniqueBuffer allocateBuffer(const vk::BufferCreateInfo&    bufferCreateInfo,
                                const VmaAllocationCreateInfo& allocCreateInfo) const;
    UniqueBuffer allocateBuffer(size_t size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage) const;

  private:
    using UniqueVmaAllocator = CustomUniquePtr<std::remove_pointer_t<VmaAllocator>, vmaDestroyAllocator>;

    static UniqueVmaAllocator createVmaAllocator(const Context& context);

    UniqueVmaAllocator m_vmaAllocator;
};

// Adds the copy to buffer command, returning the temporary buffer used so that we can properly deallocate it when
// necessary.
template <typename T>
[[nodiscard]] UniqueBuffer addCopyToBufferCommand(const vk::CommandBuffer& commandBuffer,
                                                  const GPUAllocator& gpuAllocator, const UniqueBuffer& dstBuffers,
                                                  const T& srcData)
{
    const auto srcSize = srcData.size() * sizeof(T::value_type);

    auto stagingBuffer =
        gpuAllocator.allocateBuffer(srcSize, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);

    // Copy data to the staging buffer:
    std::ranges::copy(srcData, stagingBuffer.map<T::value_type>());
    stagingBuffer.unmap();

    commandBuffer.copyBuffer(*stagingBuffer, *dstBuffers,
                              vk::BufferCopy{
                                  .size = srcSize,
                              });

    return std::move(stagingBuffer);
}

} // namespace prism