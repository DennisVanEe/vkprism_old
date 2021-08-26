#pragma once

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
    friend class Allocator;
    UniqueBuffer(VkBuffer buffer, VmaAllocation allocation, VmaAllocator allocator) :
        m_buffer(buffer), m_allocation(allocation), m_allocator(allocator)
    {}

    vk::Buffer    m_buffer;
    VmaAllocation m_allocation = nullptr;
    VmaAllocator  m_allocator  = nullptr;
};

class Allocator
{
  public:
    Allocator(const Context& context);
    Allocator(const Allocator&) = delete;
    Allocator(Allocator&&)      = delete;

    UniqueBuffer allocateBuffer(const vk::BufferCreateInfo&    bufferCreateInfo,
                                const VmaAllocationCreateInfo& allocCreateInfo) const;
    UniqueBuffer allocateBuffer(size_t size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage) const;

  private:
    using UniqueVmaAllocator = CustomUniquePtr<std::remove_pointer_t<VmaAllocator>, vmaDestroyAllocator>;

    static UniqueVmaAllocator createVmaAllocator(const Context& context);

    UniqueVmaAllocator m_vmaAllocator;
};

} // namespace prism