#include "allocator.hpp"

namespace prism {

Allocator::UniqueVmaAllocator Allocator::createVmaAllocator(const Context& context)
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
        .physicalDevice   = context.physDevInfo.physicalDevice,
        .device           = *context.device,
        .pVulkanFunctions = &vkFunctions,
        .instance         = *context.instance,
        .vulkanApiVersion = VK_API_VERSION_1_2,
    };

    VmaAllocator allocator;
    vkCall(vmaCreateAllocator(&createInfo, &allocator));

    return UniqueVmaAllocator(allocator);
}

Allocator::Allocator(const Context& context) : m_vmaAllocator(createVmaAllocator(context)) {}

UniqueBuffer Allocator::allocateBuffer(const vk::BufferCreateInfo&    bufferCreateInfo,
                                     const VmaAllocationCreateInfo& allocCreateInfo) const
{
    const VkBufferCreateInfo& convBufferCreateInfo = bufferCreateInfo;

    VkBuffer      buffer;
    VmaAllocation allocation;
    vmaCreateBuffer(m_vmaAllocator.get(), &convBufferCreateInfo, &allocCreateInfo, &buffer, &allocation, nullptr);

    return UniqueBuffer(buffer, allocation, m_vmaAllocator.get());
}

UniqueBuffer Allocator::allocateBuffer(size_t size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage) const
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