#pragma once

#include <vulkan/vulkan.hpp>

#include <context.hpp>

namespace prism {

struct Descriptor
{
    Descriptor(const Context& context, vk::ArrayProxy<const vk::DescriptorSetLayoutBinding> bindings);

    vk::UniqueDescriptorSetLayout setLayout;
    vk::UniqueDescriptorPool      pool;
    vk::DescriptorSet             set;
};

} // namespace prism