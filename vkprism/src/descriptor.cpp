#include "descriptor.hpp"

namespace prism {

Descriptor::Descriptor(const Context& context, vk::ArrayProxy<const vk::DescriptorSetLayoutBinding> bindings)
{
    setLayout = context.device().createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()});

    // Return the pool sizes, essentially combining descriptors of the same type:
    const auto poolSizes = [&]() {
        std::vector<vk::DescriptorPoolSize> poolSizes;

        for (const auto& binding : bindings) {
            auto itr = std::ranges::find_if(
                poolSizes, [&](const auto& poolSize) { return poolSize.type == binding.descriptorType; });
            if (itr == poolSizes.end()) {
                poolSizes.emplace_back(binding.descriptorType, binding.descriptorCount);
            } else {
                itr->descriptorCount += binding.descriptorCount;
            }
        }

        return poolSizes;
    }();

    pool = context.device().createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo{
        .maxSets = 1, .poolSizeCount = static_cast<uint32_t>(poolSizes.size()), .pPoolSizes = poolSizes.data()});

    set = context.device().allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo{.descriptorPool = *pool, .descriptorSetCount = 1, .pSetLayouts = &*setLayout})[0];
}

} // namespace prism