#pragma once

#include <format>
#include <memory>
#include <source_location>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace prism {

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#elif defined(__GNUC__)
#define FORCEINLINE inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
#if __has_attribute(__always_inline__)
#define FORCEINLINE inline __attribute__((__always_inline__))
#else
#define FORCEINLINE inline
#endif
#else
#define FORCEINLINE inline
#endif

// Allows for wrapping a function pointer to be a functor (so we don't have to copy around function pointers everywhere.
template <typename T, auto F>
using CustomUniquePtr = std::unique_ptr<T, std::integral_constant<decltype(F), F>>;

FORCEINLINE void vkCall(vk::Result result, const std::source_location srcLoc = std::source_location::current())
{
    if (result == vk::Result::eSuccess) {
        return;
    }

    throw std::runtime_error(std::format("Received error code: {} in file: {} in function: {} on line: {}",
                                         vk::to_string(result), srcLoc.file_name(), srcLoc.function_name(),
                                         srcLoc.line()));
}

FORCEINLINE void vkCall(VkResult result, const std::source_location srcLoc = std::source_location::current())
{
    return vkCall(vk::Result(result), srcLoc);
}

// A very basic defer class that executes a function when it's destructor is called. This is useful for resources that
// need to be manually maintained (especially with exceptions).
template <typename F>
class Defer
{
  public:
    Defer(F func) : m_func(func) {}
    Defer(const Defer&) = delete;
    Defer(Defer&&)      = delete;
    ~Defer() { m_func(); }

  private:
    F m_func;
};

} // namespace prism