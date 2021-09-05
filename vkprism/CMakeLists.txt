project(prism VERSION 1.0.0 LANGUAGES CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options("/std:c++20")
endif()

# Add source to this project's executable.
add_executable(vkprism
    "src/main.cpp"
    "src/main.hpp"
    "src/context.hpp"
    "src/context.cpp"
    "src/util.hpp" 
    "src/util.cpp"
    "src/scene.hpp"
    "src/scene.cpp"
    "src/scene.hpp"
    "src/transform.hpp"
    "src/transform.cpp"
    "src/allocator.hpp"
    "src/allocator.cpp"
    "src/pipelines.hpp"
    "src/pipelines.cpp"
    "src/shaders.hpp"
    "src/shaders.cpp"
    # Any external libraries:
    "extern/vma/vk_mem_alloc.cpp"
    "extern/miniply/miniply.cpp")

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_target_properties(vkprism
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES)
endif()

target_include_directories(vkprism PUBLIC "${PROJECT_BINARY_DIR}/include/") # To allow us to include the configure.hpp file
target_include_directories(vkprism PUBLIC "src/")
target_include_directories(vkprism PUBLIC "extern/vma/")
target_include_directories(vkprism PUBLIC "extern/miniply/")

# For now it'll just be a headless renderer:
# target_link_libraries(vkprism glfw)
target_link_libraries(vkprism glm)
target_link_libraries(vkprism spdlog)

# We need the SDK with support for ray-tracing:
add_compile_definitions(
    GLFW_INCLUDE_VULKAN
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS # TODO: when the newer SDK is released, replace this
    VMA_STATIC_VULKAN_FUNCTIONS=0     # We don't want to statically link against vulkan (we use dynamic for everything)
    VMA_DYNAMIC_VULKAN_FUNCTIONS=0    # We want to manually let VMA to fail an assertion if I'm missing a function
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

# As we are loading this dynamically, this isn't actually required:
find_package(Vulkan 1.2.162 REQUIRED)
#target_link_libraries(vkprism Vulkan::Vulkan) Don't want to link against Vulkan statically
target_include_directories(vkprism PUBLIC ${Vulkan_INCLUDE_DIRS})

target_compile_definitions(vkprism PUBLIC -DGLFW_INCLUDE_NONE)

configure_file("configure.hpp.in" "${PROJECT_BINARY_DIR}/include/configure.hpp")

#
# Compile Shaders to SPIR-V:
#

# Took this from some reddit post with some slight modifications:
function(add_shader SHADER)
    find_program(GLSLC glslc)

    set(current-shader-path ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${SHADER})
    set(current-output-path ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER}.spv)

    # Add a custom command to compile GLSL to SPIR-V.
    get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
    file(MAKE_DIRECTORY ${current-output-dir})

    add_custom_command(
           OUTPUT ${current-output-path}
           COMMAND ${GLSLC} -o ${current-output-path} ${current-shader-path}
           DEPENDS ${current-shader-path}
           IMPLICIT_DEPENDS CXX ${current-shader-path}
           VERBATIM)

    # Make sure our build depends on this output.
    set_source_files_properties(${current-output-path} PROPERTIES GENERATED TRUE)
    target_sources(vkprism PRIVATE ${current-output-path})
endfunction(add_shader)

add_shader("raygen.glsl.comp")
add_shader("raymiss.glsl.comp")
add_shader("rayclosesthit.glsl.comp")