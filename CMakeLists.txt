# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.16)

project ("vkprism")

# Extern
add_subdirectory("extern/glfw")
add_subdirectory("extern/glm")
add_subdirectory("extern/spdlog")

add_subdirectory ("vkprism")
