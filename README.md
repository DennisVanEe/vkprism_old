# vkPrism

vkPrism is a wavefront unidirectional physically based path tracer written in C++20 using Vulkan's ray tracing extension.

It is based on [pbrt-v4](https://github.com/mmp/pbrt-v4.git) by Pharr et al. along with any papers or ideas I have of my own (it's not a direct port from Optix to Vulkan).

vkPrism isn't designed to be a real-time renderer right now, it's focus is currently on photorealistic offline rendering (that's not to say it isn't suppose to be fast!).

It's only been tested with Visual Studio 2019 and 2022. It building with clang/g++ should be dependent only on what C++20 features have been implemented.

Only need CMake and Vulkan SDK to build, everything else should be included in the repo.
