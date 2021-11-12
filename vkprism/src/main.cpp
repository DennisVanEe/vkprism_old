#include "main.hpp"

#include <iostream>

#include <glm/gtx/transform.hpp>
#include <spdlog/spdlog.h>

#include <allocator.hpp>
#include <context.hpp>
#include <scene.hpp>
#include <pipelines.hpp>

using namespace prism;

int main(const int argc, const char** const argv)
{
    ContextParam param{};
    param.enableCallback   = true;
    param.enableValidation = true;

    try {
        const Context      ctx(param);
        const GPUAllocator allocator(ctx);

        // Create a simple scene:
        const auto scene = [&]() {
            SceneBuilder sceneBuilder;

            const char* path = "D:\\Dev\\vkprism\\test_files\\sphere.ply";

            const auto meshIdx      = sceneBuilder.createMesh(path);
            const auto meshGroupIdx = sceneBuilder.createMeshGroup(std::to_array({PlacedMesh{.meshIdx = meshIdx}}));
            const auto instanceIdx  = sceneBuilder.createInstance(Instance{
                .customId     = 0,
                .mask         = 1,
                .hitGroupId   = 1,
                .meshGroupIdx = meshGroupIdx,
                .transform    = Transform(glm::mat4(1.f)),
            });

            return Scene({}, ctx, allocator, sceneBuilder);
        }();

        const Pipelines pipeline(PipelineParam{.outputWidth = 1920, .outputHeight = 1080}, ctx, allocator, scene);

    } catch (const std::exception& e) {
        spdlog::error("Caught exception: {}", e.what());
    }

    std::cout << "Done!\n";
}
