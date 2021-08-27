#include "main.hpp"

#include <iostream>

#include <glm/gtx/transform.hpp>
#include <spdlog/spdlog.h>

#include <allocator.hpp>
#include <context.hpp>
#include <scene.hpp>

using namespace prism;

int main(const int argc, const char** const argv)
{
    ContextParam param{};
    param.enableCallback   = true;
    param.enableValidation = true;

    try {
        const Context   ctx(param);
        const Allocator allocator(ctx);

        const char* path = "D:\\dennis_stuff\\mesh_00022.ply";
        // const char* path = "D:\\Dev\\pbrt-v4-scenes\\barcelona-pavilion\\geometry\\mesh_00014.ply";

        Scene      scene({});
        const auto mesh      = scene.createMesh(path);
        const auto meshGroup = scene.createMeshGroup(std::to_array({MeshGroup::MeshInfo{mesh}}));

        const Instance instance{
            .customId   = 0,
            .mask       = 1,
            .hitGroupId = 1,
            .meshGroup  = meshGroup,
            .transform  = Transform(glm::mat4(1.f)),
        };

        scene.createInstance(instance);

        scene.transferToGpu(ctx, allocator);

    } catch (const std::exception& e) {
        spdlog::error("Caught exception: {}", e.what());
    }

    std::cout << "Done!\n";
}
