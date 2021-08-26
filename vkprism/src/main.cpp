#include "main.hpp"

#include <iostream>

#include <glm/gtx/transform.hpp>
#include <spdlog/spdlog.h>

#include <context.hpp>
#include <allocator.hpp>
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

        const char* path = "C:\\Users\\jan\\Downloads\\mesh_00028.ply";
        // const char* path = "D:\\Dev\\pbrt-v4-scenes\\barcelona-pavilion\\geometry\\mesh_00014.ply";

        Scene      scene({});
        const auto meshId      = scene.createMesh(path);
        const auto meshInfo    = std::to_array({Scene::MeshGroup::MeshInfo{meshId}});
        const auto meshGroupId = scene.createMeshGroup(meshInfo);

        const Scene::Instance instance{
            .customId    = 0,
            .mask        = 1,
            .hitGroupId  = 1,
            .meshGroupId = meshGroupId,
            .transform   = Transform(glm::mat4(1.f)),
        };

        scene.createInstance(instance);

        scene.transferToGpu(ctx, allocator);

    } catch (const std::exception& e) {
        spdlog::error("Caught exception: {}", e.what());
    }

    std::cout << "Done!\n";
}
