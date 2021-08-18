#include "main.hpp"

#include <iostream>

#include <glm/gtx/transform.hpp>
#include <spdlog/spdlog.h>

#include <context.hpp>
#include <scene.hpp>

using namespace prism;

int main(const int argc, const char** const argv)
{
    ContextParam param{};
    param.enableCallback   = true;
    param.enableValidation = true;

    try {
        Context ctx(param);

        const char* path = "C:\\Users\\jan\\Downloads\\mesh_00028.ply";
        //const char* path = "D:\\Dev\\pbrt-v4-scenes\\barcelona-pavilion\\geometry\\mesh_00014.ply";

        Scene scene;
        const auto meshId = scene.loadMesh(path);
        const auto meshInfo = std::to_array({Scene::MeshGroup::MeshInfo{meshId}});
        scene.loadMeshGroup(meshInfo);

        scene.transferToGpu(ctx);

    } catch (const std::exception& e) {
        spdlog::error("Caught exception: {}", e.what());
    }

    std::cout << "Done!\n";
}
