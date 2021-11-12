#include "main.hpp"

#include <iostream>
#include <fstream>

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
                .transform    = Transform(glm::mat4(1.0)),
            });

            return Scene({}, ctx, allocator, sceneBuilder);
        }();

        const Pipelines pipeline({.outputWidth = 1920, .outputHeight = 1080}, ctx, allocator, scene);
        
        const auto commandPool = ctx.device().createCommandPoolUnique(
            vk::CommandPoolCreateInfo{
                .flags = vk::CommandPoolCreateFlagBits::eTransient, // All of the command buffers will be short lived
                .queueFamilyIndex = ctx.queueFamilyIndex()
            });

        const auto commandBuffers = ctx.device().allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
            .commandPool        = *commandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 2,
        });

        commandBuffers[0]->begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        
        pipeline.addBindRTPipelineCmd(*commandBuffers[0], {.width = 1920, .height = 1080});

        submitAndWait(ctx, *commandBuffers[0], "Perform RT Pipeline Stuff construction");

        //
        // Copy Data Back to Us to Read:

        commandBuffers[1]->begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

        // Allocate a buffer to put the resulting image:
        const auto stuffSize = sizeof(glm::vec3) * 1920 * 1080;
        auto dstBuffer =
            allocator.allocateBuffer(sizeof(glm::vec3) * 1920 * 1080, vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_CPU_ONLY);

        commandBuffers[1]->copyBuffer(pipeline.getBeautyBuffer(), *dstBuffer,
            vk::BufferCopy{
                .size = stuffSize,
            });

        submitAndWait(ctx, *commandBuffers[1], "Copy Beauty to host");

        const auto dstData = dstBuffer.map<glm::vec3>();

        std::ofstream outputImage("temp.ppm");
        outputImage << "P3\n";
        outputImage << "1920 1080\n";
        outputImage << "255\n";

        for (int i = 0; i < 1920 * 1080; ++i) {
            glm::uvec3 d = glm::uvec3(dstData[i] * 255.f);
            outputImage << d.r << " " << d.g << " " << d.b << "\n";
        }

        dstBuffer.unmap();

    } catch (const std::exception& e) {
        spdlog::error("Caught exception: {}", e.what());
    }

    std::cout << "Done!\n";
}
