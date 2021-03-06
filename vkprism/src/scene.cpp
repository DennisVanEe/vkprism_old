#include "scene.hpp"

#include <format>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <miniply.h>

#include <context.hpp>
#include <util.hpp>

namespace prism {

//
// SceneBuilder
//

void SceneBuilder::addCamera(std::unique_ptr<Camera> camera) { m_camera = std::move(camera); }

MeshIndex SceneBuilder::createMesh(const std::string_view filePath)
{
    const std::string  cstrFilepath(filePath);
    miniply::PLYReader plyReader(cstrFilepath.c_str());

    if (!plyReader.valid()) {
        // TODO: replace with std::format
        throw std::runtime_error("Could not open or parse PLY file at: " + std::string(filePath));
    }

    // The data we want to work with:
    const uint32_t facesOffset    = m_faces.size();
    const uint32_t verticesOffset = m_vertices.size();

    uint32_t                        numVertices, numFaces;
    std::unique_ptr<glm::vec3[]>    pos, nrm, tan;
    std::unique_ptr<glm::vec2[]>    uvs;
    std::unique_ptr<glm::u32vec3[]> faces;
    {
        // Store the position information used by ply reader to load values:
        std::array<uint32_t, 3> triIdx, vrtIdx;

        const auto faceElement = plyReader.get_element(plyReader.find_element(miniply::kPLYFaceElement));
        if (!faceElement) {
            // TODO: replace with std::format
            throw std::runtime_error("Could not find face elements for PLY file at: " + std::string(filePath));
        }
        faceElement->convert_list_to_fixed_size(faceElement->find_property("vertex_indices"), 3, triIdx.data());

        bool hasVertices = false, hasFaces = false;
        for (; plyReader.has_element() && (!hasVertices || !hasFaces); plyReader.next_element()) {
            // If it's a vertex element:
            if (plyReader.element_is(miniply::kPLYVertexElement) && plyReader.load_element()) {
                numVertices = plyReader.num_rows();

                // Check for position data:
                if (!plyReader.find_pos(vrtIdx.data())) {
                    // TODO: replace with std::format
                    throw std::runtime_error("Missing position data in PLY file at: " + std::string(filePath));
                }
                pos.reset(new glm::vec3[numVertices]);
                plyReader.extract_properties(vrtIdx.data(), 3, miniply::PLYPropertyType::Float, pos.get());

                // Check for normals:
                if (plyReader.find_normal(vrtIdx.data())) {
                    nrm.reset(new glm::vec3[numVertices]);
                    plyReader.extract_properties(vrtIdx.data(), 3, miniply::PLYPropertyType::Float, nrm.get());
                }
                // Check for tangents:
                if (plyReader.find_properties(vrtIdx.data(), 3, "tx", "ty", "tz")) {
                    tan.reset(new glm::vec3[numVertices]);
                    plyReader.extract_properties(vrtIdx.data(), 3, miniply::PLYPropertyType::Float, tan.get());
                }
                // Check for texture coordinates:
                if (plyReader.find_texcoord(vrtIdx.data())) {
                    uvs.reset(new glm::vec2[numVertices]);
                    plyReader.extract_properties(vrtIdx.data(), 2, miniply::PLYPropertyType::Float, uvs.get());
                }

                hasVertices = true;
            } else if (plyReader.element_is(miniply::kPLYFaceElement) && plyReader.load_element()) {
                numFaces = plyReader.num_rows();
                faces.reset(new glm::u32vec3[numFaces]);
                plyReader.extract_properties(triIdx.data(), 3, miniply::PLYPropertyType::Int, faces.get());

                hasFaces = true;
            }
        }

        if (!hasVertices || !hasFaces) {
            // TODO: replace with std::format
            throw std::runtime_error("Poorly formed PLY file at: " + std::string(filePath));
        }
    }

    std::copy(faces.get(), faces.get() + numFaces, std::back_inserter(m_faces));

    for (size_t i = 0; i < numVertices; ++i) {
        m_vertices.emplace_back(Vertex{
            .pos = pos[i],
            .nrm = nrm ? nrm[i] : glm::vec3(0.f),
            .tan = tan ? tan[i] : glm::vec3(0.f),
            .uvs = uvs ? uvs[i] : glm::vec2(0.f),
        });
    }

    const uint32_t meshId = m_meshes.size();
    m_meshes.emplace_back(Mesh{
        .nrm            = static_cast<bool>(nrm),
        .tan            = static_cast<bool>(tan),
        .uvs            = static_cast<bool>(uvs),
        .verticesOffset = verticesOffset,
        .numVertices    = numVertices,
        .facesOffset    = facesOffset,
        .numFaces       = numFaces,
    });

    return MeshIndex(meshId);
}

TransformIndex SceneBuilder::createTransform(const Transform& transform)
{
    const uint32_t id = m_transforms.size();
    m_transforms.emplace_back(static_cast<vk::TransformMatrixKHR>(transform));
    return TransformIndex(id);
}

MeshGroupIndex SceneBuilder::createMeshGroup(const std::span<const PlacedMesh> placedMeshes)
{
    const uint32_t id = m_meshGroups.size();
    m_meshGroups.emplace_back(placedMeshes.begin(), placedMeshes.end());
    return MeshGroupIndex(id);
}

InstanceIndex SceneBuilder::createInstance(const Instance& instance)
{
    const uint32_t id = m_instances.size();
    m_instances.emplace_back(instance);
    return InstanceIndex(id);
}

//
// Scene
//

Scene::Scene(const SceneParam& param, const Context& context, const GPUAllocator& allocator,
             const SceneBuilder& sceneBuilder)
{
    const auto commandPool = context.device().createCommandPoolUnique(vk::CommandPoolCreateInfo{
        .flags            = vk::CommandPoolCreateFlagBits::eTransient, // All of the command buffers will be short lived
        .queueFamilyIndex = context.queueFamilyIndex()});

    m_meshGpuData = transferMeshData(context, allocator, *commandPool, sceneBuilder.m_meshes, sceneBuilder.m_vertices,
                                     sceneBuilder.m_faces, sceneBuilder.m_transforms);
    m_blases      = createBlas(context, allocator, *commandPool, m_meshGpuData, sceneBuilder.m_meshes,
                          sceneBuilder.m_meshGroups, param.enableCompaction);
    m_tlas        = createTlas(context, allocator, *commandPool, sceneBuilder.m_instances, m_blases);

    m_cameraData       = transferCamera(context, *commandPool, allocator, sceneBuilder.m_camera.get());
    m_cameraShaderName = sceneBuilder.m_camera->getShaderName();
}

Scene::MeshGpuData Scene::transferMeshData(const Context& context, const GPUAllocator& gpuAllocator,
                                           const vk::CommandPool&                        commandPool,
                                           const std::span<const SceneBuilder::Mesh>     meshes,
                                           const std::span<const Vertex>                 vertices,
                                           const std::span<const glm::u32vec3>           faces,
                                           const std::span<const vk::TransformMatrixKHR> transforms)
{
    //
    // Allocate the command buffer (not very efficient to get vector invovled, but unless this becomes a problem I won't
    // bother change it).

    const auto commandBuffer = std::move(context.device().allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
        .commandPool        = commandPool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    })[0]);

    commandBuffer->begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    //
    // Transfer mesh vertices and faces:

    //
    // Allocate buffers on the GPU where we'll send the data:
    const auto blasUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                           vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    auto gpuVertices =
        gpuAllocator.allocateBuffer(sizeof(Vertex) * vertices.size(), blasUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    auto gpuFaces =
        gpuAllocator.allocateBuffer(sizeof(glm::u32vec3) * faces.size(), blasUsage, VMA_MEMORY_USAGE_GPU_ONLY);

    const auto stagingVertices = addCopyToBufferCommand(*commandBuffer, gpuAllocator, gpuVertices, vertices);
    const auto stagingFaces    = addCopyToBufferCommand(*commandBuffer, gpuAllocator, gpuFaces, faces);

    // Transforms are optional, so we check for them, but we need to keep staging transforms on the stack:
    auto [stagingTransforms, gpuTransforms] = [&]() {
        if (!transforms.empty()) {
            auto gpuTransforms     = gpuAllocator.allocateBuffer(sizeof(vk::TransformMatrixKHR) * transforms.size(),
                                                             blasUsage, VMA_MEMORY_USAGE_GPU_ONLY);
            auto stagingTransforms = addCopyToBufferCommand(*commandBuffer, gpuAllocator, gpuTransforms, transforms);

            return std::make_tuple(std::move(stagingTransforms), std::move(gpuTransforms));
        }
        return std::make_tuple(UniqueBuffer{}, UniqueBuffer{});
    }();

    submitAndWait(context, *commandBuffer, "sending mesh data to the GPU");

    return MeshGpuData{
        .vertices   = std::move(gpuVertices),
        .faces      = std::move(gpuFaces),
        .transforms = std::move(gpuTransforms),
    };
}

std::vector<Scene::AccelStructInfo> Scene::createBlas(const Context& context, const GPUAllocator& allocator,
                                                      const vk::CommandPool&                         commandPool,
                                                      const MeshGpuData&                             meshGpuData,
                                                      const std::span<const SceneBuilder::Mesh>      meshes,
                                                      const std::span<const std::vector<PlacedMesh>> meshGroups,
                                                      const bool                                     enableCompaction)
{
    const auto gpuVerticesAddr = meshGpuData.vertices.deviceAddress(context.device());
    const auto gpuFacesAddr    = meshGpuData.faces.deviceAddress(context.device());
    const auto gpuTransformsAddr =
        meshGpuData.transforms ? meshGpuData.transforms.deviceAddress(context.device()) : vk::DeviceAddress{};

    // Stores structures required for the mesh acceleration structure:
    std::vector<vk::AccelerationStructureGeometryKHR>       geometries;
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> buildRangeInfos;

    for (const auto& meshGroup : meshGroups) {
        for (const auto& [meshIdx, transformIdx] : meshGroup) {
            const auto& mesh = meshes[meshIdx];

            geometries.emplace_back(
                vk::AccelerationStructureGeometryKHR{
                    .geometryType = vk::GeometryTypeKHR::eTriangles,
                    .geometry =
                        vk::AccelerationStructureGeometryTrianglesDataKHR{
                            .vertexFormat = vk::Format::eR32G32B32A32Sfloat, // glm::vec3
                            .vertexData =
                                vk::DeviceOrHostAddressConstKHR{.deviceAddress = gpuVerticesAddr +
                                                                                 sizeof(Vertex) * mesh.verticesOffset},
                            .vertexStride = sizeof(Vertex),
                            .maxVertex    = mesh.numVertices - 1,
                            .indexType    = vk::IndexType::eUint32,
                            .indexData =
                                vk::DeviceOrHostAddressConstKHR{.deviceAddress = gpuFacesAddr + sizeof(glm::u32vec3) *
                                                                                                    mesh.facesOffset},

                            // We specify the offset in buildRangeInfos:
                            .transformData = transformIdx
                                                 ? vk::DeviceOrHostAddressConstKHR{.deviceAddress = gpuTransformsAddr}
                                                 : vk::DeviceOrHostAddressConstKHR{},
                        },

                    .flags = vk::GeometryFlagBitsKHR::eOpaque});
            buildRangeInfos.emplace_back(vk::AccelerationStructureBuildRangeInfoKHR{
                .primitiveCount  = static_cast<uint32_t>(mesh.numFaces),
                .transformOffset = transformIdx ? *transformIdx : 0u,
            });
        }
    }

    std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> buildGeometryInfos;
    buildGeometryInfos.reserve(meshGroups.size());

    size_t currGeometryOffset = 0;
    for (const auto& meshGroup : meshGroups) {
        const auto flags = enableCompaction ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                                                  vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction
                                            : vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;

        buildGeometryInfos.emplace_back(vk::AccelerationStructureBuildGeometryInfoKHR{
            .type          = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags         = flags,
            .mode          = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = static_cast<uint32_t>(meshGroup.size()),
            .pGeometries   = &geometries[currGeometryOffset],
        });

        currGeometryOffset += meshGroup.size();
    }

    //
    // Loop over the BLAS structures we are building and check how much memory we need to construct them.

    std::vector<AccelStructInfo> blases;
    blases.reserve(meshGroups.size());

    // We keep track of the maximum amount of scratch space we need to allocate to create the acceleration structure.
    size_t maxScratchSize = 0;
    for (uint32_t i = 0; i < meshGroups.size(); ++i) {
        const auto& meshGroup         = meshGroups[i];
        auto&       buildGeometryInfo = buildGeometryInfos[i];

        std::vector<uint32_t> maxPrimitiveCounts;
        maxPrimitiveCounts.reserve(meshGroup.size());
        for (const auto& [meshIdx, _] : meshGroup) {
            maxPrimitiveCounts.emplace_back(meshes[meshIdx].numFaces);
        }

        const auto buildSizeInfo = context.device().getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, maxPrimitiveCounts);

        // Allocate space for the acceleration structure:
        auto accelStructBuff = allocator.allocateBuffer(buildSizeInfo.accelerationStructureSize,
                                                        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                                            vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                                        VMA_MEMORY_USAGE_GPU_ONLY);

        auto accelStruct = context.device().createAccelerationStructureKHRUnique(vk::AccelerationStructureCreateInfoKHR{
            //.createFlags, TODO: figure out if this is required or not... (I don't think it is).
            .buffer = *accelStructBuff,
            .size   = buildSizeInfo.accelerationStructureSize,
            .type   = vk::AccelerationStructureTypeKHR::eBottomLevel,
        });

        // Now that we have created it, we can set the destination location:
        buildGeometryInfo.dstAccelerationStructure = *accelStruct;

        blases.emplace_back(std::move(accelStructBuff), std::move(accelStruct));
        maxScratchSize = std::max(maxScratchSize, buildSizeInfo.buildScratchSize);
    }

    // We need the query pool if we are performing compaction as we need to know the new sizes of the BLAS:
    const auto queryPool = enableCompaction ? context.device().createQueryPoolUnique(vk::QueryPoolCreateInfo{
                                                  .queryType  = vk::QueryType::eAccelerationStructureCompactedSizeKHR,
                                                  .queryCount = static_cast<uint32_t>(meshGroups.size()),
                                              })
                                            : vk::UniqueQueryPool{};

    // As the Nvidia tutorial explains, we don't want windows to time-out the execution of a single command buffer when
    // processing many BLAS. To overcome this potential problem, we create a command buffer for each BLAS structure. As
    // the hardware (apperantely) can't create BLASes in parallel, we can also use just one scratch pad.

    // We have to manually manage these as otherwise it's a pain to submit them later if they were all unique handles.
    const auto commandBuffers = context.device().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
        .commandPool        = commandPool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(meshGroups.size()),
    });
    // Defer the destruction here:
    const Defer commandBufferDestructor([&]() { context.device().freeCommandBuffers(commandPool, commandBuffers); });

    // Allocate enough scratch space to construct the acceleration structure:
    const auto scratchBuffer = allocator.allocateBuffer(
        maxScratchSize, vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer,
        VMA_MEMORY_USAGE_GPU_ONLY);
    const auto scratchBufferAddr = scratchBuffer.deviceAddress(context.device());

    for (size_t i = 0, currGeometryOffset = 0; i < meshGroups.size(); ++i) {
        auto&       buildGeometryInfo = buildGeometryInfos[i];
        const auto& meshGroup         = meshGroups[i];
        const auto& commandBuffer     = commandBuffers[i];

        // Start recording:
        commandBuffer.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        });

        buildGeometryInfo.scratchData.deviceAddress = scratchBufferAddr;

        // Apparently we need an array of pointers to the range info, so we make sure to add that here:
        std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> buildRangeInfoPtrs;
        buildRangeInfoPtrs.reserve(meshGroup.size());
        for (const auto& rangeInfo : std::span(buildRangeInfos).subspan(currGeometryOffset, meshGroup.size())) {
            buildRangeInfoPtrs.emplace_back(&rangeInfo);
        }

        // Record it:
        commandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, buildRangeInfoPtrs);

        // Because we are using one scratch buffer, we have to add a barier to between these commands to make sure we
        // finish constructing the BLAS before we start the next one.
        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::DependencyFlags{},
            vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                              .dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR},
            {}, {});

        currGeometryOffset += meshGroup.size();
    }

    submitAndWait(context, commandBuffers, "BLAS construction");

    // If we turned compaction on, then we can move the values over:
    if (enableCompaction) {
        // TODO: finish this later...
    }

    return blases;
}

Scene::AccelStructInfo Scene::createTlas(const Context& context, const GPUAllocator& gpuAllocator,
                                         const vk::CommandPool& commandPool, const std::span<const Instance> instances,
                                         const std::span<const AccelStructInfo> blases)
{
    //
    // Convert instances to vulkan instances.

    std::vector<vk::AccelerationStructureInstanceKHR> vkInstances;
    vkInstances.reserve(instances.size());

    for (const auto& instance : instances) {
        vkInstances.emplace_back(vk::AccelerationStructureInstanceKHR{
            .transform                              = static_cast<vk::TransformMatrixKHR>(instance.transform),
            .instanceCustomIndex                    = instance.customId,
            .mask                                   = instance.mask,
            .instanceShaderBindingTableRecordOffset = instance.hitGroupId,
            .flags                                  = static_cast<vk::GeometryInstanceFlagsKHR::MaskType>(
                vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable),
            .accelerationStructureReference =
                context.device().getAccelerationStructureAddressKHR(vk::AccelerationStructureDeviceAddressInfoKHR{
                    .accelerationStructure = *blases[instance.meshGroupIdx].accelStruct,
                }),
        });
    }

    const uint32_t numInstances = instances.size();

    const auto commandBuffer = std::move(context.device().allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
        .commandPool        = commandPool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    })[0]);

    commandBuffer->begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    //
    // Record the command to copy the instance data to the GPU.

    const auto gpuInstances = gpuAllocator.allocateBuffer(
        sizeof(vk::AccelerationStructureInstanceKHR) * instances.size(),
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY);

    const auto stagingInstances = addCopyToBufferCommand(*commandBuffer, gpuAllocator, gpuInstances, vkInstances);

    //
    // Make sure the instances data is copied to the GPU before we start constructing the TLAS:

    commandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        vk::DependencyFlags{},
        vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                          .dstAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR},
        {}, {});

    //
    // Allocate the required memory for constructing the TLAS:

    const vk::AccelerationStructureGeometryKHR geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry =
            vk::AccelerationStructureGeometryDataKHR{.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
                                                         .arrayOfPointers = VK_FALSE,
                                                         .data = gpuInstances.deviceAddress(context.device()),
                                                     }}};

    vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{
        .type          = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode          = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = numInstances,
        .pGeometries   = &geometry,
    };

    const auto buildSizeInfo = context.device().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, numInstances);

    // Allocate the scratch buffer:
    const auto scratchBuffer = gpuAllocator.allocateBuffer(buildSizeInfo.buildScratchSize,
                                                           vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                                               vk::BufferUsageFlagBits::eStorageBuffer,
                                                           VMA_MEMORY_USAGE_GPU_ONLY);

    // Allocate the buffer where we will store the tlas:
    auto tlasBuffer = gpuAllocator.allocateBuffer(buildSizeInfo.accelerationStructureSize,
                                                  vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                                      vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                                  VMA_MEMORY_USAGE_GPU_ONLY);

    //
    // Record the creation of the TLAS in the command buffer:

    auto tlasAccelStruct = context.device().createAccelerationStructureKHRUnique(vk::AccelerationStructureCreateInfoKHR{
        //.createFlags, TODO: figure out if this is required or not... (I don't think it is).
        .buffer = *tlasBuffer,
        .size   = buildSizeInfo.accelerationStructureSize,
        .type   = vk::AccelerationStructureTypeKHR::eTopLevel,
    });

    buildGeometryInfo.dstAccelerationStructure  = *tlasAccelStruct;
    buildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress(context.device());

    const vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{.primitiveCount = numInstances};
    commandBuffer->buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);

    submitAndWait(context, *commandBuffer, "TLAS construction");

    return AccelStructInfo{.buffer = std::move(tlasBuffer), .accelStruct = std::move(tlasAccelStruct)};
}

UniqueBuffer Scene::transferCamera(const Context& context, const vk::CommandPool& commandPool,
                                   const GPUAllocator& gpuAllocator, const Camera* camera)
{
    const auto commandBuffer = std::move(context.device().allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
        .commandPool        = commandPool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    })[0]);

    commandBuffer->begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    const auto cameraShaderData = camera->getCameraShaderData();

    auto gpuShaderData = gpuAllocator.allocateBuffer(
        cameraShaderData.size(), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eUniformBuffer,
        VMA_MEMORY_USAGE_GPU_ONLY);

    const auto stagingShaderData =
        addCopyToBufferCommand(*commandBuffer, gpuAllocator, gpuShaderData, cameraShaderData);

    submitAndWait(context, *commandBuffer, "Transfering camera data");

    return gpuShaderData;
}

} // namespace prism
