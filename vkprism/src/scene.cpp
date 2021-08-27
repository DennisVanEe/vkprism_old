#include "scene.hpp"

#include <format>
#include <ranges>
#include <string>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <miniply.h>

#include <context.hpp>
#include <util.hpp>

namespace prism {

constexpr uint64_t FENCE_TIMEOUT = 6e+10; // 1 minute (not sure how long this should be...)

void Scene::createTlas(const Context& context, const Allocator& allocator, const vk::CommandPool& commandPool)
{
    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    instances.reserve(m_instances.size());

    for (const auto& instance : m_instances) {
        instances.emplace_back(vk::AccelerationStructureInstanceKHR{
            .transform                              = static_cast<vk::TransformMatrixKHR>(instance.transform),
            .instanceCustomIndex                    = instance.customId,
            .mask                                   = instance.mask,
            .instanceShaderBindingTableRecordOffset = instance.hitGroupId,
            .flags                                  = static_cast<vk::GeometryInstanceFlagsKHR::MaskType>(
                vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable),
            .accelerationStructureReference =
                context.device().getAccelerationStructureAddressKHR(vk::AccelerationStructureDeviceAddressInfoKHR{
                    .accelerationStructure = *m_blas[instance.meshGroupId].accelStruct,
                }),
        });
    }

    const auto commandBuffer = std::move(context.device().allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
        .commandPool        = commandPool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    })[0]);

    commandBuffer->begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    //
    // Record the command to copy the instance data to the GPU.

    const auto sizeOfInstancesBuff = sizeof(vk::AccelerationStructureInstanceKHR) * instances.size();

    const auto stagingInstances =
        allocator.allocateBuffer(sizeOfInstancesBuff, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);

    std::ranges::copy(instances, stagingInstances.map<vk::AccelerationStructureInstanceKHR>());
    stagingInstances.unmap();

    const auto gpuInstances = allocator.allocateBuffer(
        sizeOfInstancesBuff, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY);

    commandBuffer->copyBuffer(*stagingInstances, *gpuInstances,
                              vk::BufferCopy{
                                  .size = sizeOfInstancesBuff,
                              });

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

    const vk::AccelerationStructureGeometryKHR geometry{.geometryType = vk::GeometryTypeKHR::eInstances,
                                                        .geometry = vk::AccelerationStructureGeometryInstancesDataKHR{
                                                            .arrayOfPointers = VK_FALSE,
                                                            .data = gpuInstances.deviceAddress(context.device()),
                                                        }};

    vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{
        .type          = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode          = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = static_cast<uint32_t>(instances.size()),
        .pGeometries   = &geometry,
    };

    const auto buildSizeInfo = context.device().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo);

    // Allocate the scratch buffer:
    const auto scratchBuffer = allocator.allocateBuffer(buildSizeInfo.buildScratchSize,
                                                        vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                                            vk::BufferUsageFlagBits::eStorageBuffer,
                                                        VMA_MEMORY_USAGE_GPU_ONLY);

    m_tlas.buffer = allocator.allocateBuffer(buildSizeInfo.accelerationStructureSize,
                                             vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                                 vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                             VMA_MEMORY_USAGE_GPU_ONLY);

    //
    // Record the creation of the TLAS in the command buffer:

    m_tlas.accelStruct = context.device().createAccelerationStructureKHRUnique(vk::AccelerationStructureCreateInfoKHR{
        //.createFlags, TODO: figure out if this is required or not... (I don't think it is).
        .buffer = *m_tlas.buffer,
        .size   = buildSizeInfo.accelerationStructureSize,
        .type   = vk::AccelerationStructureTypeKHR::eTopLevel,
    });

    buildGeometryInfo.dstAccelerationStructure  = *m_tlas.accelStruct;
    buildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress(context.device());

    const vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{.primitiveCount =
                                                                        static_cast<uint32_t>(instances.size())};

    commandBuffer->buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);

    commandBuffer->end();

    //
    // Execute the command buffer and wait for it to finish:

    const auto fence = context.device().createFenceUnique({});

    context.queue().submit(
        vk::SubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers    = &*commandBuffer,
        },
        *fence);

    if (context.device().waitForFences(*fence, VK_TRUE, FENCE_TIMEOUT) == vk::Result::eTimeout) {
        throw std::runtime_error("Fence timed out when waiting for TLAS construction command.");
    }
}

void Scene::createBlas(const Context& context, const Allocator& allocator, const vk::CommandPool& commandPool)
{
    const auto gpuVerticesAddr = m_gpuVertices.deviceAddress(context.device());
    const auto gpuFacesAddr    = m_gpuFaces.deviceAddress(context.device());
    const auto gpuTransformsAddr =
        m_gpuTransforms ? m_gpuTransforms.deviceAddress(context.device()) : vk::DeviceAddress{};

    // Stores structures required for the mesh acceleration structure:
    std::vector<vk::AccelerationStructureGeometryKHR>       geometries;
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> buildRangeInfos;

    for (const auto& meshGroup : m_meshGroups) {
        for (const auto& [meshId, transformId] : meshGroup.meshes) {
            const auto& mesh = m_meshes[meshId];

            geometries.emplace_back(vk::AccelerationStructureGeometryKHR{
                .geometryType = vk::GeometryTypeKHR::eTriangles,
                .geometry =
                    vk::AccelerationStructureGeometryTrianglesDataKHR{
                        .vertexFormat = vk::Format::eR32G32B32A32Sfloat, // glm::vec3
                        .vertexData   = vk::DeviceOrHostAddressConstKHR().setDeviceAddress(
                            gpuVerticesAddr + sizeof(Vertex) * mesh.verticesOffset),
                        .vertexStride = sizeof(Vertex),
                        .maxVertex    = mesh.numVertices - 1,
                        .indexType    = vk::IndexType::eUint32,
                        .indexData    = vk::DeviceOrHostAddressConstKHR().setDeviceAddress(
                            gpuFacesAddr + sizeof(glm::u32vec3) * mesh.facesOffset),

                        // We specify the offset in buildRangeInfos:
                        .transformData = transformId
                                             ? vk::DeviceOrHostAddressConstKHR{}.setDeviceAddress(gpuTransformsAddr)
                                             : vk::DeviceOrHostAddressConstKHR{},
                    },

                .flags = vk::GeometryFlagBitsKHR::eOpaque});
            buildRangeInfos.emplace_back(vk::AccelerationStructureBuildRangeInfoKHR{
                .primitiveCount  = static_cast<uint32_t>(mesh.numFaces),
                .transformOffset = transformId ? *transformId : 0,
            });
        }
    }

    std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> buildGeometryInfos;
    buildGeometryInfos.reserve(m_meshGroups.size());

    size_t currGeometryOffset = 0;
    for (const auto& meshGroup : m_meshGroups) {
        const auto flags = m_compactAccelStruct ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                                                      vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction
                                                : vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;

        buildGeometryInfos.emplace_back(vk::AccelerationStructureBuildGeometryInfoKHR{
            .type          = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags         = flags,
            .mode          = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = static_cast<uint32_t>(meshGroup.meshes.size()),
            .pGeometries   = &geometries[currGeometryOffset],
        });

        currGeometryOffset += meshGroup.meshes.size();
    }

    //
    // Loop over the BLAS structures we are building and check how much memory we need to construct them.

    m_blas.reserve(m_meshGroups.size());

    // We keep track of the maximum amount of scratch space we need to allocate to create the acceleration structure.
    size_t maxScratchSize = 0;
    for (uint32_t i = 0; i < m_meshGroups.size(); ++i) {
        const auto& meshGroup         = m_meshGroups[i];
        auto&       buildGeometryInfo = buildGeometryInfos[i];

        std::vector<uint32_t> maxPrimitiveCounts;
        maxPrimitiveCounts.reserve(meshGroup.meshes.size());
        for (const auto& mesh : meshGroup.meshes) {
            maxPrimitiveCounts.emplace_back(m_meshes[mesh.meshId].numFaces);
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

        m_blas.emplace_back(std::move(accelStructBuff), std::move(accelStruct));
        maxScratchSize = std::max(maxScratchSize, buildSizeInfo.buildScratchSize);
    }

    // We need the query pool if we are performing compaction as we need to know the new sizes of the BLAS:
    const auto queryPool = m_compactAccelStruct
                               ? context.device().createQueryPoolUnique(vk::QueryPoolCreateInfo{
                                     .queryType  = vk::QueryType::eAccelerationStructureCompactedSizeKHR,
                                     .queryCount = static_cast<uint32_t>(m_meshGroups.size()),
                                 })
                               : vk::UniqueQueryPool{};

    // As the sample page explains, we don't want windows to time-out the execution of a single command buffer when
    // processing many BLAS. To overcome this potential problem, we create a command buffer for each BLAS structure. As
    // the hardware (apperantely) can't create BLASes in parallel, we can also use just one scratch pad.

    // We have to manually manage these as otherwise it's a pain to submit them later if they were all unique handles.
    const auto commandBuffers = context.device().allocateCommandBuffers(vk::CommandBufferAllocateInfo{
        .commandPool        = commandPool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(m_meshGroups.size()),
    });
    // Defer the destruction here:
    const Defer commandBufferDestructor([&]() { context.device().freeCommandBuffers(commandPool, commandBuffers); });

    // Allocate enough scratch space to construct the acceleration structure:
    const auto scratchBuffer = allocator.allocateBuffer(
        maxScratchSize, vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer,
        VMA_MEMORY_USAGE_GPU_ONLY);
    const auto scratchBufferAddr = scratchBuffer.deviceAddress(context.device());

    for (size_t i = 0, currGeometryOffset = 0; i < m_meshGroups.size(); ++i) {
        auto&       buildGeometryInfo = buildGeometryInfos[i];
        const auto& meshGroup         = m_meshGroups[i];
        const auto& commandBuffer     = commandBuffers[i];

        // Start recording:
        commandBuffer.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        });

        buildGeometryInfo.scratchData.deviceAddress = scratchBufferAddr;

        // Apparently we need an array of pointers to the range info, so we make sure to add that here:
        std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> buildRangeInfoPtrs;
        buildRangeInfoPtrs.reserve(meshGroup.meshes.size());
        for (const auto& rangeInfo : std::span(buildRangeInfos).subspan(currGeometryOffset, meshGroup.meshes.size())) {
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

        commandBuffer.end();

        currGeometryOffset += meshGroup.meshes.size();
    }

    // Create a fence that we will wait on for all of these operations to finish:
    const auto fence = context.device().createFenceUnique({});

    context.queue().submit(
        vk::SubmitInfo{
            .commandBufferCount = static_cast<uint32_t>(commandBuffers.size()),
            .pCommandBuffers    = commandBuffers.data(),
        },
        *fence);

    if (context.device().waitForFences(*fence, VK_TRUE, FENCE_TIMEOUT) == vk::Result::eTimeout) {
        throw std::runtime_error("Fence timed out when waiting for BLAS construction commands.");
    }

    // If we turned compaction on, then we can move the values over:
    if (m_compactAccelStruct) {
        // TODO: finish this later...
    }
}

void Scene::transferMeshData(const Context& context, const Allocator& allocator, const vk::CommandPool& commandPool)
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
    const auto sizeOfVerticesBuff = sizeof(Vertex) * m_vertices.size();
    const auto sizeOfFacesBuff    = sizeof(glm::u32vec3) * m_faces.size();

    const auto stagingVertices =
        allocator.allocateBuffer(sizeOfVerticesBuff, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);
    const auto stagingFaces =
        allocator.allocateBuffer(sizeOfFacesBuff, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);

    // Map the memory and copy it over:
    std::ranges::copy(m_vertices, stagingVertices.map<Vertex>());
    std::ranges::copy(m_faces, stagingFaces.map<glm::u32vec3>());
    stagingVertices.unmap();
    stagingFaces.unmap();

    //
    // Allocate buffers on the GPU where we'll send the data:
    const auto blasUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                           vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    m_gpuVertices = allocator.allocateBuffer(sizeOfVerticesBuff, blasUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    m_gpuFaces    = allocator.allocateBuffer(sizeOfFacesBuff, blasUsage, VMA_MEMORY_USAGE_GPU_ONLY);

    //
    // Record the copy command:
    commandBuffer->copyBuffer(*stagingVertices, *m_gpuVertices,
                              vk::BufferCopy{
                                  .size = sizeOfVerticesBuff,
                              });
    commandBuffer->copyBuffer(*stagingFaces, *m_gpuFaces,
                              vk::BufferCopy{
                                  .size = sizeOfFacesBuff,
                              });

    // Transforms are optional, so we check for them, but we do need to keep it on the stack as we don't want to destroy
    // the buffer before we copy from it.
    const auto stagingTransforms = [&]() {
        if (!m_transforms.empty()) {
            const auto sizeOfTransformBuff = sizeof(vk::TransformMatrixKHR) * m_transforms.size();

            auto stagingTransforms = allocator.allocateBuffer(
                sizeOfTransformBuff, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);
            std::ranges::copy(m_transforms, stagingTransforms.map<vk::TransformMatrixKHR>());
            stagingTransforms.unmap();

            // TODO: I believe we can use the same usage flags, but I'm not sure...
            m_gpuTransforms = allocator.allocateBuffer(sizeOfTransformBuff, blasUsage, VMA_MEMORY_USAGE_GPU_ONLY);
            commandBuffer->copyBuffer(*stagingTransforms, *m_gpuTransforms,
                                      vk::BufferCopy{
                                          .size = sizeOfTransformBuff,
                                      });
            return stagingTransforms;
        }
        return UniqueBuffer{};
    }();

    commandBuffer->end();

    // Create a fence that we will wait on for all of these operations to finish:
    const auto fence = context.device().createFenceUnique(vk::FenceCreateInfo{});
    context.queue().submit(
        vk::SubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers    = &*commandBuffer,
        },
        *fence);

    if (context.device().waitForFences(*fence, VK_TRUE, FENCE_TIMEOUT) == vk::Result::eTimeout) {
        throw std::runtime_error("Fence timed out waiting for mesh data transfer commands.");
    }
}

void Scene::transferToGpu(const Context& context, const Allocator& allocator)
{
    const auto commandPool = context.device().createCommandPoolUnique(vk::CommandPoolCreateInfo{
        .flags            = vk::CommandPoolCreateFlagBits::eTransient, // All of the command buffers will be short lived
        .queueFamilyIndex = context.queueFamilyIdx()});

    transferMeshData(context, allocator, *commandPool);
    createBlas(context, allocator, *commandPool);
    createTlas(context, allocator, *commandPool);
}

Scene::IdType Scene::createMesh(const std::string_view filePath)
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
        m_vertices.emplace_back(Scene::Vertex{
            .pos = pos[i],
            .nrm = nrm ? nrm[i] : glm::vec3(0.f),
            .tan = tan ? tan[i] : glm::vec3(0.f),
            .uvs = uvs ? uvs[i] : glm::vec2(0.f),
        });
    }

    const IdType meshId = m_meshes.size();
    m_meshes.emplace_back(Mesh{
        .nrm            = static_cast<bool>(nrm),
        .tan            = static_cast<bool>(tan),
        .uvs            = static_cast<bool>(uvs),
        .verticesOffset = verticesOffset,
        .numVertices    = numVertices,
        .facesOffset    = facesOffset,
        .numFaces       = numFaces,
    });

    return meshId;
}

Scene::IdType Scene::createTransform(const Transform& transform)
{
    const IdType id = m_transforms.size();
    m_transforms.emplace_back(static_cast<vk::TransformMatrixKHR>(transform));
    return id;
}

Scene::IdType Scene::createMeshGroup(const std::span<const MeshGroup::MeshInfo> meshInfos)
{
    // Perform some validation first:
    if (!std::ranges::all_of(meshInfos, [&](const auto& meshInfo) {
            return validMeshId(meshInfo.meshId) && validTransformId(meshInfo.transformId);
        })) {
        throw std::runtime_error("When creating mesh group invalid transform/mesh id was used.");
    }

    const IdType id = m_meshGroups.size();
    m_meshGroups.emplace_back(
        MeshGroup{.meshes = std::vector<MeshGroup::MeshInfo>(meshInfos.begin(), meshInfos.end())});
    return id;
}

Scene::IdType Scene::createInstance(const Instance& instance)
{
    if (!validMeshGroupId(instance.meshGroupId)) {
        throw std::runtime_error("When creating instance invalid mesh group id was used.");
    }

    const IdType id = m_instances.size();
    m_instances.emplace_back(instance);
    return id;
}

} // namespace prism
