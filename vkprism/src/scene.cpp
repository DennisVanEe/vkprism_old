#include "scene.hpp"

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

void Scene::createTlas(const Context& context)
{
    std::vector<vk::AccelerationStructureInstanceKHR> instances;

    for (const auto& instance : m_instances) {
        instances.emplace_back(vk::AccelerationStructureInstanceKHR{
            .transform = static_cast<vk::TransformMatrixKHR>(instance.transform),
        });
    }
}

void Scene::createBlas(const Context& context)
{
    const auto gpuVerticesAddr = m_gpuVertices.deviceAddress(*context.device);
    const auto gpuFacesAddr    = m_gpuFaces.deviceAddress(*context.device);
    const auto gpuTransformsAddr =
        m_gpuTransforms ? m_gpuTransforms.deviceAddress(*context.device) : vk::DeviceAddress{};

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
        const auto flags = context.param.compactAccelStruct
                               ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
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

    m_blasBuffers.reserve(m_meshGroups.size());

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

        const auto buildSizeInfo = context.device->getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, maxPrimitiveCounts);

        // Allocate space for the acceleration structure:
        auto accelStructureBuff = context.allocateBuffer(buildSizeInfo.accelerationStructureSize,
                                                         vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                                             vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                                         VMA_MEMORY_USAGE_GPU_ONLY);

        auto accelStructure =
            context.device->createAccelerationStructureKHRUnique(vk::AccelerationStructureCreateInfoKHR{
                //.createFlags, TODO: figure out if this is required or not... (I don't think it is).
                .buffer = *accelStructureBuff,
                .size   = buildSizeInfo.accelerationStructureSize,
                .type   = vk::AccelerationStructureTypeKHR::eBottomLevel,
            });

        // Now that we have created it, we can set the destination location:
        buildGeometryInfo.dstAccelerationStructure = *accelStructure;

        m_blasBuffers.emplace_back(std::move(accelStructure), std::move(accelStructureBuff));
        maxScratchSize = std::max(maxScratchSize, buildSizeInfo.buildScratchSize);
    }

    // We need the query pool if we are performing compaction as we need to know the new sizes of the BLAS:
    const auto queryPool = context.param.compactAccelStruct
                               ? context.device->createQueryPoolUnique(vk::QueryPoolCreateInfo{
                                     .queryType  = vk::QueryType::eAccelerationStructureCompactedSizeKHR,
                                     .queryCount = static_cast<uint32_t>(m_meshGroups.size()),
                                 })
                               : vk::UniqueQueryPool{};

    // As the sample page explains, we don't want windows to time-out the execution of a single command buffer when
    // processing many BLAS. To overcome this potential problem, we create a command buffer for each BLAS structure. As
    // the hardware (apperantely) can't create BLASes in parallel, we can also use just one scratch pad.

    // Create a command pool and command buffers:
    const auto commandPool = context.device->createCommandPoolUnique(vk::CommandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eTransient, .queueFamilyIndex = context.queues.general.familyIndex});

    // We have to manually manage these as otherwise it's a pain to submit them later if they were all unique handles.
    const auto commandBuffers = context.device->allocateCommandBuffers(vk::CommandBufferAllocateInfo{
        .commandPool        = *commandPool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(m_meshGroups.size()),
    });
    // Defer the destruction here:
    const Defer commandBufferDestructor([&]() { context.device->freeCommandBuffers(*commandPool, commandBuffers); });

    // Allocate enough scratch space to construct the acceleration structure:
    const auto scratchBuffer = context.allocateBuffer(
        maxScratchSize, vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer,
        VMA_MEMORY_USAGE_GPU_ONLY);
    const auto scratchBufferAddr = scratchBuffer.deviceAddress(*context.device);

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
    const auto fence = context.device->createFenceUnique(vk::FenceCreateInfo{});

    context.queues.general.queue.submit(
        vk::SubmitInfo{
            .commandBufferCount = static_cast<uint32_t>(commandBuffers.size()),
            .pCommandBuffers    = commandBuffers.data(),
        },
        *fence);

    if (context.device->waitForFences(*fence, VK_TRUE, FENCE_TIMEOUT) == vk::Result::eTimeout) {
        throw std::runtime_error("Fence timed out when waiting for BLAS construction commands.");
    }

    // If we turned compaction on, then we can move the values over:
    if (context.param.compactAccelStruct) {
        // TODO: finish this later...
    }
}

void Scene::transferMeshData(const Context& context)
{
    // First thing we do is create a command pool for transfers. Because the command queues we use from this are
    // short-lived, we specify the transient flag.
    const auto commandPool =
        context.queues.transfer
            ? context.device->createCommandPoolUnique(
                  vk::CommandPoolCreateInfo{.flags            = vk::CommandPoolCreateFlagBits::eTransient,
                                            .queueFamilyIndex = context.queues.transfer->familyIndex})
            : context.device->createCommandPoolUnique(
                  vk::CommandPoolCreateInfo{.flags            = vk::CommandPoolCreateFlagBits::eTransient,
                                            .queueFamilyIndex = context.queues.general.familyIndex});
    const auto queue = (context.queues.transfer ? *context.queues.transfer : context.queues.general).queue;

    //
    // Allocate the command buffer (not very efficient to get vector invovled, but unless this becomes a problem I won't
    // bother change it).
    const auto commandBuffer = std::move(context.device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
        .commandPool        = *commandPool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    })[0]);

    commandBuffer->begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    //
    // Transfer mesh vertices and faces:
    const auto sizeOfVerticesBuff = sizeof(Vertex) * m_vertices.size();
    const auto sizeOfFacesBuff    = sizeof(glm::u32vec3) * m_faces.size();

    const auto stagingVertices =
        context.allocateBuffer(sizeOfVerticesBuff, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);
    const auto stagingFaces =
        context.allocateBuffer(sizeOfFacesBuff, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);

    // Map the memory and copy it over:
    std::ranges::copy(m_vertices, stagingVertices.map<Vertex>());
    std::ranges::copy(m_faces, stagingFaces.map<glm::u32vec3>());
    stagingVertices.unmap();
    stagingFaces.unmap();

    //
    // Allocate buffers on the GPU where we'll send the data:
    const auto blasUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                           vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    m_gpuVertices = context.allocateBuffer(sizeOfVerticesBuff, blasUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    m_gpuFaces    = context.allocateBuffer(sizeOfFacesBuff, blasUsage, VMA_MEMORY_USAGE_GPU_ONLY);

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

            auto stagingTransforms = context.allocateBuffer(sizeOfTransformBuff, vk::BufferUsageFlagBits::eTransferSrc,
                                                            VMA_MEMORY_USAGE_CPU_ONLY);
            std::ranges::copy(m_transforms, stagingTransforms.map<vk::TransformMatrixKHR>());
            stagingTransforms.unmap();

            // TODO: I believe we can use the same usage flags, but I'm not sure...
            m_gpuTransforms = context.allocateBuffer(sizeOfTransformBuff, blasUsage, VMA_MEMORY_USAGE_GPU_ONLY);
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
    const auto fence = context.device->createFenceUnique(vk::FenceCreateInfo{});
    queue.submit(
        vk::SubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers    = &*commandBuffer,
        },
        *fence);

    if (context.device->waitForFences(*fence, VK_TRUE, FENCE_TIMEOUT) == vk::Result::eTimeout) {
        throw std::runtime_error("Fence timed out waiting for mesh data transfer commands.");
    }
}

void Scene::transferToGpu(const Context& context)
{
    transferMeshData(context);
    createBlas(context);
}

Scene::IdType Scene::createMesh(const std::string_view filePath)
{
    const std::string  cstrFilepath(filePath);
    miniply::PLYReader plyReader(cstrFilepath.c_str());

    if (!plyReader.valid()) {
        throw std::runtime_error(std::format("Could not open or parse PLY file at: {}", filePath));
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
            throw std::runtime_error(std::format("Could not find face elements for PLY file at: {}", filePath));
        }
        faceElement->convert_list_to_fixed_size(faceElement->find_property("vertex_indices"), 3, triIdx.data());

        bool hasVertices = false, hasFaces = false;
        for (; plyReader.has_element() && (!hasVertices || !hasFaces); plyReader.next_element()) {
            // If it's a vertex element:
            if (plyReader.element_is(miniply::kPLYVertexElement) && plyReader.load_element()) {
                numVertices = plyReader.num_rows();

                // Check for position data:
                if (!plyReader.find_pos(vrtIdx.data())) {
                    throw std::runtime_error(std::format("Missing position data in PLY file at: {}", filePath));
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
            throw std::runtime_error(std::format("Poorly formed PLY file at: {}", filePath));
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
    if (!validMeshGroupId(instance.meshGroupId) || !validTransformId(instance.transformId)) {
        throw std::runtime_error("When creating instance invalid transform/mesh group id was used.");
    }

    const IdType id = m_instances.size();
    m_instances.emplace_back(instance);
    return id;
}

} // namespace prism
