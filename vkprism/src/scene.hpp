#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <context.hpp>
#include <transform.hpp>

#include <glm/gtx/quaternion.hpp>
#include <glm/mat3x4.hpp>
#include <glm/vec3.hpp>

namespace prism {

class Scene
{
  public:
    using IdType = uint32_t; // Type used to specify the ID between mesh, nodes, etc.

    struct Vertex
    {
        glm::vec3 pos;
        glm::vec3 nrm;
        glm::vec3 tan;
        glm::vec2 uvs;
    };

    struct Mesh
    {
        bool nrm, tan, uvs; // whether or not these are present

        uint32_t verticesOffset;
        uint32_t numVertices;

        uint32_t facesOffset;
        uint32_t numFaces;
    };

    // A MeshGroup is a group of mesh that can be instanced. This is equivelant to a BLAS in Vulkan RT terminology.
    struct MeshGroup
    {
        // MeshInfo is just a pair of mesh mesh Ids and corresponding transformIds (note that if it's none, then it's
        // the identity transform).
        struct MeshInfo
        {
            IdType                meshId;
            std::optional<IdType> transformId;
        };

        std::vector<MeshInfo> meshes;
    };

    // Transfer all scene data to the GPU. After this call has finished, all of the data should be on
    // the GPU (it's not async).
    void transferToGpu(const Context& context);

    // Adds a mesh to the Scene, returning the Id of the mesh.
    IdType loadMesh(std::string_view path);

    // Adds a transform to the Scene, returning the Id of the transform.
    IdType loadTransform(const Transform& transform);

    // Creates a mesh group by passing a collection of mesh infos.
    IdType loadMeshGroup(std::span<MeshGroup::MeshInfo> meshInfos);

  private:
    void createBlas(const Context& context);
    void transferMeshData(const Context& context);

    // Raw mesh data:
    std::vector<Mesh>         m_meshes;
    std::vector<Vertex>       m_vertices;
    std::vector<glm::u32vec3> m_faces;

    // Collection of mesh groups:
    std::vector<vk::TransformMatrixKHR> m_transforms;
    std::vector<MeshGroup>              m_meshGroups;

    UniqueBuffer m_gpuVertices;
    UniqueBuffer m_gpuFaces;
    UniqueBuffer m_gpuTransforms;

    struct BlasInfo
    {
        // NOTE: destruct accel structure before buffer is probably safer...
        vk::UniqueAccelerationStructureKHR accelStructure;
        UniqueBuffer                       buffer;
    };

    std::vector<BlasInfo> m_blasBuffers;
};

} // namespace prism