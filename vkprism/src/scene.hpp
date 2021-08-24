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
        // MeshInfo is just a pair of mesh Ids and corresponding transformIds (note that if it's none, then it's
        // the identity transform).
        struct MeshInfo
        {
            IdType                meshId;
            std::optional<IdType> transformId;
        };

        std::vector<MeshInfo> meshes;
    };

    // An instance is a poiner to a MeshGroup (that will be instanced) and a corresponding transform id (if not set,
    // then it will be the identity transform). Multi-level instancing will be added later...
    // Each instance is also equiped with a custom instance id. So the shader doesn't have to change. Note that this
    // should be unique.
    struct Instance
    {
        uint32_t  customId;
        uint32_t  mask;
        uint32_t  hitGroupId;
        IdType    meshGroupId;
        Transform transform;
    };

    // Transfer all scene data to the GPU. After this call has finished, all of the data should be on
    // the GPU (it's not async).
    void transferToGpu(const Context& context);

    // Adds a mesh to the Scene, returning the Id of the mesh.
    IdType createMesh(std::string_view path);

    // Adds a transform to the Scene, returning the Id of the transform.
    IdType createTransform(const Transform& transform);

    // Creates a mesh group by passing a collection of mesh infos.
    IdType createMeshGroup(std::span<const MeshGroup::MeshInfo> meshInfos);

    // Creates an instance by passing a collection of mesh infos.
    IdType createInstance(const Instance& instance);

  private:
    void createTlas(const Context& context, const vk::CommandPool& commandPool);
    void createBlas(const Context& context, const vk::CommandPool& commandPool);
    void transferMeshData(const Context& context, const vk::CommandPool& commandPool);

    bool validTransformId(IdType id) const { return id < m_transforms.size(); }
    // Transform Ids are often optional, so we add that case here:
    bool validTransformId(std::optional<IdType> id) const
    {
        if (id) {
            return validTransformId(*id);
        }
        return true;
    }

    bool validMeshGroupId(IdType id) const { return id < m_meshGroups.size(); }
    bool validMeshId(IdType id) const { return id < m_meshes.size(); }

    // Raw mesh data:
    std::vector<Mesh>                   m_meshes;
    std::vector<Vertex>                 m_vertices;
    std::vector<glm::u32vec3>           m_faces;
    std::vector<vk::TransformMatrixKHR> m_transforms;

    // Collection of mesh groups:
    std::vector<MeshGroup> m_meshGroups;
    std::vector<Instance>  m_instances;

    UniqueBuffer m_gpuVertices;
    UniqueBuffer m_gpuFaces;
    UniqueBuffer m_gpuTransforms;

    struct AccelStructInfo
    {
        // NOTE: destruct accel structure before buffer is probably safer...
        UniqueBuffer                       buffer;
        vk::UniqueAccelerationStructureKHR accelStruct;
    };

    std::vector<AccelStructInfo> m_blas;
    AccelStructInfo              m_tlas;
};

} // namespace prism