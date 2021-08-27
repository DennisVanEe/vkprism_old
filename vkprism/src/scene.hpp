#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <allocator.hpp>
#include <context.hpp>
#include <transform.hpp>

#include <glm/gtx/quaternion.hpp>
#include <glm/mat3x4.hpp>
#include <glm/vec3.hpp>

namespace prism {

#define MAKE_HANDLE(name)                                                                                              \
    class name                                                                                                         \
    {                                                                                                                  \
        friend class Scene;                                                                                            \
        explicit name(uint32_t id) : id(id) {}                                                                         \
        uint32_t id;                                                                                                   \
    }

MAKE_HANDLE(MeshHandle);
MAKE_HANDLE(MeshGroupHandle);
MAKE_HANDLE(InstanceHandle);
MAKE_HANDLE(TransformHandle);

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
        MeshHandle                     mesh;
        std::optional<TransformHandle> transformId;
    };

    std::vector<MeshInfo> meshes;
};

// An instance is a poiner to a MeshGroup (that will be instanced) and a corresponding transform id (if not set,
// then it will be the identity transform). Multi-level instancing will be added later...
// Each instance is also equiped with a custom instance id. So the shader doesn't have to change. Note that this
// should be unique.
struct Instance
{
    uint32_t customId;
    uint32_t mask;
    uint32_t hitGroupId;

    MeshGroupHandle meshGroup;
    Transform       transform;
};

struct SceneParam
{
    bool compactAccelStruct;
};

class Scene
{
  public:
    Scene(const SceneParam& param) : m_compactAccelStruct(param.compactAccelStruct){};

    MeshHandle      createMesh(std::string_view path);
    TransformHandle createTransform(const Transform& transform);
    MeshGroupHandle createMeshGroup(std::span<const MeshGroup::MeshInfo> meshInfos);
    InstanceHandle  createInstance(const Instance& instance);

    // Transfer all scene data to the GPU synchronously.
    void transferToGpu(const Context& context, const Allocator& allocator);

  private:
    void createTlas(const Context& context, const Allocator& allocator, const vk::CommandPool& commandPool);
    void createBlas(const Context& context, const Allocator& allocator, const vk::CommandPool& commandPool);
    void transferMeshData(const Context& context, const Allocator& allocator, const vk::CommandPool& commandPool);

    // TODO: actually enable this:
    bool m_compactAccelStruct;

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
    UniqueBuffer m_gpuTransforms; // TODO: is this required after accel structure was built?

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