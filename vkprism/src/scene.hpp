#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <allocator.hpp>
#include <context.hpp>
#include <transform.hpp>

#include <glm/gtx/quaternion.hpp>
#include <glm/mat3x4.hpp>
#include <glm/vec3.hpp>

namespace prism {

#define MAKE_INDEX(name)                                                                                               \
    class name                                                                                                         \
    {                                                                                                                  \
        explicit name(uint32_t idx) : idx(idx) {}                                                                      \
        operator uint32_t() const { return idx; }                                                                      \
                                                                                                                       \
        friend class SceneBuilder;                                                                                     \
        friend class Scene;                                                                                            \
        uint32_t idx;                                                                                                  \
    }

MAKE_INDEX(MeshIndex);
MAKE_INDEX(TransformIndex);
MAKE_INDEX(MeshGroupIndex);
MAKE_INDEX(InstanceIndex);

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 nrm;
    glm::vec3 tan;
    glm::vec2 uvs;
};

struct PlacedMesh
{
    MeshIndex                     meshIdx;
    std::optional<TransformIndex> transformIdx;
};

struct Instance
{
    uint32_t customId;
    uint32_t mask;
    uint32_t hitGroupId;

    MeshGroupIndex meshGroupIdx;
    Transform      transform;
};

class SceneBuilder
{
  public:
    SceneBuilder() = default;

    MeshIndex      createMesh(std::string_view path);
    TransformIndex createTransform(const Transform& transform);
    MeshGroupIndex createMeshGroup(std::span<const PlacedMesh> placedMeshes);
    InstanceIndex  createInstance(const Instance& instance);

  private:
    friend class Scene;

    struct Mesh
    {
        bool nrm, tan, uvs;

        uint32_t verticesOffset;
        uint32_t numVertices;

        uint32_t facesOffset;
        uint32_t numFaces;
    };

  private:
    // Raw mesh data:
    std::vector<Mesh> m_meshes;

    std::vector<Vertex>                 m_vertices;
    std::vector<glm::u32vec3>           m_faces;
    std::vector<vk::TransformMatrixKHR> m_transforms;

    // Collection of mesh groups:
    std::vector<std::vector<PlacedMesh>> m_meshGroups;
    std::vector<Instance>                m_instances;
};

struct SceneParam
{
    bool enableCompaction;
};

class Scene
{
  public:
    Scene(const SceneParam& param, const Context& context, const GpuAllocator& allocator,
          const SceneBuilder& sceneBuilder);
    Scene(const Scene&) = delete;
    Scene(Scene&&)      = default;

    const vk::Buffer&                   gpuVertices() const { return *m_meshGpuData.vertices; }
    const vk::AccelerationStructureKHR& tlas() const { return *m_tlas.accelStruct; }

  private:
    struct MeshGpuData
    {
        UniqueBuffer vertices;
        UniqueBuffer faces;
        UniqueBuffer transforms;
    };

    struct AccelStructInfo
    {
        UniqueBuffer                       buffer;
        vk::UniqueAccelerationStructureKHR accelStruct;
    };

  private:
    static MeshGpuData                  transferMeshData(const Context& context, const GpuAllocator& allocator,
                                                         const vk::CommandPool& commandPool, std::span<const SceneBuilder::Mesh> meshes,
                                                         std::span<const Vertex> vertices, std::span<const glm::u32vec3> faces,
                                                         std::span<const vk::TransformMatrixKHR> transforms);
    static std::vector<AccelStructInfo> createBlas(const Context& context, const GpuAllocator& allocator,
                                                   const vk::CommandPool& commandPool, const MeshGpuData& meshGpuData,
                                                   std::span<const SceneBuilder::Mesh>      meshes,
                                                   std::span<const std::vector<PlacedMesh>> meshGroups,
                                                   bool                                     enableCompaction);
    static AccelStructInfo              createTlas(const Context& context, const GpuAllocator& allocator,
                                                   const vk::CommandPool& commandPool, std::span<const Instance> instances,
                                                   std::span<const AccelStructInfo> blases);

  private:
    MeshGpuData m_meshGpuData;

    std::vector<AccelStructInfo> m_blases;
    AccelStructInfo              m_tlas;
};

} // namespace prism