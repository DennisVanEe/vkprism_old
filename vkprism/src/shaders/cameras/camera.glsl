
//
// All of the types required by cameras:
//

#ifdef CAMERA_TYPES

struct CameraSample
{
    vec2  pLens;
    vec2  pFilm;
    float time;
};

#undef CAMERA_TYPES
#endif

//
// The function that executes the camera:
//

#ifdef CAMERA_MAIN

#include "../common.glsl"
#include "../sampler.glsl"

layout(set = 1, 

layout(set = 2, binding = 1, scalar) OutputBuffers
{
    vec3 beauty[]; // The beauty output aov...
}
g_outputBuffers;

void main()
{
    // For now, we'll just allocate the sampler:
    Sampler localSampler = sampler_create();

    // Generate the camera ray:
    const Ray cameraRay = camera_generateRay(cameraSample);
}

#undef CAMERA_MAIN
#endif