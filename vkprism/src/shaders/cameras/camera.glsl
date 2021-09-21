// All of the types required by cameras:
#ifdef CAMERA_TYPES

#include "../common.glsl"

struct CameraSample
{
    vec2  pLens;
    vec2  pFilm;
    float time;
};

#undef CAMERA_TYPES
#endif

// The main function that actually traces the rays:
#ifdef CAMERA_MAIN

#include "../sampler.glsl"

layout(set = 1, 

layout(set = 2, binding = 1) OutputBuffers
{
    float beauty[]; // The beauty output aov...
}
g_outputBuffers;

void main()
{
    // For now, we'll just allocate the sampler:
    Sampler sampler = sampler_create();

    Ray cameraRay = camera_genRay(cameraSample);
}

#undef CAMERA_MAIN
#endif