#ifndef _SAMPLER_GLSL_
#define _SAMPLER_GLSL_

#include "rand.glsl"

// This file contains the sampler. For now it's a basic uniform random number, in the future I plan
// on implementing something like PMJ02 and just transfering the sample tables to the GPU. However,
// PMJ02 requires wrapping 64-bit multiplication, which I can't seem to get any gaurantee for in GLSL...

struct Sampler
{
    RNG rng;
};

// Creates a new sampler for the specific pixel (defined by the pixelIndex):
Sampler sampler_create(uint pixelIndex)
{
    return Sampler(rng_create(pixelIndex));
}

vec2 sampler_get2D(inout Sampler self)
{
    return vec2(
        rng_getFloat(self.rng),
        rng_getFloat(self.rng)
    );
}

float sampler_get1D(inout Sampler self)
{
    return rng_getFloat(self.rng);
}

#endif // _SAMPLER_GLSL_