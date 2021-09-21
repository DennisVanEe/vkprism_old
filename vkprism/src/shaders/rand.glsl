#ifndef _RAND_GLSL_
#define _RAND_GLSL_

// This function generates a random number using a 32-bit state.
// I would like to experiment with 64-bit states, but there isn't
// anyway to gaurantee wrapping behavior with 64-bit multiplication
// in GLSL that I'm aware of.
// We're using the 32-bit RXS-M-XS from: https://en.wikipedia.org/wiki/Permuted_congruential_generator,
// the implementation is from: https://nvpro-samples.github.io/vk_mini_path_tracer/index.html#antialiasingandpseudorandomnumbergeneration/pseudorandomnumbergenerationinglsl

// Stores the state of the random number generator:
struct RNG
{
	uint state;
};

// Initializes the random number generator with a seed.
RNG rng_create(uint seed)
{
	return RNG(seed);
}

uint rng_getUint(inout RNG self)
{
	// Update the state:
	self.state = self.state * 747796405 + 1;

	uint randint = ((self.state >> ((self.state >> 28) + 4)) & self.state) * 277803737;
	return (randint >> 22) ^ randint;
}

float rng_getFloat(inout RNG self)
{
	return rng_getUint(self) / 4294967295.0f;
}

#endif // _RAND_GLSL_