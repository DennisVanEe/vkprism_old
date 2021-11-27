#pragma once

#include <cstdint>

#include "film.hpp"

namespace prism {

struct IntegratorParam
{
    int maxQueueSize; // The absolute largest we can allocate a queue.
    int numPixelSamples;
};

class Integrator
{
  public:
    Integrator(const IntegratorParam& param, const Film& film);

  private:
    int m_numPixelSamples;

    // Basically breaking up the image as pbrt-v4 does:
    int m_scanlinesPerPass;
    int m_numPasses;
    int m_maxQueueSize;
};

} // namespace prism