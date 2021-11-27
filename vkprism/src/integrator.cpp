#include "integrator.hpp"

#include <algorithm>

#include <spdlog/spdlog.h>

namespace prism {

Integrator::Integrator(const IntegratorParam& param, const Film& film) : m_numPixelSamples(param.numPixelSamples)
{
    // We use the same calculation that pbrt-v4 uses:
    const auto res = film.getResolution();

    m_scanlinesPerPass = std::max(1, param.maxQueueSize / res.x);               // Rough estimate of number of scanlines
    m_numPasses        = (res.y + m_scanlinesPerPass - 1) / m_scanlinesPerPass; // Number of passes rounded up
    m_scanlinesPerPass = (res.y + m_numPasses - 1) / m_numPasses;
    m_maxQueueSize     = res.x * m_scanlinesPerPass;

    spdlog::info("Render will run for {} passes with {} scanlines for each pass.", m_numPasses, m_scanlinesPerPass);


}

} // namespace prism