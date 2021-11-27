#pragma once

#include <glm/vec2.hpp>

namespace prism {

// For now the film class is pretty simple. I'm just rendering using RGB. Once I have the basic architecture down I can
// focus on true spectral rendering and fancy stuff.

class Film
{
  public:
    Film(const glm::ivec2& resolution) : m_resolution(resolution) {}

    glm::ivec2 getResolution() const { return m_resolution; }

  private:
    glm::ivec2 m_resolution;
};

} // namespace prism