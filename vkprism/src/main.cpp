#include "main.hpp"

#include <context.hpp>
#include <geometry/mesh.hpp>
#include <iostream>

#include <glm/gtx/transform.hpp>

#include <math.hpp>

using namespace prism;

glm::mat4 ccombine(const glm::vec3& translation, const glm::quat& rotation, const glm::vec3& scale)
{
    return glm::translate(translation) * glm::mat4(rotation) * glm::scale(scale);
    //return glm::translate(glm::mat4(1.f), translation) * glm::mat4(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

int main(const int argc, const char** const argv)
{
    // ContextParam param{};
    // param.enableCallback = true;
    // param.enableValidation = true;

    // try {
    //    Context ctx(param);
    //} catch (const std::exception& e) {
    //    std::cout << e.what() << "\n";
    //}
}
