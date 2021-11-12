#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT HitPayload PAYLOAD;

void main()
{
	PAYLOAD.hitValue = vec3(0.0);
}