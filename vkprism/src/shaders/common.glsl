#ifndef _COMMON_GLSL_
#define _COMMON_GLSL_

struct Ray
{
	vec3  org;
	vec3  dir;
	float time;
	float tNear;
	float tFar;
};

#endif // _COMMON_GLSL_