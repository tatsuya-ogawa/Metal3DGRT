//
//  GaussianRaytracingShaderTypes.h
//  Metal3DGRT
//

#ifndef GaussianRaytracingShaderTypes_h
#define GaussianRaytracingShaderTypes_h

#include <simd/simd.h>

#define MAX_INTERSECTION_COUNT_LIMIT 128
#define MAX_STRIDE_SIZE_LIMIT 8

typedef struct
{
    uint32_t width;
    uint32_t height;
    uint32_t gaussianCount;
    uint32_t visualizeRawPolygonHit;
    struct Camera camera;
    vector_float4 hitColor;
    vector_float4 missColor;
    float kernelMinResponse;
    uint32_t maxIntersectionCount;
    vector_float4 sceneAabbMin;
    vector_float4 sceneAabbMax;
    uint32_t strideSize;
    uint32_t stridePhaseX;
    uint32_t stridePhaseY;
} GaussianRaytracingUniforms;

#endif /* GaussianRaytracingShaderTypes_h */
