//
//  GaussianBLASShaderTypes.h
//  Metal3DGRT
//

#ifndef GaussianBLASShaderTypes_h
#define GaussianBLASShaderTypes_h

#include <simd/simd.h>

typedef NS_ENUM(uint32_t, GaussianBLASPrimitiveType)
{
    GaussianBLASPrimitiveTypeIcosahedron = 0,
    GaussianBLASPrimitiveTypeOctahedron = 1,
    GaussianBLASPrimitiveTypeTriHexahedron = 2,
    GaussianBLASPrimitiveTypeTriSurfel = 3,
    GaussianBLASPrimitiveTypeMinAxisTriSurfel = 4,
    GaussianBLASPrimitiveTypeTetrahedron = 5,
    GaussianBLASPrimitiveTypeDiamond = 6,
};

typedef struct {
    vector_float4 positionDensity;
    vector_float4 rotationWXYZ;
    vector_float4 scaleReserved;
    float sh[48];
} GaussianBLASGaussian;

typedef struct {
    uint32_t primitiveType;
    uint32_t gaussianCount;
    uint32_t verticesPerGaussian;
    uint32_t trianglesPerGaussian;
    uint32_t kernelOptions;
    uint32_t emitsSurfelMetadata;
    float kernelMinResponse;
    float kernelDegree;
} GaussianBLASBuildUniforms;

#endif /* GaussianBLASShaderTypes_h */
