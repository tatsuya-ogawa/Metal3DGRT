//
//  GaussianBLASGenerator.metal
//  Metal3DGRT
//

#include <metal_stdlib>
#include <simd/simd.h>
#import "ShaderTypes.h"

using namespace metal;

constant uint GaussianBLASKernelOptionAdaptiveDensityClamping = 1u << 0;
constant float GaussianBLASKernelSupportRadius = 3.0f;
constant float GaussianBLASLinearKernelNormalization = -0.329630334487f;
constant float GaussianBLASExponentialKernelCoefficient = -4.5f;
constant float GaussianBLASAdaptiveDensityClampMaxResponse = 0.97f;
constant float GaussianBLASQuaternionMatrixFactor = 2.0f;

inline float gaussianKernelScale(float density, float minResponse, uint options, float degree) {
    const float responseModulation = (options & GaussianBLASKernelOptionAdaptiveDensityClamping) != 0 ? density : 1.0f;
    const float modulatedMinResponse = min(minResponse / responseModulation, GaussianBLASAdaptiveDensityClampMaxResponse);

    if (degree < 0.0f) {
        const float k = fabs(degree);
        const float s = 1.0f / pow(GaussianBLASKernelSupportRadius, k);
        const float ks = pow((1.0f / (log(modulatedMinResponse) - 1.0f) + 1.0f) / s, 1.0f / k);
        return ks;
    }

    if (degree == 0.0f) {
        return ((1.0f - modulatedMinResponse) / GaussianBLASKernelSupportRadius) / GaussianBLASLinearKernelNormalization;
    }

    const float a = GaussianBLASExponentialKernelCoefficient / pow(GaussianBLASKernelSupportRadius, degree);
    return pow(log(modulatedMinResponse) / a, 1.0f / degree);
}

inline void quaternionWXYZToRows(float4 q,
                                 thread float3& row0,
                                 thread float3& row1,
                                 thread float3& row2) {
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    row0 = float3(1.0f - GaussianBLASQuaternionMatrixFactor * (y * y + z * z),
                  GaussianBLASQuaternionMatrixFactor * (x * y - r * z),
                  GaussianBLASQuaternionMatrixFactor * (x * z + r * y));
    row1 = float3(GaussianBLASQuaternionMatrixFactor * (x * y + r * z),
                  1.0f - GaussianBLASQuaternionMatrixFactor * (x * x + z * z),
                  GaussianBLASQuaternionMatrixFactor * (y * z - r * x));
    row2 = float3(GaussianBLASQuaternionMatrixFactor * (x * z - r * y),
                  GaussianBLASQuaternionMatrixFactor * (y * z + r * x),
                  1.0f - GaussianBLASQuaternionMatrixFactor * (x * x + y * y));
}

inline float3 rotatePoint(float3 point,
                          float3 row0,
                          float3 row1,
                          float3 row2) {
    return float3(dot(row0, point), dot(row1, point), dot(row2, point));
}

inline float3 transformVertex(float3 localVertex,
                              float3 scaledAxes,
                              float3 row0,
                              float3 row1,
                              float3 row2,
                              float3 position) {
    return rotatePoint(localVertex * scaledAxes, row0, row1, row2) + position;
}

inline void writeTriangleIndices(device uint* outIndices,
                                 uint triangleIndex,
                                 uint baseVertexIndex,
                                 uint3 triangle) {
    const uint offset = triangleIndex * 3u;
    outIndices[offset + 0] = baseVertexIndex + triangle.x;
    outIndices[offset + 1] = baseVertexIndex + triangle.y;
    outIndices[offset + 2] = baseVertexIndex + triangle.z;
}

inline float3 safeNormalize(float3 v) {
    const float lengthSquared = dot(v, v);
    if (lengthSquared <= FLT_EPSILON) {
        return float3(0.0f);
    }
    return v * rsqrt(lengthSquared);
}

inline uint surfelAxis(uint primitiveType, float3 scale) {
    if (primitiveType == GaussianBLASPrimitiveTypeMinAxisTriSurfel) {
        if (scale.y < scale.x) {
            return scale.z < scale.y ? 2u : 1u;
        }
        return scale.z < scale.x ? 2u : 0u;
    }

    return 2u;
}

kernel void gaussianBLASBuildKernel(
    uint gaussianIndex [[thread_position_in_grid]],
    device const GaussianBLASGaussian* gaussians [[buffer(0)]],
    constant GaussianBLASBuildUniforms& uniforms [[buffer(1)]],
    device float3* outVertices [[buffer(2)]],
    device uint* outIndices [[buffer(3)]],
    device float4* outSurfelNormalDensity [[buffer(4)]]
) {
    if (gaussianIndex >= uniforms.gaussianCount) {
        return;
    }

    const GaussianBLASGaussian gaussian = gaussians[gaussianIndex];
    const float3 position = gaussian.positionDensity.xyz;
    const float density = gaussian.positionDensity.w;
    const float4 rotation = gaussian.rotationWXYZ;
    const float3 scale = gaussian.scaleReserved.xyz;

    const float kernelScale = gaussianKernelScale(density,
                                                  uniforms.kernelMinResponse,
                                                  uniforms.kernelOptions,
                                                  uniforms.kernelDegree);
    const float3 scaledAxes = scale * kernelScale;

    float3 row0;
    float3 row1;
    float3 row2;
    quaternionWXYZToRows(rotation, row0, row1, row2);

    const uint baseVertexIndex = gaussianIndex * uniforms.verticesPerGaussian;
    const uint baseTriangleIndex = gaussianIndex * uniforms.trianglesPerGaussian;

    switch (uniforms.primitiveType) {
        case GaussianBLASPrimitiveTypeOctahedron: {
            const float diag = 1.7320508075688774f;
            const float3 localVertices[6] = {
                float3(0, 0, -diag), float3(0, diag, 0), float3(-diag, 0, 0),
                float3(0, -diag, 0), float3(diag, 0, 0), float3(0, 0, diag)
            };
            const uint3 triangles[8] = {
                uint3(2, 1, 0), uint3(1, 4, 0), uint3(4, 3, 0), uint3(3, 2, 0),
                uint3(4, 1, 5), uint3(3, 4, 5), uint3(2, 3, 5), uint3(1, 2, 5)
            };

            for (uint i = 0; i < 6; ++i) {
                outVertices[baseVertexIndex + i] = transformVertex(localVertices[i], scaledAxes, row0, row1, row2, position);
            }
            for (uint i = 0; i < 8; ++i) {
                writeTriangleIndices(outIndices, baseTriangleIndex + i, baseVertexIndex, triangles[i]);
            }
            break;
        }

        case GaussianBLASPrimitiveTypeTriHexahedron: {
            const float diag = 1.4142135623730951f;
            const float3 localVertices[6] = {
                float3(0, 0, -diag), float3(0, diag, 0), float3(-diag, 0, 0),
                float3(0, -diag, 0), float3(diag, 0, 0), float3(0, 0, diag)
            };
            const uint3 triangles[6] = {
                uint3(0, 1, 5), uint3(0, 5, 3), uint3(0, 5, 4),
                uint3(0, 2, 5), uint3(4, 1, 3), uint3(2, 1, 3)
            };

            for (uint i = 0; i < 6; ++i) {
                outVertices[baseVertexIndex + i] = transformVertex(localVertices[i], scaledAxes, row0, row1, row2, position);
            }
            for (uint i = 0; i < 6; ++i) {
                writeTriangleIndices(outIndices, baseTriangleIndex + i, baseVertexIndex, triangles[i]);
            }
            break;
        }

        case GaussianBLASPrimitiveTypeTriSurfel:
        case GaussianBLASPrimitiveTypeMinAxisTriSurfel: {
            const float diag = 1.4142135623730951f;
            const uint axis = surfelAxis(uniforms.primitiveType, scale);
            float3 localVertices[4];

            if (axis == 0u) {
                localVertices[0] = float3(0, diag, 0);
                localVertices[1] = float3(0, -diag, 0);
                localVertices[2] = float3(0, 0, diag);
                localVertices[3] = float3(0, 0, -diag);
            } else if (axis == 1u) {
                localVertices[0] = float3(0, 0, diag);
                localVertices[1] = float3(0, 0, -diag);
                localVertices[2] = float3(diag, 0, 0);
                localVertices[3] = float3(-diag, 0, 0);
            } else {
                localVertices[0] = float3(diag, 0, 0);
                localVertices[1] = float3(-diag, 0, 0);
                localVertices[2] = float3(0, diag, 0);
                localVertices[3] = float3(0, -diag, 0);
            }

            const uint3 triangles[2] = {
                uint3(0, 1, 2),
                uint3(0, 1, 3)
            };

            float3 worldVertices[4];
            for (uint i = 0; i < 4; ++i) {
                worldVertices[i] = transformVertex(localVertices[i], scaledAxes, row0, row1, row2, position);
                outVertices[baseVertexIndex + i] = worldVertices[i];
            }
            for (uint i = 0; i < 2; ++i) {
                writeTriangleIndices(outIndices, baseTriangleIndex + i, baseVertexIndex, triangles[i]);
            }

            if (uniforms.emitsSurfelMetadata != 0u && outSurfelNormalDensity != nullptr) {
                const float3 normal = safeNormalize(cross(worldVertices[1] - worldVertices[0],
                                                          worldVertices[2] - worldVertices[0]));
                outSurfelNormalDensity[gaussianIndex] = float4(normal, density);
            }
            break;
        }

        case GaussianBLASPrimitiveTypeTetrahedron: {
            const float edge = 4.898979485566356f;
            const float height = 4.0f;
            const float faceHeight = 4.242640687119285f;
            const float faceInRadius = 1.4142135623730951f;
            const float3 localVertices[4] = {
                float3(-0.5f * edge, -faceInRadius, -1.0f),
                float3(0, faceHeight - faceInRadius, -1.0f),
                float3(0, 0, height - 1.0f),
                float3(0.5f * edge, -faceInRadius, -1.0f)
            };
            const uint3 triangles[4] = {
                uint3(0, 2, 1), uint3(0, 3, 2), uint3(0, 1, 3), uint3(1, 2, 3)
            };

            for (uint i = 0; i < 4; ++i) {
                outVertices[baseVertexIndex + i] = transformVertex(localVertices[i], scaledAxes, row0, row1, row2, position);
            }
            for (uint i = 0; i < 4; ++i) {
                writeTriangleIndices(outIndices, baseTriangleIndex + i, baseVertexIndex, triangles[i]);
            }
            break;
        }

        case GaussianBLASPrimitiveTypeDiamond: {
            const float edge = 3.464101615137755f;
            const float height = 2.8284271247461903f;
            const float faceHeight = 3.0f;
            const float3 localVertices[5] = {
                float3(0, height, 0), float3(0, -height, 0), float3(-0.5f * edge, 0, -1.0f),
                float3(0, 0, faceHeight - 1.0f), float3(0.5f * edge, 0, -1.0f)
            };
            const uint3 triangles[6] = {
                uint3(0, 2, 3), uint3(0, 4, 2), uint3(0, 3, 4),
                uint3(1, 3, 2), uint3(1, 2, 4), uint3(1, 4, 3)
            };

            for (uint i = 0; i < 5; ++i) {
                outVertices[baseVertexIndex + i] = transformVertex(localVertices[i], scaledAxes, row0, row1, row2, position);
            }
            for (uint i = 0; i < 6; ++i) {
                writeTriangleIndices(outIndices, baseTriangleIndex + i, baseVertexIndex, triangles[i]);
            }
            break;
        }

        case GaussianBLASPrimitiveTypeIcosahedron: {
            const float goldenRatio = 1.618033988749895f;
            const float edge = 1.323169076499215f;
            const float vertexScale = 0.5f * edge;
            const float3 localVertices[12] = {
                float3(-1, goldenRatio, 0), float3(1, goldenRatio, 0), float3(0, 1, -goldenRatio),
                float3(-goldenRatio, 0, -1), float3(-goldenRatio, 0, 1), float3(0, 1, goldenRatio),
                float3(goldenRatio, 0, 1), float3(0, -1, goldenRatio), float3(-1, -goldenRatio, 0),
                float3(0, -1, -goldenRatio), float3(goldenRatio, 0, -1), float3(1, -goldenRatio, 0)
            };
            const uint3 triangles[20] = {
                uint3(0, 1, 2), uint3(0, 2, 3), uint3(0, 3, 4), uint3(0, 4, 5), uint3(0, 5, 1),
                uint3(6, 1, 5), uint3(6, 5, 7), uint3(6, 7, 11), uint3(6, 11, 10), uint3(6, 10, 1),
                uint3(8, 4, 3), uint3(8, 3, 9), uint3(8, 9, 11), uint3(8, 11, 7), uint3(8, 7, 4),
                uint3(9, 3, 2), uint3(9, 2, 10), uint3(9, 10, 11), uint3(5, 4, 7), uint3(1, 10, 2)
            };

            for (uint i = 0; i < 12; ++i) {
                outVertices[baseVertexIndex + i] = transformVertex(localVertices[i] * vertexScale,
                                                                   scaledAxes,
                                                                   row0,
                                                                   row1,
                                                                   row2,
                                                                   position);
            }
            for (uint i = 0; i < 20; ++i) {
                writeTriangleIndices(outIndices, baseTriangleIndex + i, baseVertexIndex, triangles[i]);
            }
            break;
        }

        default: {
            break;
        }
    }
}
