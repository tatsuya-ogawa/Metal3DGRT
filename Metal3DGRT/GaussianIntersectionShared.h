//
//  GaussianIntersectionShared.h
//  Metal3DGRT
//

#ifndef GaussianIntersectionShared_h
#define GaussianIntersectionShared_h

#import "ShaderTypes.h"
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

// Function constants to eliminate branches at compile-time
#ifdef __METAL_VERSION__
constant float constantKernelDegree [[function_constant(0)]];
constant uint constantPrimitiveType [[function_constant(1)]];
constant uint constantSHDegree [[function_constant(2)]];
constant uint constantIntersectionMode [[function_constant(3)]];
#endif

constant float GaussianQuaternionMatrixFactor = 2.0f;
constant float GaussianKernelSupportRadius = 3.0f;
constant float GaussianLinearKernelNormalization = 0.329630334487f;
constant float GaussianExponentialKernelCoefficient = -4.5f;
constant float GaussianScaleEpsilon = 1.0e-6f;
constant float GaussianParallelRayEpsilon = 1.0e-6f;
constant float GaussianNegativeDistanceSign = -1.0f;
constant float GaussianPositiveDistanceSign = 1.0f;
constant float GaussianKernelDegreeLinear = 1.0f;
constant float GaussianKernelDegreeCubic = 3.0f;
constant float GaussianKernelDegreeQuartic = 4.0f;
constant float GaussianKernelDegreeQuintic = 5.0f;
constant float GaussianKernelDegreeOctic = 8.0f;
constant float GaussianKernelDegreePiecewiseLinear = 0.0f;
constant uint GaussianRayIntersectionModeTriangle = 0u;
constant uint GaussianRayIntersectionModeBoundingBox = 1u;

constant float GaussianSH_C0 = 0.28209479177387814f;
constant float GaussianSH_C1 = 0.4886025119029199f;
constant float GaussianSH_C2[5] = {1.0925484305920792f, -1.0925484305920792f,
    0.31539156525252005f, -1.0925484305920792f,
    0.5462742152960396f};
constant float GaussianSH_C3[7] = {-0.5900435899266435f, 2.890611442640554f,
    -0.4570457994644658f, 0.3731763325901154f,
    -0.4570457994644658f, 1.445305721320277f,
    -0.5900435899266435f};

struct GaussianRayIntersectionEvaluation {
    bool accept;
    float hitDistance;
    float squaredDistance;
    float3 normal;
    float response;
};

struct GaussianRayIntersectionDistanceOnly {
    bool accept;
    float hitDistance;
};

inline float3 safeNormalizeIntersection(float3 value) {
    const float lengthSquared = dot(value, value);
    if (lengthSquared <= FLT_EPSILON) {
        return float3(0.0f);
    }
    return value * rsqrt(lengthSquared);
}

inline void quaternionWXYZToRowsIntersection(float4 q, thread float3 &row0,
                                             thread float3 &row1,
                                             thread float3 &row2) {
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;
    
    row0 = float3(1.0f - GaussianQuaternionMatrixFactor * (y * y + z * z),
                  GaussianQuaternionMatrixFactor * (x * y - r * z),
                  GaussianQuaternionMatrixFactor * (x * z + r * y));
    row1 = float3(GaussianQuaternionMatrixFactor * (x * y + r * z),
                  1.0f - GaussianQuaternionMatrixFactor * (x * x + z * z),
                  GaussianQuaternionMatrixFactor * (y * z - r * x));
    row2 = float3(GaussianQuaternionMatrixFactor * (x * z - r * y),
                  GaussianQuaternionMatrixFactor * (y * z + r * x),
                  1.0f - GaussianQuaternionMatrixFactor * (x * x + y * y));
}

inline float3 rotatePointIntersection(float3 point, float3 row0, float3 row1,
                                      float3 row2) {
    return float3(dot(row0, point), dot(row1, point), dot(row2, point));
}

inline float gaussianKernelScaleIntersection(float minResponse) {
    if (constantKernelDegree < 0.0f) {
        const float k = fabs(constantKernelDegree);
        const float s = 1.0f / pow(GaussianKernelSupportRadius, k);
        const float ks =
        pow((1.0f / (log(minResponse) - 1.0f) + 1.0f) / s, 1.0f / k);
        return ks;
    }
    
    if (constantKernelDegree == GaussianKernelDegreePiecewiseLinear) {
        return (1.0f - minResponse) / GaussianLinearKernelNormalization /
        GaussianKernelSupportRadius;
    }
    
    const float a = GaussianExponentialKernelCoefficient /
    pow(GaussianKernelSupportRadius, constantKernelDegree);
    return pow(log(minResponse) / a, 1.0f / constantKernelDegree);
}

inline float gaussianKernelResponseIntersection(float squaredDistance,
                                                float minResponse) {
    if (constantKernelDegree == GaussianKernelDegreeOctic) {
        return exp(log(minResponse) * squaredDistance * squaredDistance);
    }
    if (constantKernelDegree == GaussianKernelDegreeQuintic) {
        return exp(log(minResponse) * squaredDistance * squaredDistance *
                   sqrt(squaredDistance));
    }
    if (constantKernelDegree == GaussianKernelDegreeQuartic) {
        return exp(log(minResponse) * squaredDistance * squaredDistance);
    }
    if (constantKernelDegree == GaussianKernelDegreeCubic) {
        return exp(log(minResponse) * squaredDistance * sqrt(squaredDistance));
    }
    if (constantKernelDegree == GaussianKernelDegreeLinear) {
        return exp(log(minResponse) * sqrt(squaredDistance));
    }
    if (constantKernelDegree == GaussianKernelDegreePiecewiseLinear) {
        return max(1.0f - ((1.0f - minResponse) / GaussianKernelSupportRadius) *
                   sqrt(squaredDistance),
                   0.0f);
    }
    return exp(log(minResponse) * squaredDistance);
}

inline float4 evaluateGaussianSH(const thread float *sh, float3 rayDir) {
    float3 color = float3(0.5f) + float3(sh[0], sh[1], sh[2]) * GaussianSH_C0;
    
    if (constantSHDegree > 0) {
        float x = rayDir.x;
        float y = rayDir.y;
        float z = rayDir.z;
        
        color = color - float3(sh[3], sh[4], sh[5]) * (GaussianSH_C1 * y) +
        float3(sh[6], sh[7], sh[8]) * (GaussianSH_C1 * z) -
        float3(sh[9], sh[10], sh[11]) * (GaussianSH_C1 * x);
        
        if (constantSHDegree > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            
            color = color + float3(sh[12], sh[13], sh[14]) * (GaussianSH_C2[0] * xy) +
            float3(sh[15], sh[16], sh[17]) * (GaussianSH_C2[1] * yz) +
            float3(sh[18], sh[19], sh[20]) *
            (GaussianSH_C2[2] * (2.0f * zz - xx - yy)) +
            float3(sh[21], sh[22], sh[23]) * (GaussianSH_C2[3] * xz) +
            float3(sh[24], sh[25], sh[26]) * (GaussianSH_C2[4] * (xx - yy));
            
            if (constantSHDegree > 2) {
                color =
                color +
                float3(sh[27], sh[28], sh[29]) *
                (GaussianSH_C3[0] * y * (3.0f * xx - yy)) +
                float3(sh[30], sh[31], sh[32]) * (GaussianSH_C3[1] * xy * z) +
                float3(sh[33], sh[34], sh[35]) *
                (GaussianSH_C3[2] * y * (4.0f * zz - xx - yy)) +
                float3(sh[36], sh[37], sh[38]) *
                (GaussianSH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy)) +
                float3(sh[39], sh[40], sh[41]) *
                (GaussianSH_C3[4] * x * (4.0f * zz - xx - yy)) +
                float3(sh[42], sh[43], sh[44]) *
                (GaussianSH_C3[5] * z * (xx - yy)) +
                float3(sh[45], sh[46], sh[47]) *
                (GaussianSH_C3[6] * x * (xx - 3.0f * yy));
            }
        }
    }
    
    return float4(max(color, 0.0f), 1.0f);
}

inline bool isGaussianSurfelPrimitive() {
    return constantPrimitiveType == GaussianBLASPrimitiveTypeTriSurfel ||
           constantPrimitiveType == GaussianBLASPrimitiveTypeMinAxisTriSurfel;
}

inline uint surfelAxisIntersection(float3 scale) {
    if (constantPrimitiveType == GaussianBLASPrimitiveTypeMinAxisTriSurfel) {
        if (scale.y < scale.x) {
            return scale.z < scale.y ? 2u : 1u;
        }
        return scale.z < scale.x ? 2u : 0u;
    }
    return 2u;
}

inline GaussianRayIntersectionDistanceOnly evaluateGaussianDistanceOnly(
    float3 origin, float3 direction, float minDistance, float maxDistance,
    float3 position, float4 rotation,
    float3 scale, float kernelMinResponse) {
    GaussianRayIntersectionDistanceOnly evaluation;
    evaluation.accept = false;
    evaluation.hitDistance = INFINITY;
    
    float3 row0;
    float3 row1;
    float3 row2;
    quaternionWXYZToRowsIntersection(rotation, row0, row1, row2);
    
    const float3 clampedScale = max(scale, float3(GaussianScaleEpsilon));
    const float3 inverseScale = 1.0f / clampedScale;
    const float3 localOrigin = origin - position;
    
    const float3x3 worldToLocalRotation = float3x3(row0, row1, row2);
    const float3 canonicalOrigin =
    inverseScale * (worldToLocalRotation * localOrigin);
    const float3 canonicalDirectionUnnormalized =
    inverseScale * (worldToLocalRotation * direction);
    const float3 canonicalDirection =
    safeNormalizeIntersection(canonicalDirectionUnnormalized);
    
    float squaredDistance = 0.0f;
    float hitDistance = 0.0f;
    
    if (!isGaussianSurfelPrimitive()) {
        const float projectionDistance = -dot(canonicalDirection, canonicalOrigin);
        const float3 worldProjectedOffset =
        clampedScale * canonicalDirection * projectionDistance;
        hitDistance = (projectionDistance < 0.0f ? GaussianNegativeDistanceSign
                       : GaussianPositiveDistanceSign) *
        length(worldProjectedOffset);
        if (!(hitDistance > minDistance && hitDistance < maxDistance)) {
            return evaluation;
        }
        
        const float3 closestOffset =
        canonicalOrigin + canonicalDirection * projectionDistance;
        squaredDistance = dot(closestOffset, closestOffset);
    } else {
        const uint axis = surfelAxisIntersection(clampedScale);
        const float denominator = canonicalDirection[axis];
        if (fabs(denominator) <= GaussianParallelRayEpsilon) {
            return evaluation;
        }
        
        const float projectionDistance = -canonicalOrigin[axis] / denominator;
        const float3 worldProjectedOffset =
        clampedScale * canonicalDirection * projectionDistance;
        hitDistance = (projectionDistance < 0.0f ? GaussianNegativeDistanceSign
                       : GaussianPositiveDistanceSign) *
        length(worldProjectedOffset);
        if (!(hitDistance > minDistance && hitDistance < maxDistance)) {
            return evaluation;
        }
        
        const float3 closestOffset =
        canonicalOrigin + canonicalDirection * projectionDistance;
        squaredDistance = dot(closestOffset, closestOffset);
    }
    
    const float kernelScale =
    gaussianKernelScaleIntersection(kernelMinResponse);
    const float maxSquaredDistance = kernelScale * kernelScale;
    if (squaredDistance >= maxSquaredDistance) {
        return evaluation;
    }
    
    evaluation.accept = true;
    evaluation.hitDistance = hitDistance;
    return evaluation;
}

inline GaussianRayIntersectionEvaluation evaluateGaussianShading(
    float3 origin, float3 direction, float hitDistance,
    float3 position, float density, float4 rotation,
    float3 scale, float kernelMinResponse) {
    
    GaussianRayIntersectionEvaluation evaluation;
    evaluation.accept = false;
    evaluation.hitDistance = hitDistance;
    
    float3 row0;
    float3 row1;
    float3 row2;
    quaternionWXYZToRowsIntersection(rotation, row0, row1, row2);
    
    const float3 clampedScale = max(scale, float3(GaussianScaleEpsilon));
    const float3 inverseScale = 1.0f / clampedScale;
    
    const float3 hitPoint = origin + direction * hitDistance;
    const float3 localHit = hitPoint - position;
    
    const float3x3 worldToLocalRotation = float3x3(row0, row1, row2);
    const float3 canonicalHit = inverseScale * (worldToLocalRotation * localHit);
    
    const float squaredDistance = dot(canonicalHit, canonicalHit);
    
    const float kernelScale = gaussianKernelScaleIntersection(kernelMinResponse);
    const float maxSquaredDistance = kernelScale * kernelScale;
    if (squaredDistance >= maxSquaredDistance) {
        return evaluation;
    }
    
    float3 worldNormal;
    if (!isGaussianSurfelPrimitive()) {
        worldNormal = safeNormalizeIntersection(rotatePointIntersection(canonicalHit, row0, row1, row2));
    } else {
        const uint axis = surfelAxisIntersection(clampedScale);
        float3 localNormal = float3(0.0f);
        localNormal[axis] = 1.0f;
        float3 baseWorldNormal = safeNormalizeIntersection(rotatePointIntersection(localNormal, row0, row1, row2));
        worldNormal = dot(direction, baseWorldNormal) > 0.0f ? -baseWorldNormal : baseWorldNormal;
    }
    
    evaluation.accept = true;
    evaluation.squaredDistance = squaredDistance;
    evaluation.normal = worldNormal;
    evaluation.response = gaussianKernelResponseIntersection(
                              squaredDistance / max(maxSquaredDistance, GaussianScaleEpsilon),
                              kernelMinResponse) * density;
    
    return evaluation;
}

inline GaussianRayIntersectionEvaluation evaluateGaussianExactIntersection(
    float3 origin, float3 direction, float minDistance, float maxDistance,
    float3 position, float density, float4 rotation,
    float3 scale, float kernelMinResponse) {
    GaussianRayIntersectionDistanceOnly distResult = evaluateGaussianDistanceOnly(
        origin, direction, minDistance, maxDistance,
        position, rotation, scale, kernelMinResponse);
    
    if (!distResult.accept) {
        GaussianRayIntersectionEvaluation evaluation;
        evaluation.accept = false;
        evaluation.hitDistance = INFINITY;
        evaluation.squaredDistance = INFINITY;
        evaluation.normal = float3(0.0f);
        evaluation.response = 0.0f;
        return evaluation;
    }
    
    return evaluateGaussianShading(
        origin, direction, distResult.hitDistance,
        position, density, rotation, scale, kernelMinResponse);
}
#endif /* GaussianIntersectionShared_h */
