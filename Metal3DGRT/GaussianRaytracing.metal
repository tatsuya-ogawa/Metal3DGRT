//
//  GaussianRaytracing.metal
//  Metal3DGRT
//

#include <metal_stdlib>
#include <simd/simd.h>
#import "ShaderTypes.h"
#import "GaussianIntersectionShared.h"

using namespace metal;
using namespace raytracing;

constant float GaussianPixelCenterOffset = 0.5f;
constant float GaussianNDCScale = 2.0f;
constant float GaussianNDCBias = 1.0f;
constant float GaussianPrimaryRayMinDistance = 0.0f;
constant float GaussianNormalBiasEpsilon = 1.0e-6f;
constant float GaussianMinimumLighting = 0.15f;
constant float GaussianDistanceFadeSlope = 0.05f;
constant float GaussianPreviewMinimumTransmittance = 0.02f;
constant float GaussianPreviewMinimumAlpha = 1.0e-4f;
constant uint GaussianRayMaxQueuedHits = 16u;
constant uint GaussianInvalidGaussianIndex = 0xFFFFFFFFu;
constant float3 GaussianPreviewLightDirection = float3(0.4f, 0.5f, 1.0f);

struct GaussianRaytracingPayload {
    float3 normal;
    float gaussianDistance;
    float squaredDistance;
    float response;
};

struct GaussianQueuedHit {
    uint gaussianIndex;
    float gaussianDistance;
};

inline void initializeQueuedHits(thread GaussianQueuedHit (&hits)[GaussianRayMaxQueuedHits]) {
    for (uint i = 0; i < GaussianRayMaxQueuedHits; ++i) {
        hits[i].gaussianIndex = GaussianInvalidGaussianIndex;
        hits[i].gaussianDistance = INFINITY;
    }
}

// Removed queuedHitsContainGaussian since BVH culling handles uniqueness

inline void insertQueuedHit(thread GaussianQueuedHit (&hits)[GaussianRayMaxQueuedHits],
                            thread uint& hitCount,
                            uint gaussianIndex,
                            float hitDistance) {
    GaussianQueuedHit candidate;
    candidate.gaussianIndex = gaussianIndex;
    candidate.gaussianDistance = hitDistance;
    
    for (uint i = 0; i < hitCount; ++i) {
        if (hits[i].gaussianIndex != gaussianIndex) {
            continue;
        }
        if (candidate.gaussianDistance >= hits[i].gaussianDistance) {
            return;
        }
        hits[i] = candidate;
        while (i > 0 && hits[i].gaussianDistance < hits[i - 1].gaussianDistance) {
            GaussianQueuedHit previous = hits[i - 1];
            hits[i - 1] = hits[i];
            hits[i] = previous;
            --i;
        }
        return;
    }
    
    if (hitCount == GaussianRayMaxQueuedHits &&
        candidate.gaussianDistance >= hits[GaussianRayMaxQueuedHits - 1u].gaussianDistance) {
        return;
    }
    
    uint insertIndex = min(hitCount, GaussianRayMaxQueuedHits - 1u);
    if (hitCount < GaussianRayMaxQueuedHits) {
        ++hitCount;
    }
    
    while (insertIndex > 0 && candidate.gaussianDistance < hits[insertIndex - 1u].gaussianDistance) {
        if (insertIndex < GaussianRayMaxQueuedHits) {
            hits[insertIndex] = hits[insertIndex - 1u];
        }
        --insertIndex;
    }
    hits[insertIndex] = candidate;
}

inline float2 intersectAABB(float3 aabbMin, float3 aabbMax, float3 rayOri, float3 rayDir) {
    float3 t0 = (aabbMin - rayOri) / rayDir;
    float3 t1 = (aabbMax - rayOri) / rayDir;
    float3 tmax_xyz = max(t0, t1);
    float3 tmin_xyz = min(t0, t1);
    float tmin = max(0.0f, max(tmin_xyz.x, max(tmin_xyz.y, tmin_xyz.z)));
    float tmax = min(tmax_xyz.x, min(tmax_xyz.y, tmax_xyz.z));
    return float2(tmin, tmax);
}

inline void collectTriangleQueuedHits(ray traceRay,
                                      device const GaussianBLASGaussian* gaussians,
                                      acceleration_structure<instancing> accelerationStructure,
                                      thread GaussianQueuedHit (&queuedHits)[GaussianRayMaxQueuedHits],
                                      thread uint& queuedHitCount,
                                      constant GaussianRaytracingUniforms& uniforms,
                                      int intersectionCount,
                                      int maxIntersectionCount) {
    intersection_query<instancing, triangle_data> query(traceRay, accelerationStructure);
    while (query.next()) {
        if (query.get_candidate_intersection_type() != intersection_type::triangle) {
            continue;
        }
        
        if (!isGaussianSurfelPrimitive() && !query.is_candidate_triangle_front_facing()) {
            continue;
        }
        
        float candidateDistance = query.get_candidate_triangle_distance();
        if (queuedHitCount == GaussianRayMaxQueuedHits &&
            candidateDistance > queuedHits[GaussianRayMaxQueuedHits - 1].gaussianDistance) {
            continue;
        }
        
        const uint instanceIndex = query.get_candidate_instance_id();
        insertQueuedHit(queuedHits, queuedHitCount, instanceIndex, candidateDistance);
        if (intersectionCount + (int)queuedHitCount >= maxIntersectionCount) {
            break;
        }
    }
}

inline void collectBoundingBoxQueuedHits(ray traceRay,
                                         device const GaussianBLASGaussian* gaussians,
                                         acceleration_structure<instancing> accelerationStructure,
                                         thread GaussianQueuedHit (&queuedHits)[GaussianRayMaxQueuedHits],
                                         thread uint& queuedHitCount,
                                         constant GaussianRaytracingUniforms& uniforms,
                                         int intersectionCount,
                                         int maxIntersectionCount) {
    intersection_query<instancing> query(traceRay, accelerationStructure);
    while (query.next()) {
        if (query.get_candidate_intersection_type() != intersection_type::bounding_box) {
            continue;
        }
        
        const uint instanceIndex = query.get_candidate_instance_id();
        
        const GaussianBLASGaussian gaussian = gaussians[instanceIndex];
        const GaussianRayIntersectionDistanceOnly evaluation = evaluateGaussianDistanceOnly(traceRay.origin,
                                                                                          traceRay.direction,
                                                                                          traceRay.min_distance,
                                                                                          traceRay.max_distance,
                                                                                          gaussian.positionDensity.xyz,
                                                                                          gaussian.rotationWXYZ,
                                                                                          gaussian.scaleReserved.xyz,
                                                                                          uniforms.kernelMinResponse);
        if (evaluation.accept) {
            insertQueuedHit(queuedHits, queuedHitCount, instanceIndex, evaluation.hitDistance);
        }
        if (intersectionCount + (int)queuedHitCount >= maxIntersectionCount) {
            break;
        }
    }
}

kernel void gaussianTriangleRaytracingKernel(
                                             uint2 tid [[thread_position_in_grid]],
                                             constant GaussianRaytracingUniforms& uniforms [[buffer(0)]],
                                             device const GaussianBLASGaussian* gaussians [[buffer(1)]],
                                             acceleration_structure<instancing> accelerationStructure [[buffer(2)]],
                                             texture2d<float, access::write> outTexture [[texture(0)]]
                                             ) {
    // Map strided thread index to actual pixel coordinate
    const uint stride = max(uniforms.strideSize, 1u);
    const uint2 pixelCoord = tid * stride + uint2(uniforms.stridePhaseX, uniforms.stridePhaseY);
    
    if (pixelCoord.x >= uniforms.width || pixelCoord.y >= uniforms.height) {
        return;
    }
    
    float2 pixel = float2(pixelCoord) + GaussianPixelCenterOffset;
    float2 uv = pixel / float2(uniforms.width, uniforms.height);
    uv.y = 1.0f - uv.y;
    uv = uv * GaussianNDCScale - GaussianNDCBias;
    
    ray traceRay;
    traceRay.origin = uniforms.camera.position;
    traceRay.direction = normalize(uv.x * uniforms.camera.right +
                                   uv.y * uniforms.camera.up +
                                   uniforms.camera.forward);
    float2 minMaxT = intersectAABB(uniforms.sceneAabbMin.xyz, uniforms.sceneAabbMax.xyz, traceRay.origin, traceRay.direction);
    
    if (minMaxT.x > minMaxT.y) {
        outTexture.write(uniforms.missColor, pixelCoord);
        return;
    }
    
    if (uniforms.visualizeRawPolygonHit &&
        constantIntersectionMode == GaussianRayIntersectionModeTriangle) {
        traceRay.min_distance = max(GaussianPrimaryRayMinDistance, minMaxT.x - 1e-4f);
        traceRay.max_distance = minMaxT.y;
        intersection_query<instancing, triangle_data> query(traceRay, accelerationStructure);
        while (query.next()) {
            if (query.get_candidate_intersection_type() == intersection_type::triangle) {
                const uint instanceIndex = query.get_candidate_instance_id();
                const GaussianBLASGaussian gaussian = gaussians[instanceIndex];
                const float3 viewDirection = normalize(gaussian.positionDensity.xyz - uniforms.camera.position);
                const float4 evaluatedColor = evaluateGaussianSH(gaussian.sh, viewDirection);
                
                // Since we do not have vertex normals bound in this shader, we use barycentric coordinates
                // to generate a fake lighting/shading effect for visualization.
                float2 bary = query.get_candidate_triangle_barycentric_coord();
                float fakeLighting = bary.x * 0.3f + bary.y * 0.6f + (1.0f - bary.x - bary.y) * 0.9f;
                float lighting = clamp(fakeLighting, 0.2f, 1.0f);
                outTexture.write(float4(evaluatedColor.xyz * lighting, 1.0f), pixelCoord);
                return;
            }
        }
        outTexture.write(uniforms.missColor, pixelCoord);
        return;
    }
    
    float rayLastHitDistance = max(GaussianPrimaryRayMinDistance, minMaxT.x - 1e-4f);
    float3 accumulatedColor = float3(0.0f);
    float transmittance = 1.0f;
    const int maxIntersectionCount = int(uniforms.maxIntersectionCount);
    int intersectionCount = 0;
    
    // Set segment length (e.g., small percentage of scene diagonal length, or a constant value)
    float sceneSize = length(uniforms.sceneAabbMax.xyz - uniforms.sceneAabbMin.xyz);
    const float baseStep = max(0.5f, sceneSize * 0.05f);
    float rayStep = baseStep;
    
    while (rayLastHitDistance < minMaxT.y && transmittance > GaussianPreviewMinimumTransmittance && intersectionCount < maxIntersectionCount) {
        float segmentMaxDistance = min(minMaxT.y, rayLastHitDistance + rayStep);
        traceRay.min_distance = rayLastHitDistance + 1e-4f;
        traceRay.max_distance = segmentMaxDistance;
        
        GaussianQueuedHit queuedHits[GaussianRayMaxQueuedHits];
        initializeQueuedHits(queuedHits);
        uint queuedHitCount = 0u;
        if (constantIntersectionMode == GaussianRayIntersectionModeBoundingBox) {
            collectBoundingBoxQueuedHits(traceRay,
                                         gaussians,
                                         accelerationStructure,
                                         queuedHits,
                                         queuedHitCount,
                                         uniforms,
                                         intersectionCount,
                                         maxIntersectionCount);
        } else {
            collectTriangleQueuedHits(traceRay,
                                      gaussians,
                                      accelerationStructure,
                                      queuedHits,
                                      queuedHitCount,
                                      uniforms,
                                      intersectionCount,
                                      maxIntersectionCount);
        }
        
        if (queuedHitCount == 0) {
            // This segment space is completely empty, so warp to the end of the segment at once
            rayLastHitDistance = segmentMaxDistance;
            // Adaptive stepping: Since it's empty, widen the step size further
            rayStep = min(sceneSize, rayStep * 2.0f);
            continue;
        }
        
        float maxProcessedDistance = rayLastHitDistance;
        for (uint i = 0; i < queuedHitCount && transmittance > GaussianPreviewMinimumTransmittance; ++i) {
            if (intersectionCount >= maxIntersectionCount) {
                break;
            }
            
            const GaussianQueuedHit hit = queuedHits[i];
            maxProcessedDistance = max(maxProcessedDistance, hit.gaussianDistance);
            
            const GaussianBLASGaussian gaussian = gaussians[hit.gaussianIndex];
            GaussianRayIntersectionEvaluation evaluation;
            
            if (constantIntersectionMode == GaussianRayIntersectionModeBoundingBox) {
                evaluation = evaluateGaussianShading(traceRay.origin,
                                                     traceRay.direction,
                                                     hit.gaussianDistance,
                                                     gaussian.positionDensity.xyz,
                                                     gaussian.positionDensity.w,
                                                     gaussian.rotationWXYZ,
                                                     gaussian.scaleReserved.xyz,
                                                     uniforms.kernelMinResponse);
            } else {
                evaluation = evaluateGaussianExactIntersection(traceRay.origin,
                                                          traceRay.direction,
                                                          traceRay.min_distance,
                                                          traceRay.max_distance,
                                                          gaussian.positionDensity.xyz,
                                                          gaussian.positionDensity.w,
                                                          gaussian.rotationWXYZ,
                                                          gaussian.scaleReserved.xyz,
                                                          uniforms.kernelMinResponse);
            }
            
            if (!evaluation.accept) {
                continue;
            }
            
            const float alpha = min(max(evaluation.response, 0.0f), 1.0f);
            if (alpha <= GaussianPreviewMinimumAlpha) {
                continue;
            }
            
            const float weight = alpha * transmittance;
            const float3 viewDirection = normalize(gaussian.positionDensity.xyz - uniforms.camera.position);
            const float3 color = evaluateGaussianSH(gaussian.sh, viewDirection).xyz;
            accumulatedColor += color * weight;
            transmittance *= (1.0f - alpha);
            intersectionCount++;
        }
        
        // Advance the starting point (min_distance) of the next ray
        if (queuedHitCount < GaussianRayMaxQueuedHits) {
            // Queue not overflowed = All intersections up to segmentMaxDistance have been exhausted
            // = This segment interval is resolved to the end, so we can jump
            rayLastHitDistance = segmentMaxDistance;
            // Adaptive stepping: Space is empty, so widen the step size
            rayStep = min(sceneSize, rayStep * 2.0f);
        } else {
            // Queue full = The 17th intersection might be hidden just behind maxProcessedDistance
            // = Therefore, the ray must be resumed from the furthest processed hit position
            rayLastHitDistance = maxProcessedDistance;
            // Adaptive stepping: It's crowded, so revert to the base step size
            rayStep = baseStep;
        }
        
        if (intersectionCount >= maxIntersectionCount) {
            break;
        }
    }
    
    const float3 finalColor = accumulatedColor + uniforms.missColor.xyz * transmittance;
    outTexture.write(float4(finalColor, 1.0f), pixelCoord);
}
