//
//  ShaderTypes.h
//  Raytracing Shared
//
//  Created by Jaap Wijnen on 21/11/2021.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#else
#import <Foundation/Foundation.h>
#endif

#define GEOMETRY_MASK_TRIANGLE 1

struct Camera {
    vector_float3 position;
    vector_float3 right;
    vector_float3 up;
    vector_float3 forward;
};

#import "GaussianBLASShaderTypes.h"
#import "GaussianRaytracingShaderTypes.h"

#endif /* ShaderTypes_h */
