//
//  Shaders.metal
//  Metal3DGRT
//

#include <metal_stdlib>

using namespace metal;

struct Vertex {
    float4 position [[position]];
    float2 uv;
};

constant float2 quadVertices[] = {
    float2(-1.0, -1.0),
    float2(-1.0,  1.0),
    float2( 1.0,  1.0),
    float2(-1.0, -1.0),
    float2( 1.0,  1.0),
    float2( 1.0, -1.0)
};

vertex Vertex vertexShader(unsigned short vertexID [[vertex_id]]) {
    const float2 position = quadVertices[vertexID];
    Vertex out;
    out.position = float4(position, 0.0, 1.0);
    out.uv = float2(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
    return out;
}

fragment float4 fragmentShader(Vertex in [[stage_in]],
                               texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(min_filter::linear,
                        mag_filter::linear,
                        mip_filter::none);
    float3 color = tex.sample(s, in.uv).xyz;
    color = color / (1.0f + color);
    return float4(color, 1.0f);
}
