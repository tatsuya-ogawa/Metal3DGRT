//
//  SPZModelLoader.swift
//  Metal3DGRT
//
//  Created by Antigravity on 2026-03-25.
//

import Foundation
import simd

struct SPZModelLoader {
    enum Error: Swift.Error {
        case fileNotFound
        case loadingFailed(Swift.Error)
    }

    /// Loads a Gaussian Splatting model from an .spz file.
    /// - Parameters:
    ///   - url: The URL of the .spz file.
    ///   - targetCoordinateSystem: The coordinate system of the renderer (defaulting to .rub).
    /// - Returns: An array of GaussianSplat objects.
    static func load(from url: URL, targetCoordinateSystem: CoordinateSystem = .rub) throws -> [GaussianSplat] {
        let options = UnpackOptions(to: targetCoordinateSystem)
        
        let cloud: GaussianCloud
        do {
            cloud = try loadSpz(from: url, options: options)
        } catch {
            throw Error.loadingFailed(error)
        }
        
        let pointCount = Int(cloud.numPoints)
        let shDimInt = Int(dimForDegree(cloud.shDegree))
        let shPerPoint = shDimInt * 3

        var gaussians = [GaussianSplat?](repeating: nil, count: pointCount)
        
        return cloud.positions.withUnsafeBufferPointer { pPos in
        cloud.rotations.withUnsafeBufferPointer { pRot in
        cloud.scales.withUnsafeBufferPointer { pScale in
        cloud.alphas.withUnsafeBufferPointer { pAlpha in
        cloud.colors.withUnsafeBufferPointer { pColor in
        cloud.sh.withUnsafeBufferPointer { pSH in
            DispatchQueue.concurrentPerform(iterations: pointCount) { i in
                let position = SIMD3<Float>(pPos[i * 3 + 0],
                                          pPos[i * 3 + 1],
                                          pPos[i * 3 + 2])
                
                let rotationWXYZ = SIMD4<Float>(pRot[i * 4 + 3],
                                              pRot[i * 4 + 0],
                                              pRot[i * 4 + 1],
                                              pRot[i * 4 + 2])
                
                let scale = SIMD3<Float>(exp(pScale[i * 3 + 0]),
                                       exp(pScale[i * 3 + 1]),
                                       exp(pScale[i * 3 + 2]))
                
                let density = sigmoid(pAlpha[i])
                
                var sh = [Float](repeating: 0, count: 48)
                sh[0] = pColor[i * 3 + 0]
                sh[1] = pColor[i * 3 + 1]
                sh[2] = pColor[i * 3 + 2]
                
                if shPerPoint > 0 {
                    let startPos = i * shPerPoint
                    for j in 0..<shPerPoint {
                        if j + 3 < 48 {
                            sh[j + 3] = pSH[startPos + j]
                        }
                    }
                }
                
                gaussians[i] = GaussianSplat(position: position,
                                           rotationWXYZ: rotationWXYZ,
                                           scale: scale,
                                           density: density,
                                           sh: sh)
            }
            return gaussians.compactMap { $0 }
        }}}}}}
    }
}
