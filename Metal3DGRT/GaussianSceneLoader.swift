//
//  GaussianSceneLoader.swift
//  Metal3DGRT
//

import Foundation
import simd

struct RendererProgressSnapshot {
    let strideSize: Int
    let completedPhases: Int
    let totalPhases: Int

    static let empty = RendererProgressSnapshot(strideSize: 1, completedPhases: 0, totalPhases: 0)

    var statusLine: String {
        let clampedTotal = max(totalPhases, 0)
        let clampedCompleted = min(max(completedPhases, 0), clampedTotal)
        let progressPct = clampedTotal > 0 ? Double(clampedCompleted) / Double(clampedTotal) * 100.0 : 100.0
        let barWidth = 20
        let filledCount = clampedTotal > 0 ? clampedCompleted * barWidth / clampedTotal : barWidth
        let emptyCount = max(barWidth - filledCount, 0)
        let progressBar = String(repeating: "\u{2588}", count: filledCount) + String(repeating: "\u{2591}", count: emptyCount)
        let progressStatus = clampedTotal > 0 && clampedCompleted >= clampedTotal ? "Done" : "\(clampedCompleted)/\(clampedTotal)"
        return "Stride \(strideSize)x\(strideSize) [\(progressBar)] \(String(format: "%.0f", progressPct))% \(progressStatus)"
    }
}

protocol RendererProgressDelegate: AnyObject {
    func renderer(_ renderer: Renderer, didResetProgress progress: RendererProgressSnapshot)
    func rendererDidAdvanceProgress(_ renderer: Renderer)
}

final class GaussianSceneProgress: RendererProgressDelegate {
    private(set) var snapshot: RendererProgressSnapshot = .empty

    func renderer(_ renderer: Renderer, didResetProgress progress: RendererProgressSnapshot) {
        snapshot = progress
    }

    func rendererDidAdvanceProgress(_ renderer: Renderer) {
        snapshot = RendererProgressSnapshot(
            strideSize: snapshot.strideSize,
            completedPhases: min(snapshot.completedPhases + 1, snapshot.totalPhases),
            totalPhases: snapshot.totalPhases
        )
    }
}

final class GaussianScene: Hashable {
    let id: String
    let displayName: String
    let remoteURL: URL?
    let initialTransform: simd_float4x4
    let progress = GaussianSceneProgress()

    static let butterfly = GaussianScene(
        id: "butterfly",
        displayName: "Butterfly",
        remoteURL: URL(string: "https://sparkjs.dev/assets/splats/butterfly.spz"),
        initialTransform: simd_float4x4(simd_quaternion(.pi, SIMD3<Float>(1, 0, 0)))
    )
    static let debug = GaussianScene(id: "debug", displayName: "Debug Grid", remoteURL: nil)
    static let hornedlizard = GaussianScene(
        id: "hornedlizard",
        displayName: "Horned Lizard",
        remoteURL: URL(string: "https://raw.githubusercontent.com/nianticlabs/spz/main/samples/hornedlizard.spz")
    )

    static let allCases: [GaussianScene] = [.butterfly, .debug, .hornedlizard]
    static var defaultScene: GaussianScene = .butterfly

    init(id: String, displayName: String, remoteURL: URL?, initialTransform: simd_float4x4 = matrix_identity_float4x4) {
        self.id = id
        self.displayName = displayName
        self.remoteURL = remoteURL
        self.initialTransform = initialTransform
    }

    static func custom(displayName: String) -> GaussianScene {
        GaussianScene(id: "custom:\(UUID().uuidString)", displayName: displayName, remoteURL: nil)
    }

    static func == (lhs: GaussianScene, rhs: GaussianScene) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

enum GaussianSceneLoader {
    private static var cacheDirectory: URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("gaussians", isDirectory: true)
    }

    static func loadScene(_ scene: GaussianScene, completion: @escaping (Result<[GaussianSplat], Error>) -> Void) {
        if scene == .debug {
            let gaussians = debugGaussians()
            completion(.success(applyTransform(gaussians: gaussians, transform: scene.initialTransform)))
            return
        }

        guard let remoteURL = scene.remoteURL else {
            completion(.failure(NSError(domain: "GaussianSceneLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "No URL for scene"])))
            return
        }

        let fileName = remoteURL.lastPathComponent
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let cachedURL = cacheDirectory.appendingPathComponent(fileName)

        if FileManager.default.fileExists(atPath: cachedURL.path) {
            loadGaussians(from: cachedURL, transform: scene.initialTransform, completion: completion)
            return
        }

        SPZDownloader.shared.downloadSPZ(from: remoteURL, to: cachedURL) { result in
            switch result {
            case let .success(url):
                loadGaussians(from: url, transform: scene.initialTransform, completion: completion)
            case let .failure(error):
                completion(.failure(error))
            }
        }
    }

    static func loadGaussians(from url: URL, completion: @escaping (Result<[GaussianSplat], Error>) -> Void) {
        loadGaussians(from: url, transform: matrix_identity_float4x4, completion: completion)
    }

    private static func loadGaussians(from url: URL, transform: simd_float4x4, completion: @escaping (Result<[GaussianSplat], Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let gaussians = try SPZModelLoader.load(from: url)
                let transformed = applyTransform(gaussians: gaussians, transform: transform)
                completion(.success(transformed))
            } catch {
                completion(.failure(error))
            }
        }
    }

    private static func applyTransform(gaussians: [GaussianSplat], transform: simd_float4x4) -> [GaussianSplat] {
        if transform == matrix_identity_float4x4 {
            return gaussians
        }

        let rotationMatrix = simd_float3x3(
            SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
            SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
            SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        )
        let rotationQuat = simd_quaternion(transform)

        return gaussians.map { g in
            var mutated = g
            // Transform position
            let p4 = transform * SIMD4<Float>(g.position.x, g.position.y, g.position.z, 1.0)
            mutated.position = SIMD3<Float>(p4.x, p4.y, p4.z)

            // Transform rotation (quaternion multiplication)
            // GaussianSplat.rotationWXYZ is (W, X, Y, Z)
            let oldQuat = simd_quaternion(SIMD4<Float>(g.rotationWXYZ.y, g.rotationWXYZ.z, g.rotationWXYZ.w, g.rotationWXYZ.x))
            let newQuat = rotationQuat * oldQuat
            mutated.rotationWXYZ = SIMD4<Float>(newQuat.vector.w, newQuat.vector.x, newQuat.vector.y, newQuat.vector.z)

            // Transform scale (approximate if non-uniform, but usually uniform or just rotation)
            // We apply rotation to the scale axes if needed, but GS scales are local.
            // If the matrix has scaling, we should apply it to mutated.scale.
            let sx = simd_length(transform.columns.0.xyz)
            let sy = simd_length(transform.columns.1.xyz)
            let sz = simd_length(transform.columns.2.xyz)
            mutated.scale *= SIMD3<Float>(sx, sy, sz)

            return mutated
        }
    }

    private static func debugGaussians() -> [GaussianSplat] {
        var results: [GaussianSplat] = []
        let gridSize = 4
        let spacing: Float = 1.2
        let C0: Float = 0.28209479177387814
        
        for y in 0..<gridSize {
            for x in 0..<gridSize {
                for z in 0..<gridSize {
                    let fx = Float(x) - Float(gridSize-1) * 0.5
                    let fy = Float(y) - Float(gridSize-1) * 0.5
                    let fz = Float(z) - Float(gridSize-1) * 0.5
                    
                    let position = SIMD3<Float>(fx * spacing, fy * spacing + 1.0, fz * spacing)
                    
                    let angle = Float(x + y + z) * 0.5
                    let axis = simd_normalize(SIMD3<Float>(sin(Float(x)), cos(Float(y)), sin(Float(z) * 0.7)))
                    let s = sin(angle * 0.5)
                    let rotation = SIMD4<Float>(cos(angle * 0.5), axis.x * s, axis.y * s, axis.z * s)
                    
                    let scaleBase = 0.3 + 0.2 * sin(Float(x) * 1.5)
                    let scale = SIMD3<Float>(scaleBase, scaleBase * 1.5, scaleBase * 0.7)
                    
                    let r = 0.5 + 0.5 * sin(Float(x) * 1.0)
                    let g = 0.5 + 0.5 * sin(Float(y) * 1.2)
                    let b = 0.5 + 0.5 * sin(Float(z) * 1.4)
                    
                    var sh = [Float](repeating: 0, count: 48)
                    sh[0] = (r - 0.5) / C0
                    sh[1] = (g - 0.5) / C0
                    sh[2] = (b - 0.5) / C0
                    
                    sh[3] = 0.5 * sin(Float(x))
                    sh[4] = 0.5 * cos(Float(y))
                    sh[5] = -0.5 * sin(Float(z))
                    sh[6] = 0.5 * cos(Float(z))
                    sh[7] = 0.5 * sin(Float(x))
                    sh[8] = -0.5 * cos(Float(y))
                    
                    let density = 0.6 + 0.4 * cos(Float(x + y + z) * 0.8)
                    
                    results.append(GaussianSplat(position: position,
                                                 rotationWXYZ: rotation,
                                                 scale: scale,
                                                 density: density,
                                                 sh: sh))
                }
            }
        }
        return results
    }
}

extension SIMD4 {
    var xyz: SIMD3<Scalar> {
        return SIMD3<Scalar>(x, y, z)
    }
}
