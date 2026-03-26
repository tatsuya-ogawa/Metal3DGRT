//
//  SPZDownloader.swift
//  Metal3DGRT
//
//  Created by Antigravity on 2026-03-25.
//

import Foundation

class SPZDownloader {
    static let shared = SPZDownloader()
    
    private init() {}
    
    func downloadSPZ(from url: URL, to destinationURL: URL, completion: @escaping (Result<URL, Error>) -> Void) {
        // If the file already exists, return it immediately.
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            completion(.success(destinationURL))
            return
        }
        
        let task = URLSession.shared.downloadTask(with: url) { localURL, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let localURL = localURL else {
                completion(.failure(NSError(domain: "SPZDownloader", code: -1, userInfo: [NSLocalizedDescriptionKey: "Download failed: no local URL"])))
                return
            }
            
            do {
                // Ensure the directory exists
                let directory = destinationURL.deletingLastPathComponent()
                if !FileManager.default.fileExists(atPath: directory.path) {
                    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                }
                
                // Move the file to the destination
                if FileManager.default.fileExists(atPath: destinationURL.path) {
                    try FileManager.default.removeItem(at: destinationURL)
                }
                try FileManager.default.moveItem(at: localURL, to: destinationURL)
                completion(.success(destinationURL))
            } catch {
                completion(.failure(error))
            }
        }
        
        task.resume()
    }
}
