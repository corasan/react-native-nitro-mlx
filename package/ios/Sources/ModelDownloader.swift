import Foundation

actor ModelDownloader: NSObject {
    static let shared = ModelDownloader()
    static var debug: Bool = false

    private let fileManager = FileManager.default

    private func log(_ message: String) {
        if Self.debug {
            print("[Downloader] \(message)")
        }
    }

    private let downloadableExtensions: Set<String> = [
        "json", "safetensors", "txt", "model", "tiktoken", "py"
    ]

    private func fetchFileList(modelId: String) async throws -> [String] {
        let urlString = "https://huggingface.co/api/models/\(modelId)"
        guard let url = URL(string: urlString) else {
            throw NSError(domain: "ModelDownloader", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Invalid model URL for: \(modelId)"
            ])
        }

        let (data, response) = try await URLSession.shared.data(from: url)

        if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode != 200 {
            throw NSError(domain: "ModelDownloader", code: httpResponse.statusCode, userInfo: [
                NSLocalizedDescriptionKey: "Failed to fetch model info for \(modelId): HTTP \(httpResponse.statusCode)"
            ])
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = json["siblings"] as? [[String: Any]]
        else {
            throw NSError(domain: "ModelDownloader", code: -2, userInfo: [
                NSLocalizedDescriptionKey: "Invalid model metadata format for: \(modelId)"
            ])
        }

        return siblings.compactMap { $0["rfilename"] as? String }
            .filter { name in
                let ext = (name as NSString).pathExtension.lowercased()
                return downloadableExtensions.contains(ext)
            }
    }

    func download(
        modelId: String,
        progressCallback: @escaping (Double) -> Void
    ) async throws -> URL {
        let files = try await fetchFileList(modelId: modelId)
        let modelDir = getModelDirectory(modelId: modelId)
        try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)

        log("Model directory: \(modelDir.path)")
        log("Files to download: \(files)")

        var downloaded = 0

        for file in files {
            let destURL = modelDir.appendingPathComponent(file)

            if fileManager.fileExists(atPath: destURL.path) {
                log("File exists, skipping: \(file)")
                downloaded += 1
                progressCallback(Double(downloaded) / Double(files.count))
                continue
            }

            let urlString = "https://huggingface.co/\(modelId)/resolve/main/\(file)"
            guard let url = URL(string: urlString) else {
                log("Invalid URL: \(urlString)")
                continue
            }

            log("Downloading: \(file)")

            let (tempURL, response) = try await URLSession.shared.download(from: url)

            guard let httpResponse = response as? HTTPURLResponse else {
                log("Invalid response for: \(file)")
                continue
            }

            log("Response status: \(httpResponse.statusCode) for \(file)")

            guard httpResponse.statusCode == 200 else {
                log("Failed to download: \(file) - Status: \(httpResponse.statusCode)")
                throw NSError(domain: "ModelDownloader", code: httpResponse.statusCode, userInfo: [
                    NSLocalizedDescriptionKey: "Failed to download \(file): HTTP \(httpResponse.statusCode)"
                ])
            }

            if fileManager.fileExists(atPath: destURL.path) {
                try fileManager.removeItem(at: destURL)
            }
            try fileManager.moveItem(at: tempURL, to: destURL)
            log("Saved: \(file)")

            downloaded += 1
            progressCallback(Double(downloaded) / Double(files.count))
        }

        return modelDir
    }

    func isDownloaded(modelId: String) -> Bool {
        let modelDir = getModelDirectory(modelId: modelId)

        guard fileManager.fileExists(atPath: modelDir.appendingPathComponent("config.json").path) else {
            log("isDownloaded(\(modelId)): false (missing config.json)")
            return false
        }

        // Check for single safetensors file or sharded pattern (model-00001-of-NNNNN.safetensors)
        let hasSafetensors = fileManager.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors").path)
        let hasShardedIndex = fileManager.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors.index.json").path)

        let result = hasSafetensors || hasShardedIndex
        log("isDownloaded(\(modelId)): \(result)")
        return result
    }

    func getModelDirectory(modelId: String) -> URL {
        let docsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docsDir
            .appendingPathComponent("huggingface/models")
            .appendingPathComponent(modelId.replacingOccurrences(of: "/", with: "_"))
    }

    func deleteModel(modelId: String) throws {
        let modelDir = getModelDirectory(modelId: modelId)
        if fileManager.fileExists(atPath: modelDir.path) {
            try fileManager.removeItem(at: modelDir)
        }
    }
}
