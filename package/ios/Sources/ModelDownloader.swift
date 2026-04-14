import Foundation

enum ModelDownloadError: LocalizedError {
    case invalidModelMetadata(String)
    case invalidResponse(statusCode: Int, file: String?)
    case noDownloadableFiles(String)

    var errorDescription: String? {
        switch self {
        case .invalidModelMetadata(let modelId):
            return "Could not read downloadable files for model '\(modelId)'."
        case .invalidResponse(let statusCode, let file):
            if let file {
                return "Download failed for '\(file)' with HTTP status \(statusCode)."
            }
            return "Model request failed with HTTP status \(statusCode)."
        case .noDownloadableFiles(let modelId):
            return "No downloadable files were found for model '\(modelId)'."
        }
    }
}

actor ModelDownloader: NSObject {
    static let shared = ModelDownloader()
    static var debug: Bool = false

    private let fileManager = FileManager.default
    private let manifestFileName = ".download-manifest.json"

    private func log(_ message: String) {
        if Self.debug {
            print("[Downloader] \(message)")
        }
    }

    private let downloadableExtensions: Set<String> = [
        "json", "safetensors", "txt", "model", "tiktoken", "py"
    ]

    private struct DownloadManifest: Codable {
        let modelId: String
        let files: [String]
        let completedAt: Date
    }

    private func fetchFileList(modelId: String) async throws -> [String] {
        let urlString = "https://huggingface.co/api/models/\(modelId)"
        guard let url = URL(string: urlString) else {
            throw ModelDownloadError.invalidModelMetadata(modelId)
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw ModelDownloadError.invalidModelMetadata(modelId)
        }
        guard httpResponse.statusCode == 200 else {
            throw ModelDownloadError.invalidResponse(statusCode: httpResponse.statusCode, file: nil)
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = json["siblings"] as? [[String: Any]]
        else {
            throw ModelDownloadError.invalidModelMetadata(modelId)
        }

        let files = siblings.compactMap { $0["rfilename"] as? String }
            .filter { name in
                let ext = (name as NSString).pathExtension.lowercased()
                return downloadableExtensions.contains(ext)
            }

        guard !files.isEmpty else {
            throw ModelDownloadError.noDownloadableFiles(modelId)
        }

        return files
    }

    private func manifestURL(for modelDir: URL) -> URL {
        modelDir.appendingPathComponent(manifestFileName)
    }

    private func readManifest(for modelDir: URL) -> DownloadManifest? {
        let url = manifestURL(for: modelDir)
        guard let data = try? Data(contentsOf: url) else {
            return nil
        }
        return try? JSONDecoder().decode(DownloadManifest.self, from: data)
    }

    private func writeManifest(modelId: String, files: [String], to modelDir: URL) throws {
        let manifest = DownloadManifest(
            modelId: modelId,
            files: files.sorted(),
            completedAt: Date()
        )
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: manifestURL(for: modelDir), options: .atomic)
    }

    private func validateDownload(at modelDir: URL, manifest: DownloadManifest) -> Bool {
        manifest.files.allSatisfy { file in
            let fileURL = modelDir.appendingPathComponent(file)
            guard fileManager.fileExists(atPath: fileURL.path) else {
                return false
            }

            guard let attributes = try? fileManager.attributesOfItem(atPath: fileURL.path),
                  let size = attributes[.size] as? NSNumber else {
                return false
            }

            return size.int64Value > 0
        }
    }

    private func hasLegacyDownload(at modelDir: URL) -> Bool {
        let configURL = modelDir.appendingPathComponent("config.json")
        guard fileManager.fileExists(atPath: configURL.path) else {
            return false
        }

        guard let enumerator = fileManager.enumerator(at: modelDir, includingPropertiesForKeys: [.isRegularFileKey]) else {
            return false
        }

        var hasWeights = false

        for case let fileURL as URL in enumerator {
            let ext = fileURL.pathExtension.lowercased()
            if ext == "safetensors" || ext == "model" {
                hasWeights = true
                break
            }
        }

        return hasWeights
    }

    private func isExistingFileValid(at url: URL) -> Bool {
        guard fileManager.fileExists(atPath: url.path),
              let attributes = try? fileManager.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else {
            return false
        }

        return size.int64Value > 0
    }

    func download(
        modelId: String,
        progressCallback: @escaping (Double) -> Void
    ) async throws -> URL {
        let files = try await fetchFileList(modelId: modelId)
        let modelDir = getModelDirectory(modelId: modelId)
        try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)

        if let manifest = readManifest(for: modelDir),
           manifest.modelId == modelId,
           manifest.files.sorted() == files.sorted(),
           validateDownload(at: modelDir, manifest: manifest) {
            progressCallback(1.0)
            log("Using existing validated download for: \(modelId)")
            return modelDir
        }

        log("Model directory: \(modelDir.path)")
        log("Files to download: \(files)")

        var downloaded = 0

        for file in files {
            let destURL = modelDir.appendingPathComponent(file)
            let parentDir = destURL.deletingLastPathComponent()
            try fileManager.createDirectory(at: parentDir, withIntermediateDirectories: true)

            if isExistingFileValid(at: destURL) {
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
                throw ModelDownloadError.invalidModelMetadata(modelId)
            }

            log("Response status: \(httpResponse.statusCode) for \(file)")

            guard httpResponse.statusCode == 200 else {
                throw ModelDownloadError.invalidResponse(statusCode: httpResponse.statusCode, file: file)
            }

            if fileManager.fileExists(atPath: destURL.path) {
                try fileManager.removeItem(at: destURL)
            }
            try fileManager.moveItem(at: tempURL, to: destURL)
            log("Saved: \(file)")

            downloaded += 1
            progressCallback(Double(downloaded) / Double(files.count))
        }

        try writeManifest(modelId: modelId, files: files, to: modelDir)
        return modelDir
    }

    func isDownloaded(modelId: String) -> Bool {
        let modelDir = getModelDirectory(modelId: modelId)

        let isComplete: Bool
        if let manifest = readManifest(for: modelDir), manifest.modelId == modelId {
            isComplete = validateDownload(at: modelDir, manifest: manifest)
        } else {
            isComplete = hasLegacyDownload(at: modelDir)
        }

        log("isDownloaded(\(modelId)): \(isComplete)")
        return isComplete
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
