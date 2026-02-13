import Foundation
import NitroModules
internal import MLX
internal import MLXAudioSTT
internal import MLXAudioCore

enum STTError: Error {
  case notLoaded
  case notListening
  case alreadyListening
}

class HybridSTT: HybridSTTSpec {
  private var model: GLMASRModel?
  private var activeTask: Task<Any, Error>?
  private var loadTask: Task<Void, Error>?
  private var captureManager: AudioCaptureManager?
  private var listeningOnToken: ((String) -> Void)?
  private var listeningOnError: ((String) -> Void)?

  var isLoaded: Bool { model != nil }
  var isTranscribing: Bool { activeTask != nil }
  var isListening: Bool { captureManager?.isCapturing ?? false }
  var modelId: String = ""

  private func arrayBufferToMLXArray(_ buffer: ArrayBuffer) -> MLXArray {
    let count = buffer.size / MemoryLayout<Float>.size
    let rawPtr = UnsafeRawPointer(buffer.data)
    let floatPtr = rawPtr.bindMemory(to: Float.self, capacity: count)
    let floatBuffer = UnsafeBufferPointer(start: floatPtr, count: count)
    return MLXArray(Array(floatBuffer))
  }

  func load(modelId: String, options: STTLoadOptions?) throws -> Promise<Void> {
    self.loadTask?.cancel()

    return Promise.async { [self] in
      let task = Task { @MainActor in
        self.activeTask?.cancel()
        self.activeTask = nil
        self.model = nil
        MLX.Memory.clearCache()

        let loadedModel = try await GLMASRModel.fromPretrained(modelId)

        try Task.checkCancellation()

        self.model = loadedModel
        self.modelId = modelId

        options?.onProgress?(1.0)
      }

      self.loadTask = task
      try await task.value
    }
  }

  func transcribe(audio: ArrayBuffer) throws -> Promise<String> {
    guard let model else {
      throw STTError.notLoaded
    }

    return Promise.async { [self] in
      let task = Task<Any, Error> {
        let mlxAudio = self.arrayBufferToMLXArray(audio)
        let output = model.generate(audio: mlxAudio)
        return output.text as Any
      }

      self.activeTask = task
      defer { self.activeTask = nil }

      return try await task.value as! String
    }
  }

  func transcribeStream(
    audio: ArrayBuffer,
    onToken: @escaping (_ token: String) -> Void
  ) throws -> Promise<String> {
    guard let model else {
      throw STTError.notLoaded
    }

    return Promise.async { [self] in
      let task = Task<Any, Error> {
        let mlxAudio = self.arrayBufferToMLXArray(audio)
        let stream = model.generateStream(audio: mlxAudio)
        var finalText = ""

        for try await event in stream {
          if Task.isCancelled { break }

          switch event {
          case .token(let token):
            onToken(token)
          case .result(let output):
            finalText = output.text
          case .info:
            break
          }
        }

        return finalText as Any
      }

      self.activeTask = task
      defer { self.activeTask = nil }

      return try await task.value as! String
    }
  }

  func startListening(
    onToken: @escaping (_ token: String) -> Void,
    onError: @escaping (_ error: String) -> Void
  ) throws -> Promise<Void> {
    guard model != nil else {
      throw STTError.notLoaded
    }
    guard captureManager == nil || !captureManager!.isCapturing else {
      throw STTError.alreadyListening
    }

    return Promise.async { [self] in
      self.listeningOnToken = onToken
      self.listeningOnError = onError
      let manager = AudioCaptureManager()
      self.captureManager = manager
      try await manager.startCapturing()
    }
  }

  func stopListening() throws -> Promise<String> {
    guard let model else {
      throw STTError.notLoaded
    }
    guard let manager = captureManager, manager.isCapturing else {
      throw STTError.notListening
    }

    let onToken = self.listeningOnToken

    return Promise.async { [self] in
      let audio = manager.stopCapturing()
      self.captureManager = nil

      let task = Task<Any, Error> {
        let stream = model.generateStream(audio: audio)
        var finalText = ""

        for try await event in stream {
          if Task.isCancelled { break }

          switch event {
          case .token(let token):
            onToken?(token)
          case .result(let output):
            finalText = output.text
          case .info:
            break
          }
        }

        return finalText as Any
      }

      self.activeTask = task
      defer {
        self.activeTask = nil
        self.listeningOnToken = nil
        self.listeningOnError = nil
      }

      return try await task.value as! String
    }
  }

  func stop() throws {
    activeTask?.cancel()
    activeTask = nil
    if let manager = captureManager, manager.isCapturing {
      _ = manager.stopCapturing()
    }
    captureManager = nil
    listeningOnToken = nil
    listeningOnError = nil
  }

  func unload() throws {
    loadTask?.cancel()
    loadTask = nil
    activeTask?.cancel()
    activeTask = nil
    if let manager = captureManager, manager.isCapturing {
      _ = manager.stopCapturing()
    }
    captureManager = nil
    listeningOnToken = nil
    listeningOnError = nil
    model = nil
    modelId = ""
    Memory.clearCache()
  }
}
