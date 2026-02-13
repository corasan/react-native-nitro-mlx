import AVFoundation
import Foundation
internal import MLX

class AudioCaptureManager {
  private let audioEngine = AVAudioEngine()
  private var audioBuffer: [Float] = []
  private let bufferLock = NSLock()
  private let targetSampleRate: Double = 16000

  var isCapturing: Bool { audioEngine.isRunning }

  func startCapturing() async throws {
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.record, mode: .measurement)
    try session.setActive(true)

    let inputNode = audioEngine.inputNode
    let inputFormat = inputNode.outputFormat(forBus: 0)
    let outputFormat = AVAudioFormat(
      commonFormat: .pcmFormatFloat32,
      sampleRate: targetSampleRate,
      channels: 1,
      interleaved: false
    )!

    guard
      let converter = AVAudioConverter(
        from: inputFormat, to: outputFormat)
    else {
      throw NSError(
        domain: "AudioCaptureManager",
        code: -1,
        userInfo: [
          NSLocalizedDescriptionKey:
            "Failed to create audio converter"
        ]
      )
    }

    bufferLock.lock()
    audioBuffer.removeAll()
    bufferLock.unlock()

    inputNode.installTap(
      onBus: 0, bufferSize: 4096, format: inputFormat
    ) { [weak self] buffer, _ in
      guard let self else { return }

      let frameCount = AVAudioFrameCount(
        targetSampleRate * Double(buffer.frameLength)
          / inputFormat.sampleRate
      )
      guard
        let convertedBuffer = AVAudioPCMBuffer(
          pcmFormat: outputFormat, frameCapacity: frameCount)
      else { return }

      var error: NSError?
      converter.convert(to: convertedBuffer, error: &error) {
        _, outStatus in
        outStatus.pointee = .haveData
        return buffer
      }

      if error == nil, let channelData = convertedBuffer.floatChannelData {
        let frames = Int(convertedBuffer.frameLength)
        self.bufferLock.lock()
        self.audioBuffer.append(
          contentsOf: UnsafeBufferPointer(
            start: channelData[0], count: frames))
        self.bufferLock.unlock()
      }
    }

    audioEngine.prepare()
    try audioEngine.start()
  }

  func snapshotAndClear() -> MLXArray? {
    bufferLock.lock()
    let samples = audioBuffer
    audioBuffer.removeAll()
    bufferLock.unlock()

    guard samples.count >= 8000 else { return nil }
    return MLXArray(samples)
  }

  func snapshot() -> MLXArray? {
    bufferLock.lock()
    let samples = audioBuffer
    bufferLock.unlock()

    guard samples.count >= 16000 else { return nil }
    return MLXArray(samples)
  }

  func stopCapturing() -> MLXArray {
    audioEngine.inputNode.removeTap(onBus: 0)
    audioEngine.stop()

    bufferLock.lock()
    let samples = audioBuffer
    audioBuffer.removeAll()
    bufferLock.unlock()

    return MLXArray(samples)
  }
}
