import Foundation

func dictionaryToJson(_ dict: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: dict),
          let json = String(data: data, encoding: .utf8) else {
        return "{}"
    }
    return json
}
