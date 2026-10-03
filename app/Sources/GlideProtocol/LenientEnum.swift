import Foundation

/// A string enum that decodes an unknown value to `fallback` instead of failing, so a newer core can add a
/// state, phase or kind without breaking an older app. The app shows `fallback` neutrally.
public protocol LenientEnum: RawRepresentable, Codable, Sendable, Equatable, CaseIterable where RawValue == String {
    static var fallback: Self { get }
}

extension LenientEnum {
    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = Self(rawValue: raw) ?? Self.fallback
    }
}
