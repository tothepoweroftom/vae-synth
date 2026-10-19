#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include <functional>

namespace vae::params
{

/** The value type of a parameter. */
enum class ParamKind
{
    Float,
    Int,
    Bool,
    Choice
};

/** Plain, copyable descriptor of one parameter. See docs/specs/parameter-manager.md section 2. */
struct ParamSpec
{
    using ToText      = std::function<juce::String (float raw)>;
    using FromText    = std::function<float (const juce::String& text)>;
    using AudioChange = std::function<void (float raw)>;

    juce::String id;                                    ///< Unique id; also the APVTS id. Required.
    juce::String name;                                  ///< Host-visible display name (defaults to id).
    ParamKind kind = ParamKind::Float;
    juce::NormalisableRange<float> range;               ///< Raw-unit range. Default-constructed = "unset".
    float defaultValue = 0.0f;                          ///< Raw (not normalised) default.
    juce::String unit;                                  ///< Suffix used by the default text formatter.
    juce::StringArray choices;                          ///< Choice names (Choice kind only).
    bool hostVisible = true;                            ///< false = hidden parameter.
    bool restoreFromPreset = true;                      ///< false = loadPreset keeps the current value.
    bool automatable = true;
    bool discrete = false;
    bool meta = false;
    int versionHint = 1;                                ///< Passed to juce::ParameterID.
    juce::AudioProcessorParameter::Category category = juce::AudioProcessorParameter::genericParameter;
    ToText toText;                                      ///< Optional custom formatter.
    FromText fromText;                                  ///< Optional custom parser.
    AudioChange onAudioChange;                          ///< Invoked on the audio thread from ParamStore::dispatch().

    /** Logarithmic-feeling frequency range: skewed so the geometric centre sqrt(lo*hi) maps to 0.5. */
    static juce::NormalisableRange<float> frequencyRange (float lo, float hi);

    /** Formats a raw value using toText, or the kind's default formatter when toText is unset. */
    juce::String valueToText (float raw) const;

    /** Parses text to a raw value using fromText, or the kind's default parser when fromText is unset. */
    float textToValue (const juce::String& text) const;

    /** Returns a copy whose range/default are completed for its kind
        (Bool -> [0,1] step 1, Choice with unset range -> [0, n-1] step 1, Int -> step >= 1,
        default snapped into the range, empty name -> id). */
    ParamSpec completed() const;

    /** True when range is still the default-constructed value (start 0, end 1, no interval, no skew). */
    bool rangeIsUnset() const noexcept;
};

} // namespace vae::params
