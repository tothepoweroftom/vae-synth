// Shared helpers for the engine test files.
#pragma once

#include <juce_core/juce_core.h>

#include "engine/SynthEngine.h"
#include "schema/SurgeCurves.h"

namespace vae
{
inline constexpr double kRate = 48000.0;

struct RenderStats
{
    float peak = 0.0f;
    double sumSquares = 0.0;
    int64_t samples = 0;
    bool finite = true;

    float rms() const { return samples > 0 ? float (std::sqrt (sumSquares / double (samples))) : 0.0f; }
};

/** A note event for render(): sample position, note, velocity (0 = note off). */
struct NoteEvent
{
    int at;
    int note;
    int velocity;
};

/** Render `totalSamples` in blocks of `blockSizes` (cycled), playing `events`,
    measuring from `startMeasuringAt`. */
inline RenderStats renderEvents (SynthEngine& engine, std::initializer_list<int> blockSizes, int totalSamples,
                                 std::vector<NoteEvent> events, int startMeasuringAt = 0,
                                 std::vector<float>* captureLeft = nullptr)
{
    RenderStats stats;
    int position = 0;
    auto size = blockSizes.begin();
    while (position < totalSamples)
    {
        const int n = juce::jmin (*size, totalSamples - position);
        if (++size == blockSizes.end())
            size = blockSizes.begin();

        juce::AudioBuffer<float> buffer (2, juce::jmax (1, n));
        buffer.setSize (2, n, false, false, true);
        juce::MidiBuffer midi;
        for (const auto& e : events)
            if (e.at >= position && e.at < position + n)
                midi.addEvent (e.velocity > 0 ? juce::MidiMessage::noteOn (1, e.note, (juce::uint8) e.velocity)
                                              : juce::MidiMessage::noteOff (1, e.note),
                               e.at - position);

        engine.process (buffer, midi);

        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < n; ++i)
            {
                const float s = buffer.getSample (ch, i);
                stats.finite = stats.finite && std::isfinite (s);
                if (position + i >= startMeasuringAt)
                {
                    stats.peak = juce::jmax (stats.peak, std::abs (s));
                    stats.sumSquares += double (s) * double (s);
                    ++stats.samples;
                }
            }
        if (captureLeft != nullptr)
            captureLeft->insert (captureLeft->end(), buffer.getReadPointer (0), buffer.getReadPointer (0) + n);
        position += n;
    }
    return stats;
}

/** Note 60 on at 0, optional note off at `noteOffAt`. */
inline RenderStats render (SynthEngine& engine, std::initializer_list<int> blockSizes, int totalSamples,
                           int noteOffAt = -1, int startMeasuringAt = 0)
{
    std::vector<NoteEvent> events { { 0, 60, 110 } };
    if (noteOffAt >= 0)
        events.push_back ({ noteOffAt, 60, 0 });
    return renderEvents (engine, blockSizes, totalSamples, std::move (events), startMeasuringAt);
}

inline schema::Snapshot randomSnapshot (juce::Random& rng)
{
    schema::Snapshot s;
    for (int i = 0; i < schema::kControlCount; ++i)
    {
        const auto& c = schema::control (i);
        switch (c.kind)
        {
            case schema::Kind::Continuous:  s[i] = rng.nextFloat(); break;
            case schema::Kind::Toggle:      s[i] = rng.nextBool() ? 1.0f : 0.0f; break;
            case schema::Kind::Categorical: s[i] = float (rng.nextInt (c.numOptions)); break;
        }
    }
    return s;
}

inline const char* engineLabel (EngineKind k) { return k == EngineKind::Surge ? "Surge" : "Lite"; }

} // namespace vae
