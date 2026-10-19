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

namespace testutil
{
/** RMS of x[from, to). */
inline float rmsOf (const std::vector<float>& x, size_t from, size_t to)
{
    to = std::min (to, x.size());
    double sum = 0.0;
    for (size_t i = from; i < to; ++i)
        sum += double (x[i]) * double (x[i]);
    return to > from ? float (std::sqrt (sum / double (to - from))) : 0.0f;
}

/** Pitch of x[from, to) from positive-going zero crossings: crossings counted
    between the first and the last one, so partial periods at the edges don't
    bias the estimate. */
inline float zeroCrossingHz (const std::vector<float>& x, size_t from, size_t to, double sampleRate = kRate)
{
    int crossings = 0;
    size_t first = 0, last = 0;
    for (size_t i = from + 1; i < std::min (to, x.size()); ++i)
        if (x[i - 1] <= 0.0f && x[i] > 0.0f)
        {
            if (crossings++ == 0)
                first = i;
            last = i;
        }
    return crossings > 1 ? float (double (crossings - 1) * sampleRate / double (last - first)) : 0.0f;
}

/** Drives an engine block by block with arbitrary MIDI; blocks may be empty
    (0 samples), as some hosts send to flush MIDI. Keeps the left channel. */
struct Player
{
    explicit Player (SynthEngine& e)
        : engine (e)
    {
    }

    /** One block of `n` samples (n >= 0) with these messages at offset 0. */
    void block (int n, std::initializer_list<juce::MidiMessage> messages = {})
    {
        juce::AudioBuffer<float> buffer (2, juce::jmax (1, n));
        buffer.setSize (2, n, false, false, true);
        juce::MidiBuffer midi;
        for (const auto& m : messages)
            midi.addEvent (m, 0);
        engine.process (buffer, midi);
        for (int i = 0; i < n; ++i)
        {
            const float s = buffer.getSample (0, i);
            finite = finite && std::isfinite (s);
            left.push_back (s);
        }
    }

    /** Render `n` samples in blocks of 256 with no MIDI. */
    void run (int n)
    {
        for (int done = 0; done < n; done += 256)
            block (juce::jmin (256, n - done));
    }

    float peak (size_t from, size_t to) const
    {
        float p = 0.0f;
        for (size_t i = from; i < juce::jmin (to, left.size()); ++i)
            p = juce::jmax (p, std::abs (left[i]));
        return p;
    }
    float tailPeak (size_t samples) const
    {
        return peak (left.size() - juce::jmin (samples, left.size()), left.size());
    }

    float zeroCrossingHz (size_t from, size_t to, double sampleRate) const
    {
        return testutil::zeroCrossingHz (left, from, to, sampleRate);
    }

    SynthEngine& engine;
    std::vector<float> left;
    bool finite = true;
};

/** Default patch with osc 1 set to a sine (easy pitch measurement). */
inline schema::Snapshot sinePatch()
{
    auto s = schema::Snapshot::defaults();
    s[schema::kOsc1Wave] = float (*schema::optionIndex (schema::kOsc1Wave, "sine"));
    return s;
}
} // namespace testutil

} // namespace vae
