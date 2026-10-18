#include "Wavetables.h"

#include <juce_dsp/juce_dsp.h>

#include <algorithm>
#include <cmath>

namespace ablaze
{

namespace
{
constexpr int kFrameSize = Wavetable::kFrameSize;
constexpr int kNumBins = kFrameSize / 2 + 1;   // DC .. Nyquist of a kFrameSize FFT
constexpr float kTwoPi = juce::MathConstants<float>::twoPi;

int orderOf (int size)
{
    int order = 0;
    while ((1 << order) < size)
        ++order;
    return order;
}

float frac (float x) { return x - std::floor (x); }

//==============================================================================
// Built-in table definitions.

// Naive (aliased) single-cycle shapes: the FFT + per-level harmonic cut is
// what band-limits them.
float basicShape (int shape, float p)
{
    switch (shape)
    {
        case 0:  return std::sin (kTwoPi * p);
        case 1:  return 1.0f - 4.0f * std::abs (p - 0.5f);
        case 2:  return 2.0f * p - 1.0f;
        default: return p < 0.5f ? 1.0f : -1.0f;
    }
}

/** Time-domain definition of built-in table `table` at morph position t in [0, 1]. */
void fillTimeDomain (int table, float t, float* out)
{
    for (int i = 0; i < kFrameSize; ++i)
    {
        const float p = static_cast<float> (i) / static_cast<float> (kFrameSize);
        float y = 0.0f;

        switch (table)
        {
            case 0:   // Basic: sine -> triangle -> saw -> square
            {
                const float x = t * 3.0f;
                const int a = std::min (2, static_cast<int> (x));
                const float m = x - static_cast<float> (a);
                y = basicShape (a, p) * (1.0f - m) + basicShape (a + 1, p) * m;
                break;
            }
            case 1:   // PWM: square narrowing to a 3 % pulse
                y = p < 0.5f - 0.47f * t ? 1.0f : -1.0f;
                break;

            case 4:   // Sync: hard-synced saw, slave ratio 1 -> 8
                y = 2.0f * frac (p * (1.0f + 7.0f * t)) - 1.0f;
                break;

            case 5:   // Fold: sine through a sine wavefolder, gain 1 -> 8
                y = std::sin (0.5f * juce::MathConstants<float>::pi * (1.0f + 7.0f * t) * std::sin (kTwoPi * p));
                break;

            default:
                break;
        }

        out[i] = y;
    }
}

/** Spectral definition (harmonic amplitudes, sine phase) for additive tables. */
float harmonicAmplitude (int table, float t, int h)
{
    const float n = static_cast<float> (h);

    if (table == 2)   // Harmonic: saw partials fade in one by one
    {
        const float top = 1.0f + 127.0f * t * t;
        return (1.0f / n) * juce::jlimit (0.0f, 1.0f, top - n + 1.0f);
    }

    // Formant: two resonant peaks sweeping up the harmonic series
    const float c1 = 2.0f + 10.0f * t;
    const float c2 = 7.0f + 25.0f * t;
    const float w1 = 1.2f + 1.5f * t;
    const float w2 = 2.0f + 3.0f * t;
    const float f1 = std::exp (-((n - c1) * (n - c1)) / (w1 * w1));
    const float f2 = 0.6f * std::exp (-((n - c2) * (n - c2)) / (w2 * w2));
    return (h == 1 ? 0.35f : 0.0f) + (f1 + f2) / std::sqrt (n);
}

bool isSpectral (int table) { return table == 2 || table == 3; }

} // namespace

//==============================================================================
Wavetable::Wavetable (int frames) : numFrames (frames)
{
    size_t offset = 0;
    for (int level = 0; level < kNumLevels; ++level)
    {
        levelOffset[level] = offset;
        offset += static_cast<size_t> (numFrames) * static_cast<size_t> (levelSize (level) + 1);
    }
    data.assign (offset, 0.0f);
}

int Wavetable::levelForMaxHarmonics (float maxHarmonics) noexcept
{
    if (maxHarmonics >= 1024.0f)
        return 0;
    if (maxHarmonics <= 1.0f)
        return kNumLevels - 1;

    // Smallest L with (1024 >> L) <= maxHarmonics.
    const int level = static_cast<int> (std::ceil (std::log2 (1024.0f / maxHarmonics)));
    return juce::jlimit (0, kNumLevels - 1, level);
}

std::unique_ptr<Wavetable> Wavetable::fromSpectra (const std::vector<float>& spectra, int frames, Normalise normalise)
{
    frames = juce::jlimit (1, kMaxFrames, frames);
    jassert (spectra.size() >= static_cast<size_t> (frames) * kNumBins * 2);

    std::unique_ptr<Wavetable> table (new Wavetable (frames));

    // One inverse FFT per level size. JUCE's FFT engines may scale their
    // output differently per size, so calibrate: a unit harmonic must come
    // out at the same amplitude at every size.
    struct SizedFft { std::unique_ptr<juce::dsp::FFT> fft; float scale = 1.0f; };
    SizedFft ffts[16];
    std::vector<float> work (2 * kFrameSize);

    auto unitAmplitude = [&] (juce::dsp::FFT& fft, int size)
    {
        std::fill (work.begin(), work.begin() + 2 * size, 0.0f);
        work[3] = -1.0f;   // harmonic 1, sine phase
        fft.performRealOnlyInverseTransform (work.data());
        float peak = 0.0f;
        for (int i = 0; i < size; ++i)
            peak = std::max (peak, std::abs (work[static_cast<size_t> (i)]));
        return peak;
    };

    float referenceAmplitude = 0.0f;
    for (int level = 0; level < kNumLevels; ++level)
    {
        const int size = levelSize (level);
        auto& f = ffts[orderOf (size)];
        if (f.fft == nullptr)
        {
            f.fft = std::make_unique<juce::dsp::FFT> (orderOf (size));
            const float a = unitAmplitude (*f.fft, size);
            if (level == 0)
                referenceAmplitude = a;
            f.scale = a > 0.0f ? referenceAmplitude / a : 1.0f;
        }
    }

    // Band-limit every frame at every level.
    for (int frame = 0; frame < frames; ++frame)
    {
        const float* spectrum = spectra.data() + static_cast<size_t> (frame) * kNumBins * 2;

        for (int level = 0; level < kNumLevels; ++level)
        {
            const int size = levelSize (level);
            const auto& f = ffts[orderOf (size)];
            const int maxHarmonic = std::min (1024 >> level, size / 2 - 1);

            std::fill (work.begin(), work.begin() + 2 * size, 0.0f);
            for (int h = 1; h <= maxHarmonic; ++h)   // bin 0 (DC) always dropped
            {
                work[static_cast<size_t> (2 * h)]     = spectrum[2 * h];
                work[static_cast<size_t> (2 * h + 1)] = spectrum[2 * h + 1];
            }
            f.fft->performRealOnlyInverseTransform (work.data());

            auto* dst = const_cast<float*> (table->getFrame (level, frame));
            for (int i = 0; i < size; ++i)
                dst[i] = work[static_cast<size_t> (i)] * f.scale;
            dst[size] = dst[0];
        }
    }

    // Normalise from level 0 (the full-bandwidth version), applying the same
    // gain to every level of a frame so switching levels never changes loudness.
    auto peakOf = [&] (int frame)
    {
        const float* p = table->getFrame (0, frame);
        float peak = 0.0f;
        for (int i = 0; i < kFrameSize; ++i)
            peak = std::max (peak, std::abs (p[i]));
        return peak;
    };

    float globalPeak = 0.0f;
    if (normalise == Normalise::WholeTable)
        for (int frame = 0; frame < frames; ++frame)
            globalPeak = std::max (globalPeak, peakOf (frame));

    for (int frame = 0; frame < frames; ++frame)
    {
        const float peak = normalise == Normalise::WholeTable ? globalPeak : peakOf (frame);
        const float gain = peak > 1.0e-9f ? 1.0f / peak : 0.0f;
        for (int level = 0; level < kNumLevels; ++level)
        {
            auto* p = const_cast<float*> (table->getFrame (level, frame));
            for (int i = 0; i <= levelSize (level); ++i)
                p[i] *= gain;
        }
    }

    return table;
}

std::unique_ptr<Wavetable> Wavetable::fromFrames (const float* frames, int numFramesIn, Normalise normalise)
{
    numFramesIn = juce::jlimit (1, kMaxFrames, numFramesIn);

    juce::dsp::FFT fft (orderOf (kFrameSize));
    std::vector<float> spectra (static_cast<size_t> (numFramesIn) * kNumBins * 2);
    std::vector<float> work (2 * kFrameSize);

    for (int frame = 0; frame < numFramesIn; ++frame)
    {
        std::fill (work.begin(), work.end(), 0.0f);
        std::copy (frames + static_cast<size_t> (frame) * kFrameSize,
                   frames + static_cast<size_t> (frame + 1) * kFrameSize, work.begin());
        fft.performRealOnlyForwardTransform (work.data(), true);
        std::copy (work.begin(), work.begin() + kNumBins * 2,
                   spectra.begin() + static_cast<std::ptrdiff_t> (static_cast<size_t> (frame) * kNumBins * 2));
    }

    return fromSpectra (spectra, numFramesIn, normalise);
}

std::unique_ptr<Wavetable> Wavetable::fromPcm16 (const int16_t* pcm, int numFramesIn)
{
    numFramesIn = juce::jlimit (1, kMaxFrames, numFramesIn);
    std::vector<float> frames (static_cast<size_t> (numFramesIn) * kFrameSize);
    for (size_t i = 0; i < frames.size(); ++i)
        frames[i] = static_cast<float> (pcm[i]) * (1.0f / 32768.0f);
    return fromFrames (frames.data(), numFramesIn, Normalise::WholeTable);
}

//==============================================================================
WavetableBank::WavetableBank()
{
    const int frames = kFramesPerTable;

    for (int table = 0; table < kNumTables; ++table)
    {
        if (isSpectral (table))
        {
            std::vector<float> spectra (static_cast<size_t> (frames) * kNumBins * 2, 0.0f);
            for (int frame = 0; frame < frames; ++frame)
            {
                const float t = static_cast<float> (frame) / static_cast<float> (frames - 1);
                float* bins = spectra.data() + static_cast<size_t> (frame) * kNumBins * 2;
                for (int h = 1; h < kNumBins - 1; ++h)
                    bins[2 * h + 1] = -harmonicAmplitude (table, t, h);
            }
            tables.push_back (Wavetable::fromSpectra (spectra, frames, Wavetable::Normalise::PerFrame));
        }
        else
        {
            std::vector<float> samples (static_cast<size_t> (frames) * kFrameSize);
            for (int frame = 0; frame < frames; ++frame)
                fillTimeDomain (table, static_cast<float> (frame) / static_cast<float> (frames - 1),
                                samples.data() + static_cast<size_t> (frame) * kFrameSize);
            tables.push_back (Wavetable::fromFrames (samples.data(), frames, Wavetable::Normalise::PerFrame));
        }
    }
}

} // namespace ablaze
