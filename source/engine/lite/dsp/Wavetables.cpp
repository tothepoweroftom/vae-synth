#include "Wavetables.h"

#include <juce_dsp/juce_dsp.h>

#include <algorithm>
#include <cmath>

namespace vae::dsp
{

namespace
{
constexpr int kFrameSize = Wavetable::kFrameSize;
constexpr int kNumBins = kFrameSize / 2 + 1; // DC .. Nyquist of a kFrameSize FFT

int orderOf (int size)
{
    int order = 0;
    while ((1 << order) < size)
        ++order;
    return order;
}

} // namespace

//==============================================================================
Wavetable::Wavetable (int frames)
    : numFrames (frames)
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

std::unique_ptr<Wavetable> Wavetable::fromSpectra (const std::vector<float>& spectra, int frames)
{
    frames = juce::jlimit (1, kMaxFrames, frames);
    jassert (spectra.size() >= static_cast<size_t> (frames) * kNumBins * 2);

    std::unique_ptr<Wavetable> table (new Wavetable (frames));

    // One inverse FFT per level size. JUCE's FFT engines may scale their
    // output differently per size, so calibrate: a unit harmonic must come
    // out at the same amplitude at every size.
    struct SizedFft
    {
        std::unique_ptr<juce::dsp::FFT> fft;
        float scale = 1.0f;
    };
    SizedFft ffts[16];
    std::vector<float> work (2 * kFrameSize);

    auto unitAmplitude = [&] (juce::dsp::FFT& fft, int size)
    {
        std::fill (work.begin(), work.begin() + 2 * size, 0.0f);
        work[3] = -1.0f; // harmonic 1, sine phase
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
            for (int h = 1; h <= maxHarmonic; ++h) // bin 0 (DC) always dropped
            {
                work[static_cast<size_t> (2 * h)] = spectrum[2 * h];
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

    for (int frame = 0; frame < frames; ++frame)
    {
        const float peak = peakOf (frame);
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

//==============================================================================
std::unique_ptr<Wavetable> makeDefaultTable()
{
    std::vector<float> spectrum (kNumBins * 2, 0.0f);
    for (int h = 1; h <= kDefaultTableHarmonics; ++h)
        spectrum[static_cast<size_t> (2 * h + 1)] = -defaultTableHarmonic (h);
    return Wavetable::fromSpectra (spectrum, 1);
}

} // namespace vae::dsp
