/*
  Band-limited wavetables for the Lite engine.

  A Wavetable is 1..kMaxFrames single-cycle frames. Each frame is stored at
  kNumLevels mip levels (one per octave): level L keeps harmonics
  1..(1024 >> L), so the voice can always pick a level whose top harmonic
  stays below Nyquist. Level L is stored with levelSize(L) samples (halving
  per level, never below kMinLevelSize).

  Construction allocates and runs FFTs: never on the audio thread.
*/

#pragma once

#include <cmath>
#include <memory>
#include <vector>

namespace vae::dsp
{

class Wavetable
{
public:
    static constexpr int kFrameSize = 2048; // source frame size
    static constexpr int kNumLevels = 11;   // 1024 harmonics down to 1
    static constexpr int kMaxFrames = 256;
    static constexpr int kMinLevelSize = 1024; // oversample the mips; Surge's table is 1024 samples

    /** Samples per frame at a mip level (a guard sample follows each frame). */
    static constexpr int levelSize (int level) noexcept
    {
        return (kFrameSize >> level) > kMinLevelSize ? (kFrameSize >> level) : kMinLevelSize;
    }

    /** The mip level whose harmonics all stay below `maxHarmonics`. */
    static int levelForMaxHarmonics (float maxHarmonics) noexcept;

    /** Builds from per-frame harmonic spectra: numFrames * (kFrameSize/2 + 1)
        interleaved (re, im) bins, as JUCE's real-only FFT lays them out.
        Every frame is normalised to peak at 1. */
    static std::unique_ptr<Wavetable> fromSpectra (const std::vector<float>& spectra, int numFrames);

    int getNumFrames() const noexcept { return numFrames; }

    /** One frame at one mip level: levelSize(level) samples + guard (== sample 0). */
    const float* getFrame (int level, int frame) const noexcept
    {
        return data.data() + levelOffset[level]
               + static_cast<size_t> (frame) * static_cast<size_t> (levelSize (level) + 1);
    }

private:
    explicit Wavetable (int numFrames);

    int numFrames;
    std::vector<float> data;
    size_t levelOffset[kNumLevels] = {};
};

/** Harmonic amplitude (sine phase) of the one-frame table the schema's
    "wavetable" wave plays in both engines: a 1/h^0.8 spectrum with a resonant
    hump around harmonic 5, up to harmonic kDefaultTableHarmonics. */
inline constexpr int kDefaultTableHarmonics = 96;
inline float defaultTableHarmonic (int h)
{
    const float d = (float (h) - 5.0f) / 3.0f;
    return std::pow (float (h), -0.8f) * (1.0f + 4.0f * std::exp (-d * d));
}

/** The table the Lite "wavetable" wave plays (defaultTableHarmonic). */
std::unique_ptr<Wavetable> makeDefaultTable();

} // namespace vae::dsp
