/*
  Band-limited wavetables.

  A Wavetable is 1..kMaxFrames single-cycle frames that the voice morphs
  through with the Position parameter. Each frame is stored at kNumLevels mip
  levels (one per octave): level L keeps harmonics 1..(1024 >> L), so the
  voice can always pick a level whose top harmonic stays below Nyquist.
  Level L is stored with levelSize(L) samples (halving per level, never below
  kMinLevelSize), which keeps a 256-frame user table at ~5.8 MB.

  WavetableBank holds the built-in tables, generated procedurally in its
  constructor — no assets to ship. User tables (from .wav files) are built
  with Wavetable::fromPcm16 and handed to SynthEngine::setUserWavetable.

  All construction allocates and runs FFTs: never on the audio thread.
*/

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "Params.h"

namespace ablaze
{

class Wavetable
{
public:
    static constexpr int kFrameSize = 2048;     // source frame size (Vital/Serum standard)
    static constexpr int kNumLevels = 11;       // 1024 harmonics down to 1
    static constexpr int kMaxFrames = 256;
    static constexpr int kMinLevelSize = 256;

    /** Samples per frame at a mip level (a guard sample follows each frame). */
    static constexpr int levelSize (int level) noexcept
    {
        return (kFrameSize >> level) > kMinLevelSize ? (kFrameSize >> level) : kMinLevelSize;
    }

    /** The mip level whose harmonics all stay below `maxHarmonics`. */
    static int levelForMaxHarmonics (float maxHarmonics) noexcept;

    enum class Normalise
    {
        PerFrame,    // every frame peaks at 1 (built-in tables)
        WholeTable,  // the loudest frame peaks at 1, relative levels kept (user tables)
    };

    /** Builds from time-domain frames: numFrames * kFrameSize samples. */
    static std::unique_ptr<Wavetable> fromFrames (const float* frames, int numFrames, Normalise);

    /** Builds from 16-bit frames (the storage format of user tables in
        presets, plugin state and the WASM build). */
    static std::unique_ptr<Wavetable> fromPcm16 (const int16_t* frames, int numFrames);

    /** Builds from per-frame harmonic spectra: numFrames * (kFrameSize/2 + 1)
        interleaved (re, im) bins, as JUCE's real-only FFT lays them out. */
    static std::unique_ptr<Wavetable> fromSpectra (const std::vector<float>& spectra, int numFrames, Normalise);

    int getNumFrames() const noexcept { return numFrames; }

    /** One frame at one mip level: levelSize(level) samples + guard (== sample 0). */
    const float* getFrame (int level, int frame) const noexcept
    {
        return data.data() + levelOffset[level]
               + static_cast<size_t> (frame) * static_cast<size_t> (levelSize (level) + 1);
    }

    size_t getMemoryBytes() const noexcept { return data.size() * sizeof (float); }

private:
    explicit Wavetable (int numFrames);

    int numFrames;
    std::vector<float> data;
    size_t levelOffset[kNumLevels] = {};
};

//==============================================================================
/** The built-in tables (see kWavetableNames in Params.h). */
class WavetableBank
{
public:
    static constexpr int kNumTables = kNumBuiltinWavetables;
    static constexpr int kFramesPerTable = 16;

    WavetableBank();

    const Wavetable& getTable (int index) const noexcept { return *tables[static_cast<size_t> (index)]; }

private:
    std::vector<std::unique_ptr<Wavetable>> tables;
};

} // namespace ablaze
