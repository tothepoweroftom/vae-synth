/*
  Audio features and distances for comparing renders of the same preset
  (tools/compare). All renders are mono, 48 kHz, analysed with a 2048-point
  Hann window and a 512-sample hop.
*/

#pragma once

#include <juce_dsp/juce_dsp.h>

#include <array>
#include <vector>

namespace vae::compare
{
inline constexpr int kMelBands = 40;
inline constexpr int kFftOrder = 11;
inline constexpr int kFftSize = 1 << kFftOrder;
inline constexpr int kHop = 512;
inline constexpr float kFloorDb = -120.0f;
inline constexpr float kSilentDb = -60.0f;
inline constexpr float kRangeDb = 60.0f; // spectra are clamped to peak - 60 dB (residue below is inaudible)

using MelVector = std::array<float, kMelBands>;

struct Features
{
    bool silent = true;
    float levelDb = kFloorDb;
    float centroidHz = 0.0f;
    float flatness = 0.0f;
    float attackS = 0.0f;
    float releaseS = 0.0f;
    MelVector meanMel {};            // long-term spectrum shape, dB, mean over bands removed
    std::vector<MelVector> frameMel; // per frame, dB, clamped and shifted like meanMel
    std::vector<float> envelope;     // frame RMS / peak frame RMS
};

struct Distances
{
    float melDb = 0.0f;      // RMS over bands of the meanMel difference
    float frameMelDb = 0.0f; // mean over frames of the per-frame RMS-over-bands difference
    float envCorr = 0.0f;    // Pearson correlation of the envelopes
    float levelDb = 0.0f;    // |level difference|
    float centroidOct = 0.0f;
    float flatness = 0.0f;
};

/** Times in the probe (seconds) the envelope measurements refer to. */
struct ProbeTimes
{
    double note1On = 0.0, note1Off = 1.5, chordOn = 2.5;
};

Features analyse (const std::vector<float>& mono, double sampleRate, const ProbeTimes& probe);
Distances compare (const Features& a, const Features& b);

} // namespace vae::compare
