#include "Features.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace vae::compare
{
namespace
{
float hzToMel (float hz) { return 2595.0f * std::log10 (1.0f + hz / 700.0f); }
float melToHz (float mel) { return 700.0f * (std::pow (10.0f, mel / 2595.0f) - 1.0f); }

/** Triangular mel filters over FFT bins, 20 Hz .. Nyquist. */
std::vector<std::vector<float>> melFilters (double sampleRate)
{
    const int bins = kFftSize / 2 + 1;
    const float lo = hzToMel (20.0f), hi = hzToMel (float (sampleRate * 0.5));
    std::array<float, kMelBands + 2> edges {};
    for (int i = 0; i < kMelBands + 2; ++i)
        edges[(size_t) i] = melToHz (lo + (hi - lo) * float (i) / float (kMelBands + 1));

    std::vector<std::vector<float>> filters ((size_t) kMelBands, std::vector<float> ((size_t) bins, 0.0f));
    for (int b = 0; b < kMelBands; ++b)
        for (int k = 0; k < bins; ++k)
        {
            const float f = float (k * sampleRate / kFftSize);
            const float l = edges[(size_t) b], c = edges[(size_t) b + 1], r = edges[(size_t) b + 2];
            float w = 0.0f;
            if (f > l && f <= c)
                w = (f - l) / (c - l);
            else if (f > c && f < r)
                w = (r - f) / (r - c);
            filters[(size_t) b][(size_t) k] = w;
        }
    return filters;
}

float toDb (double power) { return float (10.0 * std::log10 (std::max (power, 1.0e-12))); }

float pearson (const std::vector<float>& a, const std::vector<float>& b)
{
    const size_t n = std::min (a.size(), b.size());
    if (n < 2)
        return 0.0f;
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i)
    {
        ma += a[i];
        mb += b[i];
    }
    ma /= double (n);
    mb /= double (n);
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; ++i)
    {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return saa > 0 && sbb > 0 ? float (sab / std::sqrt (saa * sbb)) : 0.0f;
}

float rmsOverBands (const MelVector& a, const MelVector& b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += double (a[i] - b[i]) * double (a[i] - b[i]);
    return float (std::sqrt (sum / double (a.size())));
}
} // namespace

Features analyse (const std::vector<float>& mono, double sampleRate, const ProbeTimes& probe)
{
    static const auto filters = melFilters (sampleRate);
    Features out;

    double sumSquares = 0.0;
    for (float s : mono)
        sumSquares += double (s) * double (s);
    out.levelDb = std::max (kFloorDb, toDb (mono.empty() ? 0.0 : sumSquares / double (mono.size())));
    // Surge leaves inaudible residue (~-70 dB) where Lite is exactly 0; both count as silent.
    out.silent = out.levelDb < kSilentDb;

    juce::dsp::FFT fft (kFftOrder);
    juce::dsp::WindowingFunction<float> window ((size_t) kFftSize, juce::dsp::WindowingFunction<float>::hann, false);
    std::vector<float> buffer ((size_t) kFftSize * 2);
    const int bins = kFftSize / 2 + 1;
    const int frames = mono.size() >= (size_t) kFftSize ? int ((mono.size() - (size_t) kFftSize) / kHop) + 1 : 0;

    std::vector<std::array<double, kMelBands>> melPower ((size_t) frames);
    std::array<double, kMelBands> meanPower {};
    std::vector<float> frameRms ((size_t) frames), frameEnergy ((size_t) frames), frameCentroid ((size_t) frames),
        frameFlatness ((size_t) frames);

    for (int f = 0; f < frames; ++f)
    {
        const float* x = mono.data() + (size_t) f * kHop;
        double e = 0.0;
        for (int i = 0; i < kFftSize; ++i)
            e += double (x[i]) * double (x[i]);
        frameRms[(size_t) f] = float (std::sqrt (e / kFftSize));

        std::fill (buffer.begin(), buffer.end(), 0.0f);
        std::copy (x, x + kFftSize, buffer.begin());
        window.multiplyWithWindowingTable (buffer.data(), (size_t) kFftSize);
        fft.performFrequencyOnlyForwardTransform (buffer.data(), true);

        double total = 0.0, weighted = 0.0, logSum = 0.0;
        for (int k = 1; k < bins; ++k)
        {
            const double p = double (buffer[(size_t) k]) * double (buffer[(size_t) k]);
            total += p;
            weighted += p * k * sampleRate / kFftSize;
            logSum += std::log (p + 1.0e-20);
        }
        frameEnergy[(size_t) f] = float (total);
        frameCentroid[(size_t) f] = total > 0.0 ? float (weighted / total) : 0.0f;
        const double arith = total / (bins - 1);
        frameFlatness[(size_t) f] = arith > 0.0 ? float (std::exp (logSum / (bins - 1)) / arith) : 0.0f;

        for (int b = 0; b < kMelBands; ++b)
        {
            double p = 0.0;
            for (int k = 0; k < bins; ++k)
                p += double (filters[(size_t) b][(size_t) k]) * double (buffer[(size_t) k])
                     * double (buffer[(size_t) k]);
            melPower[(size_t) f][(size_t) b] = p;
            meanPower[(size_t) b] += p / std::max (1, frames);
        }
    }

    // Spectral summaries: energy-weighted centroid, flatness over frames with sound.
    const float maxEnergy = frames > 0 ? *std::max_element (frameEnergy.begin(), frameEnergy.end()) : 0.0f;
    double cw = 0.0, ew = 0.0, flat = 0.0;
    int flatFrames = 0;
    for (int f = 0; f < frames; ++f)
    {
        cw += double (frameEnergy[(size_t) f]) * frameCentroid[(size_t) f];
        ew += frameEnergy[(size_t) f];
        if (frameEnergy[(size_t) f] > maxEnergy * 1.0e-6f && maxEnergy > 0.0f)
        {
            flat += frameFlatness[(size_t) f];
            ++flatFrames;
        }
    }
    out.centroidHz = ew > 0.0 ? float (cw / ew) : 0.0f;
    out.flatness = flatFrames > 0 ? float (flat / flatFrames) : 0.0f;

    // Long-term spectrum shape: dB, clamped to peak - range, mean over bands removed.
    float meanPeak = kFloorDb;
    for (auto p : meanPower)
        meanPeak = std::max (meanPeak, toDb (p));
    float avg = 0.0f;
    for (int b = 0; b < kMelBands; ++b)
    {
        out.meanMel[(size_t) b] = std::max (meanPeak - kRangeDb, toDb (meanPower[(size_t) b]));
        avg += out.meanMel[(size_t) b] / kMelBands;
    }
    for (auto& v : out.meanMel)
        v -= avg;

    // Per-frame spectra, clamped and shifted by the render's overall level.
    float framePeak = kFloorDb;
    for (const auto& frame : melPower)
        for (auto p : frame)
            framePeak = std::max (framePeak, toDb (p));
    out.frameMel.resize ((size_t) frames);
    for (int f = 0; f < frames; ++f)
        for (int b = 0; b < kMelBands; ++b)
            out.frameMel[(size_t) f][(size_t) b] =
                std::max (framePeak - kRangeDb, toDb (melPower[(size_t) f][(size_t) b])) - framePeak;

    // Envelope and its times around note 1.
    const float rmsPeak = frames > 0 ? *std::max_element (frameRms.begin(), frameRms.end()) : 0.0f;
    out.envelope.resize ((size_t) frames);
    for (int f = 0; f < frames; ++f)
        out.envelope[(size_t) f] = rmsPeak > 0.0f ? frameRms[(size_t) f] / rmsPeak : 0.0f;

    const double frameS = double (kHop) / sampleRate;
    auto frameAt = [&] (double t) { return std::clamp (int (t / frameS), 0, std::max (0, frames - 1)); };
    const int on = frameAt (probe.note1On), off = frameAt (probe.note1Off), next = frameAt (probe.chordOn);
    float notePeak = 0.0f;
    for (int f = on; f <= off && f < frames; ++f)
        notePeak = std::max (notePeak, out.envelope[(size_t) f]);

    out.attackS = float (probe.note1Off - probe.note1On);
    for (int f = on; f <= off && f < frames; ++f)
        if (notePeak > 0.0f && out.envelope[(size_t) f] >= 0.9f * notePeak)
        {
            out.attackS = float ((f - on) * frameS);
            break;
        }

    out.releaseS = float (probe.chordOn - probe.note1Off); // censored: still sounding at the next note
    for (int f = off; f < next && f < frames; ++f)
        if (out.envelope[(size_t) f] < 0.01f * notePeak)
        {
            out.releaseS = float ((f - off) * frameS);
            break;
        }

    return out;
}

Distances compare (const Features& a, const Features& b)
{
    Distances d;
    d.melDb = rmsOverBands (a.meanMel, b.meanMel);

    const size_t n = std::min (a.frameMel.size(), b.frameMel.size());
    double sum = 0.0;
    for (size_t f = 0; f < n; ++f)
        sum += rmsOverBands (a.frameMel[f], b.frameMel[f]);
    d.frameMelDb = n > 0 ? float (sum / double (n)) : 0.0f;

    d.envCorr = pearson (a.envelope, b.envelope);
    d.levelDb = std::abs (a.levelDb - b.levelDb);
    d.centroidOct = std::abs (std::log2 (std::max (a.centroidHz, 20.0f) / std::max (b.centroidHz, 20.0f)));
    d.flatness = std::abs (a.flatness - b.flatness);
    return d;
}

} // namespace vae::compare
