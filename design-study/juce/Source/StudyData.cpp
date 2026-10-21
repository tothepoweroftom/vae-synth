#include "StudyData.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace
{
std::vector<float> halfFloats (const juce::var& value)
{
    juce::MemoryOutputStream stream;
    if (! juce::Base64::convertFromBase64 (stream, value.toString())) return {};
    const auto* bytes = static_cast<const unsigned char*> (stream.getData());
    std::vector<float> result (stream.getDataSize() / 2);
    for (size_t i = 0; i < result.size(); ++i)
    {
        const auto h = unsigned (bytes[i * 2]) | (unsigned (bytes[i * 2 + 1]) << 8);
        const int exponent = (h >> 10) & 31, fraction = h & 1023;
        const float sign = (h & 32768) ? -1.0f : 1.0f;
        result[i] = exponent == 0 ? sign * std::ldexp (float (fraction), -24)
                    : sign * std::ldexp (1.0f + float (fraction) / 1024.0f, exponent - 15);
    }
    return result;
}
float dot (const std::array<float, 32>& a, const std::array<float, 32>& b)
{
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += double (a[i]) * b[i];
    return float (sum);
}
std::array<float, 32> vector32 (const juce::var& value)
{
    std::array<float, 32> result {};
    for (int i = 0; i < std::min (32, value.size()); ++i) result[size_t (i)] = float (value[i]);
    return result;
}
}

bool StudyData::load (const void* bytes, size_t length)
{
    const auto data = juce::JSON::parse (juce::String::fromUTF8 (static_cast<const char*> (bytes), int (length)));
    if (int (data["z_dim"]) != 32 || ! data["dec_a"].isArray()) return false;
    axes.clear(); presets.clear(); labels.clear(); names.clear(); decoder.clear(); bandHz.clear();
    categories = int (data["n_cls"]);
    for (const auto& entry : *data["directions"].getArray())
        axes.push_back ({entry["name"].toString(), entry["low"].toString(), entry["high"].toString(),
                         vector32 (entry["u"]), float (entry["mean"]), float (entry["sd"])});
    for (const auto& entry : *data["dims"].getArray())
    {
        const int dimension = int (entry["d"]);
        if (dimension < 0 || dimension >= 32) return false;
        std::vector<std::pair<float, juce::String>> sections;
        if (auto* object = entry["sections"].getDynamicObject())
            for (const auto& p : object->getProperties()) sections.emplace_back (float (p.value), p.name.toString());
        std::sort (sections.begin(), sections.end(), [] (const auto& a, const auto& b) { return a.first > b.first; });
        juce::String name = "Mixed";
        if (! sections.empty() && sections[0].first > 0.17f)
        {
            name = sections[0].second;
            if (sections.size() > 1 && sections[1].first > 0.15f) name += " + " + sections[1].second.toLowerCase();
        }
        StudyAxis axis {"z" + juce::String (dimension) + " · " + name, "−", "+", {}, float (entry["mean"]), float (entry["sd"])};
        axis.direction[size_t (dimension)] = 1;
        axes.push_back (axis);
    }
    const auto bank = data["presets"];
    const auto z = halfFloats (bank["z"]);
    if (z.size() != size_t (bank["label"].size()) * 32) return false;
    for (int i = 0; i < bank["label"].size(); ++i)
    {
        std::array<float, 32> row;
        std::copy_n (z.begin() + i * 32, 32, row.begin());
        presets.push_back (row); labels.push_back (int (bank["label"][i])); names.push_back (bank["name"][i].toString());
    }
    initialLatent = vector32 (data["mean_z"]["Bass"]);
    for (const auto& entry : *data["dec_a"].getArray())
    {
        Layer layer {int (entry["shape"][1]), int (entry["shape"][0]), halfFloats (entry["w"]), halfFloats (entry["b"])};
        if (layer.weights.size() != size_t (layer.inputs * layer.outputs) || layer.biases.size() != size_t (layer.outputs)) return false;
        decoder.push_back (std::move (layer));
    }
    const auto pca = data["pca"];
    components = halfFloats (pca["components"]); mean = halfFloats (pca["mean"]);
    scale = float (pca["scale"]); frames = int (pca["frames"]);
    for (const auto& value : *data["band_hz"].getArray()) bandHz.push_back (float (value));
    dt = float (data["dt"]); f0 = float (data["f0"]); holdFrames = int (data["hold_frames"]);
    return ! decoder.empty() && decoder.front().inputs == 32 + categories
           && mean.size() == size_t (frames) * bandHz.size()
           && components.size() == mean.size() * size_t (decoder.back().outputs);
}

std::array<float, 3> StudyData::project (const std::array<float, 32>& z, std::array<int, 3> assigned) const
{
    std::array<float, 3> result {};
    for (size_t i = 0; i < 3; ++i)
    {
        const auto& axis = axes.at (size_t (assigned[i]));
        result[i] = (dot (axis.direction, z) - axis.mean) / (5 * axis.sd) + 0.5f;
    }
    return result;
}

void StudyData::moveTo (std::array<float, 32>& z, std::array<int, 3> assigned, std::array<float, 3> target) const
{
    // Minimum-norm displacement, including the browser's 1e-6 regulariser.
    double matrix[3][3] {}, residual[3] {}, coefficients[3] {};
    for (int i = 0; i < 3; ++i)
    {
        const auto& axis = axes.at (size_t (assigned[size_t (i)]));
        residual[i] = axis.mean + (target[size_t (i)] - 0.5) * 5 * axis.sd - dot (axis.direction, z);
        for (int j = 0; j < 3; ++j) matrix[i][j] = dot (axis.direction, axes.at (size_t (assigned[size_t (j)])).direction) + (i == j ? 1e-6 : 0);
    }
    for (int c = 0; c < 3; ++c)
    {
        int pivot = c;
        for (int k = c + 1; k < 3; ++k) if (std::abs (matrix[k][c]) > std::abs (matrix[pivot][c])) pivot = k;
        for (int j = 0; j < 3; ++j) std::swap (matrix[c][j], matrix[pivot][j]);
        std::swap (residual[c], residual[pivot]);
        for (int k = c + 1; k < 3; ++k)
        {
            const double factor = matrix[k][c] / matrix[c][c];
            for (int j = c; j < 3; ++j) matrix[k][j] -= factor * matrix[c][j];
            residual[k] -= factor * residual[c];
        }
    }
    for (int c = 2; c >= 0; --c)
    {
        double sum = residual[c];
        for (int j = c + 1; j < 3; ++j) sum -= matrix[c][j] * coefficients[j];
        coefficients[c] = sum / matrix[c][c];
    }
    for (int i = 0; i < 3; ++i)
        for (size_t k = 0; k < 32; ++k) z[k] += float (coefficients[i] * axes.at (size_t (assigned[size_t (i)])).direction[k]);
}

juce::Image StudyData::spectrogram (const std::array<float, 32>& z, int category)
{
    if (decoder.empty()) return {};
    std::vector<float> h (size_t (32 + categories), 0);
    std::copy (z.begin(), z.end(), h.begin());
    h[size_t (32 + juce::jlimit (0, categories - 1, category))] = 1;
    for (size_t k = 0; k < decoder.size(); ++k)
    {
        const auto& layer = decoder[k];
        std::vector<float> next (size_t (layer.outputs), 0);
        for (int o = 0; o < layer.outputs; ++o)
        {
            double sum = layer.biases[size_t (o)];
            for (int i = 0; i < layer.inputs; ++i) sum += double (layer.weights[size_t (o * layer.inputs + i)]) * h[size_t (i)];
            next[size_t (o)] = float (k + 1 < decoder.size() ? sum / (1 + std::exp (-sum)) : sum);
        }
        h = std::move (next);
    }
    auto spectrum = mean;
    for (size_t k = 0; k < h.size(); ++k)
        for (size_t j = 0; j < spectrum.size(); ++j) spectrum[j] += h[k] * scale * components[k * spectrum.size() + j];
    const float peak = *std::max_element (spectrum.begin(), spectrum.end());
    for (auto& value : spectrum) value = std::max (value - peak, -80.0f);
    const size_t bands = bandHz.size();
    std::vector<double> power (spectrum.size()), energy (size_t (frames), 0), average (bands, 0);
    for (size_t j = 0; j < power.size(); ++j) power[j] = std::pow (10.0, spectrum[j] / 10.0);
    for (int t = 0; t < frames; ++t)
    {
        double sum = 0;
        for (size_t b = 0; b < bands; ++b) sum += power[size_t (t) * bands + b];
        energy[size_t (t)] = 10 * std::log10 (sum + 1e-12);
    }
    const double maximum = *std::max_element (energy.begin(), energy.end());
    int active = 0;
    for (int t = 0; t < frames; ++t)
        if (energy[size_t (t)] >= maximum - 30)
        {
            ++active;
            for (size_t b = 0; b < bands; ++b) average[b] += power[size_t (t) * bands + b];
        }
    double total = 0, frequency = 0, logs = 0;
    for (size_t b = 0; b < bands; ++b)
    {
        average[b] /= std::max (active, 1); total += average[b];
        frequency += average[b] * bandHz[b]; logs += std::log (average[b] + 1e-12);
    }
    int attack = 0, last = frames - 1;
    while (attack < frames && energy[size_t (attack)] < maximum - 3) ++attack;
    while (last > 0 && energy[size_t (last)] < maximum - 40) --last;
    double movement = 0; int count = 0;
    for (int t = 1; t < std::min (holdFrames, frames); ++t)
        for (size_t b = 0; b < bands; ++b)
        {
            const auto a = spectrum[size_t (t) * bands + b], previous = spectrum[size_t (t - 1) * bands + b];
            if (std::max (a, previous) > -50) { movement += std::abs (a - previous); ++count; }
        }
    tooltip = "Model-predicted spectrum (not rendered audio)\nBrightness: " + juce::String (std::log2 (frequency / total / f0), 2)
              + " octaves above fundamental\nAttack: " + juce::String (attack * dt * 1000, 0)
              + " ms\nLength: " + juce::String ((last + 1) * dt, 2)
              + " s\nNoise: " + juce::String (std::exp (logs / double (bands)) / (total / double (bands)), 3)
              + "\nMovement: " + juce::String (count ? movement / count : 0, 3);
    juce::Image image (juce::Image::RGB, frames, int (bands), false);
    const juce::Colour background (0xff252525), accent (0xff70ff63);
    for (int t = 0; t < frames; ++t)
        for (size_t b = 0; b < bands; ++b)
        {
            const float intensity = std::pow (juce::jlimit (0.0f, 1.0f, (spectrum[size_t (t) * bands + b] + 70) / 70), .8f);
            image.setPixelAt (t, int (bands - b - 1), background.interpolatedWith (accent, intensity));
        }
    return image;
}
