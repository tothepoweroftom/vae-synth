#include "LiteEngine.h"

namespace vae
{
using lite::PlayMode;

namespace
{
// Output guard: an extreme patch (e.g. +48 dB EQ boost into heavy resonance)
// can get very loud; clamp to +12 dBFS rather than hand the host anything wilder.
constexpr float kOutputLimit = 4.0f;
} // namespace

LiteEngine::LiteEngine() = default;
LiteEngine::~LiteEngine() = default;

void LiteEngine::prepare (double newSampleRate, int)
{
    sampleRate = float (newSampleRate);
    if (bank == nullptr)
        bank = std::make_shared<const ablaze::WavetableBank>();

    for (size_t i = 0; i < voices.size(); ++i)
        voices[i].prepare (sampleRate, bank.get(), uint32_t (0x5eed + 977 * i));
    for (size_t i = 0; i < freeLfo.size(); ++i)
        freeLfo[i].seed (uint32_t (0xf00d + i));

    fx.prepare (newSampleRate);
    patch.setAll (controls);
    prepared = true;
    reset();
}

void LiteEngine::reset()
{
    for (auto& v : voices)
        v.kill();
    numHeld = 0;
    lastNoteSemis = -1.0f;
    for (size_t i = 0; i < freeLfo.size(); ++i)
        freeLfo[i].restart (patch.lfo[i].phase);
    fx.reset();
}

void LiteEngine::setControl (int index, float value)
{
    if (index < 0 || index >= schema::kControlCount || ! std::isfinite (value))
        return;

    value = schema::sanitiseValue (index, value);
    if (juce::exactlyEqual (controls[index], value))
        return;

    controls[index] = value;
    patch.set (index, controls);
}

int LiteEngine::numActiveVoices() const noexcept
{
    int n = 0;
    for (const auto& v : voices)
        n += v.isActive() ? 1 : 0;
    return n;
}

float LiteEngine::glideSource() const noexcept
{
    return patch.glideSeconds > 0.0f ? lastNoteSemis : -1.0f;
}

void LiteEngine::noteOn (int note)
{
    // Track held keys (a re-pressed key moves to the top).
    for (int i = 0; i < numHeld; ++i)
        if (held[(size_t) i] == note)
        {
            std::copy (held.begin() + i + 1, held.begin() + numHeld, held.begin() + i);
            --numHeld;
            break;
        }
    if (numHeld < int (held.size()))
        held[(size_t) numHeld++] = note;

    if (patch.playMode == PlayMode::Poly)
    {
        // Same key already sounding: retrigger it. Else a free voice, else
        // steal the oldest releasing voice, else the oldest voice (envelopes
        // restart from their current level, so stealing does not click).
        lite::Voice* chosen = nullptr;
        for (auto& v : voices)
            if (v.isActive() && v.getNote() == note)
                chosen = &v;
        if (chosen == nullptr)
            for (auto& v : voices)
                if (! v.isActive())
                {
                    chosen = &v;
                    break;
                }
        for (int pass = 0; chosen == nullptr && pass < 2; ++pass)
        {
            for (auto& v : voices)
                if ((pass == 1 || v.isReleasing()) && (chosen == nullptr || v.getAge() < chosen->getAge()))
                    chosen = &v;
        }
        chosen->start (patch, note, glideSource(), ++noteCounter);
    }
    else
    {
        auto& v = voices[0];
        const bool gliding = v.isActive();
        const float from = gliding ? (patch.glideSeconds > 0.0f ? v.getCurrentSemis() : -1.0f) : glideSource();
        if (patch.playMode == PlayMode::Legato && gliding && ! v.isReleasing())
            v.glideTo (patch, note, from);
        else
            v.start (patch, note, from, ++noteCounter);
    }
    lastNoteSemis = float (note);
}

void LiteEngine::noteOff (int note)
{
    for (int i = 0; i < numHeld; ++i)
        if (held[(size_t) i] == note)
        {
            std::copy (held.begin() + i + 1, held.begin() + numHeld, held.begin() + i);
            --numHeld;
            break;
        }

    if (patch.playMode == PlayMode::Poly)
    {
        for (auto& v : voices)
            if (v.isActive() && ! v.isReleasing() && v.getNote() == note)
                v.release (patch);
        return;
    }

    auto& v = voices[0];
    if (! v.isActive() || v.getNote() != note)
        return;

    if (numHeld > 0)
    {
        // Return to the most recent held key: mono retriggers, legato glides.
        const int back = held[(size_t) (numHeld - 1)];
        const float from = patch.glideSeconds > 0.0f ? v.getCurrentSemis() : -1.0f;
        v.glideTo (patch, back, from);
        if (patch.playMode == PlayMode::Mono)
            v.retriggerEnvelopes (patch);
        lastNoteSemis = float (back);
    }
    else
    {
        v.release (patch);
    }
}

void LiteEngine::allNotesOff()
{
    numHeld = 0;
    for (auto& v : voices)
        if (v.isActive())
            v.release (patch);
}

void LiteEngine::renderVoices (float* left, float* right, int numSamples)
{
    for (int start = 0; start < numSamples; start += kControlBlock)
    {
        const int n = std::min (kControlBlock, numSamples - start);
        const float seconds = float (n) / sampleRate;
        for (size_t i = 0; i < freeLfo.size(); ++i)
            freeLfoValues.value[i] = freeLfo[i].advance (patch.lfo[i], seconds);
        for (auto& v : voices)
            v.render (left + start, right + start, n, patch, freeLfoValues);
    }
}

void LiteEngine::process (juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi)
{
    buffer.clear();
    const int numSamples = buffer.getNumSamples();
    if (! prepared || buffer.getNumChannels() < 2 || numSamples <= 0)
        return;

    auto* left = buffer.getWritePointer (0);
    auto* right = buffer.getWritePointer (1);

    // Render between events so every event lands on its sample.
    int position = 0;
    for (const auto meta : midi)
    {
        const int at = std::clamp (meta.samplePosition, 0, numSamples);
        if (at > position)
        {
            renderVoices (left + position, right + position, at - position);
            position = at;
        }

        const auto msg = meta.getMessage();
        if (msg.isNoteOn())
            noteOn (msg.getNoteNumber());
        else if (msg.isNoteOff())
            noteOff (msg.getNoteNumber());
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
            allNotesOff();
    }
    if (position < numSamples)
        renderVoices (left + position, right + position, numSamples - position);

    fx.process (left, right, numSamples, patch.fx);

    float check = 0.0f;
    for (int i = 0; i < numSamples; ++i)
    {
        left[i] = std::clamp (left[i] * patch.masterGain, -kOutputLimit, kOutputLimit);
        right[i] = std::clamp (right[i] * patch.masterGain, -kOutputLimit, kOutputLimit);
        check += left[i] + right[i];
    }
    if (! ablaze::isFiniteBits (check))
    {
        // Should be unreachable; recover instead of recirculating NaN in FX tails.
        buffer.clear();
        fx.reset();
        for (auto& v : voices)
            v.kill();
    }
}

} // namespace vae
