#include "LiteEngine.h"

#include <algorithm>

namespace vae
{
using lite::PlayMode;

namespace
{
// Output guard: an extreme patch (e.g. +48 dB EQ boost into heavy resonance)
// can get very loud; clamp to +12 dBFS rather than hand the host anything wilder.
constexpr float kOutputLimit = 4.0f;
constexpr float kSmoothTau = 0.020f;   // seconds, continuous controls
constexpr float kDuckSeconds = 0.005f; // each way, discrete controls
constexpr float kSnapDelta = 1.0e-4f;
} // namespace

LiteEngine::LiteEngine() = default;
LiteEngine::~LiteEngine() = default;

void LiteEngine::prepare (double newSampleRate, int)
{
    sampleRate = float (newSampleRate);
    if (wavetable == nullptr)
        wavetable = dsp::makeDefaultTable();

    for (size_t i = 0; i < voices.size(); ++i)
        voices[i].prepare (sampleRate, wavetable.get(), uint32_t (0x5eed + 977 * i));
    for (size_t i = 0; i < freeLfo.size(); ++i)
        freeLfo[i].seed (uint32_t (0xf00d + i));

    fx.prepare (newSampleRate);
    smoothRate = -1.0f / (kSmoothTau * sampleRate);
    smoothCoeff = 1.0f - std::exp (smoothRate * float (kControlBlock));
    duckStep = 1.0f / (kDuckSeconds * sampleRate);
    snapControls();
    prepared = true;
    reset();
}

void LiteEngine::reset()
{
    for (auto& v : voices)
        v.kill();
    numHeld = 0;
    sustainDown = false;
    blockValues.pitchBendSemis = 0.0f;
    lastNoteSemis = -1.0f;
    for (size_t i = 0; i < freeLfo.size(); ++i)
        freeLfo[i].restart (patch.lfo[i].phase);
    fx.reset();
    snapControls();
}

void LiteEngine::snapControls()
{
    controls = targets;
    patch.setAll (controls);
    duck = Duck::Idle;
    duckGain = 1.0f;
}

void LiteEngine::applyDiscrete()
{
    for (int i = 0; i < schema::kControlCount; ++i)
        if (schema::control (i).kind != schema::Kind::Continuous && controls[i] != targets[i])
        {
            controls[i] = targets[i];
            patch.set (i, controls);
        }
}

// One-pole toward the targets, once per control block; only moving controls cost a patch.set.
void LiteEngine::smoothControls (int numSamples)
{
    const float a = numSamples == kControlBlock ? smoothCoeff : 1.0f - std::exp (smoothRate * float (numSamples));
    for (int i = 0; i < schema::kControlCount; ++i)
    {
        if (schema::control (i).kind != schema::Kind::Continuous || controls[i] == targets[i])
            continue;
        const float d = targets[i] - controls[i];
        controls[i] = std::abs (d) < kSnapDelta ? targets[i] : controls[i] + d * a;
        patch.set (i, controls);
    }
}

void LiteEngine::setControl (int index, float value)
{
    if (index < 0 || index >= schema::kControlCount || ! std::isfinite (value))
        return;

    value = schema::sanitiseValue (index, value);
    if (juce::exactlyEqual (targets[index], value))
        return;

    targets[index] = value;
    if (numActiveVoices() == 0) // nothing to click or glide: apply at once
    {
        controls[index] = value;
        patch.set (index, controls);
    }
    else if (schema::control (index).kind != schema::Kind::Continuous && duck != Duck::Down)
    {
        duck = Duck::Down;
        ++ducksStarted;
        duckCountdown = std::max (1, int (std::ceil (duckGain / duckStep))); // samples until gain 0
    }
}

int LiteEngine::numActiveVoices() const noexcept
{
    int n = 0;
    for (const auto& v : voices)
        n += v.isActive() ? 1 : 0;
    return n;
}

float LiteEngine::glideSource() const noexcept { return patch.glideSeconds > 0.0f ? lastNoteSemis : -1.0f; }

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
                if ((pass == 1 || v.isReleasing())
                    && (chosen == nullptr || v.getStartOrder() < chosen->getStartOrder()))
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

    if (! sustainDown)
        releaseNote (note);
}

void LiteEngine::releaseNote (int note)
{
    // Poly releases every voice on the note. Mono/legato also do this for
    // voices other than voice 0, which can still be sounding from before a
    // switch out of poly.
    for (size_t i = patch.playMode == PlayMode::Poly ? 0 : 1; i < voices.size(); ++i)
    {
        auto& other = voices[i];
        if (other.isActive() && ! other.isReleasing() && other.getNote() == note)
            other.release (patch);
    }
    if (patch.playMode == PlayMode::Poly)
        return;

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

void LiteEngine::setSustain (bool down)
{
    if (down == sustainDown)
        return;
    sustainDown = down;
    if (down)
        return;

    // Pedal up: release every sounding note whose key is no longer held.
    for (auto& v : voices)
    {
        if (! v.isActive() || v.isReleasing())
            continue;
        const int note = v.getNote();
        if (std::find (held.begin(), held.begin() + numHeld, note) == held.begin() + numHeld)
            releaseNote (note);
    }
}

void LiteEngine::allNotesOff()
{
    numHeld = 0;
    sustainDown = false;
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
        smoothControls (n);
        for (size_t i = 0; i < freeLfo.size(); ++i)
            blockValues.freeLfo[i] = freeLfo[i].advance (patch.lfo[i], seconds);
        for (auto& v : voices)
            v.render (left + start, right + start, n, patch, blockValues);
    }
}

// FX and output gain (with the duck ramp) for samples [pos, pos + len).
float LiteEngine::finishSpan (float* left, float* right, int len)
{
    fx.process (left, right, len, patch.fx);

    float check = 0.0f;
    for (int i = 0; i < len; ++i)
    {
        if (duck == Duck::Down)
            duckGain = std::max (0.0f, duckGain - duckStep);
        else if (duck == Duck::Up)
        {
            duckGain = std::min (1.0f, duckGain + duckStep);
            if (duckGain == 1.0f)
                duck = Duck::Idle;
        }
        duckedSamples += duckGain < 1.0f ? 1 : 0;
        const float g = patch.masterGain * duckGain;
        left[i] = std::clamp (left[i] * g, -kOutputLimit, kOutputLimit);
        right[i] = std::clamp (right[i] * g, -kOutputLimit, kOutputLimit);
        check += left[i] + right[i];
    }
    return check;
}

void LiteEngine::process (juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi)
{
    buffer.clear();
    const int numSamples = buffer.getNumSamples();
    if (! prepared || buffer.getNumChannels() < 2)
        return; // a 0-sample block still runs the event loop: hosts flush MIDI that way

    auto* left = buffer.getWritePointer (0);
    auto* right = buffer.getWritePointer (1);

    // A duck in progress is finished span by span, so the discrete controls swap
    // on the exact sample where the gain reaches 0. Otherwise FX/gain run once.
    const bool ducking = duck == Duck::Down;
    float check = 0.0f;
    auto span = [&] (int pos, int len)
    {
        while (len > 0)
        {
            const int seg = duck == Duck::Down ? std::min (len, duckCountdown) : len;
            renderVoices (left + pos, right + pos, seg);
            if (ducking)
                check += finishSpan (left + pos, right + pos, seg);
            if (duck == Duck::Down && (duckCountdown -= seg) == 0)
            {
                applyDiscrete();
                duck = Duck::Up;
            }
            pos += seg;
            len -= seg;
        }
    };

    // Render between events so every event lands on its sample.
    int position = 0;
    for (const auto meta : midi)
    {
        const int at = std::clamp (meta.samplePosition, 0, numSamples);
        if (at > position)
        {
            span (position, at - position);
            position = at;
        }

        const auto msg = meta.getMessage();
        if (msg.isNoteOn())
            noteOn (msg.getNoteNumber());
        else if (msg.isNoteOff())
            noteOff (msg.getNoteNumber());
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
            allNotesOff();
        else if (msg.isPitchWheel())
            blockValues.pitchBendSemis = kPitchBendRange * float (msg.getPitchWheelValue() - 8192) / 8192.0f;
        else if (msg.isSustainPedalOn())
            setSustain (true);
        else if (msg.isSustainPedalOff())
            setSustain (false);
    }
    if (position < numSamples)
        span (position, numSamples - position);
    if (! ducking)
        check = finishSpan (left, right, numSamples);

    if (! dsp::isFiniteBits (check))
    {
        // Should be unreachable; recover instead of recirculating NaN in FX tails.
        buffer.clear();
        fx.reset();
        for (auto& v : voices)
            v.kill();
    }
}

} // namespace vae
