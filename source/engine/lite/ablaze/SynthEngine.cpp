#include "SynthEngine.h"

#include <cmath>

namespace ablaze
{

SynthEngine::SynthEngine (float sr, int maxVoices, std::shared_ptr<const WavetableBank> wavetables)
    : sampleRate (sr),
      bank (wavetables != nullptr ? std::move (wavetables) : std::make_shared<const WavetableBank>())
{
    for (int n = 0; n < 128; ++n)
        noteHz[n] = static_cast<float> (juce::MidiMessage::getMidiNoteInHertz (n));

    for (int i = 0; i < kNumParams; ++i)
    {
        params[i] = kParams[i].def;
        smoothed[i].reset (static_cast<double> (sr), 0.020);   // ~20 ms
        smoothed[i].setCurrentAndTargetValue (params[i]);
    }

    synth.setCurrentPlaybackSampleRate (sr);
    synth.setNoteStealingEnabled (true);
    // Default is 32: events closer than that get applied early, which breaks
    // sample-accurate timing. 1 + strict = every event lands on its sample.
    synth.setMinimumRenderingSubdivisionSize (1, true);

    synth.addSound (new WavetableSound());
    for (int i = 0; i < juce::jmax (1, maxVoices); ++i)
        synth.addVoice (new WavetableVoice (voiceParams, noteHz, i));

    // Preallocate MidiBuffer storage so per-block addEvent never grows it.
    midi.ensureSize (static_cast<size_t> (kMaxEvents) * 8);
    chunkMidi.ensureSize (static_cast<size_t> (kMaxEvents) * 8);

    fx.prepare (static_cast<double> (sr));
    lastGain = params[kParamGain];
    updateControlBlock (0);
}

void SynthEngine::noteOn (int32_t noteId, float freqHz, float velocity, int32_t frameOffset)
{
    pushEvent ({ EventType::NoteOn, noteId, freqHz, velocity, frameOffset });
}

void SynthEngine::noteOff (int32_t noteId, int32_t frameOffset)
{
    pushEvent ({ EventType::NoteOff, noteId, 0.0f, 0.0f, frameOffset });
}

void SynthEngine::allNotesOff (int32_t frameOffset)
{
    pushEvent ({ EventType::AllNotesOff, 0, 0.0f, 0.0f, frameOffset });
}

void SynthEngine::render (float* outL, float* outR, int numFrames)
{
    if (numFrames <= 0)
        return;

    midi.clear();
    for (int i = 0; i < numEvents; ++i)
        applyEventToMidiBuffer (events[i], numFrames);
    numEvents = 0;

    // Wrap the caller's pointers without copying.
    float* channels[] = { outL, outR };
    juce::AudioBuffer<float> buffer (channels, 2, numFrames);
    buffer.clear();

    auto nextEvent = midi.cbegin();

    for (int start = 0; start < numFrames; start += kControlBlock)
    {
        const int n = juce::jmin (kControlBlock, numFrames - start);

        updateControlBlock (n);

        // Synthesiser::renderNextBlock fires every event left in the buffer
        // at the end of the range it renders, so it must only see this
        // control block's events. Raw bytes: no MidiMessage construction.
        chunkMidi.clear();
        for (; nextEvent != midi.cend() && (*nextEvent).samplePosition < start + n; ++nextEvent)
        {
            const auto meta = *nextEvent;
            chunkMidi.addEvent (meta.data, meta.numBytes, meta.samplePosition);
        }

        // Synthesiser splits this range at each MidiBuffer event position:
        // this is what makes the events sample-accurate.
        synth.renderNextBlock (buffer, chunkMidi, start, n);

        fx.process (outL + start, outR + start, n, fxParams);

        // Master gain, ramped linearly across the control block.
        const float target = smoothed[kParamGain].getCurrentValue();
        const float step = (target - lastGain) / static_cast<float> (n);
        float g = lastGain;
        for (int i = start; i < start + n; ++i)
        {
            g += step;
            outL[i] *= g;
            outR[i] *= g;
        }
        lastGain = target;
    }

    // Never emit NaN/Inf. Should be unreachable; if it ever happens, recover
    // (clear the FX state that would otherwise keep recirculating it).
    // NaN/Inf propagate through the sum, so one bit test covers the block.
    float check = 0.0f;
    for (int i = 0; i < numFrames; ++i)
        check += outL[i] + outR[i];
    if (! isFiniteBits (check))
    {
        buffer.clear();
        fx.reset();
    }
}

void SynthEngine::reset()
{
    // Reset is the one place a hard cut is allowed: clear the Synthesiser's
    // note bookkeeping, then kill every voice outright (allNotesOff alone
    // would go through the voices' anti-click fade).
    synth.allNotesOff (0, false);
    for (int i = 0; i < synth.getNumVoices(); ++i)
        if (auto* voice = dynamic_cast<WavetableVoice*> (synth.getVoice (i)))
            voice->kill();

    for (auto& m : mappings)
        m.used = false;
    numEvents = 0;
    midi.clear();

    snapParams();
    fx.reset();
}

void SynthEngine::setParam (int index, float value)
{
    if (index < 0 || index >= kNumParams || ! isFiniteBits (value))
        return;

    const auto& info = kParams[index];
    value = juce::jlimit (info.min, info.max, value);

    if (info.kind != ParamKind::Continuous)
        value = std::round (value);   // toggles, choices and integers are whole numbers

    params[index] = value;

    if (isSmoothed (info.kind))
        smoothed[index].setTargetValue (value);
    else
        smoothed[index].setCurrentAndTargetValue (value);
}

void SynthEngine::snapParams()
{
    for (int i = 0; i < kNumParams; ++i)
        smoothed[i].setCurrentAndTargetValue (params[i]);
    lastGain = params[kParamGain];
    updateControlBlock (0);
}

int SynthEngine::getNumActiveVoices() const
{
    int n = 0;
    for (int i = 0; i < synth.getNumVoices(); ++i)
        if (synth.getVoice (i)->isVoiceActive())
            ++n;
    return n;
}

void SynthEngine::updateControlBlock (int numSamples)
{
    float v[kNumParams];
    for (int i = 0; i < kNumParams; ++i)
        v[i] = numSamples > 0 ? smoothed[i].skip (numSamples) : smoothed[i].getCurrentValue();

    auto asInt = [&] (int idx) { return static_cast<int> (std::lround (v[idx])); };

    auto& p = voiceParams;
    p.oscAGain = v[kParamOscAOn] * v[kParamOscALevel];
    const int tableIndex = asInt (kParamOscATable);
    const Wavetable* user = userTable.load (std::memory_order_acquire);
    p.oscATable = tableIndex == kUserWavetableIndex
                      ? (user != nullptr ? user : &bank->getTable (0))
                      : &bank->getTable (juce::jlimit (0, WavetableBank::kNumTables - 1, tableIndex));
    p.oscAPos = v[kParamOscAPos];
    p.oscASemis = 12.0f * v[kParamOscAOctave] + v[kParamOscASemi] + 0.01f * v[kParamOscAFine];
    p.unison = asInt (kParamOscAUnison);
    p.detuneCents = v[kParamOscADetune];
    p.spread = v[kParamOscASpread];

    p.oscBGain = v[kParamOscBOn] * v[kParamOscBLevel];
    p.oscBWave = static_cast<AnalogWave> (asInt (kParamOscBWave));
    p.oscBPw = v[kParamOscBPw];
    p.oscBSemis = 12.0f * v[kParamOscBOctave] + v[kParamOscBSemi] + 0.01f * v[kParamOscBFine];

    p.subGain = v[kParamSubOn] * v[kParamSubLevel];
    p.subWave = static_cast<SubWave> (asInt (kParamSubWave));
    static constexpr float kSubRatios[] = { 0.5f, 0.25f, 0.125f, 1.0f, 2.0f, 4.0f, 8.0f };
    p.subRatio = kSubRatios[juce::jlimit (0, 6, asInt (kParamSubOctave))];

    p.noiseGain = v[kParamNoiseLevel] * 0.5f;

    p.filterMix = v[kParamFilterOn];
    p.filterType = static_cast<FilterType> (asInt (kParamFilterType));
    p.cutoffHz = v[kParamFilterCutoff];
    p.filterK = 2.0f * (1.0f - 0.985f * v[kParamFilterReso]);
    const float drive = v[kParamFilterDrive];
    p.driveGain = 1.0f + 9.0f * drive * drive;
    p.driveComp = 1.0f / std::sqrt (p.driveGain);
    p.keytrack = v[kParamFilterKeytrack];
    p.envOctaves = 7.0f * v[kParamFilterEnvAmt];
    p.velOctaves = 3.0f * v[kParamFilterVel];

    p.filterEnv = EnvCoefs::make (v[kParamFenvAttack], v[kParamFenvDecay],
                                  v[kParamFenvSustain], v[kParamFenvRelease], sampleRate);
    p.ampEnv = EnvCoefs::make (v[kParamAmpAttack], v[kParamAmpDecay],
                               v[kParamAmpSustain], v[kParamAmpRelease], sampleRate);
    p.ampVel = v[kParamAmpVel];

    auto& f = fxParams;
    f.shaperOn = v[kParamShaperOn];
    f.shaperType = static_cast<ShaperType> (asInt (kParamShaperType));
    f.shaperDrive = v[kParamShaperDrive];
    f.shaperMix = v[kParamShaperMix];

    f.eqOn = v[kParamEqOn];
    f.eqLowFreq = v[kParamEqLowFreq];
    f.eqLowGain = v[kParamEqLowGain];
    f.eqMidFreq = v[kParamEqMidFreq];
    f.eqMidGain = v[kParamEqMidGain];
    f.eqMidQ = v[kParamEqMidQ];
    f.eqHighFreq = v[kParamEqHighFreq];
    f.eqHighGain = v[kParamEqHighGain];

    f.chorusOn = v[kParamChorusOn];
    f.chorusRate = v[kParamChorusRate];
    f.chorusDepth = v[kParamChorusDepth];
    f.chorusMix = v[kParamChorusMix];

    f.delayOn = v[kParamDelayOn];
    f.delayTimeMs = v[kParamDelayTime];
    f.delayFeedback = v[kParamDelayFeedback];
    f.delayDamp = v[kParamDelayDamp];
    f.delayPingPong = v[kParamDelayPingPong] >= 0.5f;
    f.delayMix = v[kParamDelayMix];

    f.reverbOn = v[kParamReverbOn];
    f.reverbSize = v[kParamReverbSize];
    f.reverbDamp = v[kParamReverbDamp];
    f.reverbWidth = v[kParamReverbWidth];
    f.reverbMix = v[kParamReverbMix];
}

int SynthEngine::hzToMidi (float freqHz) const
{
    if (! (freqHz > 0.0f) || ! isFiniteBits (freqHz))
        return 69;
    const auto m = juce::roundToInt (69.0 + 12.0 * std::log2 (freqHz / 440.0));
    return juce::jlimit (0, 127, m);
}

void SynthEngine::pushEvent (const Event& e)
{
    if (numEvents < kMaxEvents)
        events[numEvents++] = e;
    // Ring full: drop. Preferable to blocking or allocating.
}

void SynthEngine::applyEventToMidiBuffer (const Event& e, int numFrames)
{
    const int offset = juce::jlimit (0, numFrames - 1, static_cast<int> (e.frameOffset));

    switch (e.type)
    {
        case EventType::NoteOn:
        {
            const int midiNote = hzToMidi (e.freqHz);
            // Honour the exact Hz the host sent (microtuning, detuned hosts).
            if (e.freqHz > 0.0f && isFiniteBits (e.freqHz))
                noteHz[midiNote] = juce::jlimit (1.0f, 0.45f * sampleRate, e.freqHz);

            // A repeated note-on for a still-held id reuses its slot (a leaked
            // slot would eventually leave new notes without a note-off).
            NoteMapping* slot = nullptr;
            for (auto& m : mappings)
                if (m.used && m.noteId == e.noteId)
                    slot = &m;
            for (auto& m : mappings)
                if (slot == nullptr && ! m.used)
                    slot = &m;
            if (slot != nullptr)
            {
                if (slot->used && slot->midiNote != midiNote)
                    midi.addEvent (juce::MidiMessage::noteOff (kMidiChannel, slot->midiNote), offset);
                *slot = { e.noteId, midiNote, true };
            }
            const float velocity = juce::jlimit (0.0f, 1.0f, e.velocity);
            // Velocity 0 would be read as note-off by MidiMessage.
            midi.addEvent (juce::MidiMessage::noteOn (kMidiChannel, midiNote, juce::jmax (1.0f / 127.0f, velocity)),
                           offset);
            break;
        }

        case EventType::NoteOff:
            // Unknown / already-freed noteId is a documented no-op.
            for (auto& m : mappings)
            {
                if (m.used && m.noteId == e.noteId)
                {
                    m.used = false;
                    midi.addEvent (juce::MidiMessage::noteOff (kMidiChannel, m.midiNote), offset);
                    break;
                }
            }
            break;

        case EventType::AllNotesOff:
            for (auto& m : mappings)
                m.used = false;
            // Synthesiser::handleMidiEvent turns this into allNotesOff with
            // tail-off, at the right sample position.
            midi.addEvent (juce::MidiMessage::allNotesOff (kMidiChannel), offset);
            break;
    }
}

} // namespace ablaze
