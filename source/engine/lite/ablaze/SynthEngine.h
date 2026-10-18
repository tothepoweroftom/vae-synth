/*
  The engine behind the C ABI (and behind the JUCE plugin, which drives it
  exactly the way the web host does): owns the juce::Synthesiser, the event
  queue, the noteId->MIDI-note mapping, parameter smoothing and the FX chain.

  The event path — pushEvent -> MidiBuffer -> Synthesiser — is the timing
  contract with the host; keep it intact.
*/

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include <atomic>
#include <cstdint>
#include <memory>

#include "FxChain.h"
#include "Params.h"
#include "SynthVoice.h"
#include "Wavetables.h"

namespace ablaze
{

class SynthEngine
{
public:
    /** `wavetables` lets several engines share one bank (the plugin rebuilds
        its engine on sample-rate changes); null builds a private one. */
    SynthEngine (float sampleRate, int maxVoices,
                 std::shared_ptr<const WavetableBank> wavetables = nullptr);

    /** Queue an event for the NEXT render call, at the given frame offset. */
    void noteOn (int32_t noteId, float freqHz, float velocity, int32_t frameOffset);
    void noteOff (int32_t noteId, int32_t frameOffset);
    void allNotesOff (int32_t frameOffset);

    /** Render one block; outL/outR are caller-owned. Any numFrames works. */
    void render (float* outL, float* outR, int numFrames);

    /** Hard-cut everything and clear queued events, FX tails and smoothing. */
    void reset();

    /** Set a parameter (plain value, clamped to its range). Smoothed params
        glide over ~20 ms; choice/integer params switch at the next block. */
    void setParam (int index, float value);
    float getParam (int index) const { return params[index]; }

    /** Jump all smoothers to their targets (after loading a preset at init). */
    void snapParams();

    const WavetableBank& getWavetables() const noexcept { return *bank; }

    /** The table Osc A plays when osca_table is "User". Not owned: the
        caller keeps it alive while set, and for ~10 ms after replacing it
        (voices crossfade out of the old one). Lock-free; callable from any
        thread. Null = none loaded, "User" falls back to the Basic table. */
    void setUserWavetable (const Wavetable* table) noexcept { userTable.store (table, std::memory_order_release); }
    int getNumActiveVoices() const;

    /** Control rate: smoothed params are re-read every this many samples. */
    static constexpr int kControlBlock = 16;
    static constexpr int kMaxEvents = 256;

private:
    enum class EventType : int32_t { NoteOn, NoteOff, AllNotesOff };

    struct Event
    {
        EventType type;
        int32_t noteId;
        float freqHz;
        float velocity;
        int32_t frameOffset;
    };

    // juce::Synthesiser keys notes by MIDI note number, so we map the ABI's
    // opaque noteId to the nearest equal-tempered note. The exact Hz the host
    // sent is kept per note (noteHz) and used by the voice for its pitch.
    struct NoteMapping { int32_t noteId; int midiNote; bool used; };

    static constexpr int kMidiChannel = 1;

    int hzToMidi (float freqHz) const;
    void pushEvent (const Event& e);
    void applyEventToMidiBuffer (const Event& e, int numFrames);
    void updateControlBlock (int numSamples);

    float sampleRate;
    std::shared_ptr<const WavetableBank> bank;
    std::atomic<const Wavetable*> userTable { nullptr };

    juce::Synthesiser synth;
    juce::MidiBuffer midi;        // this render call's events
    juce::MidiBuffer chunkMidi;   // the slice for the current control block
    VoiceParams voiceParams;
    FxParams fxParams;
    FxChain fx;

    float params[kNumParams] = {};
    juce::SmoothedValue<float> smoothed[kNumParams];
    float lastGain = 0.8f;

    float noteHz[128] = {};
    NoteMapping mappings[128] = {};
    Event events[kMaxEvents] = {};
    int numEvents = 0;
};

} // namespace ablaze
