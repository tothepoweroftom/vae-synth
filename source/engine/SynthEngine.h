/*
  The backend-neutral synth engine interface. A backend renders stereo audio
  from MIDI and interprets the schema control vector (source/schema) its own
  way. The plug-in owns one instance per compiled-in backend and switches
  between them; nothing outside source/engine knows which DSP is running.

  Threading: everything here is called on the audio thread, except
  construction/destruction and prepare(), which the plug-in calls from
  prepareToPlay.
*/

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include "schema/ControlSchema.h"

#include <memory>
#include <vector>

namespace vae
{

enum class EngineKind : int
{
    Lite = 0,   // permissive subset engine (always built)
    Surge = 1,  // Surge XT DSP (VAE_WITH_SURGE, GPL)
};

class SynthEngine
{
public:
    virtual ~SynthEngine() = default;

    virtual EngineKind kind() const noexcept = 0;
    virtual const char* displayName() const noexcept = 0;

    /** Allocate for the given rate / maximum block. Message or audio-idle thread. */
    virtual void prepare (double sampleRate, int maxBlockSize) = 0;

    /** Hard-cut all voices and tails. */
    virtual void reset() = 0;

    /** Set one schema control (normalised 0..1). Cheap; may be called per block. */
    virtual void setControl (int index, float normalised) = 0;

    /** Apply a whole vector (after a preset, model output or engine switch). */
    virtual void setControls (const schema::Snapshot& snapshot)
    {
        for (int i = 0; i < schema::kControlCount; ++i)
            setControl (i, snapshot[i]);
    }

    /** Render `buffer` (stereo, overwritten) from the block's MIDI events,
        which carry sample offsets within the block. */
    virtual void process (juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi) = 0;
};

/** Backends compiled into this build, in EngineKind order. */
std::vector<EngineKind> availableEngines();

/** Create a backend; nullptr when it is not compiled in. */
std::unique_ptr<SynthEngine> createEngine (EngineKind kind);

} // namespace vae
