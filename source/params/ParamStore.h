#pragma once

#include "params/LinkRule.h"
#include "params/ParamSpec.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace vae::params
{

/** Dense index of a parameter in declaration order; use for hot paths instead of string ids. */
struct ParamIndex
{
    int value = -1;

    constexpr bool isValid() const noexcept { return value >= 0; }
    friend constexpr bool operator== (ParamIndex a, ParamIndex b) noexcept { return a.value == b.value; }
};

/** Whether a set() call notifies the host. */
enum class Notify
{
    host,           ///< Through the JUCE parameter's setValueNotifyingHost (default).
    listenersOnly   ///< Internal state + store listeners only; no host notification.
};

/**
    Single owner of a plug-in's parameter state. See docs/specs/parameter-manager.md.

    Must be a member of the AudioProcessor it is built for (it must not outlive the
    processor's parameters).
*/
class ParamStore
{
public:
    /** Store listener, called synchronously on the thread that set the value. */
    struct Listener
    {
        virtual ~Listener() = default;
        virtual void paramChanged (const juce::String& id, float raw) = 0;
    };

    ParamStore();
    ~ParamStore();

    ParamStore (const ParamStore&) = delete;
    ParamStore& operator= (const ParamStore&) = delete;

    /** Creates the parameters and the APVTS. Call exactly once, at the end of the processor's constructor. */
    void build (std::vector<ParamSpec> specs,
                juce::AudioProcessor& processor,
                const juce::Identifier& treeType,
                juce::UndoManager* undoManager = nullptr);

    /** True once build() has run. */
    bool isBuilt() const noexcept;

    /** Audio thread: delivers every change since the previous call (coalesced, arrival order)
        to the specs' onAudioChange callbacks. Lock-free and allocation-free. */
    void dispatch() noexcept;

    /** Id -> index lookup; invalid index for unknown ids (or before build). */
    ParamIndex indexOf (const juce::String& id) const noexcept;

    /** Raw value; 0 for unknown ids. Lock-free. */
    float get (const juce::String& id) const noexcept;
    float get (ParamIndex index) const noexcept;

    /** Normalised (0..1) value. */
    float getNormalised (const juce::String& id) const noexcept;
    float getNormalised (ParamIndex index) const noexcept;

    /** Snaps to the range, stores, marks dirty and notifies listeners (and the host if requested). */
    void set (const juce::String& id, float raw, Notify notify = Notify::host);
    void set (ParamIndex index, float raw, Notify notify = Notify::host);

    /** As set(), from a normalised value (clamped to 0..1). */
    void setNormalised (const juce::String& id, float normalised, Notify notify = Notify::host);
    void setNormalised (ParamIndex index, float normalised, Notify notify = Notify::host);

    /** Sets the parameter(s) to their default value, notifying the host. */
    void resetToDefault (const juce::String& id);
    void resetToDefault (ParamIndex index);
    void resetAllToDefaults();

    /** Forwards to the JUCE parameter's change gestures (no-op for hidden params). Must be balanced. */
    void beginGesture (const juce::String& id);
    void beginGesture (ParamIndex index);
    void endGesture (const juce::String& id);
    void endGesture (ParamIndex index);

    /** Descriptor (completed as by ParamSpec::completed()). Throws std::out_of_range for unknown ids. */
    const ParamSpec& spec (const juce::String& id) const;
    const ParamSpec& spec (ParamIndex index) const;

    /** The JUCE parameter, or nullptr for hidden/unknown ids. */
    juce::RangedAudioParameter* hostParam (const juce::String& id) const noexcept;
    juce::RangedAudioParameter* hostParam (ParamIndex index) const noexcept;

    /** The APVTS, or nullptr before build(). */
    juce::AudioProcessorValueTreeState* apvts() const noexcept;

    /** Number of parameters (visible and hidden). */
    size_t size() const noexcept;

    /** All ids in declaration order. */
    juce::StringArray ids() const;

    /** Per-id listener registration. Safe to call while callbacks run on another thread. */
    void addListener (const juce::String& id, Listener* listener);
    void removeListener (const juce::String& id, Listener* listener);

    /** Serialises the full state (APVTS state plus hidden values) as binary XML. */
    void save (juce::MemoryBlock& destination);

    /** Restores the full state. Returns false (and changes nothing) on malformed data or wrong tree type. */
    bool load (const void* data, int sizeInBytes);

    /** As load(), but parameters with restoreFromPreset == false keep their current value. */
    bool loadPreset (const void* data, int sizeInBytes);

    /** Adds a link between two parameters. Setup-time only (after build, before audio starts). */
    void addLink (const LinkRule& rule);

    /** Nearest value for id that satisfies every enabled Ordered rule it takes part in. */
    float constrain (const juce::String& id, float candidateRaw) const noexcept;

private:
    struct Entry;
    struct Link;
    class IndexQueue;

    Entry* entryAt (ParamIndex index) const noexcept;
    Entry* entryFor (const juce::String& id) const noexcept;

    void onHostValueChanged (Entry& entry, float normalised);
    void handleChange (Entry& entry, float raw);
    float storeChange (Entry& entry, float raw);
    void propagate (Entry& entry, float oldRaw, float newRaw);
    void markDirty (Entry& entry) noexcept;
    void deliver (std::uint32_t index) noexcept;
    void notifyListeners (Entry& entry, float raw);

    void applyLinks (Entry& entry, float oldRaw, float newRaw);
    void applyEnables (Entry& entry, float raw);
    void applyRule (Link& link, int movedIndex, float oldRaw, float newRaw);
    void onLinkEnabled (Link& link);
    void enforceOrdered (Link& link, bool movedIsFirst);
    void setFollower (int index, float raw);

    bool loadState (const void* data, int sizeInBytes, bool asPreset);

    std::vector<std::unique_ptr<Entry>> entries;
    std::unordered_map<juce::String, int> lookup;
    std::vector<std::unique_ptr<Link>> links;
    std::unique_ptr<IndexQueue> queue;
    std::vector<std::uint32_t> scratch;
    std::unique_ptr<juce::AudioProcessorValueTreeState> state;
    std::atomic<bool> built { false };
    std::atomic<bool> loading { false };
    std::atomic<bool> overflowed { false };
};

} // namespace vae::params
