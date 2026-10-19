#include "params/ParamStore.h"

#include <stdexcept>
#include <thread>

namespace vae::params
{

namespace
{
    const juce::Identifier paramTag { "PARAM" };    // matches the APVTS child type
    const juce::Identifier idProp { "id" };
    const juce::Identifier valueProp { "value" };
    const juce::Identifier hiddenTag { "VAE_HIDDEN_PARAMS" };

    static_assert (std::atomic<float>::is_always_lock_free);
    static_assert (std::atomic<bool>::is_always_lock_free);
    static_assert (std::atomic<std::size_t>::is_always_lock_free);

    /** Adds a meta flag that can be forced after construction (parameters that take part in a link). */
    template <typename Base>
    class StoreParameter : public Base
    {
    public:
        using Base::Base;

        bool isMetaParameter() const override   { return forcedMeta.load() || Base::isMetaParameter(); }

        std::atomic<bool> forcedMeta { false };
    };

    /** Float parameter that also passes the spec's discrete hint through. */
    class StoreFloatParameter final : public StoreParameter<juce::AudioParameterFloat>
    {
    public:
        template <typename... Args>
        explicit StoreFloatParameter (bool discreteHint, Args&&... args)
            : StoreParameter<juce::AudioParameterFloat> (std::forward<Args> (args)...), discrete (discreteHint)
        {
        }

        bool isDiscrete() const override        { return discrete; }

    private:
        const bool discrete;
    };

    float nearlyEqualTolerance (float a, float b) noexcept
    {
        return 1.0e-6f * (1.0f + juce::jmax (std::abs (a), std::abs (b)));
    }

    bool nearlyEqual (float a, float b) noexcept
    {
        return std::abs (a - b) <= nearlyEqualTolerance (a, b);
    }
}

//==============================================================================
/** Bounded lock-free MPMC queue of indices (used as MPSC), sequence-numbered cells. */
class ParamStore::IndexQueue
{
public:
    explicit IndexQueue (size_t minCapacity)
    {
        size_t capacity = 1;

        while (capacity < juce::jmax<size_t> (minCapacity, 2))
            capacity <<= 1;

        mask = capacity - 1;
        cells = std::make_unique<Cell[]> (capacity);

        for (size_t i = 0; i < capacity; ++i)
            cells[i].sequence.store (i, std::memory_order_relaxed);
    }

    bool push (std::uint32_t value) noexcept
    {
        auto pos = tail.load (std::memory_order_relaxed);

        for (;;)
        {
            auto& cell = cells[pos & mask];
            const auto seq = cell.sequence.load (std::memory_order_acquire);
            const auto diff = (std::intptr_t) seq - (std::intptr_t) pos;

            if (diff == 0)
            {
                if (tail.compare_exchange_weak (pos, pos + 1, std::memory_order_relaxed))
                {
                    cell.value = value;
                    cell.sequence.store (pos + 1, std::memory_order_release);
                    return true;
                }
            }
            else if (diff < 0)
            {
                return false; // full
            }
            else
            {
                pos = tail.load (std::memory_order_relaxed);
            }
        }
    }

    bool pop (std::uint32_t& value) noexcept
    {
        auto pos = head.load (std::memory_order_relaxed);

        for (;;)
        {
            auto& cell = cells[pos & mask];
            const auto seq = cell.sequence.load (std::memory_order_acquire);
            const auto diff = (std::intptr_t) seq - (std::intptr_t) (pos + 1);

            if (diff == 0)
            {
                if (head.compare_exchange_weak (pos, pos + 1, std::memory_order_relaxed))
                {
                    value = cell.value;
                    cell.sequence.store (pos + mask + 1, std::memory_order_release);
                    return true;
                }
            }
            else if (diff < 0)
            {
                return false; // empty (or the next push is not yet published)
            }
            else
            {
                pos = head.load (std::memory_order_relaxed);
            }
        }
    }

private:
    struct Cell
    {
        std::atomic<size_t> sequence { 0 };
        std::uint32_t value = 0;
    };

    std::unique_ptr<Cell[]> cells;
    size_t mask = 0;
    alignas (64) std::atomic<size_t> head { 0 };
    alignas (64) std::atomic<size_t> tail { 0 };
};

//==============================================================================
struct ParamStore::Entry final : public juce::AudioProcessorParameter::Listener
{
    Entry (ParamStore& o, int i, ParamSpec s)
        : owner (o), index (i), spec (std::move (s)), value (spec.defaultValue), requestedRaw (spec.defaultValue)
    {
    }

    void parameterValueChanged (int, float normalised) override   { owner.onHostValueChanged (*this, normalised); }
    void parameterGestureChanged (int, bool) override              {}

    ParamStore& owner;
    const int index;
    ParamSpec spec;
    juce::RangedAudioParameter* host = nullptr;
    std::atomic<bool>* forcedMeta = nullptr;

    std::atomic<float> value;
    std::atomic<float> requestedRaw;   // exact raw value behind the last host-bound set (precision hint)
    std::atomic<bool> dirty { false };
    std::atomic<int> gestureDepth { 0 };

    // Link-propagation deferral for set() calls (see ParamStore::set). The plain fields are
    // only touched by the thread that currently owns deferThread.
    std::atomic<std::thread::id> deferThread {};
    bool deferCaptured = false;
    float deferOld = 0.0f, deferNew = 0.0f;

    juce::CriticalSection listenerLock;
    juce::Array<ParamStore::Listener*> listeners;

    std::vector<int> linkIndices;      // rules this parameter is a member of
    std::vector<int> enablerIndices;   // rules this parameter enables
};

//==============================================================================
struct ParamStore::Link
{
    explicit Link (const LinkRule& r) : rule (r) {}

    float toSpace (const Entry& e, float raw) const noexcept
    {
        return rule.space == LinkRule::Space::Raw ? raw : e.spec.range.convertTo0to1 (raw);
    }

    float fromSpace (const Entry& e, float v) const noexcept
    {
        return rule.space == LinkRule::Space::Raw ? v : e.spec.range.convertFrom0to1 (juce::jlimit (0.0f, 1.0f, v));
    }

    float spaceMin (const Entry& e) const noexcept   { return rule.space == LinkRule::Space::Raw ? e.spec.range.start : 0.0f; }
    float spaceMax (const Entry& e) const noexcept   { return rule.space == LinkRule::Space::Raw ? e.spec.range.end : 1.0f; }

    const LinkRule rule;
    int first = -1, second = -1, enabler = -1;
    std::atomic<bool> enabled { false };
    std::atomic<bool> busy { false };
    std::atomic<float> offset { 0.0f };
};

//==============================================================================
ParamStore::ParamStore() = default;

ParamStore::~ParamStore()
{
    for (auto& e : entries)
        if (e->host != nullptr)
            e->host->removeListener (e.get());

    state.reset();
}

bool ParamStore::isBuilt() const noexcept   { return built.load(); }

void ParamStore::build (std::vector<ParamSpec> specs,
                        juce::AudioProcessor& processor,
                        const juce::Identifier& treeType,
                        juce::UndoManager* undoManager)
{
    if (built.load())
    {
        jassertfalse; // build() must be called exactly once
        return;
    }

    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    for (auto& source : specs)
    {
        if (source.id.isEmpty() || lookup.find (source.id) != lookup.end())
        {
            jassertfalse; // empty or duplicate id: ignored
            continue;
        }

        const auto index = (int) entries.size();
        auto entry = std::make_unique<Entry> (*this, index, source.completed());
        const auto& s = entry->spec;

        if (s.hostVisible)
        {
            const juce::ParameterID pid { s.id, s.versionHint };
            auto toText = [s] (auto v, int maxLen)
            {
                auto t = s.valueToText ((float) v);
                return maxLen > 0 ? t.substring (0, maxLen) : t;
            };

            auto configure = [&s] (auto attributes)
            {
                return attributes.withCategory (s.category)
                                 .withMeta (s.meta)
                                 .withAutomatable (s.automatable);
            };

            std::unique_ptr<juce::RangedAudioParameter> param;

            switch (s.kind)
            {
                case ParamKind::Float:
                {
                    auto attr = configure (juce::AudioParameterFloatAttributes())
                                    .withStringFromValueFunction (toText)
                                    .withValueFromStringFunction ([s] (const juce::String& t) { return s.textToValue (t); });
                    auto p = std::make_unique<StoreFloatParameter> (s.discrete, pid, s.name, s.range, s.defaultValue, attr);
                    entry->forcedMeta = &p->forcedMeta;
                    param = std::move (p);
                    break;
                }

                case ParamKind::Int:
                {
                    auto attr = configure (juce::AudioParameterIntAttributes())
                                    .withStringFromValueFunction (toText)
                                    .withValueFromStringFunction ([s] (const juce::String& t) { return juce::roundToInt (s.textToValue (t)); });
                    auto p = std::make_unique<StoreParameter<juce::AudioParameterInt>> (pid, s.name,
                                                                                          juce::roundToInt (s.range.start),
                                                                                          juce::roundToInt (s.range.end),
                                                                                          juce::roundToInt (s.defaultValue), attr);
                    entry->forcedMeta = &p->forcedMeta;
                    param = std::move (p);
                    break;
                }

                case ParamKind::Bool:
                {
                    auto attr = configure (juce::AudioParameterBoolAttributes())
                                    .withStringFromValueFunction (toText)
                                    .withValueFromStringFunction ([s] (const juce::String& t) { return s.textToValue (t) >= 0.5f; });
                    auto p = std::make_unique<StoreParameter<juce::AudioParameterBool>> (pid, s.name, s.defaultValue >= 0.5f, attr);
                    entry->forcedMeta = &p->forcedMeta;
                    param = std::move (p);
                    break;
                }

                case ParamKind::Choice:
                {
                    auto attr = configure (juce::AudioParameterChoiceAttributes())
                                    .withStringFromValueFunction (toText)
                                    .withValueFromStringFunction ([s] (const juce::String& t) { return juce::roundToInt (s.textToValue (t)); });
                    auto p = std::make_unique<StoreParameter<juce::AudioParameterChoice>> (pid, s.name, s.choices,
                                                                                             juce::roundToInt (s.defaultValue), attr);
                    entry->forcedMeta = &p->forcedMeta;
                    param = std::move (p);
                    break;
                }
            }

            entry->host = param.get();

            // The JUCE parameter's range is authoritative for discrete kinds.
            if (s.kind != ParamKind::Float)
                entry->spec.range = entry->host->getNormalisableRange();

            layout.add (std::move (param));
        }

        lookup.emplace (entry->spec.id, index);
        entries.push_back (std::move (entry));
    }

    state = std::make_unique<juce::AudioProcessorValueTreeState> (processor, undoManager, treeType, std::move (layout));

    for (auto& e : entries)
        if (e->host != nullptr)
            e->host->addListener (e.get());

    jassert (entries.empty() || entries.front()->deferThread.is_lock_free());

    queue = std::make_unique<IndexQueue> (entries.size());
    scratch.assign (entries.size(), 0);
    built.store (true);

    for (auto& e : entries)
        markDirty (*e);
}

//==============================================================================
ParamStore::Entry* ParamStore::entryAt (ParamIndex index) const noexcept
{
    if (! built.load() || ! juce::isPositiveAndBelow (index.value, (int) entries.size()))
        return nullptr;

    return entries[(size_t) index.value].get();
}

ParamStore::Entry* ParamStore::entryFor (const juce::String& id) const noexcept
{
    if (! built.load())
        return nullptr; // calls before build() are silent no-ops

    if (auto* e = entryAt (indexOf (id)))
        return e;

    jassertfalse; // unknown parameter id
    return nullptr;
}

ParamIndex ParamStore::indexOf (const juce::String& id) const noexcept
{
    if (! built.load())
        return {};

    const auto it = lookup.find (id);
    return it != lookup.end() ? ParamIndex { it->second } : ParamIndex {};
}

//==============================================================================
void ParamStore::markDirty (Entry& e) noexcept
{
    // Only the clean -> dirty transition enqueues, so each index is queued at most once and
    // a queue of capacity >= size() cannot overflow. The fallback scan covers it anyway.
    if (! e.dirty.exchange (true))
        if (! queue->push ((std::uint32_t) e.index))
            overflowed.store (true);
}

void ParamStore::deliver (std::uint32_t index) noexcept
{
    auto& e = *entries[index];

    if (! e.dirty.exchange (false))
        return;

    const auto v = e.value.load();

    if (e.spec.onAudioChange)
        e.spec.onAudioChange (v);
}

void ParamStore::dispatch() noexcept
{
    if (! built.load())
        return;

    // Snapshot what is queued now; anything set during delivery waits for the next call.
    size_t count = 0;
    std::uint32_t index = 0;

    while (count < scratch.size() && queue->pop (index))
        scratch[count++] = index;

    for (size_t i = 0; i < count; ++i)
        deliver (scratch[i]);

    if (overflowed.exchange (false))
        for (size_t i = 0; i < entries.size(); ++i)
            deliver ((std::uint32_t) i);
}

//==============================================================================
float ParamStore::get (const juce::String& id) const noexcept
{
    auto* e = entryFor (id);
    return e != nullptr ? e->value.load() : 0.0f;
}

float ParamStore::get (ParamIndex index) const noexcept
{
    auto* e = entryAt (index);
    return e != nullptr ? e->value.load() : 0.0f;
}

float ParamStore::getNormalised (const juce::String& id) const noexcept
{
    auto* e = entryFor (id);
    return e != nullptr ? e->spec.range.convertTo0to1 (e->value.load()) : 0.0f;
}

float ParamStore::getNormalised (ParamIndex index) const noexcept
{
    auto* e = entryAt (index);
    return e != nullptr ? e->spec.range.convertTo0to1 (e->value.load()) : 0.0f;
}

void ParamStore::set (const juce::String& id, float raw, Notify notify)
{
    if (auto* e = entryFor (id))
        set (ParamIndex { e->index }, raw, notify);
}

void ParamStore::set (ParamIndex index, float raw, Notify notify)
{
    auto* e = entryAt (index);

    if (e == nullptr)
        return;

    const auto& range = e->spec.range;
    raw = range.snapToLegalValue (juce::jlimit (range.start, range.end, raw));

    if (e->host != nullptr)
    {
        const auto normalised = e->host->convertTo0to1 (raw);

        if (notify == Notify::host)
        {
            // Our parameter listener picks the change up synchronously (onHostValueChanged).
            // Link propagation is deferred until the JUCE parameter has released its
            // listener lock, so linked parameters never take each other's locks in
            // opposite orders.
            const auto me = std::this_thread::get_id();
            auto noOwner = std::thread::id {};
            const auto deferring = e->deferThread.compare_exchange_strong (noOwner, me);

            if (deferring)
                e->deferCaptured = false;

            e->requestedRaw.store (raw);
            e->host->setValueNotifyingHost (normalised);

            if (deferring)
            {
                const auto captured = e->deferCaptured;
                const auto oldRaw = e->deferOld;
                const auto newRaw = e->deferNew;
                e->deferThread.store (std::thread::id {});

                if (captured)
                    propagate (*e, oldRaw, newRaw);
            }

            return;
        }

        e->host->setValue (normalised);
    }

    handleChange (*e, raw);
}

void ParamStore::setNormalised (const juce::String& id, float normalised, Notify notify)
{
    if (auto* e = entryFor (id))
        setNormalised (ParamIndex { e->index }, normalised, notify);
}

void ParamStore::setNormalised (ParamIndex index, float normalised, Notify notify)
{
    if (auto* e = entryAt (index))
        set (index, e->spec.range.convertFrom0to1 (juce::jlimit (0.0f, 1.0f, normalised)), notify);
}

void ParamStore::resetToDefault (const juce::String& id)
{
    if (auto* e = entryFor (id))
        resetToDefault (ParamIndex { e->index });
}

void ParamStore::resetToDefault (ParamIndex index)
{
    if (auto* e = entryAt (index))
        set (index, e->spec.defaultValue, Notify::host);
}

void ParamStore::resetAllToDefaults()
{
    if (! built.load())
        return;

    for (size_t i = 0; i < entries.size(); ++i)
        resetToDefault (ParamIndex { (int) i });
}

void ParamStore::beginGesture (const juce::String& id)
{
    if (auto* e = entryFor (id))
        beginGesture (ParamIndex { e->index });
}

void ParamStore::beginGesture (ParamIndex index)
{
    if (auto* e = entryAt (index))
        if (e->gestureDepth.fetch_add (1) == 0 && e->host != nullptr)
            e->host->beginChangeGesture();
}

void ParamStore::endGesture (const juce::String& id)
{
    if (auto* e = entryFor (id))
        endGesture (ParamIndex { e->index });
}

void ParamStore::endGesture (ParamIndex index)
{
    auto* e = entryAt (index);

    if (e == nullptr)
        return;

    auto depth = e->gestureDepth.load();

    do
    {
        if (depth <= 0)
        {
            jassertfalse; // endGesture() without a matching beginGesture()
            return;
        }
    }
    while (! e->gestureDepth.compare_exchange_weak (depth, depth - 1));

    if (depth == 1 && e->host != nullptr)
        e->host->endChangeGesture();
}

const ParamSpec& ParamStore::spec (const juce::String& id) const
{
    return spec (indexOf (id));
}

const ParamSpec& ParamStore::spec (ParamIndex index) const
{
    if (auto* e = entryAt (index))
        return e->spec;

    throw std::out_of_range ("vae::params::ParamStore::spec: unknown parameter");
}

juce::RangedAudioParameter* ParamStore::hostParam (const juce::String& id) const noexcept
{
    return hostParam (indexOf (id));
}

juce::RangedAudioParameter* ParamStore::hostParam (ParamIndex index) const noexcept
{
    auto* e = entryAt (index);
    return e != nullptr ? e->host : nullptr;
}

juce::AudioProcessorValueTreeState* ParamStore::apvts() const noexcept
{
    return built.load() ? state.get() : nullptr;
}

size_t ParamStore::size() const noexcept
{
    return built.load() ? entries.size() : 0;
}

juce::StringArray ParamStore::ids() const
{
    juce::StringArray result;

    if (built.load())
        for (auto& e : entries)
            result.add (e->spec.id);

    return result;
}

//==============================================================================
void ParamStore::addListener (const juce::String& id, Listener* listener)
{
    jassert (listener != nullptr);

    if (auto* e = entryFor (id); e != nullptr && listener != nullptr)
    {
        const juce::ScopedLock sl (e->listenerLock);
        e->listeners.addIfNotAlreadyThere (listener);
    }
}

void ParamStore::removeListener (const juce::String& id, Listener* listener)
{
    if (! built.load())
        return;

    if (auto* e = entryAt (indexOf (id)))
    {
        // Blocks while a callback is running on another thread, so after this returns the
        // listener is never called again.
        const juce::ScopedLock sl (e->listenerLock);
        e->listeners.removeFirstMatchingValue (listener);
    }
}

void ParamStore::notifyListeners (Entry& e, float raw)
{
    const juce::ScopedLock sl (e.listenerLock);

    for (int i = e.listeners.size(); --i >= 0;)
        if (i < e.listeners.size())
            e.listeners.getUnchecked (i)->paramChanged (e.spec.id, raw);
}

//==============================================================================
void ParamStore::onHostValueChanged (Entry& e, float normalised)
{
    auto raw = e.host->convertFrom0to1 (normalised);
    const auto requested = e.requestedRaw.load();

    // Prefer the exact value that was asked for when it maps to the same host value.
    if (juce::exactlyEqual (e.host->convertTo0to1 (requested), normalised))
        raw = requested;

    if (e.deferThread.load() == std::this_thread::get_id() && ! e.deferCaptured)
    {
        // Inside our own set() on this thread: store now, propagate links after the
        // JUCE parameter's notification has finished.
        e.deferOld = storeChange (e, raw);
        e.deferNew = raw;
        e.deferCaptured = true;
        return;
    }

    handleChange (e, raw);
}

float ParamStore::storeChange (Entry& e, float raw)
{
    const auto old = e.value.exchange (raw);

    // Two threads writing the same host parameter at once (automation and the
    // UI) can deliver their notifications in the opposite order to their
    // writes. Settle on the host's final value so the store never keeps the
    // loser. Compared in normalised space; bounded, as each pass re-reads.
    if (e.host != nullptr)
        for (int pass = 0; pass < 8; ++pass)
        {
            const auto hostNorm = e.host->getValue();
            if (std::abs (e.host->convertTo0to1 (raw) - hostNorm) <= 1.0e-6f)
                break;
            raw = e.host->convertFrom0to1 (hostNorm);
            e.value.store (raw);
        }

    markDirty (e);
    notifyListeners (e, raw);
    return old;
}

void ParamStore::propagate (Entry& e, float oldRaw, float newRaw)
{
    if (loading.load())
        return;

    applyLinks (e, oldRaw, newRaw);
    applyEnables (e, newRaw);
}

void ParamStore::handleChange (Entry& e, float raw)
{
    propagate (e, storeChange (e, raw), raw);
}

//==============================================================================
void ParamStore::addLink (const LinkRule& rule)
{
    if (! built.load())
        return;

    auto link = std::make_unique<Link> (rule);
    link->first = indexOf (rule.first).value;
    link->second = indexOf (rule.second).value;
    link->enabler = rule.enabledBy.isEmpty() ? -1 : indexOf (rule.enabledBy).value;

    if (link->first < 0 || link->second < 0 || link->first == link->second
        || (rule.enabledBy.isNotEmpty() && link->enabler < 0))
    {
        jassertfalse; // unknown or invalid ids in the rule
        return;
    }

    const auto k = (int) links.size();
    auto& a = *entries[(size_t) link->first];
    auto& b = *entries[(size_t) link->second];

    a.linkIndices.push_back (k);
    b.linkIndices.push_back (k);

    for (auto* member : { &a, &b })
    {
        member->spec.meta = true;

        if (member->forcedMeta != nullptr)
            member->forcedMeta->store (true);
    }

    if (link->enabler >= 0)
        entries[(size_t) link->enabler]->enablerIndices.push_back (k);

    link->enabled.store (link->enabler < 0 || entries[(size_t) link->enabler]->value.load() >= 0.5f);
    links.push_back (std::move (link));

    if (links.back()->enabled.load())
        onLinkEnabled (*links.back());
}

void ParamStore::setFollower (int index, float raw)
{
    if (! nearlyEqual (entries[(size_t) index]->value.load(), raw))
        set (ParamIndex { index }, raw, Notify::host);
}

void ParamStore::applyLinks (Entry& e, float oldRaw, float newRaw)
{
    for (const auto k : e.linkIndices)
    {
        auto& link = *links[(size_t) k];

        if (! link.enabled.load() || link.busy.exchange (true))
            continue; // disabled, or this rule is already updating (re-entry guard)

        const juce::ScopeGuard release { [&link] { link.busy.store (false); } };
        applyRule (link, e.index, oldRaw, newRaw);
    }
}

void ParamStore::applyEnables (Entry& e, float raw)
{
    for (const auto k : e.enablerIndices)
    {
        auto& link = *links[(size_t) k];
        const auto on = raw >= 0.5f;

        if (on && ! link.enabled.exchange (on))
            onLinkEnabled (link);
        else
            link.enabled.store (on);
    }
}

void ParamStore::onLinkEnabled (Link& link)
{
    if (link.busy.exchange (true))
        return;

    const juce::ScopeGuard release { [&link] { link.busy.store (false); } };
    auto& a = *entries[(size_t) link.first];
    auto& b = *entries[(size_t) link.second];

    switch (link.rule.mode)
    {
        case LinkRule::Mode::Mirror:
            setFollower (link.second, link.fromSpace (b, link.toSpace (a, a.value.load())));
            break;

        case LinkRule::Mode::Offset:
            link.offset.store (link.toSpace (a, a.value.load()) - link.toSpace (b, b.value.load()));
            break;

        case LinkRule::Mode::Ordered:
            enforceOrdered (link, true);
            break;

        case LinkRule::Mode::Opposite:
            break;
    }
}

void ParamStore::applyRule (Link& link, int movedIndex, float oldRaw, float newRaw)
{
    const auto movedIsFirst = movedIndex == link.first;
    const auto otherIndex = movedIsFirst ? link.second : link.first;
    auto& moved = *entries[(size_t) movedIndex];
    auto& other = *entries[(size_t) otherIndex];

    switch (link.rule.mode)
    {
        case LinkRule::Mode::Mirror:
            setFollower (otherIndex, link.fromSpace (other, link.toSpace (moved, newRaw)));
            break;

        case LinkRule::Mode::Offset:
        {
            const auto v = link.toSpace (moved, newRaw);
            const auto off = link.offset.load();
            setFollower (otherIndex, link.fromSpace (other, movedIsFirst ? v - off : v + off));
            break;
        }

        case LinkRule::Mode::Opposite:
        {
            const auto delta = link.toSpace (moved, newRaw) - link.toSpace (moved, oldRaw);
            setFollower (otherIndex, link.fromSpace (other, link.toSpace (other, other.value.load()) - delta));
            break;
        }

        case LinkRule::Mode::Ordered:
            enforceOrdered (link, movedIsFirst);
            break;
    }
}

void ParamStore::enforceOrdered (Link& link, bool movedIsFirst)
{
    auto& a = *entries[(size_t) link.first];
    auto& b = *entries[(size_t) link.second];
    const auto gap = link.rule.minGap;

    auto f = link.toSpace (a, a.value.load());
    auto s = link.toSpace (b, b.value.load());

    auto violated = [&] { return f > s - gap + nearlyEqualTolerance (f, s); };

    if (! violated())
        return;

    if (movedIsFirst)
    {
        setFollower (link.second, link.fromSpace (b, juce::jmin (f + gap, link.spaceMax (b))));
        s = link.toSpace (b, b.value.load());

        if (violated()) // the other member is at its end: clamp the moved one
            setFollower (link.first, link.fromSpace (a, s - gap));
    }
    else
    {
        setFollower (link.first, link.fromSpace (a, juce::jmax (s - gap, link.spaceMin (a))));
        f = link.toSpace (a, a.value.load());

        if (violated())
            setFollower (link.second, link.fromSpace (b, f + gap));
    }
}

float ParamStore::constrain (const juce::String& id, float candidateRaw) const noexcept
{
    if (! built.load())
        return candidateRaw;

    auto* e = entryAt (indexOf (id));

    if (e == nullptr)
        return candidateRaw;

    auto lo = e->spec.range.start;
    auto hi = e->spec.range.end;

    for (const auto k : e->linkIndices)
    {
        const auto& link = *links[(size_t) k];

        if (link.rule.mode != LinkRule::Mode::Ordered || ! link.enabled.load())
            continue;

        // The other member can be pushed up to its range end, so that end bounds this one.
        if (e->index == link.first)
            hi = juce::jmin (hi, link.fromSpace (*e, link.spaceMax (*entries[(size_t) link.second]) - link.rule.minGap));
        else
            lo = juce::jmax (lo, link.fromSpace (*e, link.spaceMin (*entries[(size_t) link.first]) + link.rule.minGap));
    }

    if (lo > hi) // no value satisfies every rule: fall back to the parameter's own range
        return juce::jlimit (e->spec.range.start, e->spec.range.end, candidateRaw);

    return juce::jlimit (lo, hi, candidateRaw);
}

//==============================================================================
void ParamStore::save (juce::MemoryBlock& destination)
{
    destination.reset();

    if (! built.load())
        return;

    auto tree = state->copyState();

    // Values written with Notify::listenersOnly bypass the APVTS cache, so write ours.
    for (auto& e : entries)
    {
        if (e->host == nullptr)
            continue;

        auto child = tree.getChildWithProperty (idProp, e->spec.id);

        if (! child.isValid())
        {
            child = juce::ValueTree (paramTag);
            child.setProperty (idProp, e->spec.id, nullptr);
            tree.appendChild (child, nullptr);
        }

        child.setProperty (valueProp, e->value.load(), nullptr);
    }

    tree.removeChild (tree.getChildWithName (hiddenTag), nullptr);
    juce::ValueTree hidden (hiddenTag);

    auto addHidden = [&hidden] (const juce::String& id, float v)
    {
        juce::ValueTree p (paramTag);
        p.setProperty (idProp, id, nullptr);
        p.setProperty (valueProp, v, nullptr);
        hidden.appendChild (p, nullptr);
    };

    for (auto& e : entries)
        if (e->host == nullptr)
            addHidden (e->spec.id, e->value.load());

    for (auto& link : links)
        if (link->rule.mode == LinkRule::Mode::Offset)
            addHidden (link->rule.offsetStateId(), link->offset.load());

    tree.appendChild (hidden, nullptr);

    if (auto xml = tree.createXml())
        juce::AudioProcessor::copyXmlToBinary (*xml, destination);
}

bool ParamStore::load (const void* data, int sizeInBytes)        { return loadState (data, sizeInBytes, false); }
bool ParamStore::loadPreset (const void* data, int sizeInBytes)  { return loadState (data, sizeInBytes, true); }

bool ParamStore::loadState (const void* data, int sizeInBytes, bool asPreset)
{
    if (! built.load() || data == nullptr || sizeInBytes <= 0)
        return false;

    const auto xml = juce::AudioProcessor::getXmlFromBinary (data, sizeInBytes);

    if (xml == nullptr)
        return false;

    auto tree = juce::ValueTree::fromXml (*xml);

    if (! tree.isValid() || ! tree.hasType (state->state.getType()))
        return false;

    const auto hidden = tree.getChildWithName (hiddenTag);
    tree.removeChild (hidden, nullptr);

    auto findValue = [] (const juce::ValueTree& parent, const juce::String& id, float& out)
    {
        for (const auto& child : parent)
        {
            if (child.hasType (paramTag) && child.getProperty (idProp).toString() == id && child.hasProperty (valueProp))
            {
                out = (float) child.getProperty (valueProp);
                return true;
            }
        }

        return false;
    };

    // Work out every target value first: missing ids -> default, preset-protected -> current.
    std::vector<float> targets (entries.size());

    for (auto& e : entries)
    {
        auto& target = targets[(size_t) e->index];

        if (asPreset && ! e->spec.restoreFromPreset)
            target = e->value.load();
        else if (! findValue (e->host != nullptr ? tree : hidden, e->spec.id, target))
            target = e->spec.defaultValue;

        const auto& range = e->spec.range;
        target = range.snapToLegalValue (juce::jlimit (range.start, range.end, target));

        if (e->host != nullptr)
        {
            auto child = tree.getChildWithProperty (idProp, e->spec.id);

            if (! child.isValid())
            {
                child = juce::ValueTree (paramTag);
                child.setProperty (idProp, e->spec.id, nullptr);
                tree.appendChild (child, nullptr);
            }

            child.setProperty (valueProp, target, nullptr);
            e->requestedRaw.store (target);
        }
    }

    {
        loading.store (true);
        const juce::ScopeGuard endLoading { [this] { loading.store (false); } };

        state->replaceState (tree);

        for (auto& e : entries)
        {
            const auto target = targets[(size_t) e->index];

            if (e->host != nullptr && ! juce::exactlyEqual (e->host->getValue(), e->host->convertTo0to1 (target)))
            {
                e->requestedRaw.store (target);
                e->host->setValueNotifyingHost (e->host->convertTo0to1 (target));
            }
            else if (! juce::exactlyEqual (e->value.load(), target))
            {
                handleChange (*e, target);
            }
        }

        // Recompute link state from the loaded values.
        for (auto& link : links)
        {
            link->enabled.store (link->enabler < 0 || entries[(size_t) link->enabler]->value.load() >= 0.5f);

            if (link->rule.mode != LinkRule::Mode::Offset)
                continue;

            float storedOffset = 0.0f;

            if (findValue (hidden, link->rule.offsetStateId(), storedOffset))
                link->offset.store (storedOffset);
            else if (link->enabled.load())
                link->offset.store (link->toSpace (*entries[(size_t) link->first], entries[(size_t) link->first]->value.load())
                                    - link->toSpace (*entries[(size_t) link->second], entries[(size_t) link->second]->value.load()));
        }
    }

    // Re-establish Mirror / Ordered invariants (no-ops for consistent data).
    for (auto& link : links)
    {
        if (! link->enabled.load())
            continue;

        if (link->rule.mode == LinkRule::Mode::Mirror || link->rule.mode == LinkRule::Mode::Ordered)
            onLinkEnabled (*link);
    }

    for (auto& e : entries)
        markDirty (*e);

    return true;
}

} // namespace vae::params
