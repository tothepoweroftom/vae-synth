#include "params/Attachments.h"
#include "params/ParamStore.h"

#include <barrier>
#include <thread>

namespace vae::params::tests
{

//==============================================================================
/** Minimal processor owning a store, as a real plug-in would. */
class TestProcessor final : public juce::AudioProcessor
{
public:
    TestProcessor() : juce::AudioProcessor (BusesProperties()) {}

    void buildStore (std::vector<ParamSpec> specs)   { store.build (std::move (specs), *this, juce::Identifier ("VaeTestState")); }

    const juce::String getName() const override                         { return "VaeTest"; }
    void prepareToPlay (double, int) override                           {}
    void releaseResources() override                                    {}
    using juce::AudioProcessor::processBlock;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override { store.dispatch(); }
    bool hasEditor() const override                                     { return false; }
    juce::AudioProcessorEditor* createEditor() override                 { return nullptr; }
    bool acceptsMidi() const override                                   { return false; }
    bool producesMidi() const override                                  { return false; }
    double getTailLengthSeconds() const override                        { return 0.0; }
    int getNumPrograms() override                                       { return 1; }
    int getCurrentProgram() override                                    { return 0; }
    void setCurrentProgram (int) override                               {}
    const juce::String getProgramName (int) override                    { return {}; }
    void changeProgramName (int, const juce::String&) override          {}
    void getStateInformation (juce::MemoryBlock& dest) override         { store.save (dest); }
    void setStateInformation (const void* data, int size) override      { store.load (data, size); }

    ParamStore store;
};

/** Records audio-thread deliveries. */
struct DeliveryLog
{
    struct Item { juce::String id; float value; std::thread::id thread; };
    std::vector<Item> items;

    ParamSpec::AudioChange recorder (const juce::String& id)
    {
        return [this, id] (float v) { items.push_back ({ id, v, std::this_thread::get_id() }); };
    }

    void clear()    { items.clear(); }
};

struct CountingListener final : ParamStore::Listener
{
    void paramChanged (const juce::String&, float raw) override
    {
        ++count;
        last = raw;
        thread = std::this_thread::get_id();
    }

    std::atomic<int> count { 0 };
    std::atomic<float> last { 0.0f };
    std::thread::id thread;
};

struct HostListener final : juce::AudioProcessorListener
{
    void audioProcessorParameterChanged (juce::AudioProcessor*, int, float) override  { ++changes; }
    void audioProcessorChanged (juce::AudioProcessor*, const ChangeDetails&) override {}
    void audioProcessorParameterChangeGestureBegin (juce::AudioProcessor*, int) override { ++gestureBegins; }
    void audioProcessorParameterChangeGestureEnd (juce::AudioProcessor*, int) override   { ++gestureEnds; }

    int changes = 0, gestureBegins = 0, gestureEnds = 0;
};

//==============================================================================
static ParamSpec floatSpec (const juce::String& id, float lo, float hi, float def, bool visible = true)
{
    ParamSpec s;
    s.id = id;
    s.kind = ParamKind::Float;
    s.range = { lo, hi };
    s.defaultValue = def;
    s.hostVisible = visible;
    return s;
}

static std::vector<ParamSpec> mixedSpecs (DeliveryLog* log = nullptr)
{
    std::vector<ParamSpec> specs;

    auto cutoff = floatSpec ("cutoff", 20.0f, 20000.0f, 1000.0f);
    cutoff.name = "Cutoff";
    cutoff.range = ParamSpec::frequencyRange (20.0f, 20000.0f);
    cutoff.unit = "Hz";
    specs.push_back (cutoff);

    ParamSpec voices;
    voices.id = "voices";
    voices.kind = ParamKind::Int;
    voices.range = { 1.0f, 16.0f, 1.0f };
    voices.defaultValue = 8.0f;
    specs.push_back (voices);

    ParamSpec bypass;
    bypass.id = "bypass";
    bypass.kind = ParamKind::Bool;
    bypass.defaultValue = 0.0f;
    specs.push_back (bypass);

    ParamSpec wave;
    wave.id = "wave";
    wave.kind = ParamKind::Choice;
    wave.choices = { "Sine", "Saw", "Square" };
    wave.defaultValue = 1.0f;
    specs.push_back (wave);

    specs.push_back (floatSpec ("uiZoom", 0.5f, 2.0f, 1.0f, false));

    if (log != nullptr)
        for (auto& s : specs)
            s.onAudioChange = log->recorder (s.id);

    return specs;
}

static juce::String paramIdOf (juce::AudioProcessorParameter* p)
{
    if (auto* withId = dynamic_cast<juce::AudioProcessorParameterWithID*> (p))
        return withId->paramID;

    return {};
}

static juce::MemoryBlock xmlToBinary (const juce::String& xmlText)
{
    juce::MemoryBlock mb;
    auto xml = juce::parseXML (xmlText);
    jassert (xml != nullptr);
    juce::AudioProcessor::copyXmlToBinary (*xml, mb);
    return mb;
}

//==============================================================================
class ParamStoreTests final : public juce::UnitTest
{
public:
    ParamStoreTests() : juce::UnitTest ("ParamStore", "vae_params") {}

    void runTest() override
    {
        testBuild();
        testDispatch();
        testCrossThread();
        testConcurrentWriters();
        testSetGetNotify();
        testHiddenPersistence();
        testPresets();
        testLoadEdgeCases();
        testLinkMirror();
        testLinkOffset();
        testLinkOpposite();
        testLinkOrdered();
        testFormatters();
        testBeforeBuild();
        testAttachments();
    }

private:
    //==========================================================================
    void testBuild()
    {
        beginTest ("1. Build with mixed kinds; host sees only visible params, in order");

        TestProcessor proc;
        auto specs = mixedSpecs();
        specs.push_back (floatSpec ("voices", 0.0f, 1.0f, 0.5f)); // duplicate: ignored (asserts in debug)
        proc.buildStore (std::move (specs));
        auto& store = proc.store;

        expect (store.isBuilt());
        expect (store.apvts() != nullptr);
        expectEquals ((int) store.size(), 5);
        expectEquals (store.ids().joinIntoString (","), juce::String ("cutoff,voices,bypass,wave,uiZoom"));

        const auto& hostParams = proc.getParameters();
        expectEquals (hostParams.size(), 4);
        expectEquals (paramIdOf (hostParams[0]), juce::String ("cutoff"));
        expectEquals (paramIdOf (hostParams[1]), juce::String ("voices"));
        expectEquals (paramIdOf (hostParams[2]), juce::String ("bypass"));
        expectEquals (paramIdOf (hostParams[3]), juce::String ("wave"));

        expect (dynamic_cast<juce::AudioParameterFloat*> (store.hostParam ("cutoff")) != nullptr);
        expect (dynamic_cast<juce::AudioParameterInt*> (store.hostParam ("voices")) != nullptr);
        expect (dynamic_cast<juce::AudioParameterBool*> (store.hostParam ("bypass")) != nullptr);
        expect (dynamic_cast<juce::AudioParameterChoice*> (store.hostParam ("wave")) != nullptr);
        expect (store.hostParam ("uiZoom") == nullptr);
        expect (store.apvts()->getParameter ("uiZoom") == nullptr);
        expect (store.hostParam ("cutoff") == store.apvts()->getParameter ("cutoff"));

        expectEquals (store.hostParam ("cutoff")->getName (64), juce::String ("Cutoff"));
        expectEquals (store.hostParam ("voices")->getName (64), juce::String ("voices")); // name defaults to id

        // Choice range derived from choices; Bool range forced.
        expectEquals (store.spec ("wave").range.start, 0.0f);
        expectEquals (store.spec ("wave").range.end, 2.0f);
        expectEquals (store.spec ("wave").range.interval, 1.0f);
        expectEquals (store.spec ("bypass").range.end, 1.0f);
        expectEquals (store.spec ("bypass").range.interval, 1.0f);

        // Defaults.
        expectWithinAbsoluteError (store.get ("cutoff"), 1000.0f, 0.01f);
        expectEquals (store.get ("voices"), 8.0f);
        expectEquals (store.get ("wave"), 1.0f);
        expectEquals (store.get ("uiZoom"), 1.0f);
        expectEquals (store.indexOf ("bypass").value, 2);
        expect (! store.indexOf ("nope").isValid());

        // Frequency range: geometric centre at 0.5.
        const auto fr = ParamSpec::frequencyRange (20.0f, 20000.0f);
        expectWithinAbsoluteError (fr.convertTo0to1 (std::sqrt (20.0f * 20000.0f)), 0.5f, 1.0e-4f);
        expectWithinAbsoluteError (store.getNormalised ("cutoff"), fr.convertTo0to1 (1000.0f), 1.0e-5f);

        bool threw = false;
        try { (void) store.spec ("nope"); }
        catch (const std::out_of_range&) { threw = true; }
        expect (threw, "spec() throws std::out_of_range for unknown ids");
    }

    //==========================================================================
    void testDispatch()
    {
        beginTest ("2. First dispatch delivers every param once; later only changed ones, coalesced");

        DeliveryLog log;
        TestProcessor proc;
        proc.buildStore (mixedSpecs (&log));
        auto& store = proc.store;

        expect (log.items.empty(), "nothing is delivered before dispatch()");
        store.dispatch();
        expectEquals ((int) log.items.size(), 5);

        if (log.items.size() == 5)
        {
            expectEquals (log.items[0].id, juce::String ("cutoff"));
            expectEquals (log.items[4].id, juce::String ("uiZoom"));
            expectEquals (log.items[1].value, 8.0f);
        }

        log.clear();
        store.dispatch();
        expect (log.items.empty(), "no changes -> no deliveries");

        store.set ("voices", 3.0f);
        store.set ("uiZoom", 1.5f);
        store.set ("voices", 4.0f);
        store.set ("voices", 5.0f);
        store.dispatch();
        expectEquals ((int) log.items.size(), 2);

        if (log.items.size() == 2)
        {
            expectEquals (log.items[0].id, juce::String ("voices"));
            expectEquals (log.items[0].value, 5.0f);
            expectEquals (log.items[1].id, juce::String ("uiZoom"));
            expectEquals (log.items[1].value, 1.5f);
        }

        // Index-based setter on the hot path.
        log.clear();
        store.set (store.indexOf ("wave"), 2.0f);
        store.dispatch();
        expectEquals ((int) log.items.size(), 1);

        // A value set from the audio callback is delivered on the next dispatch, not re-entrantly.
        {
            DeliveryLog log2;
            TestProcessor proc2;
            auto specs = mixedSpecs (&log2);
            specs[1].onAudioChange = [&] (float v)
            {
                log2.items.push_back ({ "voices", v, std::this_thread::get_id() });

                if (v < 10.0f)
                    proc2.store.set ("voices", v + 1.0f);
            };
            proc2.buildStore (std::move (specs));
            proc2.store.dispatch();
            const auto firstRound = log2.items.size();
            expectEquals ((int) firstRound, 5);
            log2.clear();
            proc2.store.dispatch();
            expectEquals ((int) log2.items.size(), 1);

            if (! log2.items.empty())
                expectEquals (log2.items[0].value, 9.0f);
        }

        // Many changes across all params coalesce to one delivery per param.
        log.clear();

        for (int round = 0; round < 50; ++round)
            for (const auto& id : store.ids())
                store.setNormalised (id, (float) round / 49.0f);

        store.dispatch();
        expectEquals ((int) log.items.size(), 5);
    }

    //==========================================================================
    void testCrossThread()
    {
        beginTest ("3. set from a non-audio thread is delivered only at dispatch()");

        DeliveryLog log;
        TestProcessor proc;
        proc.buildStore (mixedSpecs (&log));
        auto& store = proc.store;
        store.dispatch();
        log.clear();

        CountingListener listener;
        store.addListener ("cutoff", &listener);

        std::thread::id setterThread;

        std::thread setter ([&]
        {
            setterThread = std::this_thread::get_id();

            for (int i = 0; i < 100; ++i)
                store.set ("cutoff", 100.0f + (float) i);

            store.set ("uiZoom", 0.75f);
        });
        setter.join();

        expect (log.items.empty(), "no delivery on the setter thread");
        expectEquals (listener.count.load(), 100);
        expect (listener.thread == setterThread, "listeners run on the setting thread");

        store.dispatch();
        expectEquals ((int) log.items.size(), 2);

        for (auto& item : log.items)
            expect (item.thread == std::this_thread::get_id(), "delivery on the dispatching thread");

        if (log.items.size() == 2)
            expectWithinAbsoluteError (log.items[0].value, 199.0f, 0.01f);

        // Concurrent setters + dispatcher: no lost final values.
        log.clear();
        std::atomic<bool> stop { false };
        std::vector<std::thread> threads;

        for (int t = 0; t < 3; ++t)
        {
            threads.emplace_back ([&store, t]
            {
                for (int i = 0; i < 2000; ++i)
                    store.setNormalised (t == 0 ? "cutoff" : (t == 1 ? "voices" : "uiZoom"), (float) (i % 100) / 99.0f);

                store.setNormalised (t == 0 ? "cutoff" : (t == 1 ? "voices" : "uiZoom"), 1.0f);
            });
        }

        std::thread audio ([&] { while (! stop.load()) store.dispatch(); });

        for (auto& t : threads)
            t.join();

        stop = true;
        audio.join();
        store.dispatch();

        std::map<juce::String, float> latest;

        for (auto& item : log.items)
            latest[item.id] = item.value;

        expectWithinAbsoluteError (latest["cutoff"], 20000.0f, 0.5f);
        expectEquals (latest["voices"], 16.0f);
        expectEquals (latest["uiZoom"], 2.0f);

        store.removeListener ("cutoff", &listener);
    }

    //==========================================================================
    void testConcurrentWriters()
    {
        beginTest ("Concurrent writers to one parameter: store, host and delivery agree");

        float delivered = -1.0f;
        TestProcessor proc;
        auto spec = floatSpec ("x", 0.0f, 1.0f, 0.5f);
        spec.onAudioChange = [&delivered] (float v) { delivered = v; };
        proc.buildStore ({ spec });
        auto& store = proc.store;
        auto* host = store.hostParam ("x");

        int diverged = 0;
        constexpr int trials = 3000;
        for (int t = 0; t < trials; ++t)
        {
            std::barrier sync (2);
            std::thread automation ([&] { sync.arrive_and_wait(); host->setValueNotifyingHost (0.25f); });
            std::thread ui ([&] { sync.arrive_and_wait(); store.set ("x", 0.75f); });
            automation.join();
            ui.join();

            store.dispatch();
            if (std::abs (host->getValue() - store.get ("x")) > 1.0e-6f
                || std::abs (delivered - store.get ("x")) > 1.0e-6f)
                ++diverged;
            store.set ("x", 0.5f);
        }
        expectEquals (diverged, 0, "trials where store, host or delivered value disagree");
    }

    //==========================================================================
    void testSetGetNotify()
    {
        beginTest ("Values, setters, notification modes, gestures, listeners (spec section 5)");

        TestProcessor proc;
        proc.buildStore (mixedSpecs());
        auto& store = proc.store;
        HostListener host;
        proc.addListener (&host);

        // Snap to legal value and clamp.
        store.set ("voices", 3.4f);
        expectEquals (store.get ("voices"), 3.0f);
        store.set ("voices", 99.0f);
        expectEquals (store.get ("voices"), 16.0f);
        store.setNormalised ("uiZoom", 2.0f);
        expectEquals (store.get ("uiZoom"), 2.0f);
        store.setNormalised ("uiZoom", 0.0f);
        expectEquals (store.get ("uiZoom"), 0.5f);

        // Exact raw values survive the host round trip.
        store.set ("cutoff", 440.0f);
        expectEquals (store.get ("cutoff"), 440.0f);
        expectWithinAbsoluteError (store.hostParam ("cutoff")->convertFrom0to1 (store.hostParam ("cutoff")->getValue()), 440.0f, 0.01f);

        const auto before = host.changes;
        store.set ("voices", 5.0f, Notify::listenersOnly);
        expectEquals (host.changes, before, "listenersOnly does not notify the host");
        expectEquals (store.get ("voices"), 5.0f);
        expectEquals (juce::roundToInt (store.hostParam ("voices")->convertFrom0to1 (store.hostParam ("voices")->getValue())), 5);

        store.set ("voices", 6.0f);
        expectEquals (host.changes, before + 1, "Notify::host notifies the host");

        store.set ("uiZoom", 1.25f);
        expectEquals (host.changes, before + 1, "hidden params never notify the host");

        // A host-side change (automation) reaches the store.
        store.hostParam ("wave")->setValueNotifyingHost (0.0f);
        expectEquals (store.get ("wave"), 0.0f);

        // Listeners: visible and hidden identical.
        CountingListener visible, hidden;
        store.addListener ("cutoff", &visible);
        store.addListener ("uiZoom", &hidden);
        store.set ("cutoff", 2000.0f);
        store.set ("uiZoom", 1.5f, Notify::listenersOnly);
        expectEquals (visible.count.load(), 1);
        expectEquals (hidden.count.load(), 1);
        expectEquals (hidden.last.load(), 1.5f);
        store.removeListener ("cutoff", &visible);
        store.set ("cutoff", 3000.0f);
        expectEquals (visible.count.load(), 1, "removed listener not called");
        store.removeListener ("uiZoom", &hidden);

        // Gestures.
        store.beginGesture ("cutoff");
        store.beginGesture ("cutoff"); // nested: forwarded once
        store.endGesture ("cutoff");
        store.endGesture ("cutoff");
        expectEquals (host.gestureBegins, 1);
        expectEquals (host.gestureEnds, 1);
        store.beginGesture ("uiZoom");
        store.endGesture ("uiZoom");
        expectEquals (host.gestureBegins, 1, "hidden gestures are no-ops");

        // Reset.
        store.resetToDefault ("voices");
        expectEquals (store.get ("voices"), 8.0f);
        store.set ("uiZoom", 2.0f);
        store.resetAllToDefaults();
        expectEquals (store.get ("uiZoom"), 1.0f);
        expectWithinAbsoluteError (store.get ("cutoff"), 1000.0f, 0.01f);

        proc.removeListener (&host);
    }

    //==========================================================================
    void testHiddenPersistence()
    {
        beginTest ("4. Hidden params: not in host list, persist through save/load");

        juce::MemoryBlock saved;

        {
            TestProcessor proc;
            proc.buildStore (mixedSpecs());
            expect (proc.getParameters().size() == 4);
            proc.store.set ("uiZoom", 1.75f);
            proc.store.set ("cutoff", 5000.0f);
            proc.store.set ("voices", 2.0f, Notify::listenersOnly); // bypasses the APVTS cache
            proc.getStateInformation (saved);
        }

        expect (saved.getSize() > 0);

        DeliveryLog log;
        TestProcessor proc;
        proc.buildStore (mixedSpecs (&log));
        auto& store = proc.store;
        store.dispatch();
        log.clear();

        CountingListener hiddenListener;
        store.addListener ("uiZoom", &hiddenListener);

        expect (store.load (saved.getData(), (int) saved.getSize()));
        expectEquals (store.get ("uiZoom"), 1.75f);
        expectEquals (store.get ("cutoff"), 5000.0f);
        expectEquals (store.get ("voices"), 2.0f);
        expectEquals (hiddenListener.count.load(), 1, "hidden listener notified on load");

        store.dispatch();
        expectEquals ((int) log.items.size(), 5, "load marks every parameter dirty");

        // State tree carries the hidden child; host-visible children are APVTS PARAM nodes.
        auto xml = juce::AudioProcessor::getXmlFromBinary (saved.getData(), (int) saved.getSize());
        expect (xml != nullptr && xml->getChildByName ("VAE_HIDDEN_PARAMS") != nullptr);

        store.removeListener ("uiZoom", &hiddenListener);
    }

    //==========================================================================
    void testPresets()
    {
        beginTest ("5. loadPreset keeps restoreFromPreset == false params; load restores all");

        auto specs = []
        {
            auto s = mixedSpecs();
            s[1].restoreFromPreset = false; // voices (visible)
            s[4].restoreFromPreset = false; // uiZoom (hidden)
            return s;
        };

        juce::MemoryBlock preset;

        {
            TestProcessor proc;
            proc.buildStore (specs());
            proc.store.set ("voices", 12.0f);
            proc.store.set ("uiZoom", 2.0f);
            proc.store.set ("wave", 2.0f);
            proc.store.save (preset);
        }

        TestProcessor proc;
        proc.buildStore (specs());
        auto& store = proc.store;
        store.set ("voices", 4.0f);
        store.set ("uiZoom", 0.5f);
        store.set ("wave", 0.0f);

        expect (store.loadPreset (preset.getData(), (int) preset.getSize()));
        expectEquals (store.get ("voices"), 4.0f, "visible protected param kept");
        expectEquals (store.get ("uiZoom"), 0.5f, "hidden protected param kept");
        expectEquals (store.get ("wave"), 2.0f, "unprotected param restored");

        expect (store.load (preset.getData(), (int) preset.getSize()));
        expectEquals (store.get ("voices"), 12.0f, "load restores everything");
        expectEquals (store.get ("uiZoom"), 2.0f);
    }

    //==========================================================================
    void testLoadEdgeCases()
    {
        beginTest ("6. Missing ids keep defaults; unknown ids ignored; bad data returns false");

        TestProcessor proc;
        proc.buildStore (mixedSpecs());
        auto& store = proc.store;

        store.set ("cutoff", 300.0f);
        store.set ("voices", 3.0f);
        store.set ("uiZoom", 0.6f);
        store.set ("wave", 2.0f);

        const auto data = xmlToBinary (R"(<VaeTestState>
                                            <PARAM id="wave" value="0"/>
                                            <PARAM id="removedLongAgo" value="42"/>
                                            <VAE_HIDDEN_PARAMS><PARAM id="ghost" value="3"/></VAE_HIDDEN_PARAMS>
                                          </VaeTestState>)");

        expect (store.load (data.getData(), (int) data.getSize()));
        expectEquals (store.get ("wave"), 0.0f, "present id restored");
        expectWithinAbsoluteError (store.get ("cutoff"), 1000.0f, 0.01f);
        expectEquals (store.get ("voices"), 8.0f, "missing visible id -> default");
        expectEquals (store.get ("uiZoom"), 1.0f, "missing hidden id -> default");
        expectEquals (store.get ("bypass"), 0.0f);

        // Bad data: no change, false.
        store.set ("voices", 7.0f);
        const char garbage[] = "definitely not a JUCE state blob";
        expect (! store.load (garbage, (int) sizeof (garbage)));
        expect (! store.loadPreset (garbage, (int) sizeof (garbage)));
        expect (! store.load (nullptr, 0));

        const auto wrongType = xmlToBinary (R"(<SomeOtherPlugin><PARAM id="voices" value="1"/></SomeOtherPlugin>)");
        expect (! store.load (wrongType.getData(), (int) wrongType.getSize()));
        expect (! store.loadPreset (wrongType.getData(), (int) wrongType.getSize()));
        expectEquals (store.get ("voices"), 7.0f, "failed loads change nothing");

        // Out-of-range values in data are clamped/snapped.
        const auto wild = xmlToBinary (R"(<VaeTestState><PARAM id="voices" value="3.7"/><VAE_HIDDEN_PARAMS><PARAM id="uiZoom" value="9"/></VAE_HIDDEN_PARAMS></VaeTestState>)");
        expect (store.load (wild.getData(), (int) wild.getSize()));
        expectEquals (store.get ("voices"), 4.0f);
        expectEquals (store.get ("uiZoom"), 2.0f);
    }

    //==========================================================================
    static std::vector<ParamSpec> linkSpecs()
    {
        std::vector<ParamSpec> specs;
        specs.push_back (floatSpec ("a", 0.0f, 10.0f, 2.0f));
        specs.push_back (floatSpec ("b", 0.0f, 10.0f, 5.0f));
        specs.push_back (floatSpec ("c", 0.0f, 100.0f, 50.0f, false)); // hidden, different range

        ParamSpec link;
        link.id = "linkOn";
        link.kind = ParamKind::Bool;
        link.defaultValue = 0.0f;
        specs.push_back (link);
        return specs;
    }

    void testLinkMirror()
    {
        beginTest ("7a. Link: Mirror (raw & normalised), enabledBy, meta, re-entry guard");

        TestProcessor proc;
        proc.buildStore (linkSpecs());
        auto& store = proc.store;

        expect (! store.hostParam ("a")->isMetaParameter());

        LinkRule rule;
        rule.first = "a";
        rule.second = "b";
        rule.enabledBy = "linkOn";
        rule.mode = LinkRule::Mode::Mirror;
        store.addLink (rule);

        expect (store.hostParam ("a")->isMetaParameter(), "link members become meta");
        expect (store.hostParam ("b")->isMetaParameter());

        store.set ("a", 3.0f);
        expectEquals (store.get ("b"), 5.0f, "disabled rule does nothing");

        store.set ("linkOn", 1.0f);
        expectEquals (store.get ("b"), 3.0f, "on enable, second snaps to first");

        CountingListener aListener;
        store.addListener ("a", &aListener);
        store.set ("b", 7.0f);
        expectEquals (store.get ("a"), 7.0f, "first follows second");
        expectEquals (aListener.count.load(), 1, "follower set once, no recursion");
        store.set ("a", 1.0f);
        expectEquals (store.get ("b"), 1.0f);
        store.removeListener ("a", &aListener);

        store.set ("linkOn", 0.0f);
        store.set ("a", 9.0f);
        expectEquals (store.get ("b"), 1.0f, "disabled again");

        // Normalised space across different ranges (visible a <-> hidden c).
        LinkRule n;
        n.first = "a";
        n.second = "c";
        n.mode = LinkRule::Mode::Mirror;
        n.space = LinkRule::Space::Normalised;
        store.addLink (n);
        expectEquals (store.get ("c"), 90.0f, "always-on rule snaps on add");
        store.set ("c", 25.0f);
        expectEquals (store.get ("a"), 2.5f);

        // Followers notify the host so automation records.
        HostListener host;
        proc.addListener (&host);
        store.set ("c", 40.0f);
        expectEquals (host.changes, 1, "visible follower notified the host");
        proc.removeListener (&host);

        // Re-entry guard with a cycle of always-on rules: a->b, b->c, c->a.
        TestProcessor proc2;
        proc2.buildStore (linkSpecs());
        auto& s2 = proc2.store;
        auto mirror = [] (const char* f, const char* s)
        {
            LinkRule r;
            r.first = f;
            r.second = s;
            r.mode = LinkRule::Mode::Mirror;
            return r;
        };
        s2.addLink (mirror ("a", "b"));
        s2.addLink (mirror ("b", "c"));
        s2.addLink (mirror ("c", "a"));
        s2.set ("a", 4.0f);
        expectEquals (s2.get ("b"), 4.0f);
        expectEquals (s2.get ("c"), 4.0f);
    }

    void testLinkOffset()
    {
        beginTest ("7b. Link: Offset preserved and persisted");

        juce::MemoryBlock saved;
        LinkRule rule;
        rule.first = "a";
        rule.second = "b";
        rule.enabledBy = "linkOn";
        rule.mode = LinkRule::Mode::Offset;

        {
            TestProcessor proc;
            proc.buildStore (linkSpecs());
            auto& store = proc.store;
            store.addLink (rule);

            store.set ("a", 6.0f);
            store.set ("b", 4.0f);
            store.set ("linkOn", 1.0f); // captures a - b = 2

            store.set ("a", 8.0f);
            expectEquals (store.get ("b"), 6.0f);
            store.set ("b", 1.0f);
            expectEquals (store.get ("a"), 3.0f);

            // Clamped at range end; captured offset is kept.
            store.set ("a", 1.0f);
            expectEquals (store.get ("b"), 0.0f);
            store.set ("a", 5.0f);
            expectEquals (store.get ("b"), 3.0f);

            store.save (saved);
            auto xml = juce::AudioProcessor::getXmlFromBinary (saved.getData(), (int) saved.getSize());
            auto* hidden = xml != nullptr ? xml->getChildByName ("VAE_HIDDEN_PARAMS") : nullptr;
            expect (hidden != nullptr && hidden->getChildByAttribute ("id", rule.offsetStateId()) != nullptr,
                    "offset persisted as <first>__<second>__offset");
        }

        // The offset survives reload, and links are suspended during load.
        TestProcessor proc;
        proc.buildStore (linkSpecs());
        auto& store = proc.store;
        store.addLink (rule);
        expect (store.load (saved.getData(), (int) saved.getSize()));
        expectEquals (store.get ("a"), 5.0f);
        expectEquals (store.get ("b"), 3.0f);
        store.set ("b", 7.0f);
        expectEquals (store.get ("a"), 9.0f, "persisted offset 2 still applies");
    }

    void testLinkOpposite()
    {
        beginTest ("7c. Link: Opposite (re-entry guard prevents ping-pong)");

        TestProcessor proc;
        proc.buildStore (linkSpecs());
        auto& store = proc.store;

        LinkRule rule;
        rule.first = "a";
        rule.second = "b";
        rule.mode = LinkRule::Mode::Opposite;
        store.addLink (rule);

        store.set ("a", 3.0f); // +1
        expectEquals (store.get ("b"), 4.0f);
        store.set ("b", 6.0f); // +2
        expectEquals (store.get ("a"), 1.0f);

        // Normalised space across ranges.
        LinkRule n;
        n.first = "a";
        n.second = "c";
        n.mode = LinkRule::Mode::Opposite;
        n.space = LinkRule::Space::Normalised;
        store.addLink (n);
        store.set ("a", 2.0f); // +0.1 normalised -> c -10, b -1
        expectEquals (store.get ("c"), 40.0f);
        expectEquals (store.get ("b"), 5.0f);
    }

    void testLinkOrdered()
    {
        beginTest ("7d. Link: Ordered push/clamp and constrain()");

        TestProcessor proc;
        proc.buildStore (linkSpecs());
        auto& store = proc.store;

        expectEquals (store.constrain ("a", 9.5f), 9.5f, "no rules -> unconstrained");

        LinkRule rule;
        rule.first = "a";
        rule.second = "b";
        rule.enabledBy = "linkOn";
        rule.mode = LinkRule::Mode::Ordered;
        rule.minGap = 1.0f;
        store.addLink (rule);

        store.set ("a", 7.0f);
        store.set ("linkOn", 1.0f); // on enable: push second
        expectEquals (store.get ("b"), 8.0f);

        store.set ("a", 8.5f);
        expectEquals (store.get ("b"), 9.5f, "first pushes second");
        store.set ("a", 9.8f);
        expectEquals (store.get ("b"), 10.0f, "second clamped at its end");
        expectEquals (store.get ("a"), 9.0f, "first clamped when second is at its end");

        store.set ("b", 2.0f);
        expectEquals (store.get ("a"), 1.0f, "second pushes first down");
        store.set ("b", 0.2f);
        expectEquals (store.get ("a"), 0.0f);
        expectEquals (store.get ("b"), 1.0f);

        store.set ("a", 3.0f);
        store.set ("b", 6.0f);
        expectEquals (store.get ("a"), 3.0f, "no violation -> no change");

        expectEquals (store.constrain ("a", 9.7f), 9.0f);
        expectEquals (store.constrain ("a", 4.0f), 4.0f);
        expectEquals (store.constrain ("b", 0.3f), 1.0f);
        expectEquals (store.constrain ("b", 12.0f), 10.0f);

        store.set ("linkOn", 0.0f);
        expectEquals (store.constrain ("a", 9.7f), 9.7f, "disabled rule does not constrain");
        store.set ("a", 9.0f);
        expectEquals (store.get ("b"), 6.0f);

        // Normalised space, visible first vs hidden second with a different range.
        TestProcessor proc2;
        proc2.buildStore (linkSpecs());
        auto& s2 = proc2.store;
        LinkRule n;
        n.first = "a";
        n.second = "c";
        n.mode = LinkRule::Mode::Ordered;
        n.space = LinkRule::Space::Normalised;
        n.minGap = 0.1f;
        s2.addLink (n);
        s2.set ("a", 6.0f); // 0.6 -> c >= 0.7 -> 70
        expectWithinAbsoluteError (s2.get ("c"), 70.0f, 1.0e-3f);
        expectWithinAbsoluteError (s2.constrain ("a", 10.0f), 9.0f, 1.0e-4f);
    }

    //==========================================================================
    void testFormatters()
    {
        beginTest ("8. Text formatters round-trip for each kind");

        TestProcessor proc;
        auto specs = mixedSpecs();

        ParamSpec custom = floatSpec ("gain", -60.0f, 12.0f, 0.0f);
        custom.toText = [] (float v) { return juce::String (v, 1) + " dB"; };
        custom.fromText = [] (const juce::String& t) { return t.upToFirstOccurrenceOf (" ", false, false).getFloatValue(); };
        specs.push_back (custom);

        ParamSpec semis;
        semis.id = "semis";
        semis.kind = ParamKind::Int;
        semis.range = { -12.0f, 12.0f };
        semis.unit = "st";
        specs.push_back (semis);

        proc.buildStore (std::move (specs));
        auto& store = proc.store;

        const auto& cutoff = store.spec ("cutoff");
        expectEquals (cutoff.valueToText (440.0f), juce::String ("440.00 Hz"));
        expectEquals (cutoff.textToValue ("440.00 Hz"), 440.0f);
        expectEquals (cutoff.textToValue (cutoff.valueToText (1234.5f)), 1234.5f);

        const auto& st = store.spec ("semis");
        expectEquals (st.range.interval, 1.0f, "Int interval defaults to 1");
        expectEquals (st.valueToText (-7.0f), juce::String ("-7 st"));
        expectEquals (st.textToValue ("-7 st"), -7.0f);

        const auto& bypass = store.spec ("bypass");
        expectEquals (bypass.valueToText (1.0f), juce::String ("On"));
        expectEquals (bypass.valueToText (0.0f), juce::String ("Off"));

        for (auto* yes : { "on", "ON", "Yes", "y", "TRUE", "1", " On " })
            expectEquals (bypass.textToValue (yes), 1.0f, juce::String ("bool parses ") + yes);

        for (auto* no : { "off", "no", "0", "false", "banana" })
            expectEquals (bypass.textToValue (no), 0.0f, juce::String ("bool parses ") + no);

        const auto& wave = store.spec ("wave");
        expectEquals (wave.valueToText (2.0f), juce::String ("Square"));
        expectEquals (wave.textToValue ("saw"), 1.0f, "choice names match case-insensitively");
        expectEquals (wave.textToValue ("2"), 2.0f, "choice falls back to an integer index");

        const auto& gain = store.spec ("gain");
        expectEquals (gain.valueToText (-6.0f), juce::String ("-6.0 dB"));
        expectEquals (gain.textToValue ("-6.0 dB"), -6.0f);

        // Wired through the JUCE parameters, so hosts see the same text.
        auto* hostCutoff = store.hostParam ("cutoff");
        expectEquals (hostCutoff->getText (hostCutoff->convertTo0to1 (440.0f), 0), juce::String ("440.00 Hz"));
        expectWithinAbsoluteError (hostCutoff->convertFrom0to1 (hostCutoff->getValueForText ("440 Hz")), 440.0f, 0.05f);

        auto* hostWave = store.hostParam ("wave");
        expectEquals (hostWave->getText (hostWave->convertTo0to1 (0.0f), 0), juce::String ("Sine"));
        expectEquals (hostWave->convertFrom0to1 (hostWave->getValueForText ("SQUARE")), 2.0f);

        auto* hostBypass = store.hostParam ("bypass");
        expectEquals (hostBypass->getText (1.0f, 0), juce::String ("On"));
        expectEquals (hostBypass->getValueForText ("yes"), 1.0f);

        auto* hostSemis = store.hostParam ("semis");
        expectEquals (hostSemis->getText (hostSemis->convertTo0to1 (5.0f), 0), juce::String ("5 st"));
        expectEquals (hostSemis->convertFrom0to1 (hostSemis->getValueForText ("5 st")), 5.0f);

        auto* hostGain = store.hostParam ("gain");
        expectEquals (hostGain->getText (hostGain->convertTo0to1 (-6.0f), 0), juce::String ("-6.0 dB"));
        expectEquals (hostGain->getText (hostGain->convertTo0to1 (-6.0f), 3), juce::String ("-6."), "max length honoured");
    }

    //==========================================================================
    void testBeforeBuild()
    {
        beginTest ("9. Calls before build do not crash");

        ParamStore store;
        CountingListener listener;
        juce::MemoryBlock mb;

        expect (! store.isBuilt());
        expect (store.apvts() == nullptr);
        expectEquals ((int) store.size(), 0);
        expect (store.ids().isEmpty());
        expectEquals (store.get ("x"), 0.0f);
        expectEquals (store.get (ParamIndex { 0 }), 0.0f);
        expectEquals (store.getNormalised ("x"), 0.0f);
        expect (! store.indexOf ("x").isValid());
        expect (store.hostParam ("x") == nullptr);
        store.set ("x", 1.0f);
        store.set (ParamIndex { 0 }, 1.0f);
        store.setNormalised ("x", 1.0f);
        store.resetToDefault ("x");
        store.resetAllToDefaults();
        store.beginGesture ("x");
        store.endGesture ("x");
        store.addListener ("x", &listener);
        store.removeListener ("x", &listener);
        store.dispatch();
        store.save (mb);
        expectEquals ((int) mb.getSize(), 0);
        expect (! store.load ("abc", 3));
        expect (! store.loadPreset ("abc", 3));
        store.addLink ({});
        expectEquals (store.constrain ("x", 0.25f), 0.25f);

        bool threw = false;
        try { (void) store.spec ("x"); }
        catch (const std::out_of_range&) { threw = true; }
        expect (threw);
    }

    //==========================================================================
    void testAttachments()
    {
        beginTest ("UI attachments: value, slider, button, combo");

        TestProcessor proc;
        auto specs = linkSpecs();
        auto more = mixedSpecs();
        specs.insert (specs.end(), more.begin(), more.end());
        proc.buildStore (std::move (specs));
        auto& store = proc.store;

        // ValueAttachment: message-thread delivery, latest value wins.
        {
            std::vector<float> seen;
            std::thread::id callbackThread;
            ValueAttachment va (store, "uiZoom", [&] (float v) { seen.push_back (v); callbackThread = std::this_thread::get_id(); });
            va.sendInitialUpdate();
            expectEquals ((int) seen.size(), 1);
            expectEquals (seen.back(), 1.0f);

            std::thread t ([&] { store.set ("uiZoom", 1.2f); store.set ("uiZoom", 1.4f); });
            t.join();
            expectEquals ((int) seen.size(), 1, "not called on the setter thread");
            juce::MessageManager::getInstance()->runDispatchLoopUntil (50);
            expectEquals ((int) seen.size(), 2, "coalesced to one async update");
            expectEquals (seen.back(), 1.4f);
            expect (callbackThread == std::this_thread::get_id());

            va.setAsCompleteGesture (0.8f);
            expectEquals (store.get ("uiZoom"), 0.8f);

            // Destroying with an update pending is safe.
            std::thread t2 ([&] { store.set ("uiZoom", 1.9f); });
            t2.join();
        }
        juce::MessageManager::getInstance()->runDispatchLoopUntil (20);

        // SliderBinding.
        {
            juce::Slider slider;
            SliderBinding binding (store, "cutoff", slider);
            expectWithinAbsoluteError (slider.getMinimum(), 20.0, 1.0e-3);
            expectWithinAbsoluteError (slider.getMaximum(), 20000.0, 1.0e-2);
            expectWithinAbsoluteError (slider.getValue(), 1000.0, 0.01);
            expectWithinAbsoluteError (slider.valueToProportionOfLength (std::sqrt (20.0 * 20000.0)), 0.5, 1.0e-3);
            expectEquals (slider.getTextFromValue (440.0), juce::String ("440.00 Hz"));
            expect (slider.isDoubleClickReturnEnabled());
            expectWithinAbsoluteError (slider.getDoubleClickReturnValue(), 1000.0, 0.01);

            HostListener host;
            proc.addListener (&host);
            slider.setValue (2500.0, juce::sendNotificationSync);
            expectWithinAbsoluteError (store.get ("cutoff"), 2500.0f, 0.01f);
            expectEquals (host.gestureBegins, 1, "programmatic change is a complete gesture");
            expectEquals (host.gestureEnds, 1);

            store.set ("cutoff", 800.0f);
            expectWithinAbsoluteError (slider.getValue(), 800.0, 0.01);
            expectEquals (host.gestureBegins, 1, "store -> slider does not feed back");
            proc.removeListener (&host);
        }

        // Ordered limits on the slider via constrain().
        {
            LinkRule r;
            r.first = "a";
            r.second = "b";
            r.mode = LinkRule::Mode::Ordered;
            r.minGap = 2.0f;
            store.addLink (r);

            juce::Slider slider;
            SliderBinding binding (store, "a", slider);
            slider.setValue (9.5, juce::sendNotificationSync);
            expectEquals ((float) slider.getValue(), 8.0f, "slider limited via constrain");
            expectEquals (store.get ("a"), 8.0f);
            expectEquals (store.get ("b"), 10.0f, "and pushed the other member");

            juce::Slider hiddenSlider;
            SliderBinding hiddenBinding (store, "c", hiddenSlider);
            hiddenSlider.setValue (75.0, juce::sendNotificationSync);
            expectEquals (store.get ("c"), 75.0f, "hidden params bind the same way");
        }

        // ButtonBinding.
        {
            juce::ToggleButton button;
            ButtonBinding binding (store, "bypass", button);
            expect (! button.getToggleState());
            button.setToggleState (true, juce::sendNotificationSync);
            expectEquals (store.get ("bypass"), 1.0f);
            store.set ("bypass", 0.0f);
            expect (! button.getToggleState());
        }

        // ComboBinding.
        {
            juce::ComboBox combo;
            ComboBinding binding (store, "wave", combo);
            expectEquals (combo.getNumItems(), 3);
            expectEquals (combo.getItemText (2), juce::String ("Square"));
            expectEquals (combo.getSelectedId(), 2);
            combo.setSelectedId (3, juce::sendNotificationSync);
            expectEquals (store.get ("wave"), 2.0f);
            store.set ("wave", 0.0f);
            expectEquals (combo.getSelectedId(), 1);
        }
    }
};

static ParamStoreTests paramStoreTests;

} // namespace vae::params::tests
