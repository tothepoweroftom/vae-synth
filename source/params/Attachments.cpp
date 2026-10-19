#include "params/Attachments.h"

namespace vae::params
{

ValueAttachment::ValueAttachment (ParamStore& s, const juce::String& paramId, std::function<void (float)> callback)
    : store (s), id (paramId), onChange (std::move (callback))
{
    store.addListener (id, this);
}

ValueAttachment::~ValueAttachment()
{
    // removeListener waits for any callback running on another thread to finish.
    store.removeListener (id, this);
    cancelPendingUpdate();
}

void ValueAttachment::sendInitialUpdate()
{
    apply (store.get (id));
}

void ValueAttachment::setAsCompleteGesture (float raw)
{
    if (applying)
        return;

    store.beginGesture (id);
    store.set (id, raw, Notify::host);
    store.endGesture (id);
}

void ValueAttachment::beginGesture()
{
    if (! applying)
        store.beginGesture (id);
}

void ValueAttachment::setDuringGesture (float raw)
{
    if (! applying)
        store.set (id, raw, Notify::host);
}

void ValueAttachment::endGesture()
{
    if (! applying)
        store.endGesture (id);
}

void ValueAttachment::paramChanged (const juce::String&, float raw)
{
    pending.store (raw);

    if (juce::MessageManager::existsAndIsCurrentThread())
    {
        cancelPendingUpdate();
        apply (raw);
    }
    else
    {
        triggerAsyncUpdate(); // latest value wins
    }
}

void ValueAttachment::handleAsyncUpdate()
{
    apply (pending.load());
}

void ValueAttachment::apply (float raw)
{
    if (applying || ! onChange)
        return;

    const juce::ScopedValueSetter<bool> svs (applying, true);
    onChange (raw);
}

//==============================================================================
namespace
{
    const ParamSpec* findSpec (ParamStore& store, const juce::String& id)
    {
        if (! store.indexOf (id).isValid())
        {
            jassertfalse; // unknown id, or the store has not been built yet
            return nullptr;
        }

        return &store.spec (id);
    }
}

SliderBinding::SliderBinding (ParamStore& store, const juce::String& id, juce::Slider& s)
    : slider (s),
      attachment (store, id, [this] (float raw) { slider.setValue ((double) raw, juce::dontSendNotification); })
{
    if (auto* spec = findSpec (store, id))
    {
        const auto r = spec->range;
        auto* storePtr = &store;

        juce::NormalisableRange<double> sliderRange {
            (double) r.start, (double) r.end,
            [r] (double, double, double n) { return (double) r.convertFrom0to1 ((float) n); },
            [r] (double, double, double v) { return (double) r.convertTo0to1 ((float) v); },
            [r, storePtr, id] (double, double, double v)
            {
                return (double) storePtr->constrain (id, r.snapToLegalValue ((float) v));
            }
        };
        sliderRange.interval = (double) r.interval;
        slider.setNormalisableRange (sliderRange);

        const auto specCopy = *spec;
        slider.textFromValueFunction = [specCopy] (double v) { return specCopy.valueToText ((float) v); };
        slider.valueFromTextFunction = [specCopy] (const juce::String& t) { return (double) specCopy.textToValue (t); };
        slider.setDoubleClickReturnValue (true, (double) spec->defaultValue);
        slider.updateText();
    }

    slider.addListener (this);
    attachment.sendInitialUpdate();
}

SliderBinding::~SliderBinding()
{
    slider.removeListener (this);
}

void SliderBinding::sliderValueChanged (juce::Slider*)
{
    if (attachment.isApplyingFromStore())
        return;

    const auto v = (float) slider.getValue();

    if (dragging)
        attachment.setDuringGesture (v);
    else
        attachment.setAsCompleteGesture (v);
}

void SliderBinding::sliderDragStarted (juce::Slider*)
{
    dragging = true;
    attachment.beginGesture();
}

void SliderBinding::sliderDragEnded (juce::Slider*)
{
    attachment.endGesture();
    dragging = false;
}

//==============================================================================
ButtonBinding::ButtonBinding (ParamStore& store, const juce::String& id, juce::Button& b)
    : button (b),
      attachment (store, id, [this] (float raw) { button.setToggleState (raw >= 0.5f, juce::dontSendNotification); })
{
    findSpec (store, id);
    button.addListener (this);
    attachment.sendInitialUpdate();
}

ButtonBinding::~ButtonBinding()
{
    button.removeListener (this);
}

void ButtonBinding::buttonClicked (juce::Button*)
{
    if (! attachment.isApplyingFromStore())
        attachment.setAsCompleteGesture (button.getToggleState() ? 1.0f : 0.0f);
}

//==============================================================================
ComboBinding::ComboBinding (ParamStore& store, const juce::String& id, juce::ComboBox& c)
    : comboBox (c),
      attachment (store, id, [this] (float raw) { comboBox.setSelectedId (juce::roundToInt (raw) + 1, juce::dontSendNotification); })
{
    if (auto* spec = findSpec (store, id))
        if (comboBox.getNumItems() == 0)
            comboBox.addItemList (spec->choices, 1);

    comboBox.addListener (this);
    attachment.sendInitialUpdate();
}

ComboBinding::~ComboBinding()
{
    comboBox.removeListener (this);
}

void ComboBinding::comboBoxChanged (juce::ComboBox*)
{
    if (attachment.isApplyingFromStore())
        return;

    if (const auto selected = comboBox.getSelectedId(); selected > 0)
        attachment.setAsCompleteGesture ((float) (selected - 1));
}

} // namespace vae::params
