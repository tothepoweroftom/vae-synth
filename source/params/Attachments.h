#pragma once

#include "params/ParamStore.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

namespace vae::params
{

/**
    Generic message-thread binding between a store parameter and some UI state.
    Works for host-visible and hidden parameters. The store must outlive the attachment.
*/
class ValueAttachment final : private ParamStore::Listener,
                              private juce::AsyncUpdater
{
public:
    /** Registers as a store listener; onChange(raw) is always called on the message thread. */
    ValueAttachment (ParamStore& store, const juce::String& id, std::function<void (float raw)> onChange);
    ~ValueAttachment() override;

    /** Calls onChange synchronously with the current value. */
    void sendInitialUpdate();

    /** begin + set + end gesture in one call. */
    void setAsCompleteGesture (float raw);

    void beginGesture();
    void setDuringGesture (float raw);
    void endGesture();

    /** True while onChange is running (used by bindings to avoid feedback loops). */
    bool isApplyingFromStore() const noexcept   { return applying; }

private:
    void paramChanged (const juce::String& id, float raw) override;
    void handleAsyncUpdate() override;
    void apply (float raw);

    ParamStore& store;
    const juce::String id;
    std::function<void (float)> onChange;
    std::atomic<float> pending { 0.0f };
    bool applying = false;
};

//==============================================================================
/** Binds a juce::Slider: range (incl. skew), text formatting, double-click reset, gestures, Ordered limits. */
class SliderBinding final : private juce::Slider::Listener
{
public:
    SliderBinding (ParamStore& store, const juce::String& id, juce::Slider& slider);
    ~SliderBinding() override;

private:
    void sliderValueChanged (juce::Slider*) override;
    void sliderDragStarted (juce::Slider*) override;
    void sliderDragEnded (juce::Slider*) override;

    juce::Slider& slider;
    ValueAttachment attachment;
    bool dragging = false;
};

//==============================================================================
/** Binds a juce::Button's toggle state to a bool value; a click is a complete gesture. */
class ButtonBinding final : private juce::Button::Listener
{
public:
    ButtonBinding (ParamStore& store, const juce::String& id, juce::Button& button);
    ~ButtonBinding() override;

private:
    void buttonClicked (juce::Button*) override;

    juce::Button& button;
    ValueAttachment attachment;
};

//==============================================================================
/** Binds a juce::ComboBox to a choice index (item id = index + 1); populates items when empty. */
class ComboBinding final : private juce::ComboBox::Listener
{
public:
    ComboBinding (ParamStore& store, const juce::String& id, juce::ComboBox& comboBox);
    ~ComboBinding() override;

private:
    void comboBoxChanged (juce::ComboBox*) override;

    juce::ComboBox& comboBox;
    ValueAttachment attachment;
};

} // namespace vae::params
