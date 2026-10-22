#pragma once
#include "ui/PitcherControls.h"

// Study-only interaction experiment. The plugin continues to use AxisRow.
class StudyAxisRow final : public juce::Component, private juce::Timer
{
public:
    StudyAxisRow(juce::String name, int i, std::function<void(int,int)> assign,
              std::function<void(int,float)> move) : index(i)
    {
        label.setText(name.toUpperCase(),juce::dontSendNotification);
        label.setFont(PitcherLookAndFeel::monoFontFor(*this,10));
        addAndMakeVisible(label); addAndMakeVisible(valueLabel);
        valueLabel.setFont(PitcherLookAndFeel::monoFontFor(*this,10));
        valueLabel.setJustificationType(juce::Justification::centredRight);
        for(int a=0;a<3;++a) {
            auto& b=buttons[(size_t)a]; addAndMakeVisible(b);
            b.setButtonText(juce::String::charToString(char('X'+a)));
            b.setTooltip("Assign "+name+" to "+b.getButtonText());
            b.onClick=[assign,i,a]{assign(i,a);};
        }
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setRange(0,1,.001); slider.setTextBoxStyle(juce::Slider::NoTextBox,false,0,0);
        slider.setDoubleClickReturnValue(true,.5);
        slider.setTooltip("Normalized position / Shift: fine adjustment / double-click: centre");
        slider.onDragStart=[this]{if(onGestureStart)onGestureStart(index);};
        slider.onValueChange=[this,move]{if(!slider.isMouseButtonDown() && onGestureStart)onGestureStart(index);move(index,(float)slider.getValue());};
        addAndMakeVisible(slider);
    }
    std::function<void(int)> onGestureStart;
    void setReducedMotion(bool value) { reduced=value; }
    void update(const std::array<int,3>& axes,float value)
    {
        bool next=false;
        for(int a=0;a<3;++a) { bool on=axes[(size_t)a]==index;
            buttons[(size_t)a].setToggleState(on,juce::dontSendNotification); next|=on; }
        if(next && (!assigned || std::abs(value-(float)slider.getValue())>.002f)) {
            strength=1; started=juce::Time::getMillisecondCounterHiRes(); startTimerHz(30);
        }
        assigned=next; slider.setVisible(next); valueLabel.setVisible(next);
        slider.setValue(value,juce::dontSendNotification);
        valueLabel.setText(juce::String(value*100,1)+"%",juce::dontSendNotification);
        setSize(getWidth(),next?70:44); resized();
    }
    void resized() override
    {
        auto r=getLocalBounds().reduced(0,8);auto top=r.removeFromTop(25);
        auto actions=top.removeFromRight(87);label.setBounds(top);
        for(auto& b:buttons)b.setBounds(actions.removeFromLeft(29).reduced(2,0));
        auto bottom=r.removeFromTop(24);valueLabel.setBounds(bottom.removeFromRight(52));slider.setBounds(bottom);
    }
    void paint(juce::Graphics& g) override
    {
        const juce::Colour green(0xff70ff63);
        if(assigned) { g.setColour(green.withAlpha(.025f+strength*.08f));g.fillRect(getLocalBounds());
            g.setColour(green.withAlpha(.4f+strength*.6f));g.fillRect(0,8,2,getHeight()-16); }
        g.setColour(juce::Colour(0xff4b504b));g.drawHorizontalLine(getHeight()-1,0,(float)getWidth());
    }
private:
    void timerCallback() override
    {
        strength=reduced?0:juce::jmax(0.f,1.f-(float)((juce::Time::getMillisecondCounterHiRes()-started)/220));
        repaint();if(strength==0||!isShowing())stopTimer();
    }
    juce::Label label,valueLabel;
    std::array<juce::TextButton,3> buttons;
    juce::Slider slider;
    int index;bool assigned=false,reduced=false;float strength=0;double started=0;
};
