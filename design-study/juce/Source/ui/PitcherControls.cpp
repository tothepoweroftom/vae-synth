#include "PitcherControls.h"
namespace {
const juce::Colour bg(0xff252525), ink(0xfff0f0ec), muted(0xffa5aaa5), line(0xff4b504b), green(0xff70ff63);
juce::Font font(float size){return juce::Font(juce::FontOptions(size));}
}
PitcherLookAndFeel::PitcherLookAndFeel(juce::Typeface::Ptr bodyFont,juce::Typeface::Ptr boldFont,juce::Typeface::Ptr monoFont)
    :body(std::move(bodyFont)),bold(std::move(boldFont)),mono(std::move(monoFont))
{
    if(body!=nullptr)setDefaultSansSerifTypeface(body);
    setColour(juce::TextButton::textColourOffId,muted); setColour(juce::TextButton::textColourOnId,bg);
    setColour(juce::TextButton::buttonOnColourId,green);
    setColour(juce::Label::textColourId,ink);
    setColour(juce::Slider::thumbColourId,green); setColour(juce::Slider::trackColourId,green);
    setColour(juce::Slider::backgroundColourId,line);
    setColour(juce::TextEditor::backgroundColourId,bg); setColour(juce::TextEditor::textColourId,ink);
    setColour(juce::TextEditor::outlineColourId,line); setColour(juce::TextEditor::focusedOutlineColourId,green);
    setColour(juce::TooltipWindow::backgroundColourId,bg);setColour(juce::TooltipWindow::textColourId,ink);
    setColour(juce::TooltipWindow::outlineColourId,line);
    setColour(juce::ScrollBar::thumbColourId,line);
}
juce::Font PitcherLookAndFeel::monoFont(float size) const
{
    return mono!=nullptr?juce::Font(juce::FontOptions(mono)).withHeight(size):juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(),size,juce::Font::plain));
}
juce::Font PitcherLookAndFeel::monoFontFor(juce::Component& component,float size)
{
    if(auto* look=dynamic_cast<PitcherLookAndFeel*>(&component.getLookAndFeel()))return look->monoFont(size);
    return juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(),size,juce::Font::plain));
}
juce::Font PitcherLookAndFeel::getTextButtonFont(juce::TextButton& button,int) { return button.getProperties().contains("category")?font(11):monoFont(11); }
void PitcherLookAndFeel::drawButtonBackground(juce::Graphics& g,juce::Button& b,const juce::Colour&,bool over,bool down)
{
    if(b.getProperties().contains("tab")){g.setColour(bg);g.fillRect(b.getLocalBounds());g.setColour(b.getToggleState()?green:line);g.fillRect(0,b.getHeight()-2,b.getWidth(),2);return;}
    g.setColour(b.getToggleState()?green:over||down?green.withAlpha(.12f):bg);g.fillRect(b.getLocalBounds());
    g.setColour(b.getToggleState()?green:b.getProperties().contains("category")?bg:line);g.drawRect(b.getLocalBounds(),1);
}
void PitcherLookAndFeel::drawButtonText(juce::Graphics& g,juce::TextButton& b,bool over,bool down)
{
    if(b.getProperties().contains("tab")){g.setFont(monoFont(11));g.setColour(b.getToggleState()?green:muted);g.drawText(b.getButtonText(),b.getLocalBounds().reduced(4,0),juce::Justification::centredLeft);return;}
    if(b.getProperties().contains("category")){
        g.setFont(font(11));g.setColour(b.getToggleState()?bg:ink);
        g.drawText(b.getButtonText(),b.getLocalBounds().reduced(7,0),juce::Justification::centredLeft);
        g.drawText(b.getProperties()["percent"].toString(),b.getLocalBounds().reduced(7,0),juce::Justification::centredRight);return;
    }
    if(b.getProperties().contains("save")){
        g.setColour(green);juce::Path p;p.startNewSubPath(10,7);p.lineTo(24,7);p.lineTo(28,11);p.lineTo(28,25);p.lineTo(10,25);p.closeSubPath();
        p.addRectangle(14,7,9,6);p.addRectangle(14,17,10,8);g.strokePath(p,juce::PathStrokeType(1.5f));return;
    }
    LookAndFeel_V4::drawButtonText(g,b,over,down);
}
void Spectrum::paint(juce::Graphics& g)
{
    auto r=getLocalBounds();g.setColour(muted);g.setFont(PitcherLookAndFeel::monoFontFor(*this,10));g.drawText("EXPECTED SOUND",r.removeFromTop(22),juce::Justification::centredLeft);
    auto cap=r.removeFromBottom(18);g.drawText("0 s",cap,juce::Justification::centredLeft);g.drawText("2 s",cap,juce::Justification::centred);g.drawText("6 s",cap,juce::Justification::centredRight);
    if(image.isValid())g.drawImage(image,r.toFloat());g.setColour(line);g.drawRect(r);
}
AxisRow::AxisRow(juce::String name,int i,std::function<void(int,int)> assign,std::function<void(int,float)> move):index(i)
{
    label.setText(name.toUpperCase(),juce::dontSendNotification);label.setFont(PitcherLookAndFeel::monoFontFor(*this,10));addAndMakeVisible(label);
    for(int a=0;a<3;++a){auto& b=buttons[(size_t)a];b.setButtonText(juce::String::charToString(char('X'+a)));addAndMakeVisible(b);b.onClick=[assign,i,a]{assign(i,a);};}
    slider.setSliderStyle(juce::Slider::LinearHorizontal);slider.setTextBoxStyle(juce::Slider::NoTextBox,false,0,0);slider.setRange(0,1,.001);addAndMakeVisible(slider);
    slider.onDragStart=[this]{if(onGestureStart)onGestureStart(index);};
    slider.onDragEnd=[this]{if(onGestureEnd)onGestureEnd(index);};
    slider.onValueChange=[this,move]{move(index,(float)slider.getValue());};
}
void AxisRow::lookAndFeelChanged(){label.setFont(PitcherLookAndFeel::monoFontFor(*this,10));}
void AxisRow::update(const std::array<int,3>& axes,float value)
{
    assigned=false;for(int a=0;a<3;++a){bool on=axes[(size_t)a]==index;buttons[(size_t)a].setToggleState(on,juce::dontSendNotification);assigned|=on;}
    slider.setVisible(assigned);slider.setValue(value,juce::dontSendNotification);setSize(getWidth(),assigned?66:44);resized();
}
void AxisRow::resized()
{
    auto r=getLocalBounds().reduced(0,8);auto top=r.removeFromTop(25);auto buttonsArea=top.removeFromRight(87);
    label.setBounds(top);for(auto& b:buttons){b.setBounds(buttonsArea.removeFromLeft(29).reduced(2,0));}
    slider.setBounds(r.removeFromTop(24));
}
void AxisRow::paint(juce::Graphics& g){g.setColour(line);g.drawHorizontalLine(getHeight()-1,0,(float)getWidth());}

void PitcherLookAndFeel::drawLinearSlider(juce::Graphics& g,int x,int y,int width,int height,float pos,float minPos,float maxPos,juce::Slider::SliderStyle style,juce::Slider& slider)
{
    if(style!=juce::Slider::LinearHorizontal && style!=juce::Slider::LinearVertical){LookAndFeel_V4::drawLinearSlider(g,x,y,width,height,pos,minPos,maxPos,style,slider);return;}
    const bool horizontal=style==juce::Slider::LinearHorizontal;
    const float cx=x+width*.5f,cy=y+height*.5f;
    g.setOpacity(slider.isEnabled()?1.f:.4f);
    g.setColour(slider.findColour(juce::Slider::backgroundColourId));
    g.fillRect(horizontal?juce::Rectangle<float>((float)x,cy-1,(float)width,2):juce::Rectangle<float>(cx-1,(float)y,2,(float)height));
    g.setColour(slider.findColour(juce::Slider::trackColourId));
    g.fillRect(horizontal?juce::Rectangle<float>((float)x,cy-1,juce::jmax(0.f,pos-x),2):juce::Rectangle<float>(cx-1,pos,2,juce::jmax(0.f,y+height-pos)));
    g.setColour(slider.findColour(juce::Slider::thumbColourId));
    g.fillRect(horizontal?juce::Rectangle<float>(pos-5,cy-5,10,10):juce::Rectangle<float>(cx-5,pos-5,10,10));
}

namespace {
juce::TextLayout tooltipLayout(const juce::String& text,juce::Font textFont,float width)
{
    juce::AttributedString content;content.setJustification(juce::Justification::topLeft);
    content.append(text,textFont,ink);
    juce::TextLayout layout;layout.createLayout(content,width);return layout;
}
}
juce::Rectangle<int> PitcherLookAndFeel::getTooltipBounds(const juce::String& text,juce::Point<int> point,juce::Rectangle<int> parent)
{
    const int width=juce::jmin(360,parent.getWidth());
    const auto layout=tooltipLayout(text,monoFont(12),(float)juce::jmax(1,width-24));
    const int height=juce::roundToInt(std::ceil(layout.getHeight()))+24;
    return juce::Rectangle<int>(point.x>parent.getCentreX()?point.x-width-12:point.x+24,
        point.y>parent.getCentreY()?point.y-height-6:point.y+6,width,height).constrainedWithin(parent);
}
void PitcherLookAndFeel::drawTooltip(juce::Graphics& g,const juce::String& text,int width,int height)
{
    g.fillAll(bg);g.setColour(line);g.drawRect(0,0,width,height);
    auto layout=tooltipLayout(text,monoFont(12),(float)juce::jmax(1,width-24));
    layout.draw(g,juce::Rectangle<float>(12,12,(float)width-24,(float)height-24));
}
