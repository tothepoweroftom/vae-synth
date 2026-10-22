#include "StudyView.h"
#include "BinaryData.h"
#include "ui/PitcherLayout.h"
#include <numeric>

namespace {
const juce::Colour bg(0xff252525), ink(0xfff0f0ec), muted(0xffa5aaa5), line(0xff4b504b), green(0xff70ff63);
const std::array<juce::Colour,8> colours {juce::Colour(0xff70bcff),juce::Colour(0xffffad66),juce::Colour(0xff86eeac),juce::Colour(0xffffe078),juce::Colour(0xffff8fc7),juce::Colour(0xff63e0d5),juce::Colour(0xffb6a0ff),juce::Colour(0xffff918c)};
const std::array<juce::String,8> labels {"Bass","Lead","Pad","Keys","Pluck","Synth","Atmosphere","Orchestral"};
juce::Font font(float size) {return juce::Font(juce::FontOptions(size));}
}
StudyView::StudyView()
    :look(juce::Typeface::createSystemTypefaceFor(BinaryData::InterRegular_ttf,BinaryData::InterRegular_ttfSize),
          juce::Typeface::createSystemTypefaceFor(BinaryData::InterBold_ttf,BinaryData::InterBold_ttfSize),
          juce::Typeface::createSystemTypefaceFor(BinaryData::SpaceMonoRegular_ttf,BinaryData::SpaceMonoRegular_ttfSize))
{
    setLookAndFeel(&look);
    if(!data.load(BinaryData::study_json,(size_t)BinaryData::study_jsonSize)){status.setText("Could not load study data",juce::dontSendNotification);}
    latent=data.initialLatent;category=data.initialCategory;
    addAndMakeVisible(orbit);orbit.setLabelFont(look.monoFont(10));addAndMakeVisible(spectrum);
    orbit.onPositionChanged=[this](auto p){data.moveTo(latent,assigned,p);refreshPosition();};
    for(int i=0;i<8;++i){auto& b=categories[(size_t)i];addAndMakeVisible(b);b.setButtonText(labels[(size_t)i]);b.getProperties().set("category",true);b.onClick=[this,i]{highlighted=highlighted==i?-1:i;orbit.setHighlight(highlighted);for(int j=0;j<8;++j)categories[(size_t)j].setToggleState(j==highlighted,juce::dontSendNotification);repaint();};}
    for(auto* b:{&save,&axesTab,&presetsTab,&zoomOut,&zoomIn,&reset,&send,&tune})addAndMakeVisible(b);
    save.setButtonText("Save favourite");save.getProperties().set("save",true);save.setTooltip("Save current sound with a name");save.onClick=[this]{saveNamed();};
    axesTab.getProperties().set("tab",true);presetsTab.getProperties().set("tab",true);
    axesTab.onClick=[this]{selectTab(false);};presetsTab.onClick=[this]{selectTab(true);};
    zoomOut.onClick=[this]{orbit.zoom(1.f/1.15f);};zoomIn.onClick=[this]{orbit.zoom(1.15f);};reset.onClick=[this]{orbit.resetCamera();};
    addAndMakeVisible(prompt);prompt.setTextToShowWhenEmpty("Describe a sound...",muted);prompt.setFont(look.monoFont(14.4f));
    prompt.setMultiLine(false);prompt.setJustification(juce::Justification::centredLeft);
    prompt.setBorder(juce::BorderSize<int>(1));prompt.setIndents(12,0);
    send.onClick=[this]{auto text=prompt.getText().toLowerCase();int found=-1;for(int i=0;i<8;++i)if(text.contains(labels[(size_t)i].toLowerCase())){found=i;break;}
        if(found<0){status.setText("Try bass, lead, pad, keys, pluck or synth - category words only.",juce::dontSendNotification);return;}
        for(size_t i=0;i<data.presets.size();++i)if(data.labels[i]==found){latent=data.presets[i];category=found;break;}
        status.setText(labels[(size_t)found]+" selected / category matching only",juce::dontSendNotification);refreshPosition();};
    prompt.onReturnKey=send.onClick;addAndMakeVisible(status);status.setFont(font(10));status.setColour(juce::Label::textColourId,muted);
    axisScroll.setViewedComponent(&axisContent,false);axisScroll.setScrollBarsShown(true,false);addAndMakeVisible(axisScroll);
    presetScroll.setViewedComponent(&presetContent,false);presetScroll.setScrollBarsShown(true,false);addChildComponent(presetScroll);
    for(size_t i=0;i<data.axes.size();++i){auto row=std::make_unique<AxisRow>(data.axes[i].name,(int)i,[this](int index,int axis){
        auto previous=assigned[(size_t)axis];for(int a=0;a<3;++a)if(assigned[(size_t)a]==index)assigned[(size_t)a]=previous;
        assigned[(size_t)axis]=index;refreshCloud();refreshPosition();},[this](int index,float value){
        auto pos=data.project(latent,assigned);for(int a=0;a<3;++a)if(assigned[(size_t)a]==index)pos[(size_t)a]=value;
        data.moveTo(latent,assigned,pos);orbit.setPosition(pos);refreshPosition();});axisContent.addAndMakeVisible(*row);rows.push_back(std::move(row));}
    tune.onClick=[this]{showTune();};
    keyboard.setAvailableRange(48,84);keyboard.setBlackNoteLengthProportion(.61f);keyboard.setKeyPressBaseOctave(4);keyboard.setLowestVisibleKey(48);keyboard.setScrollButtonsVisible(false);
    keyboard.setColour(juce::MidiKeyboardComponent::whiteNoteColourId,juce::Colour(0xffb7bcb6));keyboard.setColour(juce::MidiKeyboardComponent::blackNoteColourId,bg);
    keyboard.setColour(juce::MidiKeyboardComponent::keyDownOverlayColourId,green.withAlpha(.8f));keyboard.setColour(juce::MidiKeyboardComponent::shadowColourId,juce::Colours::transparentBlack);
    addAndMakeVisible(keyboard);keyboard.setTitle("Visual keyboard - no audio engine in this study");
    setSize(1380,800);refreshCloud();refreshPosition();selectTab(false);startTimerHz(20);
}
StudyView::~StudyView(){stopTimer();setLookAndFeel(nullptr);}
void StudyView::refreshCloud()
{
    std::vector<PresetPoint> points;points.reserve(data.presets.size());for(size_t i=0;i<data.presets.size();++i)points.push_back({data.project(data.presets[i],assigned),data.labels[i]});orbit.setPoints(std::move(points));
}
void StudyView::refreshPosition()
{
    orbit.setPosition(data.project(latent,assigned));dirty=true;refreshRows();
    // Proximity summary is explicitly labelled; it is not the web demo's parameter forest.
    matches.fill(0);std::vector<std::pair<float,int>> distances;distances.reserve(data.presets.size());
    for(size_t i=0;i<data.presets.size();++i){float d=0;for(size_t k=0;k<32;++k){float delta=latent[k]-data.presets[i][k];d+=delta*delta;}distances.emplace_back(d,data.labels[i]);}
    std::sort(distances.begin(),distances.end());float sum=0;for(size_t i=0;i<std::min(size_t(32),distances.size());++i){auto [d,c]=distances[i];if(c>=0&&c<8){float w=1/(.5f+d);matches[(size_t)c]+=w;sum+=w;}}
    if(sum>0)for(auto& m:matches)m/=sum;for(int i=0;i<8;++i)categories[(size_t)i].getProperties().set("percent",juce::String(juce::roundToInt(matches[(size_t)i]*100))+"%");repaint();
}
void StudyView::timerCallback(){if(dirty){dirty=false;spectrum.image=data.spectrogram(latent,category);spectrum.setTooltip(data.soundTooltip()+"\nModel prediction, not an audio measurement.");spectrum.repaint();}}
void StudyView::refreshRows()
{
    int y=0;for(size_t i=0;i<rows.size();++i){const auto& a=data.axes[i];float dot=std::inner_product(latent.begin(),latent.end(),a.direction.begin(),0.f);float value=.5f+(dot-a.mean)/(5*a.sd);rows[i]->update(assigned,juce::jlimit(0.f,1.f,value));rows[i]->setBounds(0,y,juce::jmax(1,axisScroll.getWidth()-12),rows[i]->getHeight());y+=rows[i]->getHeight();}axisContent.setSize(juce::jmax(1,axisScroll.getWidth()-12),y);
}
void StudyView::selectTab(bool presets)
{
    presetsVisible=presets;axesTab.setToggleState(!presets,juce::dontSendNotification);presetsTab.setToggleState(presets,juce::dontSendNotification);axisScroll.setVisible(!presets);presetScroll.setVisible(presets);tune.setVisible(presets);resized();
}
void StudyView::resized()
{
    const auto layout=PitcherLayout::calculate(getLocalBounds());left=layout.left;centre=layout.centre;right=layout.right;
    int y=left.getY()+96;for(auto& b:categories){b.setBounds(left.getX(),y,left.getWidth()-20,27);y+=40;}save.setBounds(left.getX(),y+12,38,34);spectrum.setBounds(left.getX(),left.getBottom()-140,left.getWidth()-20,140);
    keyboard.setBounds(layout.keyboard);keyboard.setKeyWidth(layout.keyboard.getWidth()/22.f);
    status.setBounds(layout.status);send.setBounds(layout.submit);prompt.setBounds(layout.prompt);orbit.setBounds(layout.orbit);
    auto middle=layout.orbit;
    auto controls=middle.reduced(12).removeFromTop(27);reset.setBounds(controls.removeFromRight(74));controls.removeFromRight(5);zoomIn.setBounds(controls.removeFromRight(27));controls.removeFromRight(5);zoomOut.setBounds(controls.removeFromRight(27));
    auto rail=right;rail.removeFromLeft(20);auto tabs=rail.removeFromTop(38);axesTab.setBounds(tabs.removeFromLeft(tabs.getWidth()/2));presetsTab.setBounds(tabs);rail.removeFromTop(10);axisScroll.setBounds(rail);tune.setBounds(rail.removeFromBottom(34));rail.removeFromBottom(12);presetScroll.setBounds(rail);refreshRows();rebuildFavourites();
}
void StudyView::paint(juce::Graphics& g)
{
    g.fillAll(bg);g.setColour(line);g.drawVerticalLine(left.getRight(),(float)left.getY(),(float)left.getBottom());g.drawVerticalLine(right.getX(),(float)right.getY(),(float)right.getBottom());
    auto x=(float)left.getX(),y=(float)left.getY();g.setColour(green);juce::Path logo;logo.addArc(x+2,y+2,29,29,.45f,5.8f,true);logo.startNewSubPath(x+3,y+23);logo.cubicTo(x+10,y+3,x+24,y+1,x+33,y+7);logo.startNewSubPath(x+5,y+29);logo.cubicTo(x+16,y+10,x+27,y+9,x+34,y+13);g.strokePath(logo,juce::PathStrokeType(2));
    g.setColour(ink);g.setFont(juce::Font(juce::FontOptions(look.bold)).withHeight(27));g.drawText("pitcher",left.getX()+43,left.getY()-2,180,40,juce::Justification::centredLeft);
    g.setFont(look.monoFont(15).boldened());g.drawText("CATEGORY MATCH",left.getX(),left.getY()+53,left.getWidth(),22,juce::Justification::centredLeft);g.setColour(muted);g.setFont(look.monoFont(9));g.drawText("Nearby presets / study approximation",left.getX(),left.getY()+76,left.getWidth(),16,juce::Justification::centredLeft);
    for(int i=0;i<8;++i){auto r=categories[(size_t)i].getBounds();g.setColour(line);g.fillRect(r.getX()+6,r.getBottom()+3,r.getWidth()-12,3);g.setColour(colours[(size_t)i]);g.fillRect((float)r.getX()+6,(float)r.getBottom()+3,(r.getWidth()-12)*matches[(size_t)i],3.f);}
    if(presetsVisible&&favourites.empty()){g.setColour(muted);g.setFont(font(12));g.drawFittedText("Save a sound to keep it here.\nStudy favourites last for this session.",presetScroll.getBounds().reduced(10),juce::Justification::topLeft,3);}
}
void StudyView::saveNamed()
{
    auto* dialog=new juce::AlertWindow("Save favourite","Give this sound a name.",juce::MessageBoxIconType::NoIcon);dialog->addTextEditor("name",labels[(size_t)juce::jlimit(0,7,category)]+" "+juce::String((int)favourites.size()+1));dialog->addButton("Save",1,juce::KeyPress(juce::KeyPress::returnKey));dialog->addButton("Cancel",0,juce::KeyPress(juce::KeyPress::escapeKey));
    auto snapshot=Favourite{"",latent,assigned,category};juce::Component::SafePointer<StudyView> safe(this);
    dialog->enterModalState(true,juce::ModalCallbackFunction::create([safe,dialog,snapshot](int result)mutable{if(safe!=nullptr&&result==1){auto name=dialog->getTextEditorContents("name").trim();if(name.isNotEmpty()){snapshot.name=name;safe->favourites.push_back(snapshot);safe->rebuildFavourites();safe->selectTab(true);}}}),true);
}
void StudyView::rebuildFavourites()
{
    favouriteButtons.clear();int y=0;for(size_t i=0;i<favourites.size();++i){auto b=std::make_unique<juce::TextButton>(favourites[i].name);b->onClick=[this,i]{latent=favourites[i].latent;assigned=favourites[i].axes;category=favourites[i].category;refreshCloud();refreshPosition();status.setText("Favourite restored",juce::dontSendNotification);};presetContent.addAndMakeVisible(*b);b->setBounds(0,y,juce::jmax(1,presetScroll.getWidth()-12),36);y+=44;favouriteButtons.push_back(std::move(b));}presetContent.setSize(juce::jmax(1,presetScroll.getWidth()-12),y);tune.setEnabled(!favourites.empty());repaint();
}
void StudyView::showTune()
{
    juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon,"Shape my space - concept", "Your "+juce::String((int)favourites.size())+" saved sounds could guide a personal space.\n\nThis native study does not train or modify the model. The detailed selection/strength mockup remains in the browser study.","Close");
}
