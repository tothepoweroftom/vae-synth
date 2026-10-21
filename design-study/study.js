const $ = selector => document.querySelector(selector);
const directions = [
  {id:'precision',name:'Precision',tagline:'An instrument for deliberate exploration.',bg:'#191c1d',accent:'#d4f77d',title:'Find the unfamiliar.',subtitle:'A small movement. A different sound.',field:'LATENT FIELD',description:'Graphite surfaces, acid-green signals and tightly organised controls. A technical instrument that feels at home next to a serious VST synth.',strength:'Clear hierarchy and a restrained field of contours. The Lab feels exploratory without losing the precision of the synth.',risk:'Could feel like familiar audio software. Give the field enough character to make the VAE experience distinctive.'},
  {id:'paper',name:'Field notes',tagline:'A sound laboratory on graph paper.',bg:'#f2f0e7',accent:'#2169a3',title:'A study in sound.',subtitle:'Observe. Move. Listen. Repeat.',field:'FIG. 01 / LATENT SECTION',description:'An off-white laboratory notebook, blue drafting ink, graph paper and fine rules. Exploration is a process of making observations.',strength:'A light alternative to the usual dark plugin. Parameters on the left, the experiment in the middle, the starting idea on the right.',risk:'The scientific drawing must remain a visual metaphor. Contours do not represent measured audio characteristics.'},
  {id:'studio',name:'Studio 76',tagline:'New technology. Familiar hands.',bg:'#d4ccbb',accent:'#c14e2e',title:'Tune into something new.',subtitle:'A modern experiment with a familiar touch.',field:'VECTOR MONITOR',description:'Warm aluminium, dark wood edges, burnt-orange switches and a phosphor monitor. The VAE becomes another tactile piece of studio equipment.',strength:'The strongest family resemblance between Lab and synth. The screen provides a natural home for abstract latent exploration.',risk:'Avoid excessive ornament or faux physical controls. Wood and texture should frame the workflow, not dominate it.'},
  {id:'play',name:'Playground',tagline:'Less laboratory. More possibility.',bg:'#eee8ff',accent:'#fa5b35',title:'Follow your curiosity.',subtitle:'Take a sound somewhere unexpected.',field:'THE PLAY SPACE',description:'Lilac, tangerine and butter-yellow. Big type, rounded controls and overlapping organic shapes make the experiment feel inviting.',strength:'A friendly first encounter with an unfamiliar instrument. Strong visual identity, with the XY pad unmistakably the central action.',risk:'Could imply more musical predictability than the model provides. Keep the controls explicit and the motion purposeful.'},
  {id:'editorial',name:'Quiet form',tagline:'Space to hear your own ideas.',bg:'#f8f7f3',accent:'#ee512d',title:'Between one sound and another.',subtitle:'Explore the space around a single idea.',field:'02 — THE SPACE BETWEEN',description:'Warm white, black ink, serif headlines and a single vermilion accent. Generous negative space lets the field become the main event.',strength:'Calm and unusually human for a technical plugin. Progressive disclosure could let the sound lead and keep model details secondary.',risk:'Large typography costs space in a small plugin window. The final UI would need a compact mode with the same clarity.'},
  {id:'orbital',name:'Orbital',tagline:'Navigate a small universe of sound.',bg:'#111321',accent:'#b2a7ff',title:'Somewhere you haven’t heard.',subtitle:'An anchor. Two dimensions. A new direction.',field:'LOCAL ORBIT',description:'Midnight blue, luminous lilac and fine orbital lines. A spatial instrument built around a floating point and a sense of distance.',strength:'Makes anchoring, movement and returning home intuitive. Offers a visual signature for the VAE without imitating a modular synth.',risk:'Avoid suggesting a global map of every sound. This is a local two-dimensional slice, and the orbital lines are illustrative.'}
];
// Palette studies reuse the original layouts and field drawings.
for (const baseId of ['precision', 'orbital']) {
  const base = directions.find(direction => direction.id === baseId);
  directions.push({
    ...base,
    id: baseId + '-yellow',
    theme: baseId,
    palette: 'yellow',
    name: base.name + ' / Yellow',
    bg: '#121212',
    accent: '#ffff07',
    tagline: 'Carbon black. Electric yellow.',
    description: baseId === 'precision'
      ? 'The Precision layout in #121212 carbon black and #ffff07 electric yellow. Neutral grey surfaces keep the technical hierarchy; yellow marks the field, active controls and signal.'
      : 'The Orbital layout in #121212 carbon black and #ffff07 electric yellow. Yellow orbit lines and a softly glowing cursor replace the lilac, with neutral surfaces around the field.',
    risk: 'Electric yellow has strong visual weight. Keep large surfaces neutral and reserve the full accent for active controls and the current point.'
  });
}
const categories=['Bass','Lead','Pad','Keys','Pluck','Synth','Atmosphere','Orchestral','Unknown'];
let selected='precision', category='Pad', point={x:0,y:0}, pointNumber=1, view='lab', shortlist=[];
try{shortlist=JSON.parse(localStorage.getItem('vae-design-shortlist')||'[]');if(!Array.isArray(shortlist))shortlist=[];}catch{}
let shortlistOnly=false;
function toast(message){$('#toast').textContent=message;$('#toast').classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('#toast').classList.remove('show'),2600);}
function drawCards(){
  $('#directions').innerHTML=directions.map((d,i)=>`<button class="direction-card ${d.id===selected?'selected':''}" data-id="${d.id}" data-theme="${d.theme||d.id}" aria-pressed="${d.id===selected}" ${shortlistOnly&&!shortlist.includes(d.id)?'hidden':''}><div class="miniature" style="--mini-bg:${d.bg};--mini-color:${d.accent}"><span class="mini-sidebar"></span><span class="mini-field"></span><span class="mini-controls"></span></div><div class="card-caption"><span>${d.name}</span><small>${shortlist.includes(d.id)?'★':String(i+1).padStart(2,'0')}</small></div></button>`).join('');
  document.querySelectorAll('.direction-card').forEach(b=>b.onclick=()=>setDirection(b.dataset.id));
  $('#shortlist-count').textContent=shortlist.length;
  $('#saved-list').textContent=shortlist.length?'SHORTLIST: '+directions.filter(d=>shortlist.includes(d.id)).map(d=>d.name).join(' / '):'';
  $('#save-direction').textContent=shortlist.includes(selected)?'★ Shortlisted':'☆ Shortlist';
  $('#save-direction').classList.toggle('saved',shortlist.includes(selected));
}
function setDirection(id){
  const d=directions.find(d=>d.id===id)||directions[0];selected=d.id;
  $('#instrument').dataset.theme=d.theme||d.id;
  $('#instrument').dataset.palette=d.palette||'original';
  $('#direction-number').textContent=String(directions.indexOf(d)+1).padStart(2,'0')+' / '+String(directions.length).padStart(2,'0');
  for(const [key,value] of Object.entries({'direction-name':d.name,'direction-tagline':d.tagline,'lab-title':d.title,'lab-subtitle':d.subtitle,'map-label':d.field,'direction-description':d.description,'direction-strength':d.strength,'direction-risk':d.risk}))$('#'+key).textContent=value;
  history.replaceState(null,'','#'+d.id);drawCards();drawField();
}
function drawField(){
  const theme = $('#instrument').dataset.theme;
  let art='<defs><radialGradient id="glow"><stop stop-color="currentColor" stop-opacity=".22"/><stop offset="1" stop-color="currentColor" stop-opacity="0"/></radialGradient></defs>';
  if(theme==='play'){
    art+='<path d="M-80 330 C-20 60 200 -70 335 120 S650 160 780 30 L760 480 L-80 480Z" fill="#9373ce"/><path d="M-50 380 C80 130 200 200 300 250 S430 40 610 170 S700 420 800 470Z" fill="#c5b3ed"/><path d="M140 440 C230 130 340 170 410 270 S550 280 650 440" fill="#ebc1bb"/>';
    for(let i=0;i<8;i++)art+=`<path d="M${50+i*25} 420 C${80+i*20} 180 ${340+i*20} ${100+i*12} ${450+i*26} 390" stroke="#ffef8b" stroke-opacity=".28" fill="none"/>`;
  }else if(theme==='orbital'){
    art+='<ellipse cx="350" cy="200" rx="285" ry="190" fill="url(#glow)"/>';
    for(let i=0;i<14;i++)art+=`<ellipse cx="350" cy="200" rx="${80+i*15}" ry="${28+i*11}" fill="none" stroke="currentColor" stroke-opacity="${.1+i*.015}" transform="rotate(${i*9-65} 350 200)"/>`;
    for(let i=0;i<65;i++){const x=(i*137.51)%700,y=(i*81.73)%400;art+=`<circle cx="${x}" cy="${y}" r="${i%7===0?1.5:.7}" fill="currentColor" opacity=".4"/>`;}
  }else{
    if(theme==='precision'||theme==='studio')for(let x=20;x<700;x+=24)for(let y=10;y<400;y+=24)art+=`<circle cx="${x}" cy="${y}" r=".6" fill="currentColor" opacity=".16"/>`;
    const count=theme==='editorial'?18:22;
    for(let i=0;i<count;i++){
      let path='';const scale=1+i*.1;
      for(let n=0;n<=120;n++){const t=n/120*Math.PI*2,r=1+.12*Math.sin(3*t+.4)+.08*Math.cos(5*t);const x=355+Math.cos(t)*75*scale*r,y=208+Math.sin(t)*43*scale*r;path+=(n?'L':'M')+x.toFixed(1)+' '+y.toFixed(1);}
      art+=`<path d="${path}Z" fill="none" stroke="currentColor" stroke-width="${i%5===0?1.2:.65}" opacity="${theme==='editorial'?.28:.15+(count-i)*.014}" transform="rotate(-24 355 208)"/>`;
    }
    art+='<path d="M350 20V380M30 200H670" fill="none" stroke="currentColor" stroke-opacity=".16" stroke-dasharray="3 6"/>';
    if(theme==='paper')art+='<text x="500" y="95" fill="currentColor" opacity=".5" font-size="9" font-family="monospace">local section / z</text>';
  }
  $('#field-art').innerHTML=art;
}
function drawCategories(){
  $('.category-options').innerHTML=categories.map(c=>`<button class="${c===category?'active':''}" aria-pressed="${c===category}" data-category="${c}">${c}</button>`).join('');
  document.querySelectorAll('[data-category]').forEach(b=>b.onclick=()=>{category=b.dataset.category;pointNumber++;updateSound();drawCategories();$('#prompt-explanation').textContent=category+' selected manually. The point is a visual mockup.';});
}
function updateSound(){
  $('#sound-name').textContent=category+' / '+String(pointNumber).padStart(3,'0');
  let wave='';for(let i=0;i<90;i++){const h=3+Math.abs(Math.sin(i*.51+pointNumber)*Math.cos(i*.13))*40*(.3+.7*Math.sin(i/90*Math.PI));wave+=`<path d="M${i*2.9} ${27-h/2}v${h}" stroke="currentColor" stroke-width="1.5"/>`;}
  $('.waveform').innerHTML=wave;
}
function updatePoint(x,y,commit=false){
  point={x:Math.max(-1,Math.min(1,x)),y:Math.max(-1,Math.min(1,y))};
  $('#cursor').style.left=(50+point.x*42)+'%';$('#cursor').style.top=(50-point.y*40)+'%';
  const signed=n=>(n>=0?'+':'')+n.toFixed(2),r=Number($('#radius').value);
  $('#coordinates').textContent='X '+signed(point.x*r)+'  Y '+signed(point.y*r);
  $('#point-status').textContent=point.x===0&&point.y===0?'At the anchor':commit?'Study point updated':'Exploring…';
  if(commit){pointNumber++;updateSound();$('#point-description').textContent='Offset '+signed(point.x*r)+' / '+signed(point.y*r);}
}
let dragging=false;
function pointerPosition(e){const r=$('#map').getBoundingClientRect();updatePoint(((e.clientX-r.left)/r.width-.5)/.42,(.5-(e.clientY-r.top)/r.height)/.4);}
$('#map').addEventListener('pointerdown',e=>{dragging=true;$('#map').setPointerCapture(e.pointerId);pointerPosition(e);});
$('#map').addEventListener('pointermove',e=>{if(dragging)pointerPosition(e);});
$('#map').addEventListener('pointerup',()=>{if(dragging){dragging=false;updatePoint(point.x,point.y,true);}});
$('#map').addEventListener('pointercancel',()=>{dragging=false;});
$('#map').addEventListener('keydown',e=>{const step=e.shiftKey?.1:.025;const deltas={ArrowLeft:[-step,0],ArrowRight:[step,0],ArrowUp:[0,step],ArrowDown:[0,-step]};if(deltas[e.key]){e.preventDefault();updatePoint(point.x+deltas[e.key][0],point.y+deltas[e.key][1],true);}if(e.key==='Home'){e.preventDefault();updatePoint(0,0,true);}});
$('#centre').onclick=()=>updatePoint(0,0,true);
$('#random-anchor').onclick=()=>{updatePoint(0,0,true);$('#point-description').textContent='New anchor · '+category.toLowerCase();toast('New illustrative anchor · '+category);};
$('#generate').onclick=()=>{
  const text=$('#prompt').value.toLowerCase();
  const vocabulary=[['synth strings','Orchestral'],['e-piano','Keys'],['supersaw','Lead'],['atmosphere','Atmosphere'],['orchestral','Orchestral'],['strings','Orchestral'],['pluck','Pluck'],['piano','Keys'],['reese','Bass'],['808','Bass'],['bass','Bass'],['lead','Lead'],['pad','Pad'],['keys','Keys'],['synth','Synth'],['drone','Atmosphere'],['bell','Keys']];
  const match=vocabulary.find(([word])=>new RegExp('\\b'+word+'\\b').test(text));category=match?match[1]:'Unknown';
  $('#prompt-explanation').textContent=match?'“'+match[0]+'” selects '+category+'. Adjectives are not interpreted.':'No category noun found. Using Unknown.';
  drawCategories();updatePoint(0,0,true);toast('Visual category demo · '+category);
};
$('#prompt').addEventListener('keydown',e=>{if(e.key==='Enter')$('#generate').click();});
for(const id of ['x-axis','y-axis']){$('#'+id).innerHTML=Array.from({length:32},(_,i)=>`<option value="${i+1}">Dim ${String(i+1).padStart(2,'0')}</option>`).join('');}
$('#x-axis').value=2;$('#y-axis').value=7;
function updateAxes(changed){const other=changed==='x-axis'?'y-axis':'x-axis';if($('#'+changed).value===$('#'+other).value)$('#'+other).value=Number($('#'+changed).value)%32+1;$('.x-dimension').textContent=$('#x-axis').value.padStart(2,'0');$('.y-dimension').textContent=$('#y-axis').value.padStart(2,'0');$('.map-heading b').textContent=$('#x-axis').value.padStart(2,'0')+' / '+$('#y-axis').value.padStart(2,'0');updatePoint(0,0,true);}
$('#x-axis').onchange=()=>updateAxes('x-axis');$('#y-axis').onchange=()=>updateAxes('y-axis');
$('#radius').oninput=()=>{$('#radius-value').textContent=Number($('#radius').value).toFixed(2);updatePoint(point.x,point.y);};
$('#radius').onchange=()=>updatePoint(point.x,point.y,true);
$('#export').onclick=()=>{const data={kind:'vae-ui-design-study',notice:'Visual mockup only. Not a plugin preset or model output.',direction:selected,category,axes:[Number($('#x-axis').value),Number($('#y-axis').value)],radius:Number($('#radius').value),point};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='vae-design-point.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Study point exported');};
$('#save-direction').onclick=()=>{shortlist=shortlist.includes(selected)?shortlist.filter(id=>id!==selected):[...shortlist,selected];try{localStorage.setItem('vae-design-shortlist',JSON.stringify(shortlist));}catch{}if(!shortlist.length)shortlistOnly=false;drawCards();};
$('#shortlist-toggle').onclick=()=>{if(!shortlist.length){toast('Use ☆ Shortlist to save a direction first.');return;}shortlistOnly=!shortlistOnly;$('#shortlist-toggle').setAttribute('aria-pressed',String(shortlistOnly));if(shortlistOnly&&!shortlist.includes(selected))setDirection(shortlist[0]);drawCards();};
$('#focus-toggle').onclick=()=>{const focused=document.body.classList.toggle('focused');$('#focus-toggle').textContent=focused?'↙ Exit focus':'↗ Focus view';};
$('#info-toggle').onclick=()=>{$('#prototype-info').hidden=!$('#prototype-info').hidden;};
document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('#prototype-info').hidden=true;if(document.body.classList.contains('focused'))$('#focus-toggle').click();}});
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view;$('#lab-view').hidden=view!=='lab';$('#synth-view').hidden=view!=='synth';document.querySelectorAll('[data-view]').forEach(t=>{t.classList.toggle('active',t===b);t.setAttribute('aria-pressed',String(t===b));});});
const modules=[['OSCILLATOR 01','SAW',['Octave','Detune','Level'],[50,24,78]],['OSCILLATOR 02','TRIANGLE',['Octave','Detune','Level'],[50,42,62]],['FILTER 01','LOW PASS',['Cutoff','Resonance','Drive'],[65,28,18]],['AMP ENVELOPE','ADSR',['Attack','Decay','Sustain','Release'],[25,40,75,60]],['MODULATION','LFO 01',['Rate','Depth','Phase'],[35,20,45]],['SPACE','REVERB',['Size','Decay','Mix'],[70,65,30]]];
function drawModules(){
  $('#synth-modules').innerHTML=modules.map(([title,type,labels,values],m)=>{let path='';for(let i=0;i<200;i++){let y=m===0?35-(i%30):m===1?10+Math.abs((i%50)-25):m===2?32-24*Math.exp(-i/70):m===3?(i<35?40-i:i<65?5+(i-35)*.5:i<145?20:20+(i-145)*.36):22+Math.sin(i*.1)*15*(m===5?Math.exp(-i/100):1);path+=(i?'L':'M')+(i*2)+' '+y;}return `<section class="synth-module"><div class="module-title">${title}<span>${type}</span></div><svg class="osc-wave" viewBox="0 0 400 50" preserveAspectRatio="none"><path d="${path}" stroke="currentColor" fill="none" stroke-width="1.5"/></svg><div class="knob-row">${labels.map((label,i)=>`<label class="knob-label"><div class="knob" style="--angle:${values[i]*2.7}deg"></div>${label}<input type="range" min="0" max="100" value="${values[i]}" aria-label="${title} ${label}"><output class="knob-value">${values[i]}%</output></label>`).join('')}</div></section>`;}).join('');
  document.querySelectorAll('.knob-label input').forEach(input=>input.oninput=()=>{input.parentElement.querySelector('.knob').style.setProperty('--angle',Number(input.value)*2.7+'deg');input.parentElement.querySelector('output').textContent=input.value+'%';});
}
$('#init-synth').onclick=()=>{drawModules();toast('Visual synth controls reset');};
const whiteNames=['C','D','E','F','G','A','B'];
for(let i=0;i<21;i++){const note=whiteNames[i%7]+(3+Math.floor(i/7));const key=document.createElement('button');key.className='key';key.setAttribute('aria-label',note+' visual key');key.innerHTML=i%7===0?'<span>'+note+'</span>':'';key.onclick=()=>playKey(key,note);$('#keyboard').append(key);if(![2,6].includes(i%7)&&i<20){const black=document.createElement('button');black.className='key black';black.style.left=((i+1)/21*100)+'%';black.setAttribute('aria-label',note+' sharp visual key');black.onclick=()=>playKey(black,note+'♯');$('#keyboard').append(black);}}
function playKey(key,note){key.classList.add('played');$('#keyboard-note').textContent=note+' / VISUAL ONLY';setTimeout(()=>key.classList.remove('played'),300);}
drawModules();drawCategories();updateSound();setDirection(location.hash.slice(1));

window.addEventListener('hashchange', () => setDirection(location.hash.slice(1)));
