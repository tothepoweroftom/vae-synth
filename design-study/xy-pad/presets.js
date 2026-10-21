(() => {
  const key='precision-xypad-favorites-v1', api=window.XypadPresets;
  const heart=document.getElementById('favorite-patch'),status=document.getElementById('favorite-status');
  const list=document.getElementById('favorite-list');
  let favorites=[], pendingSave=null, appliedPreview=null;
  const nameDialog=document.getElementById('name-dialog'),nameInput=document.getElementById('favorite-name'),nameError=document.getElementById('name-error');
  try {
    const stored=JSON.parse(localStorage.getItem(key)||'[]');
    if(!Array.isArray(stored))throw Error();
    favorites=stored.filter(f=>f&&typeof f.id==='string'&&typeof f.name==='string'&&f.patch);
  } catch {status.textContent='Could not read saved favourites.';}
  const signature=p=>JSON.stringify(p.mode==='latent'
    ? {mode:p.mode,z:p.latent.z,category:p.latent.category,lock:p.latent.lock}
    : {mode:p.mode,values:p.values});
  function activeFavorite(){const patch=api.capture(),current=signature(patch);return favorites.find(f=>{try{return f.patch.model===patch.model&&signature(f.patch)===current;}catch{return false;}});}
  // Commit storage before updating the UI; quota/permission failures remain visible.
  function persist(next){
    try{localStorage.setItem(key,JSON.stringify(next));favorites=next;draw();updateHeart();return true;}
    catch{status.textContent='Could not save. Browser storage may be full or disabled.';return false;}
  }
  function updateHeart(){
    let active;
    try{active=activeFavorite();}catch{return;}
    heart.textContent=active?'♥':'♡';heart.setAttribute('aria-pressed',String(Boolean(active)));
    heart.setAttribute('aria-label',active?'Rename current favourite':'Save current patch as favourite');
    heart.title=active?'Rename favourite':'Save favourite';
    list.querySelectorAll('[data-favorite-id]').forEach(row=>row.classList.toggle('current',row.dataset.favoriteId===active?.id));
  }
  function draw(){
    document.getElementById('favorite-count').textContent=favorites.length;
    list.replaceChildren();
    document.getElementById('open-tuning').disabled=!favorites.length;
    if(appliedPreview && appliedPreview.ids.some(id=>!favorites.some(f=>f.id===id)))resetSpace();
    if(!favorites.length){const p=document.createElement('p');p.className='preset-empty';p.textContent='Found a sound you like? Tap ♡ to keep it here.';list.append(p);return;}
    for(const favorite of favorites){
      const row=document.createElement('div');row.className='favorite-row';row.dataset.favoriteId=favorite.id;
      const load=document.createElement('button');load.className='load-favorite';load.type='button';
      const title=document.createElement('strong');title.textContent=favorite.name;
      const subtitle=document.createElement('span');subtitle.textContent=(favorite.patch.label||'Sound')+' · '+(favorite.patch.mode==='latent'?'Latent':'Knob');
      load.append(title,subtitle);load.setAttribute('aria-label','Load '+favorite.name);
      load.onclick=()=>{try{api.restore(favorite.patch);const input=document.getElementById('chat-prompt');input.value=favorite.prompt||'';input.dispatchEvent(new Event('input'));status.textContent='Loaded '+favorite.name;document.getElementById('prompt-status').textContent='Favourite restored.';updateHeart();}catch(error){status.textContent=error.message;}};
      const remove=document.createElement('button');remove.className='remove-favorite';remove.textContent='♥';remove.type='button';remove.setAttribute('aria-label','Remove '+favorite.name+' from favourites');
      remove.onclick=()=>{if(persist(favorites.filter(f=>f.id!==favorite.id)))status.textContent='Favourite removed.';};
      const rename=document.createElement('button');rename.className='rename-favorite';rename.type='button';rename.textContent='✎';rename.setAttribute('aria-label','Rename '+favorite.name);rename.onclick=()=>openName(favorite);
      row.append(load,rename,remove);list.append(row);
    }
  }
  function openName(existing) {
    const patch=existing?.patch||api.capture(),prompt=existing?.prompt??document.getElementById('chat-prompt').value.trim();
    pendingSave=existing?{...existing}:{id:crypto.randomUUID(),name:'',prompt,patch};
    nameInput.value=existing?.name||(prompt?prompt.slice(0,70):`${patch.label} ${favorites.length+1}`);
    document.getElementById('name-title').textContent=existing?'Rename favourite':'Save favourite';
    nameError.textContent='';nameDialog.showModal();nameInput.focus();nameInput.select();
  }
  heart.onclick=()=>openName(activeFavorite());
  document.getElementById('cancel-name').onclick=()=>nameDialog.close();
  nameDialog.addEventListener('close',()=>{pendingSave=null;});
  document.getElementById('name-form').onsubmit=event=>{
    event.preventDefault();const name=nameInput.value.trim();
    if(!name){nameError.textContent='Give this sound a name.';nameInput.focus();return;}
    if(!pendingSave)return;
    const saved={...pendingSave,name},exists=favorites.some(f=>f.id===saved.id);
    const next=exists?favorites.map(f=>f.id===saved.id?saved:f):[...favorites,saved];
    if(persist(next)){nameDialog.close();status.textContent=(exists?'Renamed ':'Saved ')+name;}
    else nameError.textContent='Could not save. Check browser storage and try again.';
  };

  // A visual prototype of preference-guided exploration. No model is modified.
  const tuneDialog=document.getElementById('tune-dialog'),strength=document.getElementById('tune-strength');
  let selectedIds=new Set();
  function resetSpace(){
    appliedPreview=null;document.getElementById('space-state').textContent='Choose favourites to guide your space.';
    document.getElementById('reset-space').hidden=true;
  }
  function updatePreview(){
    const selected=favorites.filter(f=>selectedIds.has(f.id));
    const amount=Number(strength.value)/100;
    document.getElementById('tune-strength-value').textContent=strength.value+'%';
    document.getElementById('apply-tuning').disabled=!selected.length;
    document.getElementById('tune-summary').textContent=selected.length?`${selected.length} favourite${selected.length===1?'':'s'} · ${Math.round(amount*100)}% influence`:'Choose at least one favourite.';
    const labels=[...new Set(selected.map(f=>f.patch.label))];
    const clusters=Math.max(1,Math.min(labels.length,4));
    let art='<rect x="1" y="1" width="234" height="238" rx="3" fill="#121212" stroke="#363632"/><rect x="265" y="1" width="234" height="238" rx="3" fill="#121212" stroke="#363632"/>';
    for(let i=0;i<90;i++){
      const x=20+(i*97.31)%195,y=20+(i*61.73)%195;
      const group=i%clusters,angle=group/clusters*Math.PI*2;
      const cx=117+Math.cos(angle)*50,cy=120+Math.sin(angle)*50;
      const blend=selected.length?amount*.8:0;
      const px=x*(1-blend)+cx*blend,py=y*(1-blend)+cy*blend;
      art+=`<circle cx="${x}" cy="${y}" r="2" fill="#86867b"/><circle cx="${px+264}" cy="${py}" r="2" fill="${selected.length?'#ffff07':'#86867b'}" opacity=".75"/>`;
    }
    document.getElementById('tune-map').innerHTML=art;
  }
  document.getElementById('open-tuning').onclick=()=>{
    selectedIds=new Set(appliedPreview?.ids||favorites.map(f=>f.id));strength.value=appliedPreview?.strength??60;
    const container=document.getElementById('tune-favorites');container.replaceChildren();
    for(const favorite of favorites){
      const label=document.createElement('label'),check=document.createElement('input'),text=document.createElement('span');
      check.type='checkbox';check.checked=selectedIds.has(favorite.id);text.textContent=favorite.name;
      check.onchange=()=>{check.checked?selectedIds.add(favorite.id):selectedIds.delete(favorite.id);updatePreview();};
      label.append(check,text);container.append(label);
    }
    updatePreview();tuneDialog.showModal();
  };
  strength.oninput=updatePreview;
  document.getElementById('close-tuning').onclick=()=>tuneDialog.close();
  document.getElementById('cancel-tuning').onclick=()=>tuneDialog.close();
  document.getElementById('apply-tuning').onclick=()=>{
    if(!selectedIds.size)return;
    appliedPreview={ids:[...selectedIds],strength:Number(strength.value)};
    document.getElementById('space-state').textContent=`Personal space preview · ${selectedIds.size} favourites · ${strength.value}% influence. Sound unchanged.`;
    document.getElementById('reset-space').hidden=false;tuneDialog.close();
  };
  document.getElementById('reset-space').onclick=resetSpace;
  const axes=document.getElementById('axes-tab'),presets=document.getElementById('presets-tab');
  function tab(showPresets){
    for(const [button,active] of [[axes,!showPresets],[presets,showPresets]]){button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;}
    document.getElementById('axes-content').hidden=showPresets;document.getElementById('presets-content').hidden=!showPresets;
  }
  axes.onclick=()=>tab(false);presets.onclick=()=>tab(true);
  for(const button of [axes,presets])button.onkeydown=e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const next=e.key==='Home'?false:e.key==='End'?true:button===axes;tab(next);(next?presets:axes).focus();}};
  // Resize/camera work need not rebuild the list; patch changes update the saved indicator only.
  window.addEventListener('xypad-patch-change',updateHeart);
  draw();updateHeart();
})();
