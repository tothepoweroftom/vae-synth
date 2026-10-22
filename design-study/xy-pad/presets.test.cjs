const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const code=fs.readFileSync(`${__dirname}/app.js`,'utf8');
const start=code.indexOf('function capturePatch()'),end=code.indexOf('window.XypadPresets=',start);
const state={mode:'knob',base:0,vals:[.2,NaN,.8],axes:{x:'a',y:'b',z:'c'}};
const lat={base:0,z:Float32Array.from([.25,-.75]),y:1,axes:{x:'u',y:'v',z:null},lock:{act:Uint8Array.from([1,0,1]),bools:Uint8Array.from([1]),cats:[1]}};
let notesOff=0,syncs=0;
const fixture={state,lat,LT:{run:'test-model'},NK:3,NB:1,ZD:2,NC:2,CAT_SIZES:[2],L:['Bass','Pad'],
 predict:()=>[.2,.8],argmax:()=>1,BASES:[{}],LBASES:[{}],byName:{a:{},b:{},c:{}},AXI:{u:{},v:{},w:{}},
 noteOff(){notesOff++;},catSel:{},lockBtn:{setAttribute(){}},latSyncPos(){syncs++;},
 setMode(mode){state.mode=mode;},Float32Array,Uint8Array};
vm.createContext(fixture);vm.runInContext(code.slice(start,end),fixture);
const snap=fixture.capturePatch();
assert.equal(snap.values[1],null);assert.equal(snap.label,'Pad');
state.vals=[0,0,0];state.axes={x:'b',y:'c',z:null};lat.z=Float32Array.from([4,5]);lat.lock=null;
fixture.restorePatch(snap);
assert.equal(state.vals[0],.2);assert(Number.isNaN(state.vals[1]));assert.equal(state.vals[2],.8);
assert.equal(state.axes.x,'a');assert.deepEqual(Array.from(lat.z),[.25,-.75]);
assert.deepEqual(Array.from(lat.lock.act),[1,0,1]);assert.equal(lat.lock.cats[0],1);assert.equal(notesOff,1);
state.mode='latent';const latentSnap=fixture.capturePatch();lat.z.fill(0);lat.y=0;
fixture.restorePatch(latentSnap);assert.equal(state.mode,'latent');assert.equal(lat.y,1);assert.equal(lat.z[1],-.75);
for(const corrupt of [p=>p.model='other',p=>p.latent.z=[1],p=>p.latent.category=99,p=>p.axes.y=p.axes.x,p=>p.latent.lock.cats=[3]]){
 const bad=JSON.parse(JSON.stringify(snap));corrupt(bad);const before=JSON.stringify(fixture.capturePatch()),beforeNotes=notesOff;
 assert.throws(()=>fixture.restorePatch(bad));assert.equal(JSON.stringify(fixture.capturePatch()),before);assert.equal(notesOff,beforeNotes);
}
console.log('Favourite snapshots: knob/latent restore, inactive values, locked structure and rejection before mutation passed.');
