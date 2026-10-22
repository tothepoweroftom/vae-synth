const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = {window:{devicePixelRatio:1}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(`${__dirname}/xyz.js`, 'utf8'),context);
const R=context.window.XyzPad, width=800,height=500,start={x:.5,y:.5,z:.5};
const close=(a,b)=>assert(Math.abs(a-b)<1e-8,`${a} != ${b}`);
for(let i=0;i<72;i++){
  const camera={yaw:i*Math.PI/18,pitch:i*Math.PI/23,zoom:.8};
  const p=R.orbitProject(start.x,start.y,start.z,camera,width,height);
  const moved=R.orbitDrag(start,2,-3,camera,width,height,'plane');
  const q=R.orbitProject(moved.x,moved.y,moved.z,camera,width,height);
  close(q.x-p.x,2);close(q.y-p.y,-3);close(q.depth,p.depth);
  const depth=R.orbitDrag(start,0,-3,camera,width,height,'depth');
  const d=R.orbitProject(depth.x,depth.y,depth.z,camera,width,height);
  close(d.x,p.x);close(d.y,p.y);assert(d.depth>p.depth);
}
const camera=R.createCamera();
for(const axis of ['x','y','z']){
  const moved=R.orbitDrag(start,10,8,camera,width,height,axis);
  for(const other of ['x','y','z'].filter(a=>a!==axis))close(moved[other],start[other]);
  assert.notEqual(moved[axis],start[axis]);
}
const endOn=R.orbitDrag(start,30,30,{yaw:0,pitch:0,zoom:1},width,height,'z');
for(const axis of ['x','y','z'])close(endOn[axis],start[axis]);
const bounded=R.orbitDrag(start,99999,-99999,camera,width,height,'plane');
for(const value of Object.values(bounded))assert(value>=0&&value<=1);
const center=R.orbitProject(.5,.5,.5,camera,width,height);
assert.equal(R.hitOrbit(start,camera,width,height,center.x,center.y,false),'handle');
for(const {axis,tip} of R.gizmo(start,camera,width,height)){
  assert.equal(R.hitOrbit(start,camera,width,height,tip.x,tip.y,true),axis);
  assert.equal(R.hitOrbit(start,camera,width,height,tip.x,tip.y,false),null);
}
assert.equal(R.hitOrbit(start,camera,width,height,0,0,true),null);
// Exercise the actual interaction branch: camera drag must not decode or mutate sound.
const app=fs.readFileSync(`${__dirname}/app.js`,'utf8');
const fn=app.slice(app.indexOf('function padSet(e)'),app.indexOf('function nearestDot(e)'));
let decodes=0,renders=0;const point={...start};
const state={padView:'orbit',downAt:[0,0],cameraStart:camera,orbitCamera:{...camera},orbitGesture:'camera',handleStart:{...start},
  XyzPad:R,drawPad(){},cv:{getBoundingClientRect:()=>({width,height})},hasZ:()=>true,P_:()=>point,isLat:()=>true,
  latApplyPos(){},latDecode(){decodes++;},refreshSliders(){},render(){renders++;}};
vm.createContext(state);vm.runInContext(fn,state);vm.runInContext('padSet({clientX:10,clientY:10})',state);
assert.deepEqual(point,start);assert.equal(decodes,0);
state.orbitGesture='plane';vm.runInContext('padSet({clientX:10,clientY:10})',state);
assert.notDeepEqual(point,start);assert.equal(decodes,1);assert.equal(renders,1);
console.log('Orbit handle: 72 plane/depth round trips, axis constraints, bounds, hit targets and camera/edit isolation passed.');
