/* Oblique projection keeps XY dragging exact at a fixed depth. No WebGL dependency. */
window.XyzPad = (() => {
  const clamp = n => Math.max(0, Math.min(1, n));
  const project = (x, y, z) => ({x: .12 + .63*x + .18*z, y: .83 - .64*y - .17*z});
  const unproject = (x, y, z) => ({x: clamp((x-.12-.18*clamp(z))/.63), y: clamp((.83-y-.17*clamp(z))/.64)});
  const createCamera = () => ({yaw: -.55, pitch: .35, zoom: 1});
  function orbitProject(x, y, z, camera, width, height) {
    const dx=x-.5, dy=y-.5, dz=z-.5;
    const cy=Math.cos(camera.yaw), sy=Math.sin(camera.yaw), cp=Math.cos(camera.pitch), sp=Math.sin(camera.pitch);
    const rx=cy*dx+sy*dz, rz=-sy*dx+cy*dz;
    const ry=cp*dy-sp*rz, depth=sp*dy+cp*rz;
    const scale=Math.min(width,height)*.57*camera.zoom;
    return {x:width/2+rx*scale,y:height/2-ry*scale,depth};
  }
  // Inverse camera rotation maps screen-space movement into the XYZ volume.
  function orbitDrag(start, dx, dy, camera, width, height, mode='plane') {
    const scale=Math.min(width,height)*.57*camera.zoom;
    if(scale<=0)return {...start};
    const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch);
    let delta;
    if(['x','y','z'].includes(mode)) {
      const a=orbitProject(0,0,0,camera,width,height),v={x:0,y:0,z:0};v[mode]=1;
      const b=orbitProject(v.x,v.y,v.z,camera,width,height),vx=b.x-a.x,vy=b.y-a.y,n=vx*vx+vy*vy;
      // End-on arrows have no usable screen direction; orbit to expose them.
      if(n<scale*scale*.0025)return {...start};
      delta={x:0,y:0,z:0};delta[mode]=(dx*vx+dy*vy)/n;
    } else {
      const rx=mode==='depth'?0:dx/scale,ry=mode==='depth'?0:-dy/scale,d=mode==='depth'?-dy/scale:0;
      const y=cp*ry+sp*d,rz=-sp*ry+cp*d;
      delta={x:cy*rx-sy*rz,y,z:sy*rx+cy*rz};
    }
    return {x:clamp(start.x+delta.x),y:clamp(start.y+delta.y),z:clamp(start.z+delta.z)};
  }
  function gizmo(point,camera,width,height) {
    const origin=orbitProject(point.x,point.y,point.z,camera,width,height);
    return ['x','y','z'].map(axis=>{
      const end={...point};end[axis]+=(point[axis]>.75?-.22:.22);
      const tip=orbitProject(end.x,end.y,end.z,camera,width,height);
      return {axis,origin,tip};
    }).filter(({origin:a,tip:b})=>Math.hypot(a.x-b.x,a.y-b.y)>10);
  }
  function hitOrbit(point,camera,width,height,x,y,selected) {
    const p=orbitProject(point.x,point.y,point.z,camera,width,height);
    if(Math.hypot(x-p.x,y-p.y)<=14)return 'handle';
    if(!selected)return null;
    let best=null,distance=10;
    for(const {axis,origin:a,tip:b} of gizmo(point,camera,width,height)) {
      const vx=b.x-a.x,vy=b.y-a.y,t=Math.max(0,Math.min(1,((x-a.x)*vx+(y-a.y)*vy)/(vx*vx+vy*vy)));
      const d=Math.hypot(x-a.x-t*vx,y-a.y-t*vy);
      if(d<distance){distance=d;best=axis;}
    }
    return best;
  }
  function draw(ctx, width, height, {point, dots, accent, ink, line, colors, category, showDots, camera=null, selected=false}) {
    ctx.clearRect(0,0,width,height);
    const dpr = window.devicePixelRatio || 1;
    const screen = (x,y,z) => {if(camera){const p=orbitProject(x,y,z,camera,width,height);return [p.x,p.y];}const p=project(x,y,z);return [p.x*width,p.y*height];};
    function path(points, color, weight=1, close=false) {
      ctx.beginPath();points.forEach((p,i)=>{const s=screen(...p);if(i)ctx.lineTo(...s);else ctx.moveTo(...s);});
      if(close)ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=weight*dpr;ctx.stroke();
    }
    const p={x:clamp(point.x),y:clamp(point.y),z:clamp(point.z)};
    // Back wall and depth edges establish the entire available volume.
    for (const z of [0,1]) path([[0,0,z],[1,0,z],[1,1,z],[0,1,z]],line,1,true);
    for (const x of [0,1])for(const y of [0,1])path([[x,y,0],[x,y,1]],line);
    const plane=[[0,0,p.z],[1,0,p.z],[1,1,p.z],[0,1,p.z]];
    ctx.beginPath();plane.forEach((a,i)=>{const s=screen(...a);i?ctx.lineTo(...s):ctx.moveTo(...s);});ctx.closePath();
    ctx.fillStyle=accent+'0b';ctx.fill();path(plane,accent+'55',1,true);
    for(let i=1;i<4;i++){const q=i/4;path([[q,0,p.z],[q,1,p.z]],line);path([[0,q,p.z],[1,q,p.z]],line);}
    // Presets have real coordinates along all three selected directions.
    if(showDots){
      const visible=dots.filter(d=>d.slice(0,3).every(v=>Number.isFinite(v)&&v>=0&&v<=1)).sort((a,b)=>camera ? orbitProject(...a.slice(0,3),camera,width,height).depth-orbitProject(...b.slice(0,3),camera,width,height).depth : b[2]-a[2]);
      for(const [x,y,z,label] of visible){const s=screen(x,y,z);ctx.globalAlpha=(category===null||label===category?.65:.2)*(camera?1:Math.abs(z-p.z)<.1?1:.4);ctx.fillStyle=colors[label]||ink;ctx.beginPath();ctx.arc(...s,1.7*dpr,0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;
    }
    ctx.setLineDash([3*dpr,4*dpr]);
    path([[p.x,p.y,0],[p.x,p.y,1]],accent+'55');
    path([[p.x,0,p.z],[p.x,p.y,p.z],[0,p.y,p.z]],accent+'66');ctx.setLineDash([]);
    const cursor=screen(p.x,p.y,p.z);ctx.fillStyle='#121212';ctx.beginPath();ctx.arc(...cursor,8*dpr,0,Math.PI*2);ctx.fill();ctx.strokeStyle=accent;ctx.lineWidth=2*dpr;ctx.stroke();ctx.fillStyle=accent;ctx.beginPath();ctx.arc(...cursor,2.5*dpr,0,Math.PI*2);ctx.fill();
    ctx.font=`${9*dpr}px monospace`;ctx.fillStyle=ink;ctx.textAlign='center';
    for(const [name,coords] of [['X',[.5,-.07,0]],['Y',[-.06,.5,0]],['Z',[1.07,0,.5]]])ctx.fillText(name,...screen(...coords));
    if(camera && selected){
      // Use CSS-pixel geometry for the same arrow lengths and hit targets at every DPR.
      for(const {axis,origin:a,tip:b} of gizmo(p,camera,width/dpr,height/dpr)) {
        const angle=Math.atan2(b.y-a.y,b.x-a.x);
        ctx.strokeStyle=accent;ctx.lineWidth=1.5*dpr;ctx.beginPath();ctx.moveTo(a.x*dpr,a.y*dpr);ctx.lineTo(b.x*dpr,b.y*dpr);ctx.stroke();
        ctx.fillStyle=accent;ctx.beginPath();ctx.moveTo(b.x*dpr,b.y*dpr);
        ctx.lineTo((b.x-7*Math.cos(angle-.45))*dpr,(b.y-7*Math.sin(angle-.45))*dpr);
        ctx.lineTo((b.x-7*Math.cos(angle+.45))*dpr,(b.y-7*Math.sin(angle+.45))*dpr);ctx.closePath();ctx.fill();
        ctx.fillText(axis.toUpperCase(),(b.x+12*Math.cos(angle))*dpr,(b.y+12*Math.sin(angle)+3)*dpr);
      }
    }
    ctx.textAlign='left';ctx.fillStyle=accent;ctx.fillText(camera ? 'ORBIT · '+Math.round(camera.zoom*100)+'%' : 'Z '+Math.round(p.z*100)+'%',12*dpr,20*dpr);
  }
  return {project,unproject,createCamera,orbitProject,orbitDrag,gizmo,hitOrbit,draw};
})();
