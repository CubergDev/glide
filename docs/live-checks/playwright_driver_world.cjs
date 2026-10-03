// Pure offline objects. No Playwright import, browser, sockets, input or capture.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const driver = fs.readFileSync(process.argv[2], 'utf8');
const quantity = Number(process.argv[3]);
let allowed = true, pendingCreate = null;
const writes = [], all = [];
const context = {
  pages:()=>all.filter(p=>!p.closed),
  newPage:async()=>{
    if(pendingCreate)await pendingCreate;
    return makePage('about:blank');
  },
  newCDPSession:async()=>{throw Error('CDP is unavailable in this browser engine');},
};
const makePage = address => {
  const p = {address,closed:false,value:'',documentId:0,context:()=>context,url:()=>p.address,isClosed:()=>p.closed,
    goto:async(url,options)=>{assert.equal(options.waitUntil,'commit');writes.push({method:'Page.navigate',params:{url},p});p.address=url;p.documentId++;return {url:()=>p.address};},
    evaluate:async expr=>{assert.equal(typeof expr,'string');return {url:p.address,document_id:String(p.documentId),ready:true};},
    keyboard:{
      press:async key=>writes.push({method:'press',params:{key},p}),
      insertText:async text=>{writes.push({method:'Input.insertText',params:{text},p});p.value=text;},
      down:async key=>{
        writes.push({method:'Input.dispatchKeyEvent',params:{type:'keyDown',key},p});
        if(key==='Fail'){allowed=false;throw Error('Simulated interruption after keyDown');}
      },
      up:async key=>{writes.push({method:'Input.dispatchKeyEvent',params:{type:'keyUp',key},p});},
    },
    mouse:{
      move:async(x,y)=>writes.push({method:'move',params:{x,y},p}),
      down:async params=>writes.push({method:'mouseDown',params,p}),
      up:async params=>writes.push({method:'mouseUp',params,p}),
    },
    bringToFront:async()=>{},close:async()=>{p.closed=true;}};
  all.push(p);return p;
};
const first = makePage('https://unseen.test/start');
const run = async(method,params={},target='',deadline=Date.now()+5000) => {
  const request={method,params,target,deadline,gate:'http://127.0.0.1/fixture'};
  const code=driver.replace('__GLIDE_REQUEST__',JSON.stringify(request));
  const scope={AbortSignal,fetch:async()=>({ok:true,json:async()=>({allowed})})};
  return vm.runInNewContext(`(${code})`,scope)(first);
};
(async()=>{
  const initial=(await run('Target.getTargets')).targetInfos[0].targetId;
  const created=[];
  for(let n=0;n<quantity;n++)created.push((await run('Target.createTarget',{url:`https://unseen.test/${n}?q=粵語`})).targetId);
  assert.equal(new Set(created).size,quantity);
  let targets=(await run('Target.getTargets')).targetInfos;
  assert.equal(targets.length,quantity+1);
  assert.equal(targets[0].targetId,initial);
  all.reverse();
  assert.equal((await run('Target.getTargets')).targetInfos.at(-1).targetId,initial);
  await run('Target.closeTarget',{targetId:created[0]});
  const fresh=(await run('Target.createTarget',{url:'https://another.test'})).targetId;
  assert(!created.includes(fresh));
  assert.equal((await run('Target.attachToTarget',{targetId:initial})).sessionId,initial);
  const text="中文 ' ; process.exit(4); //\n$(echo not-a-command)";
  await run('Input.insertText',{text},initial);
  assert.equal(first.value,text);
  await run('Glide.selectAll',{},initial);
  assert.equal(writes.at(-1).params.key,'ControlOrMeta+A');
  assert.equal((await run('Runtime.evaluate',{expression:'fixed fixture snapshot'},initial)).result.value.url,first.address);
  const navigation=await run('Page.navigate',{url:'https://redirect.test/final'},initial);
  const frame=(await run('Page.getFrameTree',{},initial)).frameTree.frame;
  assert.equal(navigation.frameId,initial);
  assert.equal(navigation.loaderId,frame.loaderId);
  assert.equal(frame.url,'https://redirect.test/final');
  await run('Glide.releasePair',{method:'Input.dispatchKeyEvent',down:{key:'a',modifiers:10},up:{key:'a',modifiers:10}},initial);
  assert.deepEqual(writes.slice(-6).map(w=>[w.params.type,w.params.key]),[
    ['keyDown','Control'],['keyDown','Shift'],['keyDown','a'],['keyUp','a'],['keyUp','Shift'],['keyUp','Control']]);
  await run('Glide.releasePair',{method:'Input.dispatchMouseEvent',down:{x:42,y:73,button:'left',clickCount:1},up:{button:'left',clickCount:1}},initial);
  assert.deepEqual(writes.slice(-3).map(w=>w.method),['move','mouseDown','mouseUp']);
  const count=writes.length;
  await assert.rejects(run('Page.navigate',{url:'https://expired.test'},initial,Date.now()-1),/Expired/);
  allowed=false;
  await assert.rejects(run('Page.navigate',{url:'https://cancelled.test'},initial),/cancelled/);
  assert.equal(writes.length,count);
  // A key/button release is cleanup and still runs after Stop.
  await run('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter'},initial,Date.now()-1);
  allowed=true;
  await assert.rejects(run('Glide.releasePair',{method:'Input.dispatchKeyEvent',down:{type:'keyDown',key:'Fail'},up:{type:'keyUp',key:'Fail'}},initial),/interruption/);
  assert.equal(writes.at(-1).params.type,'keyUp');
  allowed=true;
  let finish;
  pendingCreate=new Promise(resolve=>finish=resolve);
  const creating=run('Target.createTarget',{url:'https://late.test'});
  await new Promise(resolve=>setImmediate(resolve));
  await assert.rejects(run('Target.getTargets'),/unresolved/);
  allowed=false;finish();
  await assert.rejects(creating,/cancelled/);
  assert(!writes.some(w=>w.params.url==='https://late.test'));
  console.log(JSON.stringify({quantity,verified:true}));
})().catch(error=>{console.error(error);process.exitCode=1;});
