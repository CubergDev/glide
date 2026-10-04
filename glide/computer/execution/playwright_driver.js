async page => {
  const request = __GLIDE_REQUEST__;
  const context = page.context();
  // Client objects, not DOM/window names or array indices, own stable tab IDs.
  const state = context.__glideDriver ||= {ids:new WeakMap(),next:0,pending:null};
  const id = p => {if(!state.ids.has(p))state.ids.set(p,`glide-${++state.next}`);return state.ids.get(p);};
  const pages = () => context.pages().filter(p=>!p.isClosed());
  const release = request.method==='Input.dispatchKeyEvent' && request.params.type==='keyUp'
    || request.method==='Input.dispatchMouseEvent' && request.params.type==='mouseReleased';
  if(state.pending) {
    // A timed-out client cannot let a new task overlap an unresolved write.
    if(!release)throw Error('An earlier browser operation is unresolved; no automatic replay');
    await state.pending;
  }
  const gate = async () => {
    if(release)return; // Key/button cleanup remains possible after cancellation.
    if(Date.now()>request.deadline)throw Error('Expired browser request; no action dispatched');
    const response=await fetch(request.gate,{signal:AbortSignal.timeout(5000)});
    if(!response.ok || !(await response.json()).allowed)throw Error('Browser request cancelled; no action dispatched');
  };
  await gate();
  const navigate = async (p,url) => {
    const response=await p.goto(url,{waitUntil:'commit',timeout:Math.max(1,request.deadline-Date.now())});
    const document=await p.evaluate('({url:location.href,document_id:String(performance.timeOrigin)})');
    if(!response||response.url()!==document.url)return {};
    return {frameId:id(p),loaderId:document.document_id};
  };
  const keyName = key => ({' ':'Space',ArrowLeft:'ArrowLeft',ArrowRight:'ArrowRight',
    ArrowUp:'ArrowUp',ArrowDown:'ArrowDown'}[key] || key);
  const keyPair = async (p,down,up) => {
    const held=[];
    try {
      for(const [mask,name] of [[1,'Alt'],[2,'Control'],[4,'Meta'],[8,'Shift']]) {
        if(!(down.modifiers & mask))continue;
        await gate();held.push(name);await p.keyboard.down(name);
      }
      await gate();
      try {await p.keyboard.down(keyName(down.key));}
      finally {await p.keyboard.up(keyName(up.key));}
    } finally {
      let cleanupError;
      for(const name of held.reverse()) {
        try {await p.keyboard.up(name);}catch(error){cleanupError ||= error;}
      }
      if(cleanupError)throw cleanupError;
    }
    return {};
  };
  const run = async () => {
    if(request.method==='Target.getTargets')return {targetInfos:pages().map(p=>({type:'page',targetId:id(p),url:p.url()}))};
    if(request.method==='Target.createTarget') {
      const p=await context.newPage();
      // A new blank page is already an effect. Never queue navigation after expiry.
      await gate();
      await navigate(p,request.params.url);
      return {targetId:id(p)};
    }
    const target=request.target || request.params.targetId;
    const p=pages().find(p=>id(p)===target);
    if(!p)throw Error('Selected browser tab disappeared');
    if(request.method==='Target.attachToTarget')return {sessionId:id(p)};
    if(request.method==='Target.activateTarget'){await p.bringToFront();return {};}
    if(request.method==='Target.closeTarget'){await p.close({runBeforeUnload:false});return {success:true};}
    await gate();
    if(request.method==='Page.navigate')return await navigate(p,request.params.url);
    if(request.method==='Page.getFrameTree') {
      const document=await p.evaluate('({url:location.href,document_id:String(performance.timeOrigin)})');
      return {frameTree:{frame:{id:id(p),loaderId:document.document_id,url:document.url}}};
    }
    if(request.method==='Runtime.evaluate')return {result:{value:await p.evaluate(request.params.expression)}};
    if(request.method==='Input.insertText'){await p.keyboard.insertText(request.params.text);return {};}
    if(request.method==='Glide.selectAll') {
      await p.keyboard.press('ControlOrMeta+A');
      return {};
    }
    if(request.method==='Input.dispatchKeyEvent') {
      const {type,key}=request.params;
      if(!['keyDown','keyUp'].includes(type))throw Error('Unsupported key event');
      await p.keyboard[type==='keyUp'?'up':'down'](keyName(key));return {};
    }
    if(request.method==='Input.dispatchMouseEvent') {
      const {type,x,y,button,clickCount}=request.params;
      if(!['mousePressed','mouseReleased'].includes(type))throw Error('Unsupported mouse event');
      if(type==='mousePressed'){await p.mouse.move(x,y);await gate();}
      await p.mouse[type==='mouseReleased'?'up':'down']({button,clickCount});return {};
    }
    if(request.method==='Glide.releasePair') {
      const {method,down,up}=request.params;
      if(method==='Input.dispatchKeyEvent')return await keyPair(p,down,up);
      if(method!=='Input.dispatchMouseEvent')throw Error('Unsupported paired input');
      await p.mouse.move(down.x,down.y);await gate();
      try {await p.mouse.down({button:down.button,clickCount:down.clickCount});}
      finally {await p.mouse.up({button:up.button,clickCount:up.clickCount});}
      return {};
    }
    throw Error('Unsupported Playwright bridge method');
  };
  const pending=run();state.pending=pending;
  try{return await pending;}finally{if(state.pending===pending)state.pending=null;}
}
