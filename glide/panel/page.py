"""The panel page: one self-contained document, no external assets. It builds the DOM with textContent only (no innerHTML),
so nothing that comes back from a server, a provider, a model or a file can become markup."""

from __future__ import annotations

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer"><title>Glide panel</title>
<style nonce="__NONCE__">
:root{color-scheme:light dark;--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--line:#d8d8de;--acc:#0a5cff;--bad:#b3261e;--ok:#17803d;--warn:#8a5a00}
@media(prefers-color-scheme:dark){:root{--bg:#18181b;--fg:#f2f2f4;--mut:#a0a0a8;--line:#34343a;--acc:#6ea0ff;--bad:#ff8a80;--ok:#6fd18e;--warn:#f0c060}}
body{margin:0 auto;max-width:56rem;padding:1rem 16px 8rem;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
h1{font-size:1.4rem;margin:.2rem 0}h2{font-size:1.1rem;margin:0 0 .5rem}h3{font-size:1rem;margin:.6rem 0 .2rem}
section.card{border:1px solid var(--line);border-radius:10px;padding:.8rem 1rem;margin:.8rem 0}
nav{display:flex;flex-wrap:wrap;gap:.3rem;margin:.6rem 0}nav button.on{background:var(--acc);color:var(--bg)}
button,select,input,textarea{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.3rem .55rem}
textarea{width:100%;box-sizing:border-box;font:13px/1.4 ui-monospace,monospace}input[type=text]{min-width:8rem}
button{cursor:pointer;border-color:var(--acc);color:var(--acc)}button.main{background:var(--acc);color:var(--bg)}
button.danger{border-color:var(--bad);color:var(--bad)}button:disabled{opacity:.5;cursor:default}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:rgba(127,127,127,.12);padding:.6rem;border-radius:6px;font-size:13px}
.mut{color:var(--mut);font-size:.9rem}.bad{color:var(--bad)}.ok{color:var(--ok)}.warn{color:var(--warn)}
.row{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.35rem 0}label{display:inline-flex;gap:.35rem;align-items:center}
table{border-collapse:collapse;width:100%;font-size:.92rem}td,th{border-bottom:1px solid var(--line);padding:.25rem .4rem;text-align:left;vertical-align:top;overflow-wrap:anywhere}
.banner{border:2px solid var(--bad);border-radius:8px;padding:.6rem .8rem;margin:.6rem 0;color:var(--bad);font-weight:600}
.note{border-left:3px solid var(--warn);padding:.2rem .6rem;margin:.4rem 0;color:var(--warn);font-size:.9rem}
#bar{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:.5rem 16px}
#bar .in{max-width:56rem;margin:0 auto}
.diff{max-height:16rem;overflow:auto}
</style></head><body>
<h1>Glide panel</h1>
<div id="file" class="mut"></div><div id="firstrun"></div>
<nav id="tabs"></nav>
<div id="providers" class="tab"></div><div id="roles" class="tab" hidden></div><div id="features" class="tab" hidden></div>
<div id="chat" class="tab" hidden></div><div id="files" class="tab" hidden></div><div id="status" class="tab" hidden></div>
<div id="bar"><div class="in"><div class="row"><span id="pending" class="mut"></span>
<button id="checkbtn">Check changes</button><button id="reviewbtn" class="main">Review diff and write</button></div>
<div id="barerr" class="bad"></div><div id="review" hidden><pre id="diff" class="diff"></pre>
<div class="row"><label><input type="checkbox" id="okwrite"> I have read the diff. Back up the old file and write.</label>
<button id="writebtn" class="main">Write glide.toml</button></div></div><div id="written" class="ok"></div></div></div>
<script nonce="__NONCE__">
"use strict";
const SESSION="__SESSION__";
let S=null;
const H={"X-Glide-Session":SESSION,"Content-Type":"application/json"};
const $=id=>document.getElementById(id);
function el(tag,text,attrs,...kids){const e=document.createElement(tag);if(text!=null&&text!=="")e.textContent=text;
  for(const k in attrs||{}){if(k==="class")e.className=attrs[k];else if(k==="onclick")e.onclick=attrs[k];else e.setAttribute(k,attrs[k])}
  for(const c of kids)if(c)e.append(c);return e}
function clear(e){while(e.firstChild)e.removeChild(e.firstChild);return e}
async function call(name,body){const r=await fetch("/api/"+name,{method:"POST",headers:H,body:JSON.stringify(body||{})});
  const j=await r.json();if(!r.ok)throw new Error(j.error||("error "+r.status));return j}
async function read(name,q){const r=await fetch("/api/read/"+name+(q||""),{headers:H});const j=await r.json();
  if(!r.ok)throw new Error(j.error||("error "+r.status));return j}
function show(e,msg,cls){e.textContent=msg||"";e.className=cls||""}
const D={providers:{},roles:{},features:{}};   // the draft: only what the person changed
const clone=o=>JSON.parse(JSON.stringify(o));
function pendingCount(){return Object.keys(D.providers).length+Object.keys(D.roles).length+Object.keys(D.features).length}
function changes(){const c={providers:{},roles:{},features:{}};
  for(const n in D.providers)c.providers[n]=D.providers[n]===null?null:{kind:D.providers[n].kind,base_url:D.providers[n].base_url,api_key_env:D.providers[n].api_key_env,options:parseOpts(D.providers[n].optionsText)};
  for(const r in D.roles)c.roles[r]=D.roles[r]===null?null:{chain:D.roles[r].chain.map(s=>({provider:s.provider,model:s.model,name:s.name,options:parseOpts(s.optionsText)})),policy:D.roles[r].policy};
  Object.assign(c.features,D.features);return c}
function parseOpts(t){const o={};for(const line of String(t||"").split("\n")){const l=line.trim();if(!l)continue;const i=l.indexOf("=");
  if(i<1)throw new Error("options are written name=value, one per line");const k=l.slice(0,i).trim(),v=l.slice(i+1).trim();
  if(v==="true"||v==="false")o[k]=v==="true";else if(v!==""&&!isNaN(Number(v)))o[k]=Number(v);else if(v[0]==="{"){try{o[k]=JSON.parse(v)}catch(e){throw new Error("option "+k+": not valid JSON")}}else o[k]=v.replace(/^"|"$/g,"")}return o}
function optsText(o){return Object.entries(o||{}).map(([k,v])=>k+"="+(typeof v==="object"?JSON.stringify(v):String(v))).join("\n")}
function updPending(){$("pending").textContent=pendingCount()?pendingCount()+" unsaved change(s)":"no unsaved changes"}
async function check(errEl){try{const r=await call("preview",{changes:changes()});show($("barerr"),"");if(errEl)show(errEl,"Loads fine.","ok");return r}
  catch(e){show($("barerr"),e.message,"bad");if(errEl)show(errEl,e.message,"bad");return null}}
$("checkbtn").onclick=()=>check(null);
let PREVIEW=null;
$("reviewbtn").onclick=async()=>{const r=await check(null);if(!r)return;PREVIEW=r;$("review").hidden=false;$("okwrite").checked=false;
  $("diff").textContent=(r.diff||"(no change)")+"\n\n"+r.comments_note};
$("writebtn").onclick=async()=>{if(!$("okwrite").checked||!PREVIEW){show($("barerr"),"read the diff and tick the box first","bad");return}
  try{const r=await call("write",{changes:changes(),expect:PREVIEW.expect,confirm:true});
    show($("written"),r.unchanged?"Nothing changed.":"Written to "+r.written+(r.backup?" (backup: "+r.backup+")":""));
    D.providers={};D.roles={};D.features={};$("review").hidden=true;PREVIEW=null;await load()}catch(e){show($("barerr"),e.message,"bad")}};

const TABS=[["providers","Providers"],["roles","Roles & chains"],["features","Features & safety"],["chat","Chat / Ask"],["files","Files"],["status","Status"]];
function drawTabs(){const n=clear($("tabs"));for(const [id,label] of TABS)n.append(el("button",label,{"data-t":id,onclick:()=>go(id)}))}
function go(id){for(const t of document.querySelectorAll(".tab"))t.hidden=t.id!==id;
  for(const b of document.querySelectorAll("nav button"))b.className=b.getAttribute("data-t")===id?"on":"";
  if(id==="status")drawStatus();if(id==="files")drawFiles();if(id==="chat")drawChat()}

// ---- providers
const TESTRES={};
function provView(name){const base=S.providers.find(p=>p.name===name)||{name,kind:"openai_compat",base_url:"",api_key_env:"",options:{},builtin:false,in_file:false,export:"",status:"missing"};
  const d=D.providers[name];return d?{...base,...d,options:null}:base}
function drawProviders(){const root=clear($("providers"));
  root.append(el("p","Built-in providers come from Glide's registry. Add your own (an OpenAI-compatible gateway, a classifier server) with the form below. A key is never written to the file: only the NAME of the environment variable.",{class:"mut"}));
  const names=[...S.providers.map(p=>p.name),...Object.keys(D.providers).filter(n=>D.providers[n]&&!S.providers.some(p=>p.name===n))];
  for(const name of names){const p=provView(name);const d=D.providers[name];const card=el("section",null,{class:"card"});
    card.append(el("h2",name+(p.builtin?"  (built-in)":"  (custom)")+(d?"  - edited, not written":"")));
    card.append(el("div","kind: "+p.kind+"   base_url: "+(p.base_url||"(adapter default)")+"   key variable: "+(p.api_key_env||"(none)"),{class:"mut"}));
    const st=p.api_key_env?(S.providers.find(x=>x.name===name)||{}).status:"none needed";
    card.append(el("div","key status: "+(st||"missing"),{class:st==="missing"?"warn":"ok"}));
    if(p.api_key_env){card.append(el("div",p.export||("export "+p.api_key_env+"='<paste your key here>'"),{class:"mut"}));
      const inp=el("input",null,{type:"password",autocomplete:"off",placeholder:"paste a key (kept in memory only)"});
      const msg=el("span");card.append(el("div",null,{class:"row"},inp,el("button","Hold in memory",{onclick:async()=>{try{const r=await call("key",{env_var:p.api_key_env,value:inp.value});inp.value="";show(msg,"key status: "+r.status,"ok");await load()}catch(e){show(msg,e.message,"bad")}}}),el("button","Forget it",{onclick:async()=>{try{await call("key",{env_var:p.api_key_env,value:""});await load()}catch(e){show(msg,e.message,"bad")}}}),msg))}
    const out=el("pre");out.hidden=!TESTRES[name];if(TESTRES[name])out.textContent=TESTRES[name];
    card.append(el("div",null,{class:"row"},
      el("button","Test this provider",{onclick:async()=>{if(!confirm("A test sends one tiny real request and spends a few tokens. Run it?"))return;
        out.hidden=false;out.textContent="testing...";try{const r=await call("test",{provider:name,confirm:true,changes:pendingCount()?changes():undefined});
          TESTRES[name]=r.role+": "+r.status+(r.latency_s!=null?" ("+r.latency_s.toFixed(2)+"s)":"")+"\n"+r.detail;out.textContent=TESTRES[name]}catch(e){out.textContent=e.message}}}),
      el("button","Edit",{onclick:()=>editProvider(name)}),
      (p.builtin&&!p.in_file&&!d)?null:el("button",p.builtin?"Reset to built-in":"Remove",{class:"danger",onclick:()=>{D.providers[name]=null;if(!p.in_file)delete D.providers[name];updPending();drawProviders()}})));
    card.append(el("div","A test spends tokens and runs only when you click it. The provider must be in a role's chain.",{class:"mut"}));card.append(out);root.append(card)}
  root.append(el("div",null,{class:"row"},el("button","Add a custom provider",{class:"main",onclick:()=>editProvider("")})));
  root.append(el("div",null,{id:"provform"}))}
function editProvider(name){const f=clear($("provform"));const p=name?provView(name):{kind:"openai_compat",base_url:"",api_key_env:"",options:{}};
  const cur=D.providers[name];const opts=cur?cur.optionsText:optsText(p.options);
  const n=el("input",null,{type:"text",value:name,maxlength:40,placeholder:"name"});if(name)n.disabled=true;
  const k=el("select");for(const x of S.kinds)k.append(el("option",x,{value:x}));k.value=p.kind;
  const u=el("input",null,{type:"text",size:44,value:p.base_url,placeholder:"https://host/v1"});
  const e=el("input",null,{type:"text",value:p.api_key_env,placeholder:"MY_GATEWAY_API_KEY"});
  const o=el("textarea",opts,{rows:3,placeholder:"name=value, one per line"});const err=el("div");
  f.append(el("section",null,{class:"card"},el("h2",name?"Edit "+name:"New custom provider"),
    el("div",null,{class:"row"},el("label","name ",null,n),el("label","kind ",null,k)),
    el("div",null,{class:"row"},el("label","base_url ",null,u)),
    el("div",null,{class:"row"},el("label","api_key_env ",null,e),el("span","the NAME of a variable, never the key",{class:"mut"})),
    el("div","options (optional, e.g. extra_headers={\"X-Org\":\"acme\"} or max_tokens=300)",{class:"mut"}),o,
    el("div",null,{class:"row"},el("button","Keep this edit",{class:"main",onclick:async()=>{
      const nm=name||n.value.trim();D.providers[nm]={kind:k.value,base_url:u.value.trim(),api_key_env:e.value.trim(),optionsText:o.value};
      try{parseOpts(o.value)}catch(x){show(err,x.message,"bad");delete D.providers[nm];return}
      updPending();if(await check(err)){drawProviders()}else{drawProviders();editProvider(nm)}}}),
      el("button","Cancel",{onclick:()=>clear(f)})),err))}

// ---- roles
function roleDraft(r){if(!D.roles[r]&&D.roles[r]!==null){const v=S.roles[r];D.roles[r]={chain:v.chain.map(s=>({provider:s.provider,model:s.model,name:s.name,optionsText:optsText(s.options)})),policy:clone(v.policy)}}return D.roles[r]}
function drawRoles(){const root=clear($("roles"));
  root.append(el("p","Each job has an ordered chain: the first slot is tried first, the next ones are the fallback. Model ids are free text: Glide makes no claim about any of them. A switch to a later slot is always shown, never silent.",{class:"mut"}));
  for(const r of S.role_order){const v=S.roles[r];const dr=D.roles[r];const edited=r in D.roles;const view=edited?dr:null;
    const card=el("section",null,{class:"card"});card.append(el("h2",r+(edited?"  - edited, not written":(v.defined?"  (in your file)":v.stands_on?"  (stands on "+v.stands_on+")":"  (built-in chain)"))));
    const chain=edited&&dr?dr.chain:(v.defined?v.chain.map(s=>({...s,optionsText:optsText(s.options)})):v.default_chain.map(s=>({...s,optionsText:optsText(s.options)})));
    card.append(el("div","fallback order: "+(chain.length?chain.map((s,i)=>(i+1)+". "+s.provider+(s.model?":"+s.model:"")).join("  ->  "):"(none)")+"   (order: "+(((edited&&dr?dr.policy:v.policy)||{}).order||"priority")+")",{class:"mut"}));
    const err=el("div");
    if(edited&&dr){const d=dr;d.chain.forEach((s,i)=>{const prov=el("select");for(const p of providerChoices(r))prov.append(el("option",p,{value:p}));prov.value=s.provider;prov.onchange=()=>{s.provider=prov.value;drawRoles()};
      const m=el("input",null,{type:"text",size:28,value:s.model,placeholder:"model id (free text)"});m.oninput=()=>{s.model=m.value};
      const nm=el("input",null,{type:"text",size:10,value:s.name,placeholder:"label (optional)"});nm.oninput=()=>{s.name=nm.value};
      const o=el("input",null,{type:"text",size:24,value:s.optionsText.replace(/\n/g,"; "),placeholder:"options a=b; c=d"});o.oninput=()=>{s.optionsText=o.value.split(";").map(x=>x.trim()).join("\n")};
      card.append(el("div",null,{class:"row"},el("span",String(i+1)+"."),prov,m,nm,o,
        el("button","up",{onclick:()=>{if(i>0){[d.chain[i-1],d.chain[i]]=[d.chain[i],d.chain[i-1]];drawRoles()}}}),
        el("button","down",{onclick:()=>{if(i<d.chain.length-1){[d.chain[i+1],d.chain[i]]=[d.chain[i],d.chain[i+1]];drawRoles()}}}),
        el("button","remove",{class:"danger",onclick:()=>{d.chain.splice(i,1);drawRoles()}})))});
      card.append(el("div",null,{class:"row"},el("button","Add a slot",{onclick:()=>{d.chain.push({provider:providerChoices(r)[0],model:"",name:"",optionsText:""});drawRoles()}})));
      const pol=el("div",null,{class:"row"});const ord=el("select");ord.append(el("option","priority (as listed)",{value:"priority"}),el("option","latency (fastest healthy first)",{value:"latency"}));ord.value=d.policy.order||"priority";ord.onchange=()=>{d.policy.order=ord.value};pol.append(el("label","order ",null,ord));
      const keys=["hedge_after_s","fail_threshold","cooldown_s","auth_cooldown_s","latency_alpha"].concat(r.startsWith("llm.")?["deadline_s"]:[]);
      for(const k of keys){const i=el("input",null,{type:"text",size:5,value:d.policy[k]==null?"":String(d.policy[k])});i.oninput=()=>{if(i.value==="")delete d.policy[k];else d.policy[k]=Number(i.value)};pol.append(el("label",k+" ",null,i))}
      card.append(pol,el("div","hedge_after_s: race the next slot when the first is this slow (leave empty for off). Empty fields keep Glide's own defaults.",{class:"mut"}));
      card.append(el("div",null,{class:"row"},el("button","Check this role",{onclick:()=>check(err)}),el("button","Discard edit",{onclick:()=>{delete D.roles[r];updPending();drawRoles()}})))}
    else{card.append(el("div",null,{class:"row"},el("button",v.defined?"Edit chain":"Edit (start from the "+(v.default_chain.length?"built-in":"empty")+" chain)",{onclick:()=>{
        if(v.defined)roleDraft(r);else D.roles[r]={chain:v.default_chain.map(s=>({provider:s.provider,model:s.model,name:"",optionsText:optsText(s.options)})),policy:{}};
        if(!D.roles[r].chain.length)D.roles[r].chain.push({provider:providerChoices(r)[0],model:"",name:"",optionsText:""});updPending();drawRoles()}}),
      v.defined?el("button","Remove from file (use built-in)",{class:"danger",onclick:()=>{D.roles[r]=null;updPending();drawRoles()}}):null))}
    if(edited&&dr===null)card.append(el("div","will be removed from the file when written",{class:"warn"}));
    card.append(err);root.append(card)}
  updPending()}
function providerChoices(role){const kinds=S.roles[role].kinds;const names=[];
  const all=[...S.providers,...Object.entries(D.providers).filter(([n,v])=>v&&!S.providers.some(p=>p.name===n)).map(([n,v])=>({name:n,kind:v.kind}))];
  for(const p of all){const k=(D.providers[p.name]||p).kind||p.kind;if(kinds.includes(k))names.push(p.name)}
  if(role==="classifier")names.push("llm.fast","llm.smart");return names}

// ---- features
const FEATS=[
 ["computer","Computer control","bool","Lets the Chat tab start computer tasks. OFF by default. Each task still needs your click, and a real run a second confirm."],
 ["confirm_acting","Confirm before acting (router)","bool","[routing] confirm_acting: the router asks before an act-capable request."],
 ["confirm_tasks","Spoken tasks run as a dry run first","bool","[speech] confirm_tasks (default on): hands-free speech needs the confirm phrase. Turning it off lets any audible speech start a task."],
 ["engine","Execution engine","engine","legacy is the default. The structured engine has not passed live checks yet: choose it on purpose."],
 ["voice","Voice","bool","Voice runs in a terminal: `glide voice`. This switch is the panel's record of your choice; see Status for the speech extra."],
 ["research_calls","Research budget (model calls per task)","int","[research] calls, 1 to 32."],
 ["point_ask","Point-and-ask","bool","Runs from the app or the pet. The panel records the choice and shows its status."],
 ["memory","Memory","bool","[memory] enabled: a local store, off by default."],
 ["memory_auto","Memory: capture automatically","bool","[memory] auto_capture."],
 ["webhooks","Webhooks","bool","OFF by default. On writes [webhooks] config; nothing runs until that JSON file says enabled: true."],
 ["files","Files tab","bool","Lets the Files tab plan and run moves. OFF by default."],
 ["record_content","Detailed recording","bool","OFF by default. On: the panel keeps a local chat log with your requests and the replies. Without it nothing like that is stored."],
 ["retention_days","Chat log retention (days)","int","An older log entry is deleted when the log is next written."]];
function featVal(n){return n in D.features?D.features[n]:S.features[n]}
function drawFeatures(){const root=clear($("features"));
  root.append(el("p","Changes are written to "+S.file.path+" only after you review the diff and confirm. "+S.file.scope+".",{class:"mut"}));
  root.append(el("div","Confirmations stay on: Glide asks before it takes over this machine. That is not switchable.",{class:"note"}));
  for(const [n,label,type,note] of FEATS){const row=el("div",null,{class:"row"});const v=featVal(n);let inp;
    if(type==="bool"){inp=el("input",null,{type:"checkbox"});inp.checked=v===true;inp.onchange=()=>{D.features[n]=inp.checked;updPending()}}
    else if(type==="engine"){inp=el("select");for(const e of S.engines)inp.append(el("option",e,{value:e}));inp.value=v;inp.onchange=()=>{D.features[n]=inp.value;updPending();drawFeatures()}}
    else{inp=el("input",null,{type:"text",size:5,value:String(v)});inp.oninput=()=>{const x=Number(inp.value);if(Number.isInteger(x))D.features[n]=x;updPending()}}
    row.append(el("label",label+" ",null,inp),n in D.features?el("span","changed",{class:"warn"}):null);root.append(row,el("div",note,{class:"mut"}))}
  if(featVal("engine")==="structured")root.append(el("div","The structured engine is selected. It has not passed live checks: legacy stays the default until it has.",{class:"note"}));
  root.append(el("div",null,{class:"row"},el("button","Check this page",{onclick:()=>check($("featerr"))})),el("div",null,{id:"featerr"}));updPending()}

// ---- chat
let POLLER=null,NEXT=0,LASTTEXT="";
function drawChat(){const root=clear($("chat"));
  const t=el("textarea",LASTTEXT,{rows:3,maxlength:2000,placeholder:"Ask a question or describe a task"});
  const eng=el("select");eng.append(el("option","engine: from glide.toml",{value:""}),el("option","legacy",{value:"legacy"}),el("option","structured (opt-in)",{value:"structured"}));
  const err=el("div");
  root.append(el("p","Requests go through the real assistant. A computer task starts as a DRY RUN: it looks at the screen and says what it would do, and only after you approve here. Computer control is "+(S.features.computer?"ON":"OFF (Features and safety)")+".",{class:"mut"}),
    t,el("div",null,{class:"row"},eng,
      el("button","Send",{class:"main",onclick:async()=>{LASTTEXT=t.value;try{show(err,"");NEXT=0;await call("chat_start",{text:t.value,engine:eng.value||null});drawChatOut();poll()}catch(e){show(err,e.message,"bad")}}}),
      el("button","Stop",{class:"danger",onclick:async()=>{try{const r=await call("chat_stop");show(err,r.stopped?"stopped":"stopped (no task was running)")}catch(e){show(err,e.message,"bad")}}})),err,el("div",null,{id:"chatout"}));
  poll()}
async function poll(){clearTimeout(POLLER);let r;try{r=await read("chat","?since="+NEXT)}catch(e){return}
  if(r.status==="idle")return;NEXT=r.next;CH.events=(CH.id===r.id?CH.events:[]).concat(r.events);CH.id=r.id;CH.status=r.status;CH.approval=r.approval;CH.result=r.result;drawChatOut();
  if(r.status==="running")POLLER=setTimeout(poll,600)}
const CH={id:0,events:[],status:"",approval:null,result:null};
function drawChatOut(){const o=$("chatout");if(!o)return;clear(o);
  if(CH.status==="running")o.append(el("div","working...",{class:"mut"}));
  for(const e of CH.events)o.append(el("div",(e.kind==="switch"?"SWITCH: ":e.kind==="notice"?"notice: ":"")+e.text,{class:e.kind==="switch"?"warn":e.kind==="notice"?"mut":""}));
  if(CH.approval)o.append(approvalBox(CH.approval));
  const r=CH.result;if(!r)return;
  o.append(el("h3","Result"));o.append(el("div","route: "+r.route+(r.error?"   error: "+r.error:""),{class:r.error?"bad":"mut"}));
  if(r.text)o.append(el("pre",r.text));
  if(r.task){const k=r.task;
    if(k.uncertain)o.append(el("div",k.uncertain_note+". "+(k.readback||""),{class:"banner"}));
    o.append(el("div","task: "+k.outcome+(k.act?" (REAL run)":" (dry run: nothing was clicked or typed)")+"   steps "+k.steps+"   "+k.seconds+"s",{class:"mut"}));
    for(const [lab,x] of [["would do",k.would_do],["answer",k.answer],["failure",k.failure]])if(x)o.append(el("pre",lab+": "+x));
    if(k.can_run_for_real){const ok=el("input",null,{type:"checkbox"});const m=el("span");
      o.append(el("div",null,{class:"row"},el("label","I understand Glide will click and type on this Mac ",null,ok),
        el("button","Run for real (--act)",{class:"danger",onclick:async()=>{if(!ok.checked){show(m,"tick the box first","bad");return}
          try{NEXT=0;await call("chat_start",{text:LASTTEXT,act:true,confirm_act:true,engine:null});poll()}catch(e){show(m,e.message,"bad")}}}),m))}}
  if(r.chains&&r.chains.length){const t=el("table");t.append(el("tr",null,null,el("th","role"),el("th","slot"),el("th","calls"),el("th","failures"),el("th","resting"),el("th","last error")));
    for(const c of r.chains)t.append(el("tr",null,null,el("td",c.role),el("td",c.slot),el("td",String(c.calls)),el("td",String(c.failures)),el("td",c.resting_s?c.resting_s+"s":"-"),el("td",c.last_error||"-")));
    o.append(el("h3","Provider chain"),t)}}
function approvalBox(a){const b=el("div",null,{class:"card"});const ok=el("input",null,{type:"checkbox"});const m=el("span");
  b.append(el("h2",a.act?"Approve a REAL run?":"Approve a dry run?"),el("pre",a.goal),
    el("div",a.act?"Glide will click and type on this Mac.":"A dry run looks at the screen and says what it would do. It does not click or type.",{class:a.act?"bad":"mut"}));
  if(a.act)b.append(el("label","I confirm ",null,ok));
  b.append(el("div",null,{class:"row"},el("button",a.act?"Approve real run":"Approve dry run",{class:a.act?"danger":"main",onclick:async()=>{try{await call("chat_decide",{approve:true,confirm_act:ok.checked});poll()}catch(e){show(m,e.message,"bad")}}}),
    el("button","Cancel",{onclick:async()=>{try{await call("chat_decide",{approve:false});poll()}catch(e){show(m,e.message,"bad")}}}),m));return b}

// ---- files
let PLAN=null;
async function drawFiles(){const root=clear($("files"));
  if(!S.features.files){root.append(el("div","The Files tab is off. Switch it on under Features and safety, write the file, then reload this tab.",{class:"note"}));return}
  const rootIn=el("input",null,{type:"text",size:40,placeholder:"/Users/you/Downloads"});const it=el("select");it.append(el("option","organize by type",{value:"type"}),el("option","sort into named folders",{value:"named"}));
  const cat=el("textarea","",{rows:3,placeholder:"optional JSON: {\"Pictures\":[\".png\"]}  (named folders: {\"Taxes\":[\"tax\"]})"});const err=el("div");const out=el("div");
  root.append(el("p","A plan decides what would move and moves nothing. To run it you must type its plan hash back. Every run writes an undo manifest.",{class:"mut"}),
    el("div",null,{class:"row"},el("label","folder ",null,rootIn),it),cat,
    el("div",null,{class:"row"},el("button","Make a plan (moves nothing)",{class:"main",onclick:async()=>{try{
      let c=cat.value.trim()?JSON.parse(cat.value):null;const body={root:rootIn.value,intent:it.value};if(c){if(it.value==="named")body.folders=c;else body.categories=c}
      PLAN=await call("files_plan",body);show(err,"");drawPlan(out)}catch(e){show(err,e.message,"bad")}}})),err,out,el("h3","Undo a run"),el("div",null,{id:"manifests"}));
  drawPlan(out);drawManifests()}
function drawPlan(out){clear(out);if(!PLAN)return;out.append(el("h3","Plan "+PLAN.plan_hash+": "+PLAN.count+" move(s)"),el("pre",PLAN.preview));
  const t=el("table");for(const m of PLAN.moves.slice(0,60))t.append(el("tr",null,null,el("td",m.source),el("td",m.destination)));out.append(t);
  const h=el("input",null,{type:"text",size:36,placeholder:"type the plan hash to approve"});const ok=el("input",null,{type:"checkbox"});const m=el("div");
  out.append(el("div",null,{class:"row"},h,el("label","run it ",null,ok),el("button","Approve and run",{class:"danger",onclick:async()=>{try{
    const r=await call("files_execute",{plan_hash:PLAN.plan_hash,approve:h.value.trim(),confirm:ok.checked});PLAN=null;clear(out);
    out.append(el("pre",r.status+"\n"+r.actions.map(a=>a.status+": "+a.source+" -> "+a.destination+(a.reason?" ("+a.reason+")":"")).join("\n")+"\nundo manifest: "+r.manifest));drawManifests()}catch(e){show(m,e.message,"bad")}}})),m)}
async function drawManifests(){const n=$("manifests");if(!n)return;clear(n);try{const r=await read("file_runs");if(!r.manifests.length)n.append(el("div","no runs yet",{class:"mut"}));
  for(const name of r.manifests){const m=el("span");n.append(el("div",null,{class:"row"},el("span",name),el("button","Undo",{onclick:async()=>{if(!confirm("Reverse this run?"))return;try{const u=await call("files_undo",{manifest:name,confirm:true});show(m,u.status,u.status==="ok"?"ok":"warn")}catch(e){show(m,e.message,"bad")}}}),m))}}catch(e){n.append(el("div",e.message,{class:"bad"}))}}

// ---- status
async function drawStatus(){const root=clear($("status"));root.append(el("div","loading...",{class:"mut"}));let r;try{r=await read("status")}catch(e){show(clear(root),e.message,"bad");return}
  clear(root);root.append(el("div",null,{class:"row"},el("button","Refresh",{onclick:drawStatus}),el("span","Offline results: nothing is sent anywhere. Use a provider's Test button to spend tokens.",{class:"mut"})));
  if(r.marker){const b=el("div",null,{class:"banner"});b.append(el("div","An unresolved real run was recorded at "+r.marker.started+(r.marker.running_now?" (it is running now)":"")+"."),
    el("div","If the panel or Glide stopped during a real run, or it ended with 'completion unknown', an action may have been sent and its effect never seen. Nothing was retried. Look at the screen and the run folder, decide what happened, then clear the marker. Marker file: "+r.marker.path,{class:"mut"}));
    if(!r.marker.running_now)b.append(el("button","I looked: clear the marker",{class:"danger",onclick:async()=>{if(!confirm("Have you checked the screen and the run folder?"))return;try{await call("marker_clear",{confirm:true});drawStatus()}catch(e){alert(e.message)}}}));root.append(b)}
  else root.append(el("div","No unresolved-run marker.",{class:"ok"}));
  if(r.error){root.append(el("div",r.error,{class:"bad"}));}
  root.append(el("div","engine: "+r.engine,{class:"mut"}));for(const w of r.warnings||[])root.append(el("div","warning: "+w,{class:"warn"}));
  if(r.defaulted&&r.defaulted.length)root.append(el("div","built-in chains in use for: "+r.defaulted.join(", "),{class:"mut"}));
  const t=el("table");t.append(el("tr",null,null,el("th","role"),el("th","slot"),el("th","status"),el("th","detail")));
  for(const x of r.rows)t.append(el("tr",null,null,el("td",x.role),el("td",x.slot),el("td",x.status),el("td",x.detail)));root.append(el("h3","Doctor (offline)"),t);
  root.append(el("h3","Features"));for(const f of r.features)root.append(el("div",f.name+": "+f.line,{class:f.ok?"mut":"bad"}));
  root.append(el("h3","Last runs"));if(!r.runs.length)root.append(el("div","none found",{class:"mut"}));for(const x of r.runs)root.append(el("div",x.when+"  "+x.name,{class:"mut"}))}

async function load(){S=await (await fetch("/api/state",{headers:H})).json();
  const f=S.file;$("file").textContent=f.path+" - "+f.scope;
  const fr=clear($("firstrun"));if(f.problem)fr.append(el("div",f.problem,{class:"banner"}));else if(!f.exists)fr.append(el("div","No glide.toml yet. `glide setup` walks through a first one, or build it here and write it.",{class:"note"}));
  drawProviders();drawRoles();drawFeatures();updPending()}
(async()=>{drawTabs();await load();go("providers")})().catch(e=>{document.body.append(el("pre",String(e)))});
</script></body></html>
"""


def render_panel(session: str, nonce: str) -> str:
    return PAGE.replace("__NONCE__", nonce).replace("__SESSION__", session)
