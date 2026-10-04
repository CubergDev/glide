"""The wizard page: one self-contained document. It builds the DOM with textContent only (no innerHTML), so nothing
that comes back from a server, a provider or a file can become markup."""

from __future__ import annotations

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer"><title>Glide setup</title>
<style nonce="__NONCE__">
:root{color-scheme:light dark;--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--line:#d8d8de;--acc:#0a5cff;--bad:#b3261e;--ok:#17803d}
@media(prefers-color-scheme:dark){:root{--bg:#18181b;--fg:#f2f2f4;--mut:#a0a0a8;--line:#34343a;--acc:#6ea0ff;--bad:#ff8a80;--ok:#6fd18e}}
body{margin:0 auto;max-width:46rem;padding:1rem 16px 4rem;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
h1{font-size:1.5rem}h2{font-size:1.15rem;margin:0 0 .5rem}section{border:1px solid var(--line);border-radius:10px;padding:1rem;margin:1rem 0}
button,select,input{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.35rem .6rem}
button{cursor:pointer;border-color:var(--acc);color:var(--acc)}button.main{background:var(--acc);color:var(--bg)}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:rgba(127,127,127,.12);padding:.6rem;border-radius:6px}
.mut{color:var(--mut);font-size:.9rem}.bad{color:var(--bad)}.ok{color:var(--ok)}.row{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.4rem 0}
label{display:block}
</style></head><body>
<h1>Glide setup</h1>
<p class="mut">This page talks only to this computer. Nothing is written until step 5, and no key is ever written.</p>
<section><h2>1. Features</h2><div id="features"></div>
<p class="mut">Confirmations stay on: Glide asks before it takes over this machine. That is not switchable.</p></section>
<section><h2>2. Providers</h2>
<div class="row"><label>One key is enough: <select id="presetsel"></select></label><button id="presetbtn">Use it for everything it can do</button></div>
<div id="roles"></div></section>
<section><h2>3. Keys</h2>
<p class="mut">Glide reads each key from an environment variable. A key pasted here is held in this process's memory only: it is used for the optional test and for starting Glide, and never written or shown again.</p>
<div id="keys"></div></section>
<section><h2>4. Test</h2>
<p class="mut">A test sends one tiny real request. It spends a few tokens and only runs when you click it.</p>
<div id="tests"></div></section>
<section><h2>5. Review and write</h2>
<div class="row"><button id="renderbtn">Show glide.toml</button><button id="writebtn" class="main">Write it</button></div>
<pre id="toml"></pre><div id="written" class="mut"></div></section>
<section><h2>6. Start Glide</h2>
<div class="row"><label>Request (for ask) <input id="asktext" size="30" maxlength="2000"></label></div>
<div class="row" id="modes"></div><pre id="launchout"></pre></section>
<script nonce="__NONCE__">
"use strict";
const SESSION="__SESSION__";
let S=null, state={features:{computer:false,webhooks:false,memory:false},roles:{}};
const $=id=>document.getElementById(id);
function el(tag,text,attrs){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;for(const k in attrs||{})e.setAttribute(k,attrs[k]);return e}
async function call(name,body){
  const r=await fetch("/api/"+name,{method:"POST",headers:{"Content-Type":"application/json","X-Glide-Session":SESSION},body:JSON.stringify(body||{})});
  const j=await r.json();if(!r.ok)throw new Error(j.error||("error "+r.status));return j}
function fail(node,e){node.textContent=String(e.message||e);node.className="bad"}
function drawFeatures(){
  const box=$("features");box.replaceChildren();
  const names={computer:"Computer control (Glide may click and type, after you confirm)",webhooks:"Webhooks (listener; needs its own enabled file)",memory:"Memory (saves what you say or type, on this machine)"};
  for(const k of Object.keys(names)){const l=el("label");const c=el("input",undefined,{type:"checkbox"});c.checked=state.features[k];
    c.addEventListener("change",()=>{state.features[k]=c.checked});l.append(c," "+names[k]);box.append(l)}}
function drawRoles(){
  const box=$("roles");box.replaceChildren();
  for(const role of Object.keys(S.roles)){
    const info=S.roles[role],chain=state.roles[role]||[];
    const wrap=el("div");wrap.append(el("strong",role));
    chain.forEach((slot,i)=>{const row=el("div");row.className="row";
      row.append(el("span",(i+1)+". "+slot.provider));
      if(slot.provider!=="llm.fast"){const m=el("input",undefined,{placeholder:"model id","aria-label":"model id for "+slot.provider,size:"28"});
        m.value=slot.model||"";m.addEventListener("input",()=>{slot.model=m.value});row.append(m)}
      const x=el("button","remove");x.addEventListener("click",()=>{chain.splice(i,1);if(!chain.length)delete state.roles[role];drawRoles();drawTests()});row.append(x);wrap.append(row)});
    const row=el("div");row.className="row";const sel=el("select");
    for(const c of info.candidates)sel.append(el("option",c,{value:c}));
    const add=el("button","add to the fallback chain");
    add.addEventListener("click",()=>{const d=info.defaults[sel.value]||{};(state.roles[role]=state.roles[role]||[]).push({provider:sel.value,model:d.model||""});drawRoles();drawTests()});
    row.append(sel,add);wrap.append(row);
    wrap.append(el("div",chain.length?"Tried in order; a switch to the next one is always reported.":"Not set: Glide's built-in chain is used.",{class:"mut"}));
    box.append(wrap)}}
function drawKeys(){
  const box=$("keys");box.replaceChildren();
  for(const p of S.providers){if(!p.env)continue;
    const wrap=el("div");const head=el("div");head.className="row";
    head.append(el("strong",p.name),el("span","reads "+p.env),el("span","status: "+p.status,{class:p.status==="missing"?"bad":"ok"}));wrap.append(head);
    const row=el("div");row.className="row";
    if(p.link){const a=el("a","Where to get a key (check the provider's site)",{href:p.link,target:"_blank",rel:"noopener noreferrer"});row.append(a)}
    const inp=el("input",undefined,{type:"password",autocomplete:"off","aria-label":"paste a key for "+p.name,size:"24",placeholder:"optional: paste a key"});
    const b=el("button","hold in memory");const msg=el("span","",{class:"mut"});
    b.addEventListener("click",async()=>{try{const r=await call("key",{provider:p.name,value:inp.value});inp.value="";p.status=r.status;drawKeys();drawTests()}catch(e){fail(msg,e)}});
    row.append(inp,b,msg);wrap.append(row);wrap.append(el("pre",p.export));box.append(wrap)}}
function drawTests(){
  const box=$("tests");box.replaceChildren();let any=false;
  for(const role of Object.keys(state.roles))for(const slot of state.roles[role]){if(slot.provider==="llm.fast")continue;any=true;
    const row=el("div");row.className="row";const out=el("span","",{class:"mut"});
    const b=el("button","Test "+slot.provider+" ("+role+")");
    b.addEventListener("click",async()=>{b.disabled=true;out.textContent="testing...";out.className="mut";
      try{const r=await call("test",{confirm:true,provider:slot.provider,role,state});out.textContent=r.status+" - "+r.detail;out.className=r.status==="ok"?"ok":"bad"}catch(e){fail(out,e)}b.disabled=false});
    row.append(b,out);box.append(row)}
  if(!any)box.append(el("div","Choose providers in step 2 to test them.",{class:"mut"}))}
$("presetbtn").addEventListener("click",async()=>{try{const r=await call("preset",{provider:$("presetsel").value});state.roles=r.roles;drawRoles();drawTests()}catch(e){fail($("toml"),e)}});
$("renderbtn").addEventListener("click",async()=>{try{const r=await call("render",state);$("toml").textContent=r.toml;$("toml").className=""}catch(e){fail($("toml"),e)}});
$("writebtn").addEventListener("click",async()=>{try{const r=await call("write",state);$("written").textContent="Wrote "+r.written+(r.backup?" (previous file saved as "+r.backup+")":"");$("written").className="ok"}catch(e){fail($("written"),e)}});
function drawModes(){const box=$("modes");
  for(const m of ["ask","chat","voice","ui"]){const b=el("button","Start: "+m);
    b.addEventListener("click",async()=>{try{
      let confirm_act=false;if(state.features.computer){confirm_act=window.confirm("Computer control is on. Glide may click and type on this Mac. Continue?");if(!confirm_act)return}
      const r=await call("launch",{mode:m,text:$("asktext").value,state,confirm_act});$("launchout").textContent="Equivalent command:\n"+r.command+"\n\n"+r.output;$("launchout").className=""}
      catch(e){fail($("launchout"),e)}});box.append(b)}}
(async()=>{
  const r=await fetch("/api/state",{headers:{"X-Glide-Session":SESSION}});S=await r.json();
  const sel=$("presetsel");for(const p of S.providers)if(p.env)sel.append(el("option",p.name,{value:p.name}));
  drawFeatures();drawRoles();drawKeys();drawTests();drawModes()})().catch(e=>{document.body.append(el("pre",String(e)))});
</script></body></html>
"""


def render_page(session: str, nonce: str) -> str:
    return PAGE.replace("__NONCE__", nonce).replace("__SESSION__", session)
