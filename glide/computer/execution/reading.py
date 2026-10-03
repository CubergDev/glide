"""Bounded, fixed-code page reading shared by every browser provider (CDP, Obscura, Playwright CLI).

The browser runs one fixed, read-only script and returns plain data. Nothing here is generated, and nothing the
page says is ever interpreted: `page_record` treats even the adapter's output as untrusted data, bounds it, and
stamps it with a host-generated observation time.
"""

from datetime import UTC, datetime

from .contracts import InvalidAction, safe_url

MAX_TEXT = 16000  # characters of visible body text kept per page
MAX_LINKS = 60  # visible links kept per page
MAX_TITLE = 500
MAX_LINK_TITLE = 240
MAX_URL = 2048
MAX_NODES = 12000  # text nodes the script visits before it stops reading
MAX_ANCHORS = 2000  # anchors the script scans before it stops collecting links

# The only DOM policy. Text and links are skipped inside these (scripts, form controls, hidden content).
SKIPPED = 'script,style,noscript,template,input,textarea,select,[contenteditable],[hidden],[aria-hidden="true"]'

_SCRIPT = r"""(() => {
  const root=document.body||document.documentElement, chunks=[], links=[];
  const skip='__SKIPPED__';
  const visible=el=>el && !el.closest(skip) && el.getClientRects().length>0
    && !['hidden','collapse'].includes(getComputedStyle(el).visibility);
  const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
  let node, size=0, visited=0;
  while((node=walker.nextNode()) && ++visited<=__NODES__ && size<__TEXT__) {
    if(!visible(node.parentElement)||node.parentElement.closest('nav,footer'))continue;
    const text=(node.textContent||'').replace(/\s+/g,' ').trim();
    if(text){const part=text.slice(0,__TEXT__-size);chunks.push(part);size+=part.length+1;}
  }
  const seen=new Set();let scannedLinks=0;
  // Navigation/footer labels are body-text noise, but their visible links are
  // evidence of real destinations (visitor information, admission, contact).
  for(const el of root.querySelectorAll('a[href]')) {
    if(links.length>=__LINKS__||++scannedLinks>__ANCHORS__)break;
    if(!visible(el))continue;
    let url;try {url=new URL(el.href,location.href);}catch {continue;}
    if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.href.length>__URL__||seen.has(url.href))continue;
    seen.add(url.href);links.push({url:url.href,title:(el.innerText||el.getAttribute('aria-label')||'').trim().slice(0,__LINK_TITLE__)});
  }
  return {url:location.href,title:document.title.slice(0,__TITLE__),document_id:String(performance.timeOrigin),
    text:chunks.join('\n').slice(0,__TEXT__),links,truncated:!!node};
})()"""

PAGE_SCRIPT = (
    _SCRIPT.replace("__SKIPPED__", SKIPPED)
    .replace("__NODES__", str(MAX_NODES))
    .replace("__TEXT__", str(MAX_TEXT))
    .replace("__LINKS__", str(MAX_LINKS))
    .replace("__ANCHORS__", str(MAX_ANCHORS))
    .replace("__URL__", str(MAX_URL))
    .replace("__LINK_TITLE__", str(MAX_LINK_TITLE))
    .replace("__TITLE__", str(MAX_TITLE))
)


def page_record(data):
    """One page as evidence: bounded, validated, deduplicated and stamped with the host's own clock.

    Anything the browser returned that does not fit the contract is an error, never silently repaired. A link that
    is merely unusable (unsafe scheme, credentials, too long) is dropped, not trusted.
    """
    if not isinstance(data, dict):
        raise InvalidAction("The browser did not return readable page data")
    for key, limit in (("url", MAX_URL), ("title", MAX_TITLE), ("text", MAX_TEXT)):
        if not isinstance(data.get(key), str) or len(data[key]) > limit:
            raise InvalidAction("The page reading exceeded its content limits")
    if not safe_url(data["url"]):
        raise InvalidAction("Open a web page before collecting evidence")
    raw_links = data.get("links")
    if not isinstance(raw_links, list) or len(raw_links) > MAX_LINKS or type(data.get("truncated")) is not bool:
        raise InvalidAction("The page reader returned an invalid link list")
    links, seen = [], set()
    for item in raw_links:
        if not isinstance(item, dict) or set(item) != {"url", "title"}:
            raise InvalidAction("The page reader returned an invalid link")
        url, title = item["url"], item["title"]
        if not isinstance(url, str) or not isinstance(title, str):
            raise InvalidAction("The page reader returned an invalid link")
        if url not in seen and len(url) <= MAX_URL and len(title) <= MAX_LINK_TITLE and safe_url(url):
            seen.add(url)
            links.append({"url": url, "title": title})
    return {
        "url": data["url"],
        "title": data["title"],
        "text": data["text"],
        "links": links,
        "truncated": data["truncated"],
        "observed_at": datetime.now(UTC).isoformat(),
    }
