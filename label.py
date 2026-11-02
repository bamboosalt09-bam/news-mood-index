"""브라우저에서 키보드로 감성 라벨 붙이기.

python label.py ko      (또는 en) → 브라우저가 열린다. 누를 때마다 labeled/ko.tsv에 저장된다.
끝낼 때는 이 창에서 Ctrl+C.
"""
import json
import random
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HOME = Path(__file__).parent
PORT = 8765
SCORES = {"-2", "-1", "0", "1", "2"}

PAGE = """<!doctype html><html lang="ko"><meta charset="utf-8"><title>뉴스 감성 라벨</title>
<style>
body{font-family:system-ui,'Malgun Gothic',sans-serif;max-width:860px;margin:40px auto;padding:0 16px;background:#fafafa;color:#222}
.meta{color:#888;font-size:14px}
#title{font-size:26px;font-weight:600;line-height:1.45;margin:18px 0 10px}
#title a{color:inherit;text-decoration:none} #title a:hover{text-decoration:underline}
#summary{color:#555;min-height:1.5em;line-height:1.5}
.btns{display:flex;gap:8px;margin:28px 0 18px}
.btns button{flex:1;padding:14px 0;font-size:16px;border:3px solid transparent;border-radius:8px;cursor:pointer}
.btns button.on{border-color:#222}
.b0{background:#f2a8a8}.b1{background:#f9d6d6}.b2{background:#e4e4e4}.b3{background:#d3eed3}.b4{background:#a3d9a3}
select{font-size:15px;padding:4px}
.help{color:#888;font-size:13px;margin-top:28px;line-height:1.7}
</style>
<div class="meta"><span id="prog"></span> · <span id="src"></span></div>
<div id="title"></div><div id="summary"></div>
<div class="btns" id="btns"></div>
<label>주제 <select id="topic"></select></label>
<div class="help">1 매우부정 · 2 부정 · 3 중립 · 4 긍정 · 5 매우긍정 → 저장하고 다음으로<br>
← → 이전/다음 · O 기사 열기 · 주제가 틀렸으면 목록에서 고친 뒤 숫자키</div>
<script>
const S=[[-2,'매우부정'],[-1,'부정'],[0,'중립'],[1,'긍정'],[2,'매우긍정']];
const $=id=>document.getElementById(id);
let items=[],i=0;
S.forEach(([v,name],k)=>{const b=document.createElement('button');b.className='b'+k;
  b.textContent=(k+1)+'  '+name;b.onclick=()=>label(v);$('btns').appendChild(b)});
fetch('/api/items').then(r=>r.json()).then(d=>{
  items=d.items;
  $('topic').innerHTML='<option value="">(미분류)</option>'+d.topics.map(t=>'<option></option>').join('');
  d.topics.forEach((t,k)=>$('topic').options[k+1].textContent=t);
  i=items.findIndex(x=>x.score==='');if(i<0)i=items.length-1;show()});
function show(){
  const it=items[i],done=items.filter(x=>x.score!=='').length;
  $('prog').textContent=done+' / '+items.length+' 완료 · '+(i+1)+'번째';
  $('src').textContent=it.source;
  const a=document.createElement('a');a.href=it.link;a.target='_blank';a.textContent=it.title;
  $('title').replaceChildren(a);$('summary').textContent=it.summary;
  $('topic').value=it.topic;
  [...$('btns').children].forEach((b,k)=>b.classList.toggle('on',it.score===String(S[k][0])));
}
async function label(v){
  // 먼저 다음 제목으로 넘기고 저장은 뒤에서: 빠르게 연타해도 제목마다 한 번씩만 저장
  const it=items[i],topic=$('topic').value,prev=[it.score,it.topic];
  it.score=String(v);it.topic=topic;if(i<items.length-1)i++;show();
  let ok=false;
  try{ok=(await fetch('/api/label',{method:'POST',body:JSON.stringify({title:it.title,score:v,topic})})).ok}catch(e){}
  if(!ok){[it.score,it.topic]=prev;show();alert('저장 실패 (서버가 꺼졌나요?): '+it.title)}
}
$('topic').onchange=()=>$('topic').blur();
document.onkeydown=e=>{
  if(e.target.tagName==='SELECT')return;
  if(e.key>='1'&&e.key<='5')label(S[e.key-1][0]);
  else if(e.key==='ArrowLeft'&&i>0){i--;show()}
  else if(e.key==='ArrowRight'&&i<items.length-1){i++;show()}
  else if(e.key==='o'||e.key==='O')window.open(items[i].link,'_blank');
};
</script></html>"""


def load(lang):
    lines = (HOME / "topics.txt").read_text(encoding="utf-8").splitlines()
    topics = {name: scope for scope, name in (l.split("\t") for l in lines if l.strip())}
    path = HOME / "labeled" / f"{lang}.tsv"
    old = {}
    if path.exists():
        for r in [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")][1:]:
            old[r[3]] = {"score": r[0], "topic": r[2]}
    items = []
    for _, source, title, link, summary in (l.rstrip("\n").split("\t") for l in open(HOME / "headlines" / f"{lang}.tsv", encoding="utf-8")):
        items.append({"title": title, "link": link, "summary": summary, "source": source,
                      **old.get(title, {"score": "", "topic": ""})})
    todo = [it for it in items if it["score"] == ""]
    random.shuffle(todo)  # 안 붙인 것은 섞어서 뒤에, 붙인 것은 앞에 (← 로 돌아가 고칠 수 있게)
    return topics, [it for it in items if it["score"] != ""] + todo, path


def save(path, items, topics):
    rows = ["감성\t범위\t주제\t제목"] + [
        f"{it['score']}\t{topics.get(it['topic'], '')}\t{it['topic']}\t{it['title']}"
        for it in items if it["score"] != "" or it["topic"]]
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(rows) + "\n", encoding="utf-8")
    tmp.replace(path)  # 저장 도중 꺼져도 파일이 깨지지 않게


def main(lang):
    topics, items, path = load(lang)
    by_title = {it["title"]: it for it in items}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body=b"", kind="text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/items":
                data = json.dumps({"items": items, "topics": list(topics)}, ensure_ascii=False)
                self.reply(200, data.encode(), "application/json; charset=utf-8")
            else:
                self.reply(200, PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            it = by_title.get(d.get("title"))
            score, topic = str(d.get("score")), d.get("topic", "")
            if self.path != "/api/label" or not it or score not in SCORES or (topic and topic not in topics):
                return self.reply(400, "잘못된 요청".encode())
            it["score"], it["topic"] = score, topic
            save(path, items, topics)
            self.reply(204)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"{lang}: {sum(it['score'] != '' for it in items)}/{len(items)} 완료. "
          f"http://127.0.0.1:{PORT} (끝내려면 Ctrl+C)")
    webbrowser.open(f"http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n저장됨 → {path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] in (["ko"], ["en"]):
        main(sys.argv[1])
    else:
        print(__doc__)
