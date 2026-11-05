'use strict';
// 오늘의 뉴스 분위기: data/latest.json (없으면 data/sample.json)만 읽어 그린다. 형식은 web/DATA.md.

const NS = 'http://www.w3.org/2000/svg';
const MINUS = '−';
const view = document.getElementById('view');
let DATA = null, CUR = null, CHARTS = [];

// ---------- 작은 도우미 ----------
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat(2)) if (c != null && c !== false) el.append(c instanceof Node ? c : String(c));
  return el;
}
function sv(tag, attrs, txt) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (txt != null) el.textContent = txt;
  return el;
}
const ok = v => v != null && Number.isFinite(v);
function signed(x, d) {
  if (!ok(x)) return '–';
  const r = Number(x.toFixed(d));
  return (r > 0 ? '+' : r < 0 ? MINUS : '') + Math.abs(r).toFixed(d);
}
const fz = x => signed(x, 1);                      // z: 소수 한 자리
const fpct = x => ok(x) ? signed(x, 2) + '%' : '자료 없음'; // 등락률: 소수 둘째
const fsc = x => signed(x, 2);                     // 원점수 평균 (−2~+2)
const f2 = x => ok(x) ? x.toFixed(2) : '–';
const fnum = x => ok(x) ? x.toLocaleString('ko-KR') : '–';
const fshare = x => ok(x) ? Math.round(x * 100) + '%' : '–';
const ftime = s => s ? String(s).slice(0, 16).replace('T', ' ') : '';
const tone = z => !ok(z) || Math.abs(z) < 0.5 ? '' : z > 0 ? 'pos' : 'neg';
const safeLink = u => /^https?:\/\//i.test(u || '') ? u : null;
function nice(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return Math.ceil((v / p) * 2) / 2 * p;
}

function card(id, title, sub, ...kids) {
  return h('section', { class: 'card', id, 'aria-labelledby': id + '-h' },
    h('h2', { id: id + '-h' }, title), sub ? h('p', { class: 'sub' }, sub) : null, kids);
}
function table(caption, head, rows) {
  return h('div', { class: 'table-scroll', tabindex: '0', role: 'region', 'aria-label': caption },
    h('table', {}, h('caption', { class: 'sr' }, caption),
      h('thead', {}, h('tr', {}, head.map(x => h('th', { scope: 'col' }, x)))),
      h('tbody', {}, rows.map(r => h('tr', {}, r.map((x, i) => i ? h('td', {}, x) : h('th', { scope: 'row' }, x)))))));
}
const tile = (label, value, note) =>
  h('div', { class: 'tile' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value), note ? h('div', { class: 'note' }, note) : null);
const key = (cls, text) => h('span', {}, h('i', { class: 'key', style: `background:var(${cls})`, 'aria-hidden': 'true' }), text);

// ---------- 읽기 ----------
async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(url + ' ' + r.status);
  const d = await r.json();
  if (!d || !d.markets || !Object.keys(d.markets).length) throw new Error(url + ' 형식 오류');
  return d;
}

async function start() {
  let sample = false;
  try { DATA = await getJSON('data/latest.json'); }
  catch (e) {
    try { DATA = await getJSON('data/sample.json'); sample = true; }
    catch (e2) {
      view.replaceChildren(h('p', { role: 'alert' }, '자료를 불러오지 못했습니다. 잠시 뒤 다시 열어 주세요.'));
      return;
    }
  }
  if (sample) {
    const n = document.getElementById('notice');
    n.textContent = '시험용 자료: 지금 보이는 숫자와 제목은 화면 확인용 가짜 자료입니다.';
    n.hidden = false;
  }
  if (DATA.frozen) {  // 공개용 고정본 (web/freeze.py): 실시간으로 바뀌지 않음
    const n = document.getElementById('notice');
    n.textContent = `고정본: ${ftime(DATA.frozen.at)} (한국 시간) 기준으로 멈춘 화면입니다. 실시간으로 바뀌지 않습니다.` + (DATA.frozen.note ? ' ' + DATA.frozen.note : '');
    n.hidden = false;
  }
  if (DATA.updated) document.getElementById('updated').textContent = '자료 갱신 ' + ftime(DATA.updated) + ' (한국 시간)';
  const ids = Object.keys(DATA.markets);
  CUR = ids[0];
  document.getElementById('tabs').replaceChildren(...ids.map(id => {  // 시장 고르기 = 두 시장 오늘 한눈에
    const m = DATA.markets[id], t = m.today || {};
    return h('button', { type: 'button', 'data-m': id, onclick: () => { CUR = id; render(); } },
      h('span', { class: 'mk' }, m.name, h('small', {}, String(m.news || '').replace(/\s*\(.*\)$/, ''))),
      h('span', { class: 'ml' }, h('i', { class: 'dot ' + tone(t.mood_rel), 'aria-hidden': 'true' }), t.mood_label || '–'),
      h('span', { class: 'mc' }, '그날 ' + fpct(t.change_pct)));
  }));
  render();
  addEventListener('resize', () => requestAnimationFrame(() => CHARTS.forEach(drawChart)));
}

function render() {
  const m = DATA.markets[CUR];
  for (const b of document.querySelectorAll('#tabs button')) b.setAttribute('aria-pressed', String(b.dataset.m === CUR));
  document.querySelector('.jump a[href="#s-summary"]').hidden = !m.summary;
  CHARTS = [];
  view.replaceChildren(...[secLive(m), secToday(m), secSummary(m), secClusters(m), secTrend(m), secKinds(m), secTopics(m), secSectors(m), secMethod(m)].filter(Boolean));
  CHARTS.forEach(drawChart);
}

// ---------- 칸들 ----------
const topText = (r, what) => r ? `${what} 가운데 ${r.side} 쪽 상위 ${r.pct}%` : '';  // 상위 몇 % (build.top)

function rankText(m) {  // z 대신 쉬운 말: 앞 60거래일 가운데 오늘보다 더 치우친 날 수
  const s = m.series || [], t = m.today || {};
  if (t.rank) return `최근 ${t.rank.n}거래일 가운데 ${t.rank.side} 쪽 상위 ${t.rank.pct}% (오늘만큼 이상 ${t.rank.side === '긍정' ? '긍정적' : '부정적'}이었던 날 ${t.rank.days}일)`;
  const prev = s.slice(-61, -1).map(d => d.mood).filter(ok);
  if (!ok(t.mood) || prev.length < 20) return '';
  const neg = !(t.mood_rel > 0);
  const k = prev.filter(v => neg ? v < t.mood : v > t.mood).length;
  return `최근 ${prev.length}거래일 가운데 오늘보다 ${neg ? '부정적' : '긍정적'}이었던 날 ${k}일`;
}

function gauge(z) {  // 라벨 경계 ±0.5·±1.5를 −2.5~+2.5 막대 위에 (다섯 칸 같은 폭)
  const pos = ok(z) ? (Math.max(-2.5, Math.min(2.5, z)) + 2.5) / 5 * 100 : 50;
  return h('div', { class: 'gauge', 'aria-hidden': 'true' },
    h('div', { class: 'gbar' }, ['매우 부정', '부정', '중립', '긍정', '매우 긍정'].map((x, i) => h('span', { class: 'g' + i }, x))),
    h('i', { class: 'gmark', style: `left:${pos.toFixed(1)}%` }));
}

function secToday(m) {
  const t = m.today || {};
  const s = (m.series || []).slice(-30);
  const mini = [{ title: `최근 ${s.length}거래일 ${m.name} 종가(회색, 왼쪽)와 뉴스 지수(보라, 오른쪽)`, short: m.name, get: d => d.close, type: 'line', fmt: fnum,
    get2: d => d.mood, short2: '뉴스 지수', fmt2: fsc }];
  return h('section', { class: 'card', id: 's-today', 'aria-labelledby': 's-today-h' },
    h('p', { class: 'eyebrow' }, `${m.news} · ${m.date} 기준`),
    h('h2', { id: 's-today-h', class: 'sr' }, '오늘 분위기'),
    h('div', { class: 'hero' }, h('span', { class: 'dot ' + tone(t.mood_rel), 'aria-hidden': 'true' }), h('strong', {}, t.mood_label || '–')),
    gauge(t.mood_rel),
    h('p', { class: 'sub' }, rankText(m) || '오늘 뉴스 분위기'),
    h('div', { class: 'tiles' },
      tile('그날 등락률', fpct(t.change_pct), ok(t.close) ? '종가 ' + fnum(t.close) : ''),
      tile('뉴스 지수', fsc(t.mood), ok(t.usual_mood) ? `평소(앞 60일) ${fsc(t.usual_mood)} ± ${Math.abs(t.usual_sd).toFixed(2)}` : '제목 점수(−2~+2) 평균'),
      tile('뉴스 충격도', t.shock_label || '–', `평소보다 센 기사가 많은지 · 호재·악재 엇갈림 ${t.spread_label || '–'}`),
      tile('판정한 제목', fnum(t.titles) + '개', `부정 ${fshare(t.neg_share)} · 긍정 ${fshare(t.pos_share)}`)),
    weekendBox(m),
    s.length > 5 ? chartBox(mini[0].title, s, mini) : null,
    h('p', { class: 'small' }, '뉴스 제목이 그날 시장을 어떻게 썼는지 보여 주는 현황판입니다. 다음 날 시장을 맞히지는 못합니다. ',
      h('a', { href: '#s-method' }, '계산 방법과 기존 지수와 다른 점')));
}

function secLive(m) {  // 실시간(잠정): 아직 마감 안 된 다음 거래일 몫, 30분마다 갱신
  const v = m.live;
  if (!v) return null;
  const md = d => d.slice(5).replace('-', '/');
  const head = (x, cls) => x && h('li', {}, h('span', { class: 'dot ' + cls, 'aria-hidden': 'true' }),
    safeLink(x.link) ? h('a', { href: x.link, target: '_blank', rel: 'noopener noreferrer' }, x.title) : x.title);
  const zone = m.id === 'en' ? '미국 동부' : '한국';
  return h('section', { class: 'card live', id: 's-live', 'aria-labelledby': 's-live-h' },
    h('p', { class: 'eyebrow' }, h('span', { class: 'badge test' }, DATA.frozen ? '잠정 · 고정본' : '실시간 · 잠정'), ` ${md(v.date)} ${v.status} · ${zone} ${v.updated} 갱신`),
    h('h2', { id: 's-live-h' }, v.mood_label ? [h('span', { class: 'dot ' + tone(v.mood_rel), 'aria-hidden': 'true' }), `지금까지 ${v.mood_label}`] : '아직 제목이 모자람'),
    h('p', { class: 'meta' }, `지금까지 제목 ${fnum(v.titles)}개` + (ok(v.mood) ? ` · 평균 ${fsc(v.mood)} (평소 ${v.usual_kind === '같은 시각까지' ? '같은 시각까지 ' : ''}${fsc(v.usual_mood)} ± ${Math.abs(v.usual_sd).toFixed(2)})` : '') +
      (ok(v.change_pct) ? ` · ${m.name} ${fpct(v.change_pct)} (${v.price_time})` : '') + (v.rank ? ' · ' + topText(v.rank, `평소 ${v.rank.n}일`) : '')),
    (v.top_negative || []).length || (v.top_positive || []).length ? h('ul', { class: 'stitles' }, head(v.top_negative[0], 'neg'), head(v.top_positive[0], 'pos')) : null,
    v.weekend ? h('p', { class: 'meta' }, `진행 중인 주말·휴장 몫: 제목 ${fnum(v.weekend.titles)}개 · 평균 ${fsc(v.weekend.mood)} (마감 뒤 다음 거래일 지수에는 넣지 않음)`) : null,
    h('p', { class: 'small' }, `${md(v.date)} 장이 끝나기 전까지 들어온 제목으로 ${DATA.frozen ? '고정 시각까지 센 값입니다' : '10분마다 다시 셉니다'}. 제목이 덜 모여 흔들릴 수 있고, 마감 뒤에는 아래 확정 값으로 바뀝니다. 전날 장 마감 뒤 기사(마감 시황 포함)도 이 날 몫입니다.`));
}

function weekendBox(m) {  // 거래일 바로 앞 주말·휴장 동안의 뉴스 분위기 (그 거래일 지수에는 넣지 않음)
  const w = m.weekend;
  if (!w) return null;
  const md = d => d.slice(5).replace('-', '/');
  const head = (x, cls) => x && h('li', {}, h('span', { class: 'dot ' + cls, 'aria-hidden': 'true' }),
    safeLink(x.link) ? h('a', { href: x.link, target: '_blank', rel: 'noopener noreferrer' }, x.title) : x.title);
  return h('div', { class: 'weekend' },
    h('p', {}, h('strong', {}, `주말·휴장 동안 (${md(w.from)}~${md(w.to)}) 뉴스 분위기: `),
      h('span', { class: 'dot ' + tone(w.mood_rel), 'aria-hidden': 'true' }), `${w.mood_label} · 제목 ${fnum(w.titles)}개 · 평균 ${fsc(w.mood)}` + (w.rank ? ' · ' + topText(w.rank, `지난 주말·휴장 ${w.rank.n}번`) : '')),
    h('ul', { class: 'stitles' }, head((w.top_negative || [])[0], 'neg'), head((w.top_positive || [])[0], 'pos')),
    h('p', { class: 'small' }, `장이 쉬는 동안 나온 기사는 ${m.date} 지수에 넣지 않고 따로 셉니다 (금요일 장 마감 뒤부터 일요일 오후까지). 평소 주말과 견준 수준입니다.`));
}

function secSummary(m) {
  const s = m.summary;
  if (!s) return null;
  const all = [...(m.top_negative || []), ...(m.top_positive || [])];
  const chips = (s.sources || []).map(id => all.find(c => c.id === id)).filter(Boolean)
    .map(c => h('a', { class: 'chip', href: '#' + c.id, title: c.title }, c.title));
  return card('s-summary', 'AI 해설', null,
    h('div', { class: 'ai-head' }, h('span', { class: 'badge' }, 'AI 요약'),
      s.model === 'dry-run' ? h('span', { class: 'badge test' }, '시험용 해설') : null),
    h('div', { class: 'ai' },
      h('h3', {}, s.headline || ''),
      h('ul', {}, (s.bullets || []).map(b => h('li', {}, b))),
      s.watch ? h('p', { class: 'watch' }, h('strong', {}, '지켜볼 점 '), s.watch) : null,
      chips.length ? [h('p', { class: 'small' }, '근거 기사 묶음 (누르면 아래 목록으로 이동)'), h('div', { class: 'chips' }, chips)] : null,
      h('p', { class: 'disclaimer' }, 'AI(Claude Haiku)가 아래 근거 제목만 보고 쓴 요약이며, 투자 조언이 아닙니다. 보도 내용을 옮긴 것으로 원인을 단정하지 않습니다.' +
        (s.generated ? ` · 작성 ${ftime(s.generated)}` : ''))));
}

function secClusters(m) {
  const list = (items, title, cls) => h('div', {},
    h('h3', {}, h('span', { class: 'dot ' + cls, 'aria-hidden': 'true' }), title),
    items && items.length
      ? h('ol', {}, items.map(c => h('li', { id: c.id, tabindex: '-1' },
          safeLink(c.link) ? h('a', { href: c.link, target: '_blank', rel: 'noopener noreferrer' }, c.title) : h('span', {}, c.title),
          h('div', { class: 'meta' }, `점수 ${signed(c.score, 1)} · 같은 내용 ${fnum(c.count)}건` + (c.topic ? ` · ${c.topic}` : '')))))
      : h('p', { class: 'small' }, '해당 묶음 없음'));
  return card('s-clusters', '오늘 지수를 움직인 기사 묶음',
    '거의 같은 제목끼리 묶어, 그날 평균을 가장 많이 끌어내린 묶음과 끌어올린 묶음을 5개씩 보여 줍니다. 대표 제목은 원문으로 이어집니다.' +
      ([...(m.top_negative || []), ...(m.top_positive || [])].some(c => c.check) ? ' 점수 모델이 가끔 방향을 거꾸로 읽어서, 대표 제목은 AI(Claude Haiku)가 한 번 더 확인해 방향이 반대인 묶음은 빼고 다음 묶음을 올립니다.' : ''),
    h('div', { class: 'cols' }, list(m.top_negative, '끌어내린 묶음', 'neg'), list(m.top_positive, '끌어올린 묶음', 'pos')));
}

function spark(pts) {  // 섹터 카드용 작은 막대: 최근 지수일 분위기 z (±2.5에서 자름)
  const n = pts.length, svg = sv('svg', { viewBox: `0 0 ${n * 4} 24`, preserveAspectRatio: 'none', class: 'spark', 'aria-hidden': 'true' });
  svg.append(sv('line', { x1: 0, x2: n * 4, y1: 12, y2: 12, class: 'base-line' }));
  pts.forEach((p, i) => {
    if (!ok(p.mood_rel)) return;
    const v = Math.max(-2.5, Math.min(2.5, p.mood_rel)) / 2.5 * 11;
    svg.append(sv('rect', { x: i * 4 + 0.5, width: 3, y: v > 0 ? 12 - v : 12, height: Math.max(0.5, Math.abs(v)), class: v > 0 ? 'bar-pos' : 'bar-neg' }));
  });
  return svg;
}

function secSectors(m) {
  const sc = m.sectors;
  if (!sc || !(sc.list || []).length) return null;
  const head = (x, cls) => x && h('li', {}, h('span', { class: 'dot ' + cls, 'aria-hidden': 'true' }),
    safeLink(x.link) ? h('a', { href: x.link, target: '_blank', rel: 'noopener noreferrer' }, x.title) : x.title);
  const LTONE = { '좋은 소식 많음': 'pos', '나쁜 소식 많음': 'neg' };
  const move = x => x.move && ok(x.move.pct)
    ? h('p', { class: 'smove' }, '그날 업체 주가 ', h('b', {}, fpct(x.move.pct)),
        ok(x.move.excess) ? ` (코스피보다 ${x.move.excess >= 0 ? '나음' : '못함'} ${fpct(Math.abs(x.move.excess)).replace(/^\+/, '')}p)` : '')
    : null;
  const one = x => h('div', { class: 'scard' },
    h('div', { class: 'shead' }, h('strong', {}, x.name, x.scoop ? h('span', { class: 'badge' }, '그날의 특종') : null),
      h('span', { class: 'ml' }, h('i', { class: 'dot ' + (LTONE[x.label] || ''), 'aria-hidden': 'true' }), x.label)),
    x.line ? h('p', { class: 'sline' }, x.line) : null,
    x.scoop ? h('p', { class: 'meta' }, `그날 전체 제목 중 ${(100 * x.scoop.share).toFixed(1)}% (평소 ${(100 * x.scoop.share_usual).toFixed(1)}%)`) : null,
    move(x),
    (x.spark || []).length > 5 ? spark(x.spark) : null,
    h('p', { class: 'meta' }, `제목 ${fnum(x.titles)}개 · 평균 점수 ${fsc(x.mood)}`),
    (x.companies || []).length ? h('p', { class: 'meta' }, x.companies.map(c => `${c.name} ${fsc(c.mood)} (${c.titles})`).join(' · ')) : null,
    h('ul', { class: 'stitles' }, head((x.top_negative || [])[0], 'neg'), head((x.top_positive || [])[0], 'pos')));
  const short = (sc.short || []).filter(x => x.titles > 0);
  const ai = sc.list.some(x => x.judge_model && x.judge_model !== 'dry-run');
  return h('section', { class: 'card', id: 's-sectors', 'aria-labelledby': 's-sectors-h' },
    h('h2', { id: 's-sectors-h' }, '섹터별 동향 ', h('span', { class: 'badge test' }, '실험 기능')),
    h('p', { class: 'sub' }, `${ai ? 'AI(Claude Haiku)가 섹터마다 그날의 서로 다른 소식을 읽고 좋은 소식이 많았는지 판정했습니다 (같은 기사가 여러 번 나와도 소식 하나로 봄, 주가는 보지 않음).' : '섹터 제목 점수의 그날 평균과 그 섹터의 앞 60일 대비 수준이 같은 쪽으로 뚜렷할 때만 좋은·나쁜 소식 많음으로 표시합니다.'} ` +
      `${m.id === 'en' ? '' : '업체 이름이 나온 제목은 같은 날·전날 그 업체 기사를 함께 읽은 AI가 그 업체 주가 관점으로 판정한 점수를 씁니다. '}` +
      `${m.id === 'en' ? '영어 섹터는 업체 주가 줄이 없습니다.' : '"그날 업체 주가"는 그 섹터 대표 업체들의 실제 등락(시가총액 무게)으로, 뉴스 판정과 나란히 견주어 보라고 둔 것입니다.'}`),
    h('div', { class: 'sgrid' }, sc.list.map(one)),
    short.length ? h('p', { class: 'small' }, '오늘 제목(20개)이나 견줄 지난 날이 모자라 뺀 섹터: ' + short.map(x => `${x.name} ${x.titles}개`).join(', ')) : null,
    sc.list.some(x => x.memory_share != null && x.memory_share < 0.9) ? h('p', { class: 'small' }, '오늘 업체 제목 일부는 아직 AI 판정 전이라 기본 점수로 셌습니다.') : null,
    h('p', { class: 'small' }, `${m.id === 'en' ? '반도체·빅테크' : '반도체'}는 늘 보이고, 그날 전체 제목 중 비중이 평소(앞 60일)보다 크게 늘어난 섹터가 있으면 그중 하나를 '그날의 특종'으로 더 보입니다. ` + '막대는 최근 지수일마다 그 섹터의 앞 60일 대비 뉴스 분위기, 기업 이름 옆 괄호는 그날 제목 수(5개 이상만)입니다. 섹터는 기사 하나가 크게 흔들 수 있어 실험 기능으로 둡니다. 투자 조언이 아닙니다.'));
}

function chartBox(label, series, panels, extra) {
  const el = h('div', { class: 'chart', tabindex: '0', role: 'group',
    'aria-label': label + '. 좌우 화살표로 날짜를 옮기면 그날 값이 나옵니다. 같은 값은 아래 표에도 있습니다.' });
  CHARTS.push({ el, series, panels, extra });
  return el;
}
const barLegend = () => h('p', { class: 'legend' },
  key('--pos', '0보다 큼 (긍정·상승)'), key('--neg', '0보다 작음 (부정·하락)'));

function secTrend(m) {
  const s = m.series || [];
  const corr = m.method && m.method.same_day_corr;
  const panels = [
    { title: `${m.name} 종가(회색, 왼쪽 눈금)와 뉴스 분위기 지수(보라, 오른쪽 눈금)`, short: m.name, get: d => d.close, type: 'line', fmt: fnum,
      get2: d => d.mood, short2: '뉴스 지수', fmt2: fsc, tall: true },
    { title: '분위기 (앞 60거래일 대비 z)', short: '분위기 z', get: d => d.mood_rel, type: 'bar', fmt: fz },
    { title: '그날 등락률 %', short: '등락률', get: d => d.change_pct, type: 'bar', fmt: fpct },
    { title: '뉴스 충격도 (평균 |점수|)', short: '충격도', get: d => d.shock, type: 'line', fmt: f2 },
    { title: '상쇄율 (호재·악재가 서로 지운 정도, 0~1)', short: '상쇄율', get: d => d.cancel, type: 'line', fmt: f2 },
  ];
  return card('s-trend', `최근 ${s.length}거래일 추이`,
    `맨 위는 시장 지수와 뉴스 분위기 지수(그날 제목 점수 평균)를 겹친 것입니다. 아래 두 막대를 견주면 같은 날 관계가 더 잘 보입니다. 같은 날 분위기와 등락률의 상관은 ${signed(corr, 2)}이지만, 다음 날 등락은 맞히지 못합니다.`,
    h('p', { class: 'legend' }, key('--ink-2', `${m.name} 종가`), key('--series', '뉴스 분위기 지수')),
    barLegend(),
    chartBox(`${m.name} 최근 ${s.length}거래일 분위기·등락률·충격도·상쇄율 차트`, s, panels,
      d => [['종가', fnum(d.close)], ['제목 수', fnum(d.titles)]]),
    h('details', {}, h('summary', {}, '표로 보기'),
      table('최근 거래일 값 (최근 날짜부터)', ['날짜', '뉴스 지수', '분위기 z', '등락률', '종가', '충격도', '상쇄율', '제목 수'],
        s.slice().reverse().map(d => [d.date, fsc(d.mood), fz(d.mood_rel), fpct(d.change_pct), fnum(d.close), f2(d.shock), f2(d.cancel), fnum(d.titles)]))));
}

function secKinds(m) {
  const s = m.series || [];
  const names = Object.keys(m.kinds || {});
  if (!names.length) return null;
  const all = s.flatMap(d => names.map(k => d.kinds && d.kinds[k])).filter(ok);
  const M = nice(Math.max(0.01, ...all.map(Math.abs)));
  const panels = names.map(k => ({ title: k, short: k, get: d => d.kinds ? d.kinds[k] : null, type: 'bar', fmt: fsc, domain: [-M, M] }));
  return card('s-kinds', '갈래별 분위기',
    '제목이 무엇을 다루는지에 따라 나눈 갈래마다 그날 평균 점수(−2~+2, 모두 같은 눈금)입니다. 제목이 5개 미만인 날은 비웁니다.',
    barLegend(),
    chartBox(`${m.name} 갈래별 최근 ${s.length}거래일 평균 점수 차트`, s, panels),
    h('h3', { class: 'panel-title' }, '오늘 갈래별'),
    table('오늘 갈래별 평균 점수와 제목 수', ['갈래', '평균 점수', '제목 수'],
      names.map(k => [k, fsc(m.kinds[k].mood), fnum(m.kinds[k].titles)])),
    h('details', {}, h('summary', {}, '추이를 표로 보기'),
      table('갈래별 평균 점수 (최근 날짜부터)', ['날짜', ...names],
        s.slice().reverse().map(d => [d.date, ...names.map(k => fsc(d.kinds && d.kinds[k]))]))));
}

function secTopics(m) {
  const t = m.topics || [];
  if (!t.length) return null;
  const max = Math.max(0.01, ...t.flatMap(x => [x.share || 0, x.share_prior7 || 0]));
  const bar = (cls, label, v) => h('div', { class: 'hbar ' + cls },
    h('i', { style: `width:calc((100% - 84px) * ${(v || 0) / max})`, 'aria-hidden': 'true' }), `${label} ${fshare(v)}`);
  return card('s-topics', '주제 비중', '오늘 제목 가운데 각 주제가 차지한 비율과 앞 7일 평균 비율입니다 (경제와 무관한 제목은 뺌).',
    h('p', { class: 'legend' }, key('--series', '오늘'), key('--prior', '앞 7일')),
    t.map(x => h('div', { class: 'topic' },
      h('div', { class: 'name' }, x.topic, h('span', {}, `평균 점수 ${fsc(x.mood)}`)),
      bar('now', '오늘', x.share), bar('prior', '앞 7일', x.share_prior7))));
}

// 실험/기존지수비교.py 결과 (docs/실험기록.md 2026-10-31 '기존 지수 견주기')
const COMPARE = {
  ko: ['한국은행 뉴스심리지수(NSI)와 상관 +0.31(일별)~+0.49(20일 평균): 일부만 겹칩니다. NSI는 경제 전반 뉴스를 한 주 정도 평균 낸 것처럼 천천히 움직입니다.',
    '같은 날 코스피 등락과의 상관은 이 지수 +0.64, NSI 변화 +0.41이고, NSI를 빼도 이 지수 쪽 +0.61이 남습니다. 그날 시장과 맞는 몫은 대부분 NSI에 없는 정보입니다.',
    '숫자 하나가 아니라 어떤 기사가 지수를 움직였는지(기사 묶음·주제)를 함께 보여 줍니다. 다음 날 방향은 이 지수도 NSI도 맞히지 못합니다.'],
  en: ['CNN 공포탐욕지수·VIX는 주가·옵션 같은 시장 자료로 만들어, 같은 날 S&P 500과 더 강하게 묶입니다 (공포탐욕 변화 +0.62, VIX 변화 +0.81, 이 지수 +0.40).',
    '이 지수는 그 둘과 일부만 겹치고(상관 +0.24~+0.44), 그 둘을 빼도 같은 날 S&P와 +0.23~+0.26이 남습니다. 시장 자료에 없는 뉴스 쪽 정보입니다.',
    '숫자 하나가 아니라 왜 그런지(끌어내린·끌어올린 기사 묶음, 주제)를 함께 보여 줍니다. 다음 날 방향은 이 지수도 공포탐욕도 맞히지 못합니다.'],
};

function secMethod(m) {
  const t = m.today || {}, me = m.method || {};
  return card('s-method', '지수 산출', `${m.name}의 오늘 숫자와 계산 방법입니다.`,
    table('오늘 지수 숫자', ['항목', '오늘 값', '계산'], [
      ['분위기 평균', fsc(t.mood), '제목 점수(−2~+2)의 그날 평균'],
      ['앞 60거래일 대비', 'z ' + fz(t.mood_rel), '(오늘 평균 − 앞 60일 평균) ÷ 앞 60일 표준편차'],
      ['충격도 (강도)', `${f2(t.shock)} · z ${fz(t.shock_rel)}`, '평균 |점수|'],
      ['엇갈림 (분산)', `${f2(t.spread)} · z ${fz(t.spread_rel)} · ${t.spread_label || '–'}`, '점수의 표준편차'],
      ['상쇄율', f2(t.cancel), '1 − |Σ점수| ÷ Σ|점수|'],
      ['부정 제목 비율', fshare(t.neg_share), '점수 ≤ −0.25'],
      ['긍정 제목 비율', fshare(t.pos_share), '점수 ≥ +0.25'],
      ['판정한 제목 수', fnum(t.titles) + '개', '그날 수집·판정한 경제 뉴스 제목'],
    ].map(r => r.map((x, i) => i === 2 ? h('span', { class: 'small' }, x) : x))),
    h('div', { class: 'formula' },
      h('p', {}, '① 작은 판정 모델이 제목마다 −2(매우 부정) ~ +2(매우 긍정) 점수를 매깁니다.'),
      h('p', {}, '② 그날 점수를 모아 ', h('code', {}, '분위기 = 평균'), ', ', h('code', {}, '강도 = 평균|점수|'), ', ',
        h('code', {}, '분산 = 표준편차'), ', ', h('code', {}, '상쇄율 = 1 − |Σ| ÷ Σ|·|'), ' 를 냅니다.'),
      h('p', {}, '③ 2026년에 수집량이 달라져 절대값 대신 앞 60거래일 대비 z를 주로 씁니다. 라벨 기준: 분위기 z ±0.5·±1.5, 충격도 z 0·1·2.')),
    h('ul', { class: 'facts' },
      h('li', {}, `판정 모델: ${me.model || '–'}`),
      h('li', {}, `쓴 자료: ${me.since || '–'}부터 ${fnum(me.days_total)}거래일, 제목 ${fnum(me.titles_total)}개`),
      h('li', {}, `같은 날 지수와 등락률의 상관 ${signed(me.same_day_corr, 2)}: 그날 시장 분위기를 꽤 반영합니다.`),
      h('li', {}, `다음 날 등락률과의 상관 ${signed(me.next_day_corr, 2)}: 다음 날 예측력은 없습니다.`),
      h('li', {}, '표본 밖 변동성 예측 시험도 통과하지 못했습니다. 이 지수로 앞날을 내다볼 수는 없습니다.')),
    COMPARE[m.id] ? [h('h3', { class: 'panel-title' }, '기존 지수와 다른 점'), h('ul', { class: 'facts' }, COMPARE[m.id].map(x => h('li', {}, x)))] : null,
    h('p', { class: 'small' }, '면책·개인정보처리방침·데이터 출처는 ', h('a', { href: 'about.html' }, '이 문서'), '에 있습니다.'));
}

// ---------- 차트 (직접 SVG, 위아래 작은 패널, 날짜 십자선 공유) ----------
function ariaFor(p, series, vals) {
  let lo = -1, hi = -1;
  vals.forEach((v, i) => { if (!ok(v)) return; if (lo < 0 || v < vals[lo]) lo = i; if (hi < 0 || v > vals[hi]) hi = i; });
  if (lo < 0) return p.title + ': 자료 없음';
  const n = series.length;
  return `${p.title}, ${series[0].date}부터 ${series[n - 1].date}까지. 최저 ${p.fmt(vals[lo])} (${series[lo].date}), 최고 ${p.fmt(vals[hi])} (${series[hi].date}), 마지막 날 ${p.fmt(vals[n - 1])}.`;
}

function drawChart(c) {
  const { el, series, panels, extra } = c;
  const W = Math.floor(el.clientWidth);
  const n = series.length;
  if (!W || !n || c.w === W) return;
  c.w = W;
  const dual = panels.some(p => p.get2);  // 두 줄을 겹치는 패널이 있으면 오른쪽에 눈금 자리
  const L = 44, R = dual ? 46 : 6, PH = 62, XA = 18, top = 6;
  const step = (W - L - R) / n, xAt = i => L + step * (i + 0.5);
  el.replaceChildren();
  const drawn = panels.map((p, pi) => {
    const last = pi === panels.length - 1;
    const vals = series.map(d => { const v = p.get(d); return ok(v) ? v : null; });
    const vs = vals.filter(ok);
    let lo, hi;
    if (p.domain) [lo, hi] = p.domain;
    else if (p.type === 'bar') { hi = nice(Math.max(0.01, ...vs.map(Math.abs))); lo = -hi; }
    else { lo = Math.min(...vs); hi = Math.max(...vs); const pad = (hi - lo) * 0.1 || 0.05; lo -= pad; hi += pad; }
    if (!vs.length) { lo = -1; hi = 1; }
    const ph = p.tall ? PH * 2 : PH, bot = ph - 6;
    const H = ph + (last ? XA : 0);
    const y = v => top + (hi - v) / (hi - lo) * (bot - top);
    const svg = sv('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': ariaFor(p, series, vals) });
    for (const v of [hi, lo]) {
      svg.append(sv('line', { class: 'grid-line', x1: L, x2: W - R, y1: y(v), y2: y(v) }));
      svg.append(sv('text', { x: L - 6, y: y(v) + 4, 'text-anchor': 'end' }, p.type === 'bar' ? p.fmt(v).replace('%', '') : p.get2 ? Math.round(v).toLocaleString('ko-KR') : f2(v)));
    }
    if (lo < 0 && hi > 0 && !p.get2) {
      svg.append(sv('line', { class: 'base-line', x1: L, x2: W - R, y1: y(0), y2: y(0) }));
      svg.append(sv('text', { x: L - 6, y: y(0) + 4, 'text-anchor': 'end' }, '0'));
    }
    if (p.type === 'bar') {
      const bw = Math.max(1, step * 0.72), y0 = y(0);
      vals.forEach((v, i) => {
        if (v == null) return;
        const yv = y(v);
        svg.append(sv('rect', { class: v >= 0 ? 'bar-pos' : 'bar-neg', x: (xAt(i) - bw / 2).toFixed(2), y: Math.min(y0, yv).toFixed(2), width: bw.toFixed(2), height: Math.max(1, Math.abs(yv - y0)).toFixed(2) }));
      });
    } else {
      let d = '', pen = false;
      vals.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? 'L' : 'M') + xAt(i).toFixed(1) + ' ' + y(v).toFixed(1); pen = true; });
      svg.append(sv('path', { class: p.get2 ? 'data-line market-line' : 'data-line', d }));
    }
    let vals2 = null, y2 = null, dot2 = null;
    if (p.get2) {  // 둘째 줄 (뉴스 지수): 자기 눈금(오른쪽)
      vals2 = series.map(d => { const v = p.get2(d); return ok(v) ? v : null; });
      const v2 = vals2.filter(ok);
      let lo2 = Math.min(...v2), hi2 = Math.max(...v2);
      const pad2 = (hi2 - lo2) * 0.1 || 0.05; lo2 -= pad2; hi2 += pad2;
      y2 = v => top + (hi2 - v) / (hi2 - lo2) * (bot - top);
      for (const v of [hi2, lo2]) svg.append(sv('text', { x: W - R + 6, y: y2(v) + 4, 'text-anchor': 'start', class: 'axis2' }, p.fmt2(v)));
      if (lo2 < 0 && hi2 > 0) svg.append(sv('line', { class: 'base-line', x1: L, x2: W - R, y1: y2(0), y2: y2(0) }));
      let d2 = '', pen2 = false;
      vals2.forEach((v, i) => { if (v == null) { pen2 = false; return; } d2 += (pen2 ? 'L' : 'M') + xAt(i).toFixed(1) + ' ' + y2(v).toFixed(1); pen2 = true; });
      svg.append(sv('path', { class: 'data-line', d: d2 }));
      dot2 = sv('circle', { class: 'hover-dot', r: 4, visibility: 'hidden' });
      svg.append(dot2);
    }
    if (last) {
      [[0, 'start'], [Math.floor(n / 2), 'middle'], [n - 1, 'end']].forEach(([i, a]) =>
        svg.append(sv('text', { x: a === 'start' ? L : a === 'end' ? W - R : xAt(i), y: ph + 13, 'text-anchor': a }, series[i].date.slice(5).replace('-', '/'))));
    }
    const xh = sv('line', { class: 'xhair', y1: top - 4, y2: bot + 4, visibility: 'hidden' });
    svg.append(xh);
    const dot = p.type === 'line' ? sv('circle', { class: p.get2 ? 'hover-dot market-dot' : 'hover-dot', r: 4, visibility: 'hidden' }) : null;
    if (dot) svg.append(dot);
    el.append(h('p', { class: 'panel-title', 'aria-hidden': 'true' }, p.title), svg);
    return { xh, dot, y, vals, dot2, y2, vals2 };
  });
  const tip = h('div', { class: 'tip', 'aria-live': 'polite' });
  tip.hidden = true;
  el.append(tip);

  let cur = -1;
  const show = i => {
    cur = i;
    const X = xAt(i), d = series[i];
    for (const s of drawn) {
      s.xh.setAttribute('x1', X); s.xh.setAttribute('x2', X); s.xh.setAttribute('visibility', 'visible');
      if (s.dot) {
        const v = s.vals[i];
        if (v == null) s.dot.setAttribute('visibility', 'hidden');
        else { s.dot.setAttribute('cx', X); s.dot.setAttribute('cy', s.y(v)); s.dot.setAttribute('visibility', 'visible'); }
      }
      if (s.dot2) {
        const v = s.vals2[i];
        if (v == null) s.dot2.setAttribute('visibility', 'hidden');
        else { s.dot2.setAttribute('cx', X); s.dot2.setAttribute('cy', s.y2(v)); s.dot2.setAttribute('visibility', 'visible'); }
      }
    }
    const row = (k, v) => h('div', {}, h('span', {}, k + ' '), v);
    tip.replaceChildren(h('b', {}, d.date), ...panels.flatMap(p => p.get2 ? [row(p.short, p.fmt(p.get(d))), row(p.short2, p.fmt2(p.get2(d)))] : [row(p.short, p.fmt(p.get(d)))]), ...(extra ? extra(d).map(([k, v]) => row(k, v)) : []));
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = Math.max(0, X + 12 + tw > W ? X - 12 - tw : X + 12) + 'px';
  };
  const hide = () => { cur = -1; tip.hidden = true; drawn.forEach(s => { s.xh.setAttribute('visibility', 'hidden'); if (s.dot) s.dot.setAttribute('visibility', 'hidden'); if (s.dot2) s.dot2.setAttribute('visibility', 'hidden'); }); };
  const at = e => Math.min(n - 1, Math.max(0, Math.floor((e.clientX - el.getBoundingClientRect().left - L) / step)));
  el.onpointermove = e => show(at(e));
  el.onpointerdown = e => show(at(e));
  el.onpointerleave = hide;
  el.onblur = hide;
  el.onkeydown = e => {
    const k = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
    if (k) show(cur < 0 ? n - 1 : Math.min(n - 1, Math.max(0, cur + k)));
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(n - 1);
    else if (e.key === 'Escape') hide();
    else return;
    e.preventDefault();
  };
}

start();
