export const meta = {
  name: 'lean-labeler',
  description: 'Lean Haiku labeler: one file read per batch (instructions + guide + titles), exact-count structured labels, no file writes',
  phases: [{ title: 'Label', detail: 'one Haiku agent per batch', model: 'haiku' }],
}
// 가벼운 라벨러 (2026-10-29). 묶음 파일은 실험/lean_label.py prep이 만든다. args = [{lang, dir, total, size[, model, agentType, only]}, ...] (model 기본 haiku, agentType 'labeler' = 읽기만 하는 전용 에이전트, only = 돌릴 묶음 이름들)
// 결과는 journal.jsonl에 {name, start, n, out} → python 실험/lean_label.py merge
const V = ['-2', '-1', '-0.5', '0', '0.5', '1', '2']
const jobs = []
for (const a of args) {
  for (let s = 0; s < a.total; s += a.size) {
    const n = Math.min(a.size, a.total - s)
    if (a.only && !a.only.includes(`${a.lang}_b${String(s / a.size).padStart(4, '0')}`)) continue  // only: 이 묶음들만 (실패분 다시)
    jobs.push({ ...a, name: `${a.lang}_b${String(s / a.size).padStart(4, '0')}`, file: `${a.dir}\\b${String(s / a.size).padStart(4, '0')}.txt`, start: s, n })
  }
}
const PAIRS = V.flatMap(u => V.map(k => `${u}/${k}`))  // 영어: "미국/한국"
const run = j => {
  const val = { type: 'string', enum: j.lang === 'en' ? PAIRS : V }
  const keys = Array.from({ length: j.n }, (_, i) => String(i + 1))
  const what = j.lang === 'en' ? '"미국/한국" 형식 (예: "0.5/0", "-1/-0.5")' : '감성 값 하나 (예: "0.5")'
  return agent(`\`${j.file}\`을 Read로 한 번 읽는다. 그 파일 맨 위의 지시와 '# 기준'대로, 파일의 제목 ${j.n}개(번호 1~${j.n}) 모두에 감성을 정한다.
제목마다 오래 생각하지 말고 기준에 비춰 바로 정한다.
답은 구조화 출력으로만: 열쇠는 제목 번호("1"~"${j.n}", 하나도 빠짐없이), 값은 ${what}. 다른 파일·도구는 쓰지 않는다.`,
    { label: j.name, phase: 'Label', model: j.model || 'haiku', ...(j.agentType ? { agentType: j.agentType } : {}), ...(/4-5/.test(j.model || '') ? {} : { effort: 'low' }),  // 4.5 세대는 effort를 안 받음
      schema: { type: 'object', properties: Object.fromEntries(keys.map(k => [k, val])), required: keys, additionalProperties: false } })
}
// 동시에 W개만 (많이 띄우면 에이전트 검토 분류기가 요청 한도에 걸려 시작도 못 하고 실패함, 2026-10-29). 실패한 묶음은 한 번 더
const W = 6
const pool = async list => {
  const out = new Array(list.length).fill(null)
  let next = 0
  const worker = async () => {
    while (next < list.length) {
      const i = next++
      try { out[i] = await run(list[i]) } catch (e) { out[i] = null }
    }
  }
  await Promise.all(Array.from({ length: W }, worker))
  return out
}
let res = await pool(jobs)
const again = jobs.filter((_, i) => !res[i])
log(`1차 실패 ${again.length}개, 한 번 더`)
const res2 = await pool(again)
let k = 0
res = res.map(r => r || res2[k++])
return res.map(r => !!r)
