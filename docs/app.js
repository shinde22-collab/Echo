import { predict } from './inference.mjs';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const state = { strategy: 'uncertainty', seed: 42, index: 2, map: 'surrogate', sample: 0, playing: false, motion: !reducedMotion, rotation: -.5, tilt: .28 };
let data;
let replayTimer;
let frameId;
let previousFrame = 0;
let dragX;
let dragY;
let signalVisible = true;
const percent = (value) => `${(value * 100).toFixed(2)}%`;
const currentRun = () => data.runs.find(r => r.strategy === state.strategy && r.seed === state.seed && r.budget === data.budgets[state.index]);

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width * ratio), height = Math.round(rect.height * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawSignal() {
  if (!data) return;
  const { ctx, width: w, height: h } = fitCanvas($('#signal'));
  ctx.clearRect(0, 0, w, h);
  const centerX = w * .5, centerY = h * .455, radius = Math.min(w * .39, h * .34);
  ctx.strokeStyle = '#d5ff7009';
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 32) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = 0; y < h; y += 32) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  const project = (x, y, z) => {
    const rx = x * Math.cos(state.rotation) + z * Math.sin(state.rotation);
    const rz = z * Math.cos(state.rotation) - x * Math.sin(state.rotation);
    const ry = y * Math.cos(state.tilt) - rz * Math.sin(state.tilt);
    const depth = rz * Math.cos(state.tilt) + y * Math.sin(state.tilt);
    const perspective = 2.9 / (2.9 + depth);
    return [centerX + rx * radius * perspective, centerY + ry * radius * perspective, depth];
  };
  for (let ring = 0; ring < 4; ring++) {
    ctx.beginPath();
    for (let a = 0; a <= Math.PI * 2 + .05; a += .04) {
      const p = ring === 0 ? project(Math.cos(a), Math.sin(a), 0) : ring === 1 ? project(0, Math.cos(a), Math.sin(a)) : ring === 2 ? project(Math.cos(a), 0, Math.sin(a)) : project(Math.cos(a) * 1.18, Math.sin(a) * .3, Math.sin(a) * .7);
      if (a === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    }
    ctx.strokeStyle = ring === 3 ? '#d5ff7040' : '#93ac8c25';
    ctx.lineWidth = ring === 3 ? 1 : .6;
    ctx.stroke();
  }
  const run = currentRun();
  const cloud = data.test.x.map((row, i) => {
    const normalized = row.slice(0, 3).map((v, j) => (v - data.scaler.mean[j]) / data.scaler.scale[j] * .32);
    const p = project(...normalized);
    return { p, i, color: run.predictions[i] !== data.test.victim[i] ? '#ff8f79' : data.test.victim[i] ? '#d5ff70' : '#8cabff' };
  }).sort((a, b) => b.p[2] - a.p[2]);
  for (let i = 0; i < cloud.length; i++) {
    const { p, color } = cloud[i];
    const alpha = Math.max(.25, Math.min(1, .65 - p[2] * .25));
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    if (i % 13 === 0) {
      ctx.strokeStyle = color + '20';
      ctx.beginPath(); ctx.moveTo(p[0], p[1]);
      const q = cloud[(i + 11) % cloud.length].p;
      ctx.lineTo(q[0], q[1]); ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(p[0], p[1], color === '#ff8f79' ? 2.6 : 1.5 + alpha * .8, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = '#d5ff7055';
  ctx.beginPath(); ctx.arc(centerX, centerY, radius * 1.17, -.4, .5); ctx.stroke();
  ctx.beginPath(); ctx.arc(centerX, centerY, radius * 1.17, 2.6, 3.45); ctx.stroke();
  ctx.fillStyle = '#7c8c74'; ctx.font = '8px monospace';
  ctx.fillText('X / VARIANCE', 20, h * .64);
  ctx.fillText('Y / SKEWNESS', w - 95, h * .28);
  ctx.fillStyle = '#d5ff70';
  ctx.fillText(`FIDELITY ${(run.fidelity * 100).toFixed(2)}%`, w - 120, h - 103);
}

function animateSignal(timestamp) {
  frameId = undefined;
  if (state.motion && !document.hidden && signalVisible) {
    state.rotation += Math.min(timestamp - previousFrame, 40) * .00012;
    drawSignal();
    frameId = requestAnimationFrame(animateSignal);
  }
  previousFrame = timestamp;
}

function startMotion() {
  if (state.motion && !frameId && !document.hidden && signalVisible) {
    previousFrame = performance.now();
    frameId = requestAnimationFrame(animateSignal);
  }
  $('#motion').textContent = state.motion ? 'Ⅱ' : '▷';
  $('#motion').setAttribute('aria-label', state.motion ? 'Pause motion' : 'Resume motion');
}

function drawCurve() {
  const width = 530, height = 275;
  const padding = { left: 42, top: 22, right: 22, bottom: 35 };
  const x = budget => padding.left + (budget - 25) / 475 * (width - padding.left - padding.right);
  const summaries = data.summary.filter(r => r.strategy === state.strategy);
  const runs = data.runs.filter(r => r.strategy === state.strategy && r.seed === state.seed);
  const minValue = Math.max(.4, Math.floor(Math.min(...summaries.map(r => r.mean - r.std), ...runs.map(r => r.fidelity)) * 10) / 10 - .025);
  const y = fidelity => height - padding.bottom - (fidelity - minValue) / (1 - minValue) * (height - padding.top - padding.bottom);
  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg"><title>Held-out fidelity across query budgets, ${state.strategy} strategy, seed ${state.seed}</title>`;
  for (let i = 0; i <= 4; i++) {
    const value = minValue + (1 - minValue) * i / 4;
    svg += `<line x1="${padding.left}" y1="${y(value)}" x2="${width-padding.right}" y2="${y(value)}" stroke="#ffffff0d"/><text x="${padding.left-10}" y="${y(value)+3}" fill="#99a59b" font-family="monospace" font-size="9" text-anchor="end">${(value*100).toFixed(0)}%</text>`;
  }
  const band = summaries.map(r => `${x(r.budget)},${y(Math.min(1,r.mean+r.std))}`).concat([...summaries].reverse().map(r=>`${x(r.budget)},${y(r.mean-r.std)}`)).join(' ');
  svg += `<polygon points="${band}" fill="#8cabff16"/><polyline points="${summaries.map(r=>`${x(r.budget)},${y(r.mean)}`).join(' ')}" fill="none" stroke="#8cabff" stroke-width="1.4" stroke-dasharray="4 4"/>`;
  const active = runs[state.index];
  svg += `<line x1="${x(active.budget)}" y1="${padding.top}" x2="${x(active.budget)}" y2="${height-padding.bottom}" stroke="#d5ff7033" stroke-dasharray="3 3"/><polyline points="${runs.map(r=>`${x(r.budget)},${y(r.fidelity)}`).join(' ')}" fill="none" stroke="#d5ff70" stroke-width="2" stroke-linejoin="round"/>`;
  runs.forEach((r,i) => {
    svg += `<g class="curve-point" tabindex="0" role="button" data-index="${i}" aria-label="${r.budget} queries: ${percent(r.fidelity)} fidelity" style="cursor:pointer"><circle cx="${x(r.budget)}" cy="${y(r.fidelity)}" r="13" fill="transparent"/><circle cx="${x(r.budget)}" cy="${y(r.fidelity)}" r="${i===state.index?5:3}" fill="${i===state.index?'#d5ff70':'#101614'}" stroke="#d5ff70" stroke-width="1.5"/><text x="${x(r.budget)}" y="${height-12}" fill="${i===state.index?'#d5ff70':'#99a59b'}" font-family="monospace" font-size="9" text-anchor="middle">${r.budget}</text></g>`;
  });
  svg += '</svg>';
  $('#curve').innerHTML = svg;
  $('#curve').setAttribute('aria-label', `${state.strategy} fidelity: ${runs.map(r=>`${r.budget} queries ${percent(r.fidelity)}`).join(', ')}`);
  $$('.curve-point').forEach(point => {
    const choose = () => { stopReplay(); state.index = Number(point.dataset.index); render(); };
    point.addEventListener('click', choose);
    point.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); } });
  });
  const summary = summaries[state.index];
  $('#curve-caption').textContent = `${active.budget} queries · mean ${percent(summary.mean)} ± ${(summary.std*100).toFixed(2)} percentage points across 5 seeds. Test n=${data.dataset.test_rows}.`;
}

function drawMap() {
  if (!data) return;
  const { ctx, width: w, height: h } = fitCanvas($('#landscape'));
  const run = currentRun();
  const slice = data.slice;
  ctx.clearRect(0, 0, w, h);
  for (let y = 0; y < slice.height; y++) {
    for (let x = 0; x < slice.width; x++) {
      const index = y * slice.width + x;
      const mismatch = run.slice[index] !== slice.victim[index];
      const label = state.map === 'victim' ? slice.victim[index] : run.slice[index];
      ctx.fillStyle = state.map === 'difference' ? mismatch ? '#9e5143' : '#17231b' : label === '1' ? '#516930' : '#263d61';
      ctx.fillRect(x*w/slice.width, h-(y+1)*h/slice.height, Math.ceil(w/slice.width)+.2, Math.ceil(h/slice.height)+.2);
    }
  }
  ctx.strokeStyle = '#ffffff09'; ctx.lineWidth = 1;
  for (let i=1;i<5;i++) { ctx.beginPath(); ctx.moveTo(i*w/5,0);ctx.lineTo(i*w/5,h);ctx.moveTo(0,i*h/5);ctx.lineTo(w,i*h/5);ctx.stroke(); }
  const [xb, yb] = slice.bounds;
  run.selected.forEach((index, i) => {
    const row = data.pool.x[index];
    const px = (row[0]-xb[0])/(xb[1]-xb[0])*w, py = h-(row[1]-yb[0])/(yb[1]-yb[0])*h;
    if (px<0||px>w||py<0||py>h) return;
    ctx.fillStyle = run.acquired_labels[i] ? '#d5ff70' : '#9fbaff';
    ctx.strokeStyle='#0a0e0d'; ctx.lineWidth=.8;
    ctx.beginPath();ctx.arc(px,py,2,0,Math.PI*2);ctx.fill();ctx.stroke();
  });
  $('#landscape').setAttribute('aria-label', `${state.map} decision slice at ${run.budget} queries. Four-dimensional test fidelity ${percent(run.fidelity)}.`);
}

function renderProbe() {
  const run = currentRun();
  const row = data.test.x[state.sample];
  const victim = data.test.victim[state.sample];
  const surrogate = predict(run.model, row, data.scaler);
  $('#sample-id').textContent = `#${String(data.test.ids[state.sample]).padStart(4,'0')}`;
  $('#sample').value = state.sample;
  $('#sample-features').innerHTML = data.features.map((feature,i)=>`<div class="feature-row"><span>${feature}</span><strong>${row[i].toFixed(4)}</strong></div>`).join('');
  $('#victim-class').textContent = `Class ${victim}`;
  $('#surrogate-class').textContent = `Class ${surrogate}`;
  const agreement = victim === surrogate;
  $('#agreement-badge').textContent = `${agreement?'✓ MATCH':'↗ DISAGREEMENT'} · TRUE LABEL ${data.test.labels[state.sample]}`;
  $('#agreement-badge').classList.toggle('different', !agreement);
}

function render() {
  const run = currentRun();
  $('#budget').value = state.index;
  $('#budget-value').innerHTML = `${run.budget}<span> / 500</span>`;
  $('#pool-percent').textContent = `${(run.budget/data.dataset.pool_rows*100).toFixed(1)}%`;
  $('#fidelity').textContent = percent(run.fidelity);
  $('#accuracy').textContent = percent(run.accuracy);
  $('#disagreements').textContent = String(run.disagreements).padStart(2,'0');
  $('#test-count').textContent = `out of ${data.dataset.test_rows} held-out samples`;
  const notes = {
    random:'Random sampling: query a seeded random ordering of the unlabeled pool. A straightforward baseline.',
    coverage:'Coverage sampling: after 25 initial random queries, greedily query points farthest from the queried set in standardized feature space.',
    uncertainty:'Uncertainty sampling: after 25 initial random queries, ask about inputs closest to the surrogate’s current decision boundary in batches of 25.'
  };
  $('#strategy-note').textContent = notes[state.strategy];
  $$('[data-strategy]').forEach(button=>button.setAttribute('aria-pressed', String(button.dataset.strategy===state.strategy)));
  $$('[data-map]').forEach(button=>button.setAttribute('aria-pressed', String(button.dataset.map===state.map)));
  $$('[data-budget]').forEach(button=>button.classList.toggle('active', Number(button.dataset.budget)===state.index));
  drawCurve(); drawMap(); renderProbe(); drawSignal();
}

function stopReplay() {
  clearTimeout(replayTimer);
  state.playing = false;
  $('#replay').innerHTML = '<span aria-hidden="true">▷</span> Replay extraction';
}

function replay() {
  if (state.playing) { stopReplay(); $('#log').textContent = 'Replay paused. Explore this checkpoint or start a new replay.'; return; }
  state.playing = true;
  state.index = 0;
  $('#replay').innerHTML = '<span aria-hidden="true">Ⅱ</span> Pause replay';
  const advance = () => {
    if (!state.playing) return;
    render();
    const run=currentRun();
    $('#log').textContent=`RECORDED CHECKPOINT ${state.index+1}/6 → ${run.queries} oracle queries → fit surrogate → ${percent(run.fidelity)} test fidelity.`;
    if (state.index === data.budgets.length-1) {
      stopReplay();
      $('#log').textContent=`Replay complete. ${run.queries} total extraction queries, ${data.dataset.test_rows} untouched test samples, ${percent(run.fidelity)} fidelity. Reproduce with python -m echo.`;
    } else {
      replayTimer=setTimeout(()=>{state.index++;advance();}, reducedMotion?650:1350);
    }
  };
  advance();
}

function bindEvents() {
  $$('[data-strategy]').forEach(button=>button.addEventListener('click',()=>{stopReplay();state.strategy=button.dataset.strategy;render();}));
  $$('[data-map]').forEach(button=>button.addEventListener('click',()=>{state.map=button.dataset.map;render();}));
  $('#seed').addEventListener('change',event=>{stopReplay();state.seed=Number(event.target.value);render();});
  $('#budget').addEventListener('input',event=>{stopReplay();state.index=Number(event.target.value);render();});
  $('#sample').max=data.dataset.test_rows-1;
  $('#sample').addEventListener('input',event=>{state.sample=Number(event.target.value);renderProbe();});
  $('#next-sample').addEventListener('click',()=>{state.sample=(state.sample+1)%data.dataset.test_rows;renderProbe();});
  $('#find-difference').addEventListener('click',()=>{
    const run=currentRun();
    for(let offset=1;offset<=data.dataset.test_rows;offset++){
      const i=(state.sample+offset)%data.dataset.test_rows;
      if(run.predictions[i]!==data.test.victim[i]) { state.sample=i;renderProbe();return; }
    }
    $('#agreement-badge').textContent='✓ ALL TEST SAMPLES AGREE AT THIS CHECKPOINT';
  });
  $('#replay').addEventListener('click',replay);
  $('#motion').addEventListener('click',()=>{state.motion=!state.motion;startMotion();});
  $('#budget-ticks').innerHTML=data.budgets.map((b,i)=>`<button data-budget="${i}" aria-label="Set budget to ${b} queries">${b}</button>`).join('');
  $$('[data-budget]').forEach(button=>button.addEventListener('click',()=>{stopReplay();state.index=Number(button.dataset.budget);render();}));
  const signal=$('#signal');
  signal.addEventListener('pointerdown',event=>{dragX=event.clientX;dragY=event.clientY;signal.setPointerCapture(event.pointerId);});
  signal.addEventListener('pointermove',event=>{
    if(dragX===undefined)return;
    state.rotation+=(event.clientX-dragX)*.008;
    state.tilt=Math.max(-1,Math.min(1,state.tilt+(event.clientY-dragY)*.006));
    dragX=event.clientX;dragY=event.clientY;drawSignal();
  });
  const release=()=>{dragX=undefined;dragY=undefined;};
  signal.addEventListener('pointerup',release);signal.addEventListener('pointercancel',release);
  signal.addEventListener('keydown',event=>{
    if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){
      event.preventDefault();state.rotation+=event.key==='ArrowLeft'?-.1:event.key==='ArrowRight'?.1:0;
      state.tilt=Math.max(-1,Math.min(1,state.tilt+(event.key==='ArrowUp'?-.1:event.key==='ArrowDown'?.1:0)));drawSignal();
    }
  });
  $('#landscape').addEventListener('pointermove',event=>{
    const canvas=$('#landscape'), rect=canvas.getBoundingClientRect();
    const px=(event.clientX-rect.left)/rect.width,py=1-(event.clientY-rect.top)/rect.height;
    const x=Math.max(0,Math.min(data.slice.width-1,Math.floor(px*data.slice.width))),y=Math.max(0,Math.min(data.slice.height-1,Math.floor(py*data.slice.height)));
    const index=y*data.slice.width+x,run=currentRun();
    const tooltip=$('#map-tooltip');tooltip.hidden=false;
    tooltip.textContent=`BLACK BOX ${data.slice.victim[index]} / SURROGATE ${run.slice[index]}`;
    tooltip.style.left=`${Math.max(4,Math.min(rect.width-220,event.clientX-rect.left+12))}px`;
    tooltip.style.top=`${Math.max(4,Math.min(rect.height-35,event.clientY-rect.top-30))}px`;
  });
  $('#landscape').addEventListener('pointerleave',()=>{$('#map-tooltip').hidden=true;});
  const resize=new ResizeObserver(()=>{drawMap();drawSignal();});resize.observe($('#landscape'));resize.observe($('#signal'));
  new IntersectionObserver(entries=>{signalVisible=entries[0].isIntersecting;startMotion();}).observe(signal);
  document.addEventListener('visibilitychange',startMotion);
}

async function init() {
  try {
    const response=await fetch(new URL('assets/experiment.json',import.meta.url));
    if(!response.ok)throw new Error(`Data request returned ${response.status}`);
    data=await response.json();
    if(data.protocol!=='echo-v1'||data.runs.length!==90)throw new Error('Unsupported experiment data');
    $('#loading').hidden=true;$('#lab').hidden=false;
    bindEvents();render();startMotion();
  } catch(error) {
    $('#loading').textContent=`The experiment data could not be loaded. Reload the page, or download the results from the repository. ${error.message}`;
    console.error(error);
  }
}

init();
