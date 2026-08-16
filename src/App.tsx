"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createSnapshot, dictionary, evaluate, Evaluation, MEANINGS, rollingAccuracy, Snapshot, TOKENS, trainBatch } from "./babel";

const arrows=["↑","→","↓","←"];

function AccuracyChart({snapshot}:{snapshot:Snapshot}){
  const w=520,h=132,points=snapshot.chart,min=points[0]?.episode??0,max=points.at(-1)?.episode??1;
  const x=(v:number)=>8+(v-min)/Math.max(1,max-min)*(w-16),y=(v:number)=>h-8-v/100*(h-16),line=points.map(p=>`${x(p.episode)},${y(p.accuracy)}`).join(" ");
  return <div className="chart-wrap"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Rolling task accuracy">
    <defs><linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#87f7c7" stopOpacity=".34"/><stop offset="100%" stopColor="#87f7c7" stopOpacity="0"/></linearGradient></defs>
    {[25,50,75,100].map(v=><g key={v}><line x1="0" y1={y(v)} x2={w} y2={y(v)} className="grid-line"/><text x="8" y={y(v)-4} className="axis-label">{v}%</text></g>)}
    {points.length>1&&<><polygon points={`${x(points[0].episode)},${h} ${line} ${x(points.at(-1)!.episode)},${h}`} fill="url(#chartFill)"/><polyline points={line} className="accuracy-line"/></>}
  </svg></div>;
}

function ProbabilityBars({values,labels}:{values:number[];labels:readonly string[]}){
  return <div className="probability-bars">{values.map((v,i)=><div className="probability-row" key={labels[i]}><span>{labels[i]}</span><div className="bar-track"><i style={{width:`${Math.max(2,v*100)}%`}}/></div><b>{Math.round(v*100)}%</b></div>)}</div>;
}

export default function Home(){
  const [snapshot,setSnapshot]=useState<Snapshot>(()=>createSnapshot(81527));
  const [running,setRunning]=useState(false),[speed,setSpeed]=useState(120),[rate,setRate]=useState(.16),[temperature,setTemperature]=useState(.82),[selectedTrace,setSelectedTrace]=useState(0),[evaluation,setEvaluation]=useState<Evaluation[]>([]),[showMethod,setShowMethod]=useState(false);
  const frameRef=useRef<number|null>(null);
  const train=useCallback((count:number)=>setSnapshot(current=>trainBatch(current,count,rate,temperature)),[rate,temperature]);
  useEffect(()=>{if(!running)return;const loop=()=>{train(speed);frameRef.current=requestAnimationFrame(loop)};frameRef.current=requestAnimationFrame(loop);return()=>{if(frameRef.current)cancelAnimationFrame(frameRef.current)}},[running,speed,train]);
  const accuracy=rollingAccuracy(snapshot),lexicon=useMemo(()=>dictionary(snapshot),[snapshot]),trace=snapshot.traces[selectedTrace]??snapshot.traces[0],latest=snapshot.traces[0],distinct=new Set(lexicon.map(i=>i.tokenIndex)).size;
  const proof=accuracy>=90&&distinct===4?"PROTOCOL DETECTED":accuracy>=55?"CONVENTION FORMING":"NO LANGUAGE YET";
  const reset=()=>{setRunning(false);setSnapshot(createSnapshot(Math.floor(Math.random()*9_000_000)));setEvaluation([]);setSelectedTrace(0)};
  const test=(mode:Evaluation["mode"])=>setEvaluation(current=>[evaluate(snapshot,mode),...current.filter(i=>i.mode!==mode)]);
  const exportWorld=()=>{const blob=new Blob([JSON.stringify(snapshot,null,2)],{type:"application/json"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=`babel-seed-${snapshot.seed}-episode-${snapshot.episodes}.json`;a.click();URL.revokeObjectURL(url)};

  return <main>
    <header className="topbar"><div className="brand"><span className="brand-mark">B</span><div><strong>BABEL</strong><small>LANGUAGE GENESIS LAB</small></div></div><div className="status-strip"><span className="local-dot"/> LOCAL COMPUTE <span className="divider"/> SEED {snapshot.seed}<button className="text-button" onClick={exportWorld}>EXPORT RUN</button></div></header>
    <section className="hero"><div><p className="eyebrow">PROTOCOL ONE · THE HIDDEN COMPASS</p><h1>Watch meaning<br/><em>come into existence.</em></h1><p className="lede">The sender sees the target. The receiver does not. Between them: four meaningless symbols, a shared outcome, and no dictionary.</p></div><div className={`proof ${proof==="PROTOCOL DETECTED"?"proved":""}`}><span>LIVE FINDING</span><strong>{proof}</strong><small>{distinct}/4 distinct symbols grounded</small></div></section>

    <section className="lab-grid">
      <article className="panel world-panel"><div className="panel-head"><div><span>01</span><h2>THE WORLD</h2></div><span className="live-tag"><i/> {running?"TRAINING":"PAUSED"}</span></div>
        <div className="world">
          {MEANINGS.map((meaning,i)=><div className={`chamber chamber-${i} ${latest?.target===i?"target":""} ${latest?.action===i?"chosen":""}`} key={meaning}><span>{arrows[i]}</span><small>{meaning}</small></div>)}
          <div className="sender agent"><span>S</span><small>SEES TARGET</small></div><div className="receiver agent"><span>R</span><small>CHOOSES</small></div>
          {latest&&<div className="signal" key={latest.id}><span>{TOKENS[latest.sent]}</span><small>MESSAGE</small></div>}
          <div className="world-caption">{latest?(latest.success?<><b>COORDINATION SUCCESS</b> Receiver chose {MEANINGS[latest.action]}</>:<><b className="miss">COORDINATION FAILED</b> Target was {MEANINGS[latest.target]}</>):"Start training to observe the first exchange."}</div>
        </div>
        <div className="controls"><button className="primary" onClick={()=>setRunning(v=>!v)}>{running?"PAUSE EVOLUTION":"RUN EVOLUTION"}</button><button onClick={()=>train(1)} disabled={running}>STEP ONCE</button><button onClick={()=>train(2500)}>+2,500 TRIALS</button><button onClick={reset}>NEW CIVILIZATION</button></div>
        <div className="sliders"><label><span>Trials / frame <b>{speed}</b></span><input type="range" min="10" max="500" step="10" value={speed} onChange={e=>setSpeed(+e.target.value)}/></label><label><span>Exploration <b>{temperature.toFixed(2)}</b></span><input type="range" min=".2" max="1.4" step=".02" value={temperature} onChange={e=>setTemperature(+e.target.value)}/></label><label><span>Learning rate <b>{rate.toFixed(2)}</b></span><input type="range" min=".03" max=".3" step=".01" value={rate} onChange={e=>setRate(+e.target.value)}/></label></div>
      </article>

      <aside className="metrics-column"><article className="panel score-panel"><div className="panel-head compact"><div><span>02</span><h2>EVIDENCE</h2></div></div><div className="score-row"><div><strong>{accuracy.toFixed(1)}%</strong><small>ROLLING ACCURACY</small></div><div><strong>{snapshot.episodes.toLocaleString()}</strong><small>TRAINING TRIALS</small></div></div><AccuracyChart snapshot={snapshot}/><div className="chance-line"><span/> CHANCE PERFORMANCE = 25%</div></article>
        <article className="panel ablation-panel"><div className="panel-head compact"><div><span>03</span><h2>BREAK THE LANGUAGE</h2></div></div><p>Run 800 frozen trials. Learning is disabled, so only the communication channel changes.</p><div className="ablation-buttons">{(["intact","scrambled","silent"] as const).map(mode=><button key={mode} onClick={()=>test(mode)}><span>{mode==="intact"?"◆":mode==="scrambled"?"⤨":"∅"}</span>{mode.toUpperCase()}</button>)}</div><div className="evaluation-results">{(["intact","scrambled","silent"] as const).map(mode=>{const result=evaluation.find(i=>i.mode===mode);return <div key={mode}><span>{mode}</span><b>{result?`${result.accuracy.toFixed(1)}%`:"—"}</b></div>})}</div></article>
      </aside>
    </section>

    <section className="lower-grid"><article className="panel lexicon-panel"><div className="panel-head"><div><span>04</span><h2>EMERGENT DICTIONARY</h2></div><small>Inferred from sender policy</small></div><div className="dictionary">{lexicon.map(item=><div className={`word ${item.meaning===item.interpretedAs?"aligned":""}`} key={item.meaning}><span className="token">{item.token}</span><div><small>SENDER USES FOR</small><strong>{item.meaning}</strong></div><div><small>RECEIVER INTERPRETS</small><strong>{item.interpretedAs}</strong></div><b>{Math.round(Math.min(item.confidence,item.interpretationConfidence))}%</b></div>)}</div></article>
      <article className="panel microscope-panel"><div className="panel-head"><div><span>05</span><h2>DECISION MICROSCOPE</h2></div><small>Exact computation · no generated explanation</small></div>{trace?<><div className="trace-picker">{snapshot.traces.slice(0,8).map((item,i)=><button className={selectedTrace===i?"active":""} onClick={()=>setSelectedTrace(i)} key={item.id}>#{item.id} <i className={item.success?"success":"failure"}/></button>)}</div><div className="decision-grid"><div className="decision-card"><span>1 · SENDER OBSERVATION</span><strong>{arrows[trace.target]} {MEANINGS[trace.target]}</strong><small>Only the sender receives this input.</small></div><div className="decision-card"><span>2 · MESSAGE POLICY</span><ProbabilityBars values={trace.senderProbabilities} labels={TOKENS}/></div><div className="decision-card signal-card"><span>3 · DISCRETE BOTTLENECK</span><strong>{TOKENS[trace.sent]}</strong><small>One symbol crosses the channel.</small></div><div className="decision-card"><span>4 · ACTION POLICY</span><ProbabilityBars values={trace.receiverProbabilities} labels={arrows}/></div></div><div className="counterfactual"><div><span>CAUSAL REPLAY</span><strong>What would the receiver do if the symbol changed?</strong></div><div className="counterfactual-grid">{trace.counterfactuals.map((values,token)=>{const choice=values.indexOf(Math.max(...values));return <div className={token===trace.sent?"actual":""} key={TOKENS[token]}><span>{TOKENS[token]}</span><b>{arrows[choice]}</b><small>{Math.round(values[choice]*100)}%</small></div>})}</div></div></>:<div className="empty-state">No decision exists yet. Step once or run evolution.</div>}</article>
    </section>

    <section className="method"><button onClick={()=>setShowMethod(v=>!v)}><span>EXPERIMENTAL METHOD</span>{showMethod?"−":"+"}</button>{showMethod&&<div><p><b>Architecture.</b> Two independent four-by-four stochastic policies. The sender maps a hidden target to a symbol; the receiver maps that symbol to an action.</p><p><b>Learning.</b> Both policies update from the same outcome using a baseline-adjusted policy-gradient rule. No correct symbol mapping is supplied.</p><p><b>Reward.</b> Successful coordination: +0.97. Incorrect choice: −0.28. The 0.03 success deduction represents channel and action cost.</p><p><b>Falsification.</b> Frozen-policy ablations scramble or remove messages. If accuracy remains high, messages were not causally necessary.</p><p><b>Scope.</b> This demonstrates a grounded signaling convention, not human language. Compositional sequences, dialects, cultural transmission, and deception require later worlds.</p></div>}</section>
    <footer><span>BABEL / PROTOCOL ONE</span><p>Nothing leaves this device. No model API. No prescribed vocabulary.</p><span>BUILD 0.1</span></footer>
  </main>;
}
