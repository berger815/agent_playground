export const MEANINGS = ["NORTH", "EAST", "SOUTH", "WEST"] as const;
export const TOKENS = ["◈", "●", "▲", "■"] as const;

export type Trace = { id:number; target:number; sent:number; action:number; success:boolean; reward:number; senderProbabilities:number[]; receiverProbabilities:number[]; counterfactuals:number[][] };
export type Snapshot = { sender:number[][]; receiver:number[][]; baseline:number; episodes:number; successes:number; recent:number[]; chart:{episode:number;accuracy:number}[]; traces:Trace[]; seed:number };
export type Evaluation = { mode:"intact"|"scrambled"|"silent"; accuracy:number; trials:number };
const SIZE = 4;

export function mulberry32(seed:number) { let value=seed>>>0; return ()=>{ value+=0x6d2b79f5; let t=value; t=Math.imul(t^(t>>>15),t|1); t^=t+Math.imul(t^(t>>>7),t|61); return ((t^(t>>>14))>>>0)/4294967296; }; }
const matrix=(random:()=>number)=>Array.from({length:SIZE},()=>Array.from({length:SIZE},()=>(random()-.5)*.08));
export function createSnapshot(seed=Math.floor(Math.random()*1_000_000)):Snapshot { const random=mulberry32(seed); return { sender:matrix(random),receiver:matrix(random),baseline:0,episodes:0,successes:0,recent:[],chart:[{episode:0,accuracy:25}],traces:[],seed }; }
export function softmax(logits:number[],temperature=1){const scaled=logits.map(v=>v/Math.max(.08,temperature));const max=Math.max(...scaled);const exps=scaled.map(v=>Math.exp(v-max));const sum=exps.reduce((a,b)=>a+b,0);return exps.map(v=>v/sum);}
function choose(p:number[],random:()=>number){let cursor=random();for(let i=0;i<p.length;i++){cursor-=p[i];if(cursor<=0)return i;}return p.length-1;}
function greedy(p:number[]){return p.reduce((best,value,index,array)=>value>array[best]?index:best,0);}
function update(row:number[],selected:number,p:number[],advantage:number,rate:number){for(let i=0;i<row.length;i++){const d=(i===selected?1:0)-p[i];row[i]=Math.max(-8,Math.min(8,row[i]+rate*advantage*d));}}

export function trainBatch(previous:Snapshot,count:number,rate:number,temperature:number){
  const state:Snapshot={...previous,sender:previous.sender.map(r=>[...r]),receiver:previous.receiver.map(r=>[...r]),recent:[...previous.recent],chart:[...previous.chart],traces:[...previous.traces]};
  const random=mulberry32(state.seed+state.episodes*7919+17);
  for(let run=0;run<count;run++){
    const target=Math.floor(random()*SIZE); const senderP=softmax(state.sender[target],temperature); const sent=choose(senderP,random); const receiverP=softmax(state.receiver[sent],temperature); const action=choose(receiverP,random); const success=action===target; const reward=success?.97:-.28;
    state.baseline=state.baseline*.985+reward*.015; const advantage=reward-state.baseline; update(state.sender[target],sent,senderP,advantage,rate); update(state.receiver[sent],action,receiverP,advantage,rate);
    state.episodes++; state.successes+=success?1:0; state.recent.push(success?1:0); if(state.recent.length>240)state.recent.shift();
    if(run===count-1){state.traces.unshift({id:state.episodes,target,sent,action,success,reward,senderProbabilities:senderP,receiverProbabilities:receiverP,counterfactuals:state.receiver.map(row=>softmax(row,.18))});state.traces=state.traces.slice(0,16);}
    if(state.episodes%100===0){const accuracy=state.recent.reduce((s,v)=>s+v,0)/Math.max(1,state.recent.length);state.chart.push({episode:state.episodes,accuracy:accuracy*100});state.chart=state.chart.slice(-80);}
  } return state;
}

export function evaluate(snapshot:Snapshot,mode:Evaluation["mode"],trials=800):Evaluation{const random=mulberry32(snapshot.seed+snapshot.episodes+mode.length*10007);let successes=0;for(let i=0;i<trials;i++){const target=Math.floor(random()*SIZE);const sent=greedy(softmax(snapshot.sender[target],.12));const received=mode==="silent"?null:mode==="scrambled"?Math.floor(random()*SIZE):sent;const action=received===null?Math.floor(random()*SIZE):greedy(softmax(snapshot.receiver[received],.12));if(action===target)successes++;}return{mode,accuracy:successes/trials*100,trials};}
export function dictionary(snapshot:Snapshot){return MEANINGS.map((meaning,target)=>{const p=softmax(snapshot.sender[target],.18);const tokenIndex=greedy(p);const rp=softmax(snapshot.receiver[tokenIndex],.18);return{meaning,token:TOKENS[tokenIndex],tokenIndex,confidence:p[tokenIndex]*100,interpretedAs:MEANINGS[greedy(rp)],interpretationConfidence:Math.max(...rp)*100};});}
export function rollingAccuracy(snapshot:Snapshot){return snapshot.recent.length?snapshot.recent.reduce((s,v)=>s+v,0)/snapshot.recent.length*100:25;}
