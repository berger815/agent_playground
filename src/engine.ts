export type Activation = "tanh" | "relu" | "sigmoid";
export type SensorKey = "energy" | "resourceVector" | "hazardVector" | "neighborVector" | "heardSignal" | "crowding";
export type TrainingMode = "autonomous" | "coach-enabled";

export const SIGNALS = ["·", "▲", "●", "◆", "■"] as const;
export const MOVES = ["hold", "north", "east", "south", "west"] as const;
export const SENSOR_LABELS: Record<SensorKey,string> = {
  energy:"Internal energy", resourceVector:"Resource vector", hazardVector:"Hazard vector",
  neighborVector:"Neighbor vector", heardSignal:"Nearby signals", crowding:"Local crowding"
};

export interface Genome {
  id:string; name:string; color:string; hidden:number[]; activation:Activation;
  learningRate:number; exploration:number; mutation:number; sensors:SensorKey[];
  vision:number; harvest:number; metabolism:number; trainingMode:TrainingMode;
}

export interface Network { sizes:number[]; weights:number[][][]; activation:Activation; }
export interface Experience { input:number[]; move:number; signal:number; spawn:boolean; }
export interface Agent {
  id:string; genome:Genome; network:Network; x:number; y:number; energy:number; age:number;
  generation:number; parentId:string|null; totalReward:number; rewardBaseline:number;
  lastReward:number; lastSignal:number; lastMove:number; lastSpawnProbability:number;
  moveProbabilities:number[]; signalProbabilities:number[]; history:Experience[];
  coachingEvents:number; births:number;
}

export interface Resource { id:string; x:number; y:number; energy:number; }
export interface Hazard { id:string; x:number; y:number; strength:number; }
export interface WorldRules {
  width:number; height:number; resourceRate:number; resourceCap:number; hazardCount:number;
  cooperation:number; signalRange:number; signalCost:number; reproductionEnergy:number;
  populationCap:number; architectureMutation:number; learningEnabled:boolean;
}
export interface EventRecord { tick:number; type:"birth"|"death"|"cooperation"|"intervention"; message:string; agentId?:string; }
export interface Simulation {
  version:5; tick:number; rngState:number; agents:Agent[]; resources:Resource[]; hazards:Hazard[];
  rules:WorldRules; events:EventRecord[]; cooperationEvents:number; totalSignals:number;
  totalDecisions:number; nextId:number;
}

export const DEFAULT_RULES:WorldRules={
  width:960,height:560,resourceRate:.055,resourceCap:44,hazardCount:7,cooperation:1,
  signalRange:150,signalCost:.0007,reproductionEnergy:2.8,populationCap:42,
  architectureMutation:.12,learningEnabled:true
};

const clamp=(v:number,min:number,max:number)=>Math.max(min,Math.min(max,v));
const activate=(v:number,a:Activation)=>a==="tanh"?Math.tanh(v):a==="relu"?Math.max(0,v):1/(1+Math.exp(-clamp(v,-30,30)));
const derivative=(activated:number,a:Activation)=>a==="tanh"?1-activated*activated:a==="relu"?(activated>0?1:0):activated*(1-activated);
const softmax=(values:number[],temperature=1)=>{
  const t=Math.max(.08,temperature),m=Math.max(...values),exps=values.map(v=>Math.exp(clamp((v-m)/t,-30,30))),sum=exps.reduce((a,b)=>a+b,0);
  return exps.map(v=>v/sum);
};
const sigmoid=(v:number)=>1/(1+Math.exp(-clamp(v,-30,30)));

export function sensorWidth(sensors:SensorKey[]){return sensors.reduce((n,s)=>n+(s==="energy"||s==="crowding"?1:s==="heardSignal"?SIGNALS.length:2),0)}

export function createNetwork(sizes:number[],activation:Activation,random=Math.random):Network{
  const weights:number[][][]=[];
  for(let layer=1;layer<sizes.length;layer++){
    const scale=Math.sqrt(2/Math.max(1,sizes[layer-1]));
    weights.push(Array.from({length:sizes[layer]},()=>Array.from({length:sizes[layer-1]+1},()=>((random()*2)-1)*scale)));
  }
  return{sizes:[...sizes],weights,activation};
}

export function forward(network:Network,input:number[]){
  let values=[...input];
  for(let layer=0;layer<network.weights.length;layer++){
    const last=layer===network.weights.length-1;
    values=network.weights[layer].map(row=>{
      let sum=row[row.length-1];
      for(let i=0;i<values.length;i++)sum+=row[i]*values[i];
      return last?sum:activate(sum,network.activation);
    });
  }
  return values;
}

function choose(probabilities:number[],random:()=>number){let p=random();for(let i=0;i<probabilities.length;i++){p-=probabilities[i];if(p<=0)return i}return probabilities.length-1}

function trainExperience(network:Network,experience:Experience,advantage:number,learningRate:number){
  const activations:number[][]=[[...experience.input]];
  let values=[...experience.input];
  for(let layer=0;layer<network.weights.length;layer++){
    const last=layer===network.weights.length-1;
    values=network.weights[layer].map(row=>{
      let sum=row[row.length-1];for(let i=0;i<values.length;i++)sum+=row[i]*values[i];
      return last?sum:activate(sum,network.activation);
    });
    activations.push(values);
  }
  const logits=activations.at(-1)!;
  const moveP=softmax(logits.slice(0,MOVES.length));
  const signalP=softmax(logits.slice(MOVES.length,MOVES.length+SIGNALS.length));
  let gradients=Array(logits.length).fill(0);
  for(let i=0;i<moveP.length;i++)gradients[i]=((i===experience.move?1:0)-moveP[i])*advantage;
  for(let i=0;i<signalP.length;i++)gradients[MOVES.length+i]=((i===experience.signal?1:0)-signalP[i])*advantage;
  const spawnIndex=MOVES.length+SIGNALS.length,spawnP=sigmoid(logits[spawnIndex]);
  gradients[spawnIndex]=((experience.spawn?1:0)-spawnP)*advantage*.35;
  for(let layer=network.weights.length-1;layer>=0;layer--){
    const inputs=activations[layer],rows=network.weights[layer],previous=Array(inputs.length).fill(0);
    for(let o=0;o<rows.length;o++){
      const g=clamp(gradients[o],-2,2);
      for(let i=0;i<inputs.length;i++){previous[i]+=rows[o][i]*g;rows[o][i]=clamp(rows[o][i]+learningRate*g*inputs[i],-8,8)}
      rows[o][inputs.length]=clamp(rows[o][inputs.length]+learningRate*g,-8,8);
    }
    if(layer>0)gradients=previous.map((g,i)=>g*derivative(inputs[i],network.activation));
  }
}

function nextRandom(sim:Simulation){sim.rngState=(sim.rngState+0x6D2B79F5)|0;let t=sim.rngState;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}
const randomRange=(sim:Simulation,min:number,max:number)=>min+nextRandom(sim)*(max-min);
const distance=(a:{x:number;y:number},b:{x:number;y:number})=>Math.hypot(a.x-b.x,a.y-b.y);
const logEvent=(sim:Simulation,event:EventRecord)=>{sim.events.unshift(event);sim.events=sim.events.slice(0,80)};

export function createGenome(overrides:Partial<Genome>={}):Genome{
  return{
    id:overrides.id??`genome_${Math.random().toString(36).slice(2,9)}`,name:overrides.name??"Founder",
    color:overrides.color??"#a8ff60",hidden:overrides.hidden??[12,8],activation:overrides.activation??"tanh",
    learningRate:overrides.learningRate??.025,exploration:overrides.exploration??.72,mutation:overrides.mutation??.08,
    sensors:overrides.sensors??["energy","resourceVector","neighborVector","heardSignal"],vision:overrides.vision??150,
    harvest:overrides.harvest??1,metabolism:overrides.metabolism??1,trainingMode:overrides.trainingMode??"autonomous"
  };
}

function makeAgent(sim:Simulation,genome:Genome,generation=0,parentId:string|null=null,network?:Network):Agent{
  const input=sensorWidth(genome.sensors),sizes=[input,...genome.hidden,MOVES.length+SIGNALS.length+1];
  return{id:`agent_${sim.nextId++}`,genome:{...genome,id:`genome_${sim.nextId}_${Math.floor(nextRandom(sim)*1e6)}`},
    network:network??createNetwork(sizes,genome.activation,()=>nextRandom(sim)),x:randomRange(sim,sim.rules.width*.32,sim.rules.width*.68),y:randomRange(sim,sim.rules.height*.3,sim.rules.height*.7),
    energy:2.05,age:0,generation,parentId,totalReward:0,rewardBaseline:0,lastReward:0,lastSignal:0,lastMove:0,lastSpawnProbability:0,
    moveProbabilities:Array(MOVES.length).fill(1/MOVES.length),signalProbabilities:Array(SIGNALS.length).fill(1/SIGNALS.length),history:[],coachingEvents:0,births:0};
}

function resource(sim:Simulation):Resource{return{id:`resource_${sim.nextId++}`,x:randomRange(sim,18,sim.rules.width-18),y:randomRange(sim,18,sim.rules.height-18),energy:randomRange(sim,1.1,1.9)}}
function hazard(sim:Simulation):Hazard{return{id:`hazard_${sim.nextId++}`,x:randomRange(sim,35,sim.rules.width-35),y:randomRange(sim,35,sim.rules.height-35),strength:randomRange(sim,.25,.52)}}

export function createSimulation(seed=815):Simulation{
  const sim:Simulation={version:5,tick:0,rngState:seed|0,agents:[],resources:[],hazards:[],rules:{...DEFAULT_RULES},events:[],cooperationEvents:0,totalSignals:0,totalDecisions:0,nextId:1};
  const scout=createGenome({name:"Lumen",color:"#a8ff60",hidden:[12,8],sensors:["energy","resourceVector","neighborVector","heardSignal"],vision:210,harvest:.55,metabolism:.92});
  const gatherer=createGenome({name:"Quarry",color:"#ffb45c",hidden:[10,10],sensors:["energy","neighborVector","heardSignal","crowding"],vision:85,harvest:1.65,metabolism:1.05});
  const generalist=createGenome({name:"Drift",color:"#75d6ff",hidden:[14],sensors:["energy","resourceVector","hazardVector","neighborVector","heardSignal"],vision:140,harvest:1,metabolism:1});
  sim.agents=[makeAgent(sim,scout),makeAgent(sim,gatherer),makeAgent(sim,generalist)];
  for(let i=0;i<28;i++)sim.resources.push(resource(sim));
  for(let i=0;i<sim.rules.hazardCount;i++)sim.hazards.push(hazard(sim));
  logEvent(sim,{tick:0,type:"intervention",message:"Three unlike founders released into an unwritten ecology."});
  return sim;
}

function vectorTo(origin:{x:number;y:number},target:{x:number;y:number}|undefined,range:number){
  if(!target)return[0,0];return[clamp((target.x-origin.x)/range,-1,1),clamp((target.y-origin.y)/range,-1,1)];
}

function sense(sim:Simulation,agent:Agent){
  const visibleResources=sim.resources.filter(r=>distance(agent,r)<=agent.genome.vision).sort((a,b)=>distance(agent,a)-distance(agent,b));
  const visibleHazards=sim.hazards.filter(h=>distance(agent,h)<=agent.genome.vision).sort((a,b)=>distance(agent,a)-distance(agent,b));
  const neighbors=sim.agents.filter(a=>a.id!==agent.id).sort((a,b)=>distance(agent,a)-distance(agent,b));
  const close=neighbors.filter(a=>distance(agent,a)<=sim.rules.signalRange),heard=close[0]?.lastSignal??0;
  const parts:Record<SensorKey,number[]>={
    energy:[clamp(agent.energy/3,0,1)],resourceVector:vectorTo(agent,visibleResources[0],agent.genome.vision),
    hazardVector:vectorTo(agent,visibleHazards[0],agent.genome.vision),neighborVector:vectorTo(agent,neighbors[0],agent.genome.vision),
    heardSignal:SIGNALS.map((_,i)=>i===heard?1:0),crowding:[clamp(close.length/8,0,1)]
  };
  return agent.genome.sensors.flatMap(sensor=>parts[sensor]);
}

function decide(sim:Simulation,agent:Agent,input:number[]){
  const logits=forward(agent.network,input),moveProbabilities=softmax(logits.slice(0,MOVES.length),agent.genome.exploration),signalProbabilities=softmax(logits.slice(MOVES.length,MOVES.length+SIGNALS.length),agent.genome.exploration);
  const spawnProbability=sigmoid(logits[MOVES.length+SIGNALS.length]);
  return{move:choose(moveProbabilities,()=>nextRandom(sim)),signal:choose(signalProbabilities,()=>nextRandom(sim)),spawn:nextRandom(sim)<spawnProbability,spawnProbability,moveProbabilities,signalProbabilities};
}

function reshapeNetwork(sim:Simulation,parent:Network,genome:Genome){
  const next=createNetwork([sensorWidth(genome.sensors),...genome.hidden,MOVES.length+SIGNALS.length+1],genome.activation,()=>nextRandom(sim));
  for(let l=0;l<Math.min(parent.weights.length,next.weights.length);l++)for(let o=0;o<Math.min(parent.weights[l].length,next.weights[l].length);o++)for(let i=0;i<Math.min(parent.weights[l][o].length,next.weights[l][o].length);i++)next.weights[l][o][i]=parent.weights[l][o][i]+(nextRandom(sim)*2-1)*genome.mutation;
  return next;
}

function descendantGenome(sim:Simulation,parent:Genome){
  const child:Genome={...parent,hidden:[...parent.hidden],sensors:[...parent.sensors],name:parent.name};
  if(nextRandom(sim)<sim.rules.architectureMutation){
    const change=nextRandom(sim);
    if(change<.45&&child.hidden.length<3)child.hidden.push(4+Math.floor(nextRandom(sim)*9));
    else if(change<.8&&child.hidden.length>0){const i=Math.floor(nextRandom(sim)*child.hidden.length);child.hidden[i]=clamp(child.hidden[i]+(nextRandom(sim)<.5?-2:2),3,32)}
    else child.activation=(["tanh","relu","sigmoid"] as Activation[])[Math.floor(nextRandom(sim)*3)];
  }
  child.learningRate=clamp(child.learningRate+(nextRandom(sim)*2-1)*.003,.001,.08);
  child.exploration=clamp(child.exploration+(nextRandom(sim)*2-1)*.05,.12,1.8);
  return child;
}

function applyReward(sim:Simulation,agent:Agent,reward:number){
  agent.lastReward=reward;agent.totalReward+=reward;
  const advantage=reward-agent.rewardBaseline;agent.rewardBaseline=agent.rewardBaseline*.985+reward*.015;
  if(!sim.rules.learningEnabled||Math.abs(advantage)<.0001)return;
  const history=agent.history;
  for(let lag=0;lag<history.length;lag++)trainExperience(agent.network,history[history.length-1-lag],advantage*Math.pow(.91,lag),agent.genome.learningRate);
}

export function stepSimulation(sim:Simulation,steps=1){
  for(let step=0;step<steps;step++){
    sim.tick++;
    while(sim.hazards.length<sim.rules.hazardCount)sim.hazards.push(hazard(sim));
    while(sim.hazards.length>sim.rules.hazardCount)sim.hazards.pop();
    if(sim.resources.length<sim.rules.resourceCap&&nextRandom(sim)<sim.rules.resourceRate)sim.resources.push(resource(sim));
    const rewards=new Map<string,number>(),decisions=new Map<string,ReturnType<typeof decide>>();
    for(const agent of sim.agents){
      agent.age++;const nearestBefore=agent.genome.sensors.includes("resourceVector")?[...sim.resources].sort((a,b)=>distance(agent,a)-distance(agent,b))[0]:undefined,beforeDistance=nearestBefore?distance(agent,nearestBefore):null;
      const input=sense(sim,agent),decision=decide(sim,agent,input);decisions.set(agent.id,decision);
      agent.moveProbabilities=decision.moveProbabilities;agent.signalProbabilities=decision.signalProbabilities;agent.lastSpawnProbability=decision.spawnProbability;
      agent.lastMove=decision.move;agent.lastSignal=decision.signal;agent.history.push({input,move:decision.move,signal:decision.signal,spawn:decision.spawn});if(agent.history.length>22)agent.history.shift();
      const speed=2.2;if(decision.move===1)agent.y-=speed;if(decision.move===2)agent.x+=speed;if(decision.move===3)agent.y+=speed;if(decision.move===4)agent.x-=speed;
      agent.x=clamp(agent.x,8,sim.rules.width-8);agent.y=clamp(agent.y,8,sim.rules.height-8);
      const cost=.00065*agent.genome.metabolism+(decision.signal>0?sim.rules.signalCost:0);agent.energy-=cost;rewards.set(agent.id,-cost);
      if(nearestBefore&&beforeDistance!==null&&beforeDistance<=agent.genome.vision){const progress=beforeDistance-distance(agent,nearestBefore);rewards.set(agent.id,(rewards.get(agent.id)??0)+clamp(progress*.0025,-.008,.008))}
      if(decision.signal>0)sim.totalSignals++;sim.totalDecisions++;
      for(const h of sim.hazards)if(distance(agent,h)<15){const hit=h.strength*.045;agent.energy-=hit;rewards.set(agent.id,(rewards.get(agent.id)??0)-hit*.8)}
    }
    for(let i=sim.resources.length-1;i>=0;i--){
      const r=sim.resources[i],participants=sim.agents.filter(a=>distance(a,r)<27);
      if(participants.length<sim.rules.cooperation){
        if(participants.length){for(const agent of participants){const graze=.004*agent.genome.harvest;agent.energy+=graze;rewards.set(agent.id,(rewards.get(agent.id)??0)+graze)}r.energy-=.004;if(r.energy<=.15)sim.resources.splice(i,1)}
        continue;
      }
      const totalHarvest=participants.reduce((sum,a)=>sum+a.genome.harvest,0),yieldValue=r.energy*(1+.18*(sim.rules.cooperation-1));
      for(const agent of participants){const share=yieldValue*agent.genome.harvest/Math.max(.01,totalHarvest);agent.energy+=share;rewards.set(agent.id,(rewards.get(agent.id)??0)+share)}
      sim.resources.splice(i,1);
      if(sim.rules.cooperation>1){sim.cooperationEvents++;if(sim.cooperationEvents%12===1)logEvent(sim,{tick:sim.tick,type:"cooperation",message:`${participants.map(a=>a.genome.name).join(" + ")} unlocked a shared resource.`})}
    }
    const newborns:Agent[]=[];
    for(const agent of sim.agents){
      const decision=decisions.get(agent.id)!;
      if(decision.spawn&&agent.energy>=sim.rules.reproductionEnergy&&sim.agents.length+newborns.length<sim.rules.populationCap){
        const genome=descendantGenome(sim,agent.genome),network=reshapeNetwork(sim,agent.network,genome),child=makeAgent(sim,genome,agent.generation+1,agent.id,network);
        child.x=clamp(agent.x+randomRange(sim,-18,18),8,sim.rules.width-8);child.y=clamp(agent.y+randomRange(sim,-18,18),8,sim.rules.height-8);
        child.energy=agent.energy*.46;agent.energy*=.54;agent.births++;newborns.push(child);rewards.set(agent.id,(rewards.get(agent.id)??0)+.05);
        logEvent(sim,{tick:sim.tick,type:"birth",agentId:child.id,message:`${agent.genome.name} produced generation ${child.generation}; architecture ${genome.hidden.join("×")||"linear"}.`});
      }
      applyReward(sim,agent,rewards.get(agent.id)??0);
    }
    sim.agents.push(...newborns);
    const dead=sim.agents.filter(a=>a.energy<=0||a.age>7200);
    for(const agent of dead)logEvent(sim,{tick:sim.tick,type:"death",agentId:agent.id,message:`${agent.genome.name} left the active ecology after ${agent.age} steps.`});
    sim.agents=sim.agents.filter(a=>a.energy>0&&a.age<=7200);
  }
  return sim;
}

export function releaseFounder(sim:Simulation,genome:Genome){const agent=makeAgent(sim,genome);sim.agents.push(agent);logEvent(sim,{tick:sim.tick,type:"intervention",agentId:agent.id,message:`Human-built founder ${genome.name} released.`});return agent}

export function cloneSelected(sim:Simulation,agentId:string,mutate=true){
  const parent=sim.agents.find(a=>a.id===agentId);if(!parent||sim.agents.length>=sim.rules.populationCap)return null;
  const genome=mutate?descendantGenome(sim,parent.genome):{...parent.genome,hidden:[...parent.genome.hidden],sensors:[...parent.genome.sensors]};
  const network=mutate?reshapeNetwork(sim,parent.network,genome):JSON.parse(JSON.stringify(parent.network)) as Network;
  const child=makeAgent(sim,genome,parent.generation+1,parent.id,network);sim.agents.push(child);
  logEvent(sim,{tick:sim.tick,type:"birth",agentId:child.id,message:`${mutate?"Mutated descendant":"Exact learned clone"} created from ${parent.genome.name}.`});return child;
}

export function coachAgent(sim:Simulation,agentId:string,reward:number){
  const agent=sim.agents.find(a=>a.id===agentId);if(!agent||agent.genome.trainingMode!=="coach-enabled")return false;
  agent.coachingEvents++;applyReward(sim,agent,reward);logEvent(sim,{tick:sim.tick,type:"intervention",agentId,message:`Coach supplied ${reward>0?"positive":"negative"} feedback to ${agent.genome.name}.`});return true;
}

export function setRules(sim:Simulation,next:Partial<WorldRules>){sim.rules={...sim.rules,...next};return sim}
export function finiteSimulation(sim:Simulation){return sim.version===5&&sim.agents.every(a=>a.network.weights.flat(2).every(Number.isFinite))}
export function populationMetrics(sim:Simulation){
  const population=sim.agents.length,meanEnergy=population?sim.agents.reduce((s,a)=>s+a.energy,0)/population:0,maxGeneration=population?Math.max(...sim.agents.map(a=>a.generation)):0;
  return{population,meanEnergy,maxGeneration,cooperationEvents:sim.cooperationEvents,signalRate:sim.totalDecisions?sim.totalSignals/sim.totalDecisions:0,resources:sim.resources.length};
}
