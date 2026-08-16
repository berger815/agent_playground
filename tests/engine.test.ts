import test from "node:test";
import assert from "node:assert/strict";
import {coachAgent,cloneSelected,createGenome,createSimulation,finiteSimulation,releaseFounder,stepSimulation} from "../src/engine.ts";

test("the open ecology remains numerically finite while policies train",()=>{
  const sim=createSimulation(815);
  stepSimulation(sim,1200);
  assert.ok(sim.tick===1200);
  assert.ok(sim.resources.length<=sim.rules.resourceCap);
  assert.ok(finiteSimulation(sim),"all active learned weights should remain finite");
  assert.ok(sim.totalDecisions>0,"agents should have acted in the ecology");
});

test("a human-built neural architecture is released exactly as configured",()=>{
  const sim=createSimulation(91);
  const genome=createGenome({name:"Test founder",hidden:[7,5,3],activation:"relu",sensors:["energy","heardSignal"],trainingMode:"coach-enabled"});
  const agent=releaseFounder(sim,genome);
  assert.deepEqual(agent.network.sizes,[6,7,5,3,11]);
  assert.equal(agent.network.activation,"relu");
  assert.equal(agent.genome.trainingMode,"coach-enabled");
});

test("coaching is isolated from autonomous lineages and recorded for enabled agents",()=>{
  const sim=createSimulation(42);
  const autonomous=sim.agents[0];
  const coached=releaseFounder(sim,createGenome({name:"Coached",trainingMode:"coach-enabled"}));
  stepSimulation(sim,8);
  assert.equal(coachAgent(sim,autonomous.id,.5),false);
  assert.equal(coachAgent(sim,coached.id,.5),true);
  assert.equal(coached.coachingEvents,1);
  assert.ok(finiteSimulation(sim));
});

test("learned policies can be cloned or structurally mutated without deleting the parent",()=>{
  const sim=createSimulation(123);
  const parent=sim.agents[0],count=sim.agents.length;
  const exact=cloneSelected(sim,parent.id,false);
  const mutated=cloneSelected(sim,parent.id,true);
  assert.ok(exact&&mutated);
  assert.equal(sim.agents.length,count+2);
  assert.ok(sim.agents.some(agent=>agent.id===parent.id));
  assert.deepEqual(exact!.network.weights,parent.network.weights);
  assert.ok(finiteSimulation(sim));
});
