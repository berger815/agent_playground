import test from "node:test";
import assert from "node:assert/strict";
import {createLab,evaluate,rollingAccuracy,runKey,trainLab} from "../src/babel.ts";

test("starter policies acquire a causal two-symbol protocol",()=>{
  let lab=createLab();
  lab=trainLab(lab,2500);
  const sender=lab.agents[0],receiver=lab.agents[1],world=lab.worlds[0];
  const run=lab.runs[runKey(sender.id,receiver.id,world.id)];
  const intact=evaluate(sender,receiver,world,"intact",800).accuracy;
  const scrambled=evaluate(sender,receiver,world,"scrambled",800).accuracy;
  const weights=lab.agents.flatMap(agent=>agent.sender.weights.flat(2).concat(agent.receiver.weights.flat(2)));
  assert.ok(weights.every(Number.isFinite),"all learned weights should remain finite");
  assert.ok(rollingAccuracy(run)>90,"the starter pair should master Hidden Compass");
  assert.ok(intact>90,"the intact protocol should coordinate reliably");
  assert.ok(intact-scrambled>30,"scrambling symbols should causally break coordination");
});
