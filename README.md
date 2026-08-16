# Babel — Artificial Development Laboratory

> A locally running artificial-development laboratory where users create agents, agents train with other agents, environments evolve around their limitations, and increasingly complex language becomes necessary for continued success.

**Build agents. Evolve worlds. Watch language emerge.**

Babel is a browser-native laboratory for emergent communication. It is deliberately small enough to inspect: every agent is an actual configurable multilayer policy, every reward is explicit, and every claim can be challenged with a causal intervention.

## Development Lab 0.4

- **Agent Foundry:** create agents with 0–3 hidden layers, 4–32 neurons per layer, selectable activation, learning rate, exploration temperature, and mutation scale.
- **Real inheritance:** clone learned weights exactly or create a perturbed descendant while keeping the parent.
- **Training partnerships:** select any sender, receiver, and world. Each combination has an independent experiment ledger.
- **World Forge:** direct a local curriculum compiler, inspect its explicit proposal, and preserve an indefinite lineage of environmental descendants.
- **Cross-play:** evaluate every sender against every receiver to distinguish transferable structure from private dialects.
- **Decision microscope:** inspect action probabilities, chosen symbols, reward, baseline-relative advantage, and recent trials.
- **Causal tests:** compare intact communication against scrambled and silent channels using frozen policies.
- **Persistent laboratory:** browser-local autosave plus portable JSON export and import.
- **Human bridge:** compile a bounded direction instruction into the learned protocol and execute it without an LLM or API.
- **Compositional tasks:** Echo Corridor combines learned meanings into two-symbol instructions such as “north, then east” and rewards only a completely correct action chain.
- **Research Expeditions:** a persistent evidence-based objective path points toward the next meaningful experiment without manufacturing a daily streak.

## Reward and learning

Both policies receive `+1.00` for a correct joint action and `−0.25` for an incorrect one. A moving reward baseline supplies the advantage signal. Policy gradients update the sender's selected symbol and receiver's selected action through their configured networks. Agents are never gained or deleted as punishment; the user performs selection by preserving, cloning, mutating, or retiring attention from a lineage.

## Run locally

```bash
npm install
npm run dev
```

No paid service, model API, account, or server is required after the static app loads.

## Honest boundary

World lineages and task sequences are open-ended within safeguards, but the current semantic substrate is bounded to eight spatial meanings and eight symbols. This release establishes genuine multi-symbol composition. Objects and attributes, ecological resources, reproduction, cultural transmission, and unrestricted natural language remain future substrate expansions—not claims made by v0.4.
