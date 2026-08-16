# Babel — Open Artificial Ecology

> A locally running artificial-development laboratory where users construct neural agents, release them beside other lineages, alter the laws of the world, and inspect what training and selection produce.

**Build a mind. Change the pressure. Follow the descendants.**

Babel is no longer organized as a ladder of scripted scenarios. Version 0.5 is a continuous ecology with no prescribed winning sequence. Resources, hazards, energy, sensing, signaling, cooperation, reproduction, and learning all operate at the same time under rules the user can change while the ecology is running.

## What is real in v0.5

- **Agent Foundry:** choose zero to three hidden layers, neurons per layer, activation function, learning rate, exploration temperature, mutation scale, sensors, sensing radius, harvest capability, and metabolic cost.
- **Actual neural policies:** every active agent runs a configurable feed-forward network whose weights update through an online policy-gradient eligibility trace.
- **Open ecology:** agents move, sense, signal, collect resources, encounter hazards, spend energy, and reproduce continuously. There is no level-completion script.
- **Unequal bodies:** observers, gatherers, generalists, and minimal agents can coexist. A user may create any other combination.
- **Communication pressure:** nearby signals are optional sensory inputs; signaling costs energy; shared resources may require multiple agents. Whether a useful protocol appears is an experimental result, not a scripted event.
- **Within-life training and inheritance:** agents update weights from recent ecological outcomes. Descendants inherit learned weights with optional mutation.
- **Architectural mutation:** descendants can change layer depth, width, activation, learning rate, and exploration when structural mutation is enabled.
- **Human-assisted versus autonomous lineages:** coach-enabled agents accept recorded positive or negative feedback. Autonomous agents remain isolated from explicit human reward.
- **Decision microscope:** inspect action probabilities, emitted symbol probabilities, reproduction impulse, architecture, weight saturation, sensors, reward, and intervention history.
- **Causal ledger:** births, deaths, cooperative harvests, and human interventions are recorded without inventing an internal narrative.
- **Persistent laboratory:** browser-local autosave plus portable JSON export and import.

## Ecology and learning

Every step changes an agent's energy. Resources create positive reward; metabolism, signaling, and hazards create negative reward. The reward is applied backward through a short eligibility trace so recent movement and signaling decisions can be reinforced. Each agent maintains its own moving reward baseline.

Reproduction is an output of the policy rather than an automatic clock. It can occur only above the world's energy threshold and below the population ceiling. The descendant inherits the parent's trained network; mutation may alter weights or network structure. An inactive lineage is not described as punished—the ecology simply stopped allocating it an active population slot.

## Run locally

```bash
npm install
npm run dev
```

Validation:

```bash
npm test
npm run build
```

No model API, paid service, account, or server is required after the static application loads.

## Honest boundary

This is a small artificial ecology, not unrestricted artificial life. Agents have bounded sensors, five movement actions, five signal states, and a compact reproduction output. They can acquire policies and produce mutated descendants, but they do not yet deliberately specify a child's complete architecture or interpret unrestricted natural-language goals. Those are future experimental layers and are not claimed by this release.
