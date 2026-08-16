# Agent Forge

A phone-first machine-learning playground: draw a course, demonstrate it with phone tilt or touch, train a small neural policy locally, then freeze learning and evaluate the result.

**Live prototype:** https://agent-playground.mike815.chatgpt.site

## The loop

1. Build an ordered course with gates, obstacles, a start, and a goal.
2. Teach by steering with device orientation or touch; the app records observation/action pairs.
3. Configure a small network (width, depth, activation) and train it in the browser.
4. Test with learning frozen and inspect observation → action → result → reward.
5. Clone and mutate the trained policy.

The score remains a visible vector: completion, time, collisions, energy, and smoothness. No model API, backend, or paid service is required.

## Run locally

    npm install
    npm run dev
