export type Activation = "tanh" | "relu" | "sigmoid";
export type SensorKind = "none" | "proximity" | "radar" | "lidar";
export type Point = { x: number; y: number };
export type Obstacle = Point & { r: number };
export type Gate = Point & { r: number };
export type Course = { name: string; start: Point & { angle: number }; finish: Gate; gates: Gate[]; obstacles: Obstacle[] };
export type EnergyLedger = { propulsion: number; sensing: number; helpers: number; comms: number; compute: number };
export type Car = Point & {
  angle: number; speed: number; gate: number; collisions: number; energy: number; energyLedger: EnergyLedger;
  jerk: number; lastSteer: number; elapsed: number; distance: number; complete: boolean;
};
export type Control = { steer: number; throttle: number };
export type Sample = { input: number[]; output: number[] };
export type Network = { sizes: number[]; activation: Activation; weights: number[][][] };
export type Score = { completion: number; time: number; collisions: number; energy: number; smoothness: number; efficiency: number; distance: number; optimal: number };
export type Decision = { observation: string; action: string; result: string; reward: number; update: string; inputs: number[]; outputs: number[] };
export type HelperRole = "scout" | "finisher";
export type Helper = { id: string; role: HelperRole; sensor: SensorKind; network: Network };
export type Team = { id: string; generation: number; driver: Network; driverSensor: SensorKind; helpers: Helper[]; parentId: string | null };
export type SensorSpec = { label: string; range: number; rays: number; cost: number; description: string };
export type MessagePacket = { from: string; role: HelperRole; bearing: number; distance: number; size: number; confidence: number; age: number };
export type HelperBody = Point & { id: string; role: HelperRole; sensor: SensorKind; angle: number; speed: number; packetCooldown: number };
export type TeamRuntime = { helpers: HelperBody[]; messages: MessagePacket[] };
export type AutoResult = { team: Team; fitness: number; score: Score; parameters: number };

export const W = 720, H = 980, INPUTS = 12;
export const SENSOR_SPECS: Record<SensorKind, SensorSpec> = {
  none: { label: "None", range: 0, rays: 0, cost: 0, description: "No obstacle data" },
  proximity: { label: "Proximity", range: 78, rays: 3, cost: 0.06, description: "Cheap, short-range feelers" },
  radar: { label: "Forward radar", range: 175, rays: 5, cost: 0.16, description: "Focused look ahead" },
  lidar: { label: "360° lidar", range: 265, rays: 12, cost: 0.42, description: "Broad view, highest draw" },
};
export const clamp = (n: number, a: number, b: number) => Math.max(a, Math.min(b, n));
const rand = (a = -1, b = 1) => a + Math.random() * (b - a);
const normAngle = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

export const defaultCourse = (): Course => ({ name: "Night Slalom", start: { x: 110, y: 880, angle: -1.42 }, finish: { x: 610, y: 100, r: 40 }, gates: [{ x: 235, y: 760, r: 34 }, { x: 510, y: 650, r: 34 }, { x: 245, y: 510, r: 34 }, { x: 500, y: 355, r: 34 }, { x: 360, y: 220, r: 34 }], obstacles: [{ x: 355, y: 740, r: 48 }, { x: 360, y: 590, r: 56 }, { x: 350, y: 420, r: 48 }, { x: 210, y: 300, r: 42 }, { x: 530, y: 230, r: 44 }] });
export const blankCourse = (): Course => ({ name: "My course", start: { x: 100, y: 870, angle: -Math.PI / 2 }, finish: { x: 620, y: 110, r: 40 }, gates: [], obstacles: [] });
export const emptyEnergy = (): EnergyLedger => ({ propulsion: 0, sensing: 0, helpers: 0, comms: 0, compute: 0 });
export const makeCar = (course: Course): Car => ({ ...course.start, speed: 0, gate: 0, collisions: 0, energy: 0, energyLedger: emptyEnergy(), jerk: 0, lastSteer: 0, elapsed: 0, distance: 0, complete: false });

function activate(x: number, a: Activation) { return a === "tanh" ? Math.tanh(x) : a === "relu" ? Math.max(0, x) : 1 / (1 + Math.exp(-clamp(x, -20, 20))); }
function derivative(y: number, a: Activation) { return a === "tanh" ? 1 - y * y : a === "relu" ? (y > 0 ? 1 : 0) : y * (1 - y); }
export function createNetwork(width = 12, depth = 2, activation: Activation = "tanh"): Network {
  const sizes = [INPUTS, ...Array(depth).fill(width), 2], weights: number[][][] = [];
  for (let l = 1; l < sizes.length; l++) weights.push(Array.from({ length: sizes[l] }, () => Array.from({ length: sizes[l - 1] + 1 }, () => rand(-1, 1) * Math.sqrt(2 / sizes[l - 1]))));
  return { sizes, activation, weights };
}
export function cloneNetwork(n: Network, mutation = 0): Network { return { sizes: [...n.sizes], activation: n.activation, weights: n.weights.map(layer => layer.map(row => row.map(v => v + rand(-mutation, mutation)))) }; }
export function parameterCount(n: Network) { return n.weights.reduce((sum, layer) => sum + layer.reduce((s, row) => s + row.length, 0), 0); }
export function teamParameters(team: Team) { return parameterCount(team.driver) + team.helpers.reduce((sum, h) => sum + parameterCount(h.network), 0); }
export function createTeam(driver: Network): Team { return { id: `team_${Math.random().toString(36).slice(2, 8)}`, generation: 1, driver: cloneNetwork(driver), driverSensor: "radar", helpers: [], parentId: null }; }
export function forward(net: Network, input: number[]) { let values = [...input]; for (let l = 0; l < net.weights.length; l++) values = net.weights[l].map(row => activate(row.slice(0, -1).reduce((s, w, i) => s + w * (values[i] ?? 0), row.at(-1)!), l === net.weights.length - 1 ? "tanh" : net.activation)); return values; }
export function teamControl(team: Team, input: number[]): Control { const out = forward(team.driver, input); return { steer: clamp(out[0], -1, 1), throttle: clamp(out[1], -1, 1) }; }
export function targetFor(course: Course, car: Pick<Car, "gate">): Gate { return car.gate < course.gates.length ? course.gates[car.gate] : course.finish; }

function rayDistance(course: Course, origin: Point, angle: number, range: number) {
  let clear = range;
  for (const o of course.obstacles) {
    const dx = o.x - origin.x, dy = o.y - origin.y, along = dx * Math.cos(angle) + dy * Math.sin(angle), side = Math.abs(-dx * Math.sin(angle) + dy * Math.cos(angle));
    if (along > 0 && side < o.r + 15) clear = Math.min(clear, Math.max(0, along - o.r - 10));
  }
  return range ? clear / range : 1;
}
export function sensorReadings(course: Course, origin: Point & { angle: number }, kind: SensorKind) {
  const spec = SENSOR_SPECS[kind]; if (!spec.range) return [1, 1, 1];
  if (kind === "lidar") {
    const sectors = [[-2.6, -.32], [-.31, .31], [.32, 2.6]];
    return sectors.map(([a, b]) => { let best = 1; for (let i = 0; i < 4; i++) best = Math.min(best, rayDistance(course, origin, origin.angle + a + (b - a) * i / 3, spec.range)); return best; });
  }
  return [-.58, 0, .58].map(offset => rayDistance(course, origin, origin.angle + offset, spec.range));
}
export function observe(course: Course, car: Car, sensor: SensorKind = "radar", packet?: MessagePacket) {
  const t = targetFor(course, car), dx = t.x - car.x, dy = t.y - car.y, bearing = normAngle(Math.atan2(dy, dx) - car.angle), d = Math.hypot(dx, dy), rays = sensorReadings(course, car, sensor), pBearing = packet ? normAngle(packet.bearing - car.angle) : 0;
  return [clamp(Math.sin(bearing), -1, 1), clamp(Math.cos(bearing), -1, 1), clamp(d / 500, 0, 1.5), clamp(car.speed / 5, -1, 1), ...rays, packet ? Math.sin(pBearing) : 0, packet ? Math.cos(pBearing) : 0, packet ? clamp(packet.distance / 450, 0, 1.5) : 1, packet ? packet.confidence : 0, clamp(car.gate / Math.max(1, course.gates.length), 0, 1)];
}

function addEnergy(car: Car, key: keyof EnergyLedger, amount: number) { car.energyLedger[key] += amount; car.energy += amount; }
export function stepCar(course: Course, car: Car, control: Control, dt = .016) {
  if (car.complete) return { hit: false, gate: false, finished: true, progress: 0 };
  const before = Math.hypot(targetFor(course, car).x - car.x, targetFor(course, car).y - car.y), scale = dt * 60, steer = clamp(control.steer, -1, 1), throttle = clamp(control.throttle, -1, 1), ox = car.x, oy = car.y;
  car.speed = clamp(car.speed + throttle * .12 * scale - car.speed * .025 * scale, -2.1, 5.2); car.angle = normAngle(car.angle + steer * (.022 + .018 * Math.abs(car.speed)) * scale); car.x += Math.cos(car.angle) * car.speed * scale; car.y += Math.sin(car.angle) * car.speed * scale; car.elapsed += dt; car.distance += Math.hypot(car.x - ox, car.y - oy); addEnergy(car, "propulsion", (Math.abs(throttle) * .55 + Math.abs(car.speed) * .055) * dt); car.jerk += Math.abs(steer - car.lastSteer); car.lastSteer = steer;
  let hit = false;
  if (car.x < 18 || car.x > W - 18 || car.y < 18 || car.y > H - 18) { car.x = clamp(car.x, 18, W - 18); car.y = clamp(car.y, 18, H - 18); car.speed *= -.25; hit = true; }
  for (const o of course.obstacles) { const dx = car.x - o.x, dy = car.y - o.y, d = Math.hypot(dx, dy), min = o.r + 15; if (d < min) { const nx = dx / (d || 1), ny = dy / (d || 1); car.x = o.x + nx * min; car.y = o.y + ny * min; car.speed *= -.18; hit = true; } }
  if (hit) car.collisions++;
  const target = targetFor(course, car), after = Math.hypot(target.x - car.x, target.y - car.y); let gate = false, finished = false;
  if (after < target.r) { gate = true; if (car.gate < course.gates.length) car.gate++; else { car.complete = true; finished = true; car.speed = 0; } }
  return { hit, gate, finished, progress: clamp((before - after) / 40, -.2, .2) };
}

export function createTeamRuntime(team: Team, course: Course, car: Car): TeamRuntime {
  const helpers = team.helpers.map((h, i) => ({ id: h.id, role: h.role, sensor: h.sensor, x: car.x + (i ? -24 : 24), y: car.y + 20, angle: course.start.angle, speed: 0, packetCooldown: i * .12 }));
  if (helpers.length) addEnergy(car, "helpers", helpers.length * 3.5);
  return { helpers, messages: [] };
}
function detectedObstacle(course: Course, body: HelperBody) {
  const range = SENSOR_SPECS[body.sensor].range; if (!range) return null;
  let found: Obstacle | null = null, best = Infinity;
  for (const o of course.obstacles) { const d = Math.hypot(o.x - body.x, o.y - body.y) - o.r; if (d < range && d < best) { best = d; found = o; } }
  return found ? { obstacle: found, distance: Math.max(0, best), confidence: clamp(1 - best / range, .12, 1) } : null;
}
export function stepTeam(course: Course, car: Car, team: Team, runtime: TeamRuntime, dt = .016) {
  runtime.messages.forEach(m => m.age += dt); runtime.messages = runtime.messages.filter(m => m.age < 1.8);
  for (let i = 0; i < runtime.helpers.length; i++) {
    const body = runtime.helpers[i], target = body.role === "finisher" && car.gate >= course.gates.length - 1 ? course.finish : targetFor(course, car), lead = body.role === "scout" ? 125 : 70;
    const goalAngle = Math.atan2(target.y - car.y, target.x - car.x), gx = car.x + Math.cos(goalAngle) * lead + (i ? 22 : -22), gy = car.y + Math.sin(goalAngle) * lead, desired = Math.atan2(gy - body.y, gx - body.x);
    body.angle = normAngle(body.angle + clamp(normAngle(desired - body.angle), -.09, .09)); body.speed += (3.2 - body.speed) * .08; body.x = clamp(body.x + Math.cos(body.angle) * body.speed * dt * 44, 16, W - 16); body.y = clamp(body.y + Math.sin(body.angle) * body.speed * dt * 44, 16, H - 16); body.packetCooldown -= dt;
    addEnergy(car, "helpers", dt * (.13 + body.speed * .025)); addEnergy(car, "sensing", dt * SENSOR_SPECS[body.sensor].cost);
    const found = detectedObstacle(course, body);
    if (found && body.packetCooldown <= 0) {
      const distanceToDriver = Math.hypot(body.x - car.x, body.y - car.y), brain = team.helpers[i]?.network, filter = brain ? forward(brain, observe(course, car, body.sensor))[0] : 0;
      if (filter > -.65 || found.confidence > .55) runtime.messages.unshift({ from: body.id, role: body.role, bearing: Math.atan2(found.obstacle.y - car.y, found.obstacle.x - car.x), distance: Math.hypot(found.obstacle.x - car.x, found.obstacle.y - car.y) - found.obstacle.r, size: found.obstacle.r, confidence: clamp(found.confidence * (.8 + Math.max(0, filter) * .2), 0, 1), age: 0 });
      runtime.messages = runtime.messages.slice(0, 4); body.packetCooldown = .28 + distanceToDriver / 900; addEnergy(car, "comms", .07 + distanceToDriver * .00014); addEnergy(car, "compute", parameterCount(team.helpers[i].network) * .00000035);
    }
  }
  const packet = runtime.messages[0], input = observe(course, car, team.driverSensor, packet); addEnergy(car, "sensing", dt * SENSOR_SPECS[team.driverSensor].cost); addEnergy(car, "compute", parameterCount(team.driver) * dt * .0000014); const control = teamControl(team, input); const result = stepCar(course, car, control, dt);
  return { input, control, result, packet };
}

export function train(net: Network, samples: Sample[], epochs = 18, rate = .012) {
  if (!samples.length) return 0; let loss = 0;
  for (let epoch = 0; epoch < epochs; epoch++) for (let k = 0; k < samples.length; k++) {
    const sample = samples[(k * 37 + epoch * 13) % samples.length], acts: number[][] = [[...sample.input]], deltas: number[][] = []; let v = [...sample.input];
    for (let l = 0; l < net.weights.length; l++) { const raw = net.weights[l].map(row => row.slice(0, -1).reduce((s, w, i) => s + w * (v[i] ?? 0), row.at(-1)!)); v = raw.map(x => activate(x, l === net.weights.length - 1 ? "tanh" : net.activation)); acts.push(v); }
    const out = acts.at(-1)!; let delta = out.map((y, i) => (y - sample.output[i]) * (1 - y * y)); loss += out.reduce((s, y, i) => s + (y - sample.output[i]) ** 2, 0) / out.length; deltas.unshift(delta);
    for (let l = net.weights.length - 2; l >= 0; l--) { const next = net.weights[l + 1], a = acts[l + 1]; delta = a.map((y, i) => next.reduce((s, row, o) => s + row[i] * delta[o], 0) * derivative(y, net.activation)); deltas.unshift(delta); }
    for (let l = 0; l < net.weights.length; l++) { const inp = acts[l], d = deltas[l]; for (let o = 0; o < net.weights[l].length; o++) { for (let i = 0; i < inp.length; i++) net.weights[l][o][i] -= rate * d[o] * inp[i]; net.weights[l][o][inp.length] -= rate * d[o]; } }
  }
  return loss / (samples.length * epochs);
}
export function coachSamples(course: Course, count = 700): Sample[] { const samples: Sample[] = []; for (let lap = 0; lap < 8 && samples.length < count; lap++) { const car = makeCar(course); for (let i = 0; i < 1800 && !car.complete && samples.length < count; i++) { const input = observe(course, car, "radar"), bearing = Math.atan2(input[0], input[1]), left = input[4], front = input[5], right = input[6]; let steer = clamp(bearing * 1.35, -1, 1); if (front < .55) steer += left > right ? -.8 : .8; steer = clamp(steer + rand(-.04, .04), -1, 1); const throttle = front < .28 ? .05 : Math.abs(bearing) > .8 ? .35 : .82; samples.push({ input, output: [steer, throttle] }); stepCar(course, car, { steer, throttle }, .032); } } return samples; }

const routeCache = new Map<string, number>();
function courseKey(c: Course) { return JSON.stringify([c.start.x, c.start.y, ...c.gates.flatMap(g => [g.x, g.y]), c.finish.x, c.finish.y, ...c.obstacles.flatMap(o => [o.x, o.y, o.r])].map(Math.round)); }
function pathSegment(course: Course, start: Point, goal: Point) {
  const cell = 20, cols = Math.floor(W / cell), rows = Math.floor(H / cell), index = (x: number, y: number) => y * cols + x, point = (i: number) => ({ x: (i % cols) * cell + cell / 2, y: Math.floor(i / cols) * cell + cell / 2 }), blocked = (p: Point) => p.x < 20 || p.x > W - 20 || p.y < 20 || p.y > H - 20 || course.obstacles.some(o => Math.hypot(p.x - o.x, p.y - o.y) < o.r + 19);
  const sx = clamp(Math.floor(start.x / cell), 0, cols - 1), sy = clamp(Math.floor(start.y / cell), 0, rows - 1), gx = clamp(Math.floor(goal.x / cell), 0, cols - 1), gy = clamp(Math.floor(goal.y / cell), 0, rows - 1), begin = index(sx, sy), end = index(gx, gy), open = new Set([begin]), g = new Map<number, number>([[begin, 0]]), f = new Map<number, number>([[begin, Math.hypot(gx - sx, gy - sy)]]);
  while (open.size) { let current = -1, best = Infinity; for (const id of open) if ((f.get(id) ?? Infinity) < best) { best = f.get(id)!; current = id; } if (current === end) return (g.get(current) ?? 0) * cell; open.delete(current); const cx = current % cols, cy = Math.floor(current / cols); for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]) { const nx = cx + dx, ny = cy + dy; if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue; const ni = index(nx, ny), p = point(ni); if (blocked(p) && ni !== end) continue; const candidate = (g.get(current) ?? 0) + Math.hypot(dx, dy); if (candidate < (g.get(ni) ?? Infinity)) { g.set(ni, candidate); f.set(ni, candidate + Math.hypot(gx - nx, gy - ny)); open.add(ni); } } }
  return Math.hypot(goal.x - start.x, goal.y - start.y) * 1.35;
}
export function optimalRouteLength(course: Course) { const key = courseKey(course); if (routeCache.has(key)) return routeCache.get(key)!; const points: Point[] = [course.start, ...course.gates, course.finish]; let total = 0; for (let i = 1; i < points.length; i++) total += pathSegment(course, points[i - 1], points[i]); routeCache.set(key, total); return total; }
export function score(car: Car, course: Course): Score { const completion = (car.gate + (car.complete ? 1 : 0)) / (course.gates.length + 1), optimal = optimalRouteLength(course), efficiency = car.complete ? clamp(optimal / Math.max(optimal, car.distance) * 100, 0, 100) : 0; return { completion, time: car.complete ? Math.max(0, 100 - car.elapsed * 1.5) : 0, collisions: Math.max(0, 100 - car.collisions * 12), energy: car.complete ? Math.max(0, 100 - car.energy * 2.2) : 0, smoothness: Math.max(0, 100 - car.jerk / Math.max(1, car.elapsed) * 5), efficiency, distance: car.distance, optimal }; }

function seeded(seed: number) { let s = seed | 0; return () => { s = (s + 0x6D2B79F5) | 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
export function varyCourse(base: Course, seed: number, variation = .18): Course { const r = seeded(seed), j = (amount: number) => (r() * 2 - 1) * amount * variation; return { name: `${base.name} · scenario ${seed}`, start: { x: clamp(base.start.x + j(180), 35, W - 35), y: clamp(base.start.y + j(180), 35, H - 35), angle: base.start.angle + j(1.1) }, finish: { ...base.finish, x: clamp(base.finish.x + j(170), 45, W - 45), y: clamp(base.finish.y + j(170), 45, H - 45) }, gates: base.gates.map(g => ({ ...g, x: clamp(g.x + j(210), 45, W - 45), y: clamp(g.y + j(210), 45, H - 45) })), obstacles: base.obstacles.map(o => ({ ...o, x: clamp(o.x + j(170), o.r + 22, W - o.r - 22), y: clamp(o.y + j(170), o.r + 22, H - o.r - 22), r: clamp(o.r + j(30), 28, 70) })) }; }
export function evaluateTeam(team: Team, course: Course) { const car = makeCar(course), runtime = createTeamRuntime(team, course, car); let reward = 0; for (let i = 0; i < 1250 && !car.complete; i++) { const { control, result } = stepTeam(course, car, team, runtime, .032); reward += result.finished ? 400 : result.gate ? 70 : result.hit ? -40 : result.progress * .12 - .004 - .002 * Math.abs(control.steer); } const s = score(car, course); reward += car.complete ? s.efficiency * 1.2 + s.energy * .45 : 0; return { car, score: s, reward, runtime }; }
function averageScores(scores: Score[]): Score { const avg = (key: keyof Score) => scores.reduce((s, v) => s + Number(v[key]), 0) / scores.length; return { completion: avg("completion"), time: avg("time"), collisions: avg("collisions"), energy: avg("energy"), smoothness: avg("smoothness"), efficiency: avg("efficiency"), distance: avg("distance"), optimal: avg("optimal") }; }
const SENSOR_ORDER: SensorKind[] = ["none", "proximity", "radar", "lidar"];
function mutateTeam(parent: Team, generation: number, index: number, budget: number): Team {
  const amount = .035 + Math.min(.11, generation * .004), team: Team = { id: `g${generation}_${index}_${Math.random().toString(36).slice(2, 5)}`, generation, parentId: parent.id, driver: cloneNetwork(parent.driver, index === 0 ? 0 : amount), driverSensor: parent.driverSensor ?? "radar", helpers: parent.helpers.map(h => ({ ...h, id: `${h.role}_${generation}_${index}`, sensor: h.sensor ?? "radar", network: cloneNetwork(h.network, index === 0 ? 0 : amount * 1.25) })) };
  if (index > 0 && Math.random() < .28) team.driverSensor = SENSOR_ORDER[Math.floor(Math.random() * SENSOR_ORDER.length)];
  if (index > 0 && team.helpers.length < 2 && Math.random() < .38) { const role: HelperRole = team.helpers.some(h => h.role === "scout") ? "finisher" : "scout"; team.helpers.push({ id: `${role}_${generation}_${index}`, role, sensor: Math.random() < .55 ? "radar" : "proximity", network: cloneNetwork(parent.driver, .12) }); }
  if (index > 0 && team.helpers.length && Math.random() < .16) team.helpers.splice(Math.floor(Math.random() * team.helpers.length), 1);
  if (index > 0 && team.helpers.length && Math.random() < .25) { const h = team.helpers[Math.floor(Math.random() * team.helpers.length)]; h.sensor = SENSOR_ORDER[1 + Math.floor(Math.random() * 3)]; }
  while (team.helpers.length && teamParameters(team) > budget) team.helpers.pop(); return team;
}
export function evolveGeneration(parent: Team, base: Course, generation: number, population = 10, scenarios = 4, variation = .18, budget = 250000): AutoResult {
  const candidates = Array.from({ length: population }, (_, i) => mutateTeam(parent, generation, i, budget)); const results = candidates.map(team => { const scores: Score[] = []; let raw = 0; for (let s = 0; s < scenarios; s++) { const result = evaluateTeam(team, varyCourse(base, 815 + s * 97, variation)); scores.push(result.score); raw += result.reward; } const avg = averageScores(scores), parameters = teamParameters(team), finished = avg.completion >= .999 ? 1 : 0, fitness = avg.completion * 100000 + finished * 20000 + avg.collisions * 75 + avg.efficiency * 28 + avg.time * 4 + avg.energy * 3 + avg.smoothness + raw / scenarios * .08 - parameters / budget * 8; return { team, fitness, score: avg, parameters }; }); return results.sort((a, b) => b.fitness - a.fitness)[0];
}
