// ForceAtlas2 (graphology's implementation: LinLog mode, Barnes-Hut) for the
// force maps -- `concordance force-map` runs it with Node. Reads
// {nodes: [{x, y}], edges: [[i, j, weight]], iterations, settings} as JSON on
// stdin, writes {x: [...], y: [...]} to stdout.
import Graph from 'graphology'
import forceAtlas2 from 'graphology-layout-forceatlas2'

let raw = ''
for await (const chunk of process.stdin) raw += chunk
const { nodes, edges, iterations = 500, settings = {} } = JSON.parse(raw)

const graph = new Graph({ type: 'undirected', multi: false })
nodes.forEach((n, i) => graph.addNode(i, { x: n.x, y: n.y }))
for (const [a, b, w] of edges) {
  if (a !== b && !graph.hasEdge(a, b)) graph.addEdge(a, b, { weight: w })
}

const started = Date.now()
forceAtlas2.assign(graph, {
  iterations,
  getEdgeWeight: 'weight',
  settings: { ...forceAtlas2.inferSettings(graph), ...settings },
})
process.stderr.write(`force_layout: ${nodes.length} nodes, ${graph.size} edges, ${iterations} iterations in ${Date.now() - started} ms\n`)

const x = new Array(nodes.length)
const y = new Array(nodes.length)
graph.forEachNode((key, attr) => { x[+key] = attr.x; y[+key] = attr.y })
process.stdout.write(JSON.stringify({ x, y }))
