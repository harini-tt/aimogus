// ============================================================
// Among Us Replay Viewer - Main Application
// ============================================================

// --- Color mapping for player colors ---
const COLOR_MAP = {
    red: '#e94560',
    blue: '#3498db',
    green: '#27ae60',
    pink: '#e91e9e',
    orange: '#e67e22',
    yellow: '#f1c40f',
    black: '#2c3e50',
    white: '#ecf0f1',
    purple: '#8e44ad',
    brown: '#8B4513',
    cyan: '#1abc9c',
    lime: '#2ecc71',
};

const ROOM_LABELS = {
    'Upper Engine': 'Upper Eng.',
    'Lower Engine': 'Lower Eng.',
    'Communications': 'Comms',
    'Navigation': 'Nav',
};

const SPRITE_SHEET_GRID = 3;
const SPRITE_FRAME_COL = 2;  // top-right frame in a 3x3 sheet
const SPRITE_FRAME_ROW = 0;
const PLAYER_SPRITE_SIZE = 52;
const PLAYER_SPRITE_GAP = -18;

// Align room overlays/players to the underlying skeld.png.
// Previously this was applied as a canvas transform (translate+scale).
// We'll keep the same transform but apply it to geometry instead.
const MAP_ALIGN = {
    tx: 75,
    ty: 8,
    s: 0.92,
};

// ============================================================
// Game State
// ============================================================
const GameState = {
    gameData: null,
    mapConfig: null,
    // Canvas-space version of mapCoords (after MAP_ALIGN transform)
    mapConfigCanvas: null,
    taskConfig: null,
    mapImage: null,
    playerSprites: {},

    // Precomputed per-turn state
    playerPositions: [],  // [{playerName: roomName, ...}, ...]
    playerAlive: [],      // [{playerName: bool, ...}, ...]
    taskProgress: [],     // [float, ...]

    // Walkway graph (separate from gameplay graph; UI-only)
    walkwayGraph: null,         // {meta, nodes, edges, roomAnchors}
    walkwayAdj: null,           // nodeId -> [{to, w}]
    walkwayPathCache: new Map(),// "fromRoom->toRoom" -> cached polyline info

    // Playback
    currentTurnIndex: 0,
    isPlaying: false,
    playbackSpeed: 1,
    playbackTimer: null,
    playbackRafId: null,
    playbackSegmentIndex: 0,
    playbackSegmentStartTs: 0,
    playbackAlpha: 0,
    _lastUiTurnIndex: null,

    get totalStates() {
        return this.playerPositions.length;
    },

    // Returns the game turn for the current index, or null for the initial state (index 0)
    get currentTurn() {
        if (!this.gameData || this.currentTurnIndex <= 0) return null;
        const turnIdx = this.currentTurnIndex - 1;
        if (turnIdx >= this.gameData.turns.length) return null;
        return this.gameData.turns[turnIdx];
    },
};

// ============================================================
// Walkway Editor State (UI-only)
// ============================================================
const WalkwayEditor = {
    uiReady: false,
    enabled: false,
    tool: 'addNode',          // addNode | addEdge | selectMove | setAnchor
    selectedNode: null,
    edgeStartNode: null,
    draggingNode: null,
    dragOffset: { x: 0, y: 0 },
    cursor: { x: 0, y: 0, valid: false },
    showGraph: true,
    showAnchors: true,
    showMagnifier: false,
    previewPath: false,
    nextNodeId: 1,
};

// ============================================================
// Initialization
// ============================================================
async function init() {
    setupControls();
    setupKeyboardShortcuts();
    setupWalkwayEditorUI();
    drawEmptyCanvas();
    // Auto-load the most recent game
    const res = await fetch('/api/games');
    const games = await res.json();
    if (games.length > 0) {
        loadGame(games[0]);
    }
}

async function loadGame(filename) {
    showLoading(true);
    pause();

    const [gameRes, mapRes, taskRes, walkwayGraph] = await Promise.all([
        fetch(`/api/game/${encodeURIComponent(filename)}`),
        fetch('/api/map-config'),
        fetch('/api/task-config'),
        loadWalkwayGraph(),
    ]);

    GameState.gameData = await gameRes.json();
    GameState.mapConfig = await mapRes.json();
    GameState.taskConfig = await taskRes.json();
    GameState.walkwayGraph = walkwayGraph;
    GameState.walkwayAdj = buildWalkwayAdjacency(walkwayGraph);
    GameState.walkwayPathCache = new Map();
    GameState.mapConfigCanvas = buildCanvasMapConfig(GameState.mapConfig);
    hydrateWalkwayEditorFromGraph();
    populateWalkwayRoomDropdown();
    showWalkwayEditorPanel(true);

    // Load map image
    GameState.mapImage = await loadImage('/assets/skeld.png');
    GameState.playerSprites = {};

    // Load color sprite sheets used by this game.
    const usedColors = [...new Set((GameState.gameData.players || []).map(p => p.color))];
    await Promise.all(usedColors.map(async color => {
        try {
            GameState.playerSprites[color] = await loadImage(`/assets/sprites/${color}.png`);
        } catch (err) {
            console.warn(`Unable to load sprite for color "${color}"`, err);
        }
    }));

    // Precompute states
    precomputeStates();

    // Reset playback
    GameState.currentTurnIndex = 0;

    // Update UI
    updateTimeline();
    updatePlayerRoster();
    showGameResult();
    renderCurrentState();
    showLoading(false);
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

function setupWalkwayEditorUI() {
    if (WalkwayEditor.uiReady) return;
    WalkwayEditor.uiReady = true;

    const panel = document.getElementById('walkway-editor');
    const btnToggle = document.getElementById('btn-edit-walkways');
    const toolSelect = document.getElementById('walkway-editor-tool');
    const roomSelect = document.getElementById('walkway-editor-room');
    const btnDownload = document.getElementById('btn-walkway-download');
    const fileInput = document.getElementById('walkway-editor-file');
    const toggleGraph = document.getElementById('walkway-toggle-graph');
    const toggleAnchors = document.getElementById('walkway-toggle-anchors');
    const togglePreview = document.getElementById('walkway-toggle-preview');
    const toggleMag = document.getElementById('walkway-toggle-magnifier');
    const magnifier = document.getElementById('walkway-magnifier');
    const previewRow = document.getElementById('walkway-preview-row');
    const roomToSelect = document.getElementById('walkway-editor-room-to');

    // Panel starts hidden; we show it after a game loads (or via hotkey).
    panel.classList.add('hidden');

    btnToggle.addEventListener('click', () => {
        WalkwayEditor.enabled = !WalkwayEditor.enabled;
        if (!panel.classList.contains('hidden')) {
            // keep visible
        } else {
            panel.classList.remove('hidden');
        }
        updateWalkwayEditorHint();
        renderCanvasFrame();
    });

    toolSelect.addEventListener('change', () => {
        WalkwayEditor.tool = toolSelect.value;
        WalkwayEditor.selectedNode = null;
        WalkwayEditor.edgeStartNode = null;
        WalkwayEditor.draggingNode = null;
        updateWalkwayEditorHint();
        renderCanvasFrame();
    });

    roomSelect.addEventListener('change', () => {
        updateWalkwayEditorHint();
    });

    btnDownload.addEventListener('click', () => {
        downloadWalkwayGraph();
    });

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            const empty = {
                meta: { version: 'canvas_v1', canvasWidth: 980, canvasHeight: 561 },
                nodes: {},
                edges: [],
                roomAnchors: {},
            };
            GameState.walkwayGraph = normalizeWalkwayGraph(data, empty);
            onWalkwayGraphChanged();
            hydrateWalkwayEditorFromGraph();
            populateWalkwayRoomDropdown();
            renderCurrentState();
        } catch (err) {
            console.warn('Failed to load walkway JSON.', err);
        } finally {
            // allow re-uploading the same file after edits
            fileInput.value = '';
        }
    });

    toggleGraph.addEventListener('change', () => {
        WalkwayEditor.showGraph = !!toggleGraph.checked;
        renderCanvasFrame();
    });
    toggleAnchors.addEventListener('change', () => {
        WalkwayEditor.showAnchors = !!toggleAnchors.checked;
        renderCanvasFrame();
    });
    togglePreview.addEventListener('change', () => {
        WalkwayEditor.previewPath = !!togglePreview.checked;
        previewRow.classList.toggle('hidden', !WalkwayEditor.previewPath);
        renderCanvasFrame();
    });
    toggleMag.addEventListener('change', () => {
        WalkwayEditor.showMagnifier = !!toggleMag.checked;
        magnifier.classList.toggle('hidden', !WalkwayEditor.showMagnifier);
        renderCanvasFrame();
    });

    roomToSelect.addEventListener('change', () => {
        renderCanvasFrame();
    });

    // Canvas interactions
    const canvas = document.getElementById('game-map');
    canvas.addEventListener('pointermove', (e) => onWalkwayPointerMove(e, canvas));
    canvas.addEventListener('pointerdown', (e) => onWalkwayPointerDown(e, canvas));
    canvas.addEventListener('pointerup', (e) => onWalkwayPointerUp(e, canvas));

    // Delete selected node
    document.addEventListener('keydown', (e) => {
        if (!WalkwayEditor.enabled) return;
        if (e.code !== 'Backspace' && e.code !== 'Delete') return;
        if (!WalkwayEditor.selectedNode) return;
        e.preventDefault();
        removeWalkwayNode(WalkwayEditor.selectedNode);
        WalkwayEditor.selectedNode = null;
        WalkwayEditor.edgeStartNode = null;
        onWalkwayGraphChanged();
        renderCanvasFrame();
    });

    updateWalkwayEditorHint();
}

function showWalkwayEditorPanel(show) {
    const panel = document.getElementById('walkway-editor');
    if (!panel) return;
    panel.classList.toggle('hidden', !show);
}

function populateWalkwayRoomDropdown() {
    const el = document.getElementById('walkway-editor-room');
    const elTo = document.getElementById('walkway-editor-room-to');
    if (!el) return;
    el.innerHTML = '';
    if (elTo) elTo.innerHTML = '';

    const rooms = GameState.mapConfig ? Object.keys(GameState.mapConfig.mapCoords || {}) : [];
    rooms.sort((a, b) => a.localeCompare(b));
    for (const room of rooms) {
        const opt = document.createElement('option');
        opt.value = room;
        opt.textContent = ROOM_LABELS[room] || room;
        el.appendChild(opt);
        if (elTo) elTo.appendChild(opt.cloneNode(true));
    }
    if (rooms.includes('Cafeteria')) el.value = 'Cafeteria';
    if (elTo) elTo.value = rooms.includes('Admin') ? 'Admin' : (rooms[0] || '');
}

function hydrateWalkwayEditorFromGraph() {
    // Pick a nextNodeId larger than any existing "n<number>" id.
    const g = GameState.walkwayGraph;
    let maxN = 0;
    if (g && g.nodes) {
        for (const id of Object.keys(g.nodes)) {
            const m = /^n(\d+)$/.exec(id);
            if (m) maxN = Math.max(maxN, parseInt(m[1], 10));
        }
    }
    WalkwayEditor.nextNodeId = maxN + 1;
}

function updateWalkwayEditorHint() {
    const hint = document.getElementById('walkway-editor-hint');
    const tool = WalkwayEditor.tool;
    const roomSel = document.getElementById('walkway-editor-room');
    const room = roomSel ? roomSel.value : '';

    if (!hint) return;
    if (!WalkwayEditor.enabled) {
        hint.textContent = 'Click “Edit walkways” to enable editing. (Hotkey: E)';
        return;
    }

    if (tool === 'addNode') hint.textContent = 'Add node: click on a hallway centerline.';
    else if (tool === 'addEdge') hint.textContent = WalkwayEditor.edgeStartNode
        ? 'Add edge: click a second node to connect. (Esc cancels)'
        : 'Add edge: click a node, then click another node to connect.';
    else if (tool === 'selectMove') hint.textContent = 'Select/drag: click a node to select; drag to move. Delete removes selected.';
    else if (tool === 'setAnchor') hint.textContent = `Set anchor: select room (${room || '—'}), then click a node to anchor it.`;
    else hint.textContent = '';
}

function eventToCanvasXY(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    return { x, y };
}

function onWalkwayPointerMove(e, canvas) {
    const { x, y } = eventToCanvasXY(e, canvas);
    WalkwayEditor.cursor = { x, y, valid: true };
    const coordsEl = document.getElementById('walkway-editor-coords');
    if (coordsEl) coordsEl.textContent = `(${x.toFixed(1)}, ${y.toFixed(1)})`;

    if (WalkwayEditor.enabled && WalkwayEditor.draggingNode) {
        const g = GameState.walkwayGraph;
        if (g && g.nodes && g.nodes[WalkwayEditor.draggingNode]) {
            g.nodes[WalkwayEditor.draggingNode].x = x + WalkwayEditor.dragOffset.x;
            g.nodes[WalkwayEditor.draggingNode].y = y + WalkwayEditor.dragOffset.y;
            onWalkwayGraphChanged();
        }
    }
    renderCanvasFrame();
}

function onWalkwayPointerDown(e, canvas) {
    if (!WalkwayEditor.enabled) return;
    if (WalkwayEditor.tool !== 'selectMove') return;
    const { x, y } = eventToCanvasXY(e, canvas);
    const nodeId = findNearestWalkwayNode(x, y, 14);
    if (!nodeId) return;
    WalkwayEditor.selectedNode = nodeId;
    const pt = GameState.walkwayGraph?.nodes?.[nodeId];
    if (!pt) return;
    WalkwayEditor.draggingNode = nodeId;
    WalkwayEditor.dragOffset = { x: pt.x - x, y: pt.y - y };
    renderCanvasFrame();
}

function onWalkwayPointerUp(e, canvas) {
    if (!WalkwayEditor.enabled) return;
    const wasDragging = !!WalkwayEditor.draggingNode;
    WalkwayEditor.draggingNode = null;
    WalkwayEditor.dragOffset = { x: 0, y: 0 };
    if (wasDragging) {
        renderCanvasFrame();
        return;
    }

    const { x, y } = eventToCanvasXY(e, canvas);
    const tool = WalkwayEditor.tool;

    if (tool === 'addNode') {
        addWalkwayNode(x, y);
        onWalkwayGraphChanged();
        renderCanvasFrame();
        return;
    }

    const hit = findNearestWalkwayNode(x, y, 14);
    if (!hit) return;

    if (tool === 'selectMove') {
        WalkwayEditor.selectedNode = hit;
        renderCanvasFrame();
        return;
    }

    if (tool === 'addEdge') {
        if (!WalkwayEditor.edgeStartNode) {
            WalkwayEditor.edgeStartNode = hit;
        } else if (WalkwayEditor.edgeStartNode !== hit) {
            addWalkwayEdge(WalkwayEditor.edgeStartNode, hit);
            WalkwayEditor.edgeStartNode = null;
            onWalkwayGraphChanged();
        }
        renderCanvasFrame();
        return;
    }

    if (tool === 'setAnchor') {
        const roomSel = document.getElementById('walkway-editor-room');
        const room = roomSel ? roomSel.value : '';
        if (!room) return;
        if (!GameState.walkwayGraph) return;
        GameState.walkwayGraph.roomAnchors = GameState.walkwayGraph.roomAnchors || {};
        GameState.walkwayGraph.roomAnchors[room] = hit;
        WalkwayEditor.selectedNode = hit;
        onWalkwayGraphChanged();
        renderCanvasFrame();
        return;
    }
}

function findNearestWalkwayNode(x, y, radiusPx) {
    const g = GameState.walkwayGraph;
    if (!g || !g.nodes) return null;
    const r2 = radiusPx * radiusPx;
    let best = null;
    let bestD2 = Infinity;
    for (const [id, pt] of Object.entries(g.nodes)) {
        const dx = pt.x - x;
        const dy = pt.y - y;
        const d2 = dx * dx + dy * dy;
        if (d2 <= r2 && d2 < bestD2) {
            best = id;
            bestD2 = d2;
        }
    }
    return best;
}

function addWalkwayNode(x, y) {
    if (!GameState.walkwayGraph) return;
    const id = `n${WalkwayEditor.nextNodeId++}`;
    GameState.walkwayGraph.nodes[id] = { x, y };
    WalkwayEditor.selectedNode = id;
}

function addWalkwayEdge(a, b) {
    const g = GameState.walkwayGraph;
    if (!g) return;
    if (!g.nodes[a] || !g.nodes[b]) return;
    // Prevent duplicates
    const exists = (g.edges || []).some(e =>
        (e.from === a && e.to === b) || (e.from === b && e.to === a)
    );
    if (exists) return;
    g.edges.push({ from: a, to: b });
}

function removeWalkwayNode(nodeId) {
    const g = GameState.walkwayGraph;
    if (!g || !g.nodes) return;
    delete g.nodes[nodeId];
    g.edges = (g.edges || []).filter(e => e.from !== nodeId && e.to !== nodeId);
    if (g.roomAnchors) {
        for (const [room, anchorId] of Object.entries(g.roomAnchors)) {
            if (anchorId === nodeId) delete g.roomAnchors[room];
        }
    }
}

function onWalkwayGraphChanged() {
    GameState.walkwayAdj = buildWalkwayAdjacency(GameState.walkwayGraph);
    GameState.walkwayPathCache = new Map();
}

function downloadWalkwayGraph() {
    const g = GameState.walkwayGraph;
    if (!g) return;
    const payload = JSON.stringify(g, null, 2);
    const blob = new Blob([payload], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'skeld_walkways_canvas_v1.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

async function loadWalkwayGraph() {
    // Graph file is expected to exist in assets/, but we fail soft if it doesn't.
    const empty = {
        meta: { version: 'canvas_v1', canvasWidth: 980, canvasHeight: 561 },
        nodes: {},
        edges: [],
        roomAnchors: {},
    };
    try {
        const res = await fetch('/assets/skeld_walkways_canvas_v1.json', { cache: 'no-store' });
        if (!res.ok) return empty;
        const data = await res.json();
        return normalizeWalkwayGraph(data, empty);
    } catch (err) {
        console.warn('Failed to load walkway graph, using empty graph.', err);
        return empty;
    }
}

function normalizeWalkwayGraph(data, fallback) {
    if (!data || typeof data !== 'object') return fallback;
    const nodes = (data.nodes && typeof data.nodes === 'object') ? data.nodes : {};
    const edges = Array.isArray(data.edges) ? data.edges : [];
    const roomAnchors = (data.roomAnchors && typeof data.roomAnchors === 'object') ? data.roomAnchors : {};
    const meta = (data.meta && typeof data.meta === 'object') ? data.meta : fallback.meta;

    // Normalize node values to numbers
    for (const [id, pt] of Object.entries(nodes)) {
        if (!pt || typeof pt !== 'object') continue;
        pt.x = Number(pt.x);
        pt.y = Number(pt.y);
    }

    // Normalize edges
    const normEdges = [];
    for (const e of edges) {
        if (!e || typeof e !== 'object') continue;
        const from = String(e.from ?? '');
        const to = String(e.to ?? '');
        if (!from || !to || from === to) continue;
        if (!nodes[from] || !nodes[to]) continue;
        normEdges.push({ from, to });
    }

    return { meta, nodes, edges: normEdges, roomAnchors };
}

function buildWalkwayAdjacency(graph) {
    const adj = {};
    if (!graph || !graph.nodes || !graph.edges) return adj;
    for (const id of Object.keys(graph.nodes)) adj[id] = [];
    for (const e of graph.edges) {
        const a = e.from;
        const b = e.to;
        const pa = graph.nodes[a];
        const pb = graph.nodes[b];
        if (!pa || !pb) continue;
        const dx = pb.x - pa.x;
        const dy = pb.y - pa.y;
        const w = Math.hypot(dx, dy) || 1;
        adj[a].push({ to: b, w });
        adj[b].push({ to: a, w });
    }
    return adj;
}

function buildCanvasMapConfig(mapConfig) {
    if (!mapConfig || !mapConfig.mapCoords) return null;
    const out = { ...mapConfig, mapCoords: {} };
    for (const [room, data] of Object.entries(mapConfig.mapCoords)) {
        const coords = data.coords || [];
        const canvasCoords = [];
        for (let i = 0; i < coords.length; i += 2) {
            const x = coords[i];
            const y = coords[i + 1];
            canvasCoords.push(MAP_ALIGN.tx + MAP_ALIGN.s * x);
            canvasCoords.push(MAP_ALIGN.ty + MAP_ALIGN.s * y);
        }
        out.mapCoords[room] = { coords: canvasCoords };
    }
    return out;
}

function buildMovementTypeMap(turn) {
    // Map playerName -> 'move' | 'vent'
    const out = {};
    if (!turn || !Array.isArray(turn.events)) return out;
    for (const ev of turn.events) {
        if (!ev || !ev.player) continue;
        if (ev.type === 'move') out[ev.player] = 'move';
        if (ev.type === 'vent') out[ev.player] = 'vent';
    }
    return out;
}

function getRoomCenter(room, mapConfig) {
    const roomData = mapConfig?.mapCoords?.[room];
    if (!roomData) return { x: 0, y: 0 };
    const coords = roomData.coords || [];
    const xs = coords.filter((_, i) => i % 2 === 0);
    const ys = coords.filter((_, i) => i % 2 !== 0);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    return { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
}

function getInterpolatedWalkPosition(fromRoom, toRoom, alpha, mapConfig) {
    // If we don't have a usable walkway graph yet, fall back to linear interpolation between room centers.
    const g = GameState.walkwayGraph;
    const adj = GameState.walkwayAdj;
    if (!g || !g.nodes || !g.edges || Object.keys(g.nodes).length === 0) {
        const a = getRoomCenter(fromRoom, mapConfig);
        const b = getRoomCenter(toRoom, mapConfig);
        return { x: lerp(a.x, b.x, alpha), y: lerp(a.y, b.y, alpha) };
    }

    const fromAnchorId = g.roomAnchors?.[fromRoom];
    const toAnchorId = g.roomAnchors?.[toRoom];
    if (!fromAnchorId || !toAnchorId || !g.nodes[fromAnchorId] || !g.nodes[toAnchorId]) {
        const a = getRoomCenter(fromRoom, mapConfig);
        const b = getRoomCenter(toRoom, mapConfig);
        return { x: lerp(a.x, b.x, alpha), y: lerp(a.y, b.y, alpha) };
    }

    const path = getOrComputeWalkwayPath(fromRoom, toRoom, fromAnchorId, toAnchorId, g, adj);
    if (!path || !path.total || path.total <= 0) {
        const a = g.nodes[fromAnchorId];
        const b = g.nodes[toAnchorId];
        return { x: lerp(a.x, b.x, alpha), y: lerp(a.y, b.y, alpha) };
    }
    return pointAlongPolyline(path, alpha);
}

function getOrComputeWalkwayPath(fromRoom, toRoom, startId, endId, g, adj) {
    const key = `${fromRoom}->${toRoom}`;
    const cached = GameState.walkwayPathCache.get(key);
    if (cached) return cached;

    const nodePath = dijkstra(adj, startId, endId);
    if (!nodePath || nodePath.length === 0) return null;
    const points = nodePath.map(id => ({ x: g.nodes[id].x, y: g.nodes[id].y }));

    const segLens = [];
    let total = 0;
    for (let i = 0; i < points.length - 1; i++) {
        const dx = points[i + 1].x - points[i].x;
        const dy = points[i + 1].y - points[i].y;
        const len = Math.hypot(dx, dy);
        segLens.push(len);
        total += len;
    }
    const pathInfo = { points, segLens, total };
    GameState.walkwayPathCache.set(key, pathInfo);
    return pathInfo;
}

function pointAlongPolyline(path, alpha) {
    const pts = path.points;
    if (!pts || pts.length === 0) return { x: 0, y: 0 };
    if (pts.length === 1) return { x: pts[0].x, y: pts[0].y };
    const target = alpha * path.total;
    let acc = 0;
    for (let i = 0; i < path.segLens.length; i++) {
        const seg = path.segLens[i];
        if (acc + seg >= target) {
            const t = seg > 0 ? (target - acc) / seg : 0;
            return {
                x: lerp(pts[i].x, pts[i + 1].x, t),
                y: lerp(pts[i].y, pts[i + 1].y, t),
            };
        }
        acc += seg;
    }
    return { x: pts[pts.length - 1].x, y: pts[pts.length - 1].y };
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

class MinHeap {
    constructor() {
        this.arr = [];
    }
    push(item) {
        this.arr.push(item);
        this._bubbleUp(this.arr.length - 1);
    }
    pop() {
        if (this.arr.length === 0) return null;
        const top = this.arr[0];
        const last = this.arr.pop();
        if (this.arr.length > 0) {
            this.arr[0] = last;
            this._bubbleDown(0);
        }
        return top;
    }
    get size() {
        return this.arr.length;
    }
    _bubbleUp(i) {
        while (i > 0) {
            const p = Math.floor((i - 1) / 2);
            if (this.arr[p][0] <= this.arr[i][0]) break;
            [this.arr[p], this.arr[i]] = [this.arr[i], this.arr[p]];
            i = p;
        }
    }
    _bubbleDown(i) {
        const n = this.arr.length;
        while (true) {
            let smallest = i;
            const l = 2 * i + 1;
            const r = 2 * i + 2;
            if (l < n && this.arr[l][0] < this.arr[smallest][0]) smallest = l;
            if (r < n && this.arr[r][0] < this.arr[smallest][0]) smallest = r;
            if (smallest === i) break;
            [this.arr[i], this.arr[smallest]] = [this.arr[smallest], this.arr[i]];
            i = smallest;
        }
    }
}

function dijkstra(adj, start, goal) {
    if (start === goal) return [start];
    const dist = {};
    const prev = {};
    const visited = new Set();
    const heap = new MinHeap();
    dist[start] = 0;
    heap.push([0, start]);

    while (heap.size > 0) {
        const [d, u] = heap.pop();
        if (visited.has(u)) continue;
        visited.add(u);
        if (u === goal) break;
        const edges = adj[u] || [];
        for (const e of edges) {
            const v = e.to;
            const nd = d + e.w;
            if (dist[v] === undefined || nd < dist[v]) {
                dist[v] = nd;
                prev[v] = u;
                heap.push([nd, v]);
            }
        }
    }

    if (prev[goal] === undefined) return null;
    const path = [];
    let cur = goal;
    while (cur !== undefined) {
        path.push(cur);
        if (cur === start) break;
        cur = prev[cur];
    }
    path.reverse();
    return path[0] === start ? path : null;
}

// ============================================================
// Precompute per-turn states
// ============================================================
function precomputeStates() {
    const data = GameState.gameData;
    const tc = GameState.taskConfig;
    const positions = [];
    const alive = [];
    const progress = [];

    // All players start in Cafeteria and alive
    let currentPositions = {};
    let currentAlive = {};

    // Build per-player task tracker: {playerName: {taskName: remainingDuration}}
    // Only for crewmates (impostors do fake tasks)
    const taskDurations = {};  // playerName -> {taskName -> remaining}
    const taskCompleted = {};  // playerName -> {taskName -> bool}

    data.players.forEach(p => {
        currentPositions[p.name] = 'Cafeteria';
        currentAlive[p.name] = true;

        if (p.role === 'Crewmate') {
            taskDurations[p.name] = {};
            taskCompleted[p.name] = {};
            p.tasks.forEach(task => {
                const duration = (tc && tc[task.name]) ? tc[task.name].duration : 1;
                taskDurations[p.name][task.name] = duration;
                taskCompleted[p.name][task.name] = false;
            });
        }
    });

    // Helper: compute task progress matching game logic (alive players only)
    function computeTaskProgress(aliveStatus) {
        let totalAlive = 0;
        let completedAlive = 0;
        for (const [playerName, tasks] of Object.entries(taskCompleted)) {
            if (!aliveStatus[playerName]) continue;  // skip dead players
            for (const [taskName, done] of Object.entries(tasks)) {
                totalAlive++;
                if (done) completedAlive++;
            }
        }
        return totalAlive > 0 ? completedAlive / totalAlive : 0;
    }

    // Push initial state: everyone alive in Cafeteria, 0% progress
    positions.push({ ...currentPositions });
    alive.push({ ...currentAlive });
    progress.push(0);

    data.turns.forEach(turn => {
        // Capture the state at the start of this turn (before turn actions).
        const turnPositions = { ...currentPositions };
        const turnAlive = { ...currentAlive };

        if (turn.phase === 'meeting') {
            // Meetings start with everyone alive in Cafeteria.
            Object.keys(turnPositions).forEach(name => {
                if (turnAlive[name]) turnPositions[name] = 'Cafeteria';
            });
        }

        positions.push(turnPositions);
        alive.push({ ...turnAlive });
        progress.push(computeTaskProgress(turnAlive));

        // Apply this turn's events to produce the next turn's starting state.
        turn.events.forEach(event => {
            const player = event.player;
            if (event.type === 'move' || event.type === 'vent') {
                currentPositions[player] = event.to;
            } else if (event.type === 'kill') {
                currentAlive[event.victim] = false;
            } else if (event.type === 'complete_task') {
                // Decrement remaining duration; mark complete when <= 0.
                if (taskDurations[player] && taskDurations[player][event.task] !== undefined) {
                    taskDurations[player][event.task]--;
                    if (taskDurations[player][event.task] <= 0) {
                        taskCompleted[player][event.task] = true;
                    }
                }
            }
        });

        // Handle voteout at end of meeting.
        if (turn.voteResult && turn.voteResult.ejected) {
            currentAlive[turn.voteResult.ejected] = false;
        }

        // Meetings end with alive players in Cafeteria for the next turn.
        if (turn.phase === 'meeting') {
            Object.keys(currentPositions).forEach(name => {
                if (currentAlive[name]) currentPositions[name] = 'Cafeteria';
            });
        }
    });

    GameState.playerPositions = positions;
    GameState.playerAlive = alive;
    GameState.taskProgress = progress;
}

// ============================================================
// Canvas Rendering
// ============================================================
function drawEmptyCanvas() {
    const canvas = document.getElementById('game-map');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#555';
    ctx.font = '20px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Select a game log to start', canvas.width / 2, canvas.height / 2);
}

function renderCurrentState() {
    if (!GameState.gameData || !GameState.mapConfig) return;

    const turnIndex = GameState.currentTurnIndex;
    const turn = GameState.currentTurn;
    const taskProg = GameState.taskProgress[turnIndex];

    renderCanvasFrame();
    updateTaskProgress(taskProg);
    updateTurnDisplay(turn);
    updateActivityLog(turnIndex);
    updateMeetingPanel(turn);
    updatePlayerRosterStatus(GameState.playerAlive[turnIndex]);
    updateTimelinePosition();
}

function renderCanvasFrame() {
    if (!GameState.gameData || !GameState.mapConfig) return;
    const turnIndex = GameState.currentTurnIndex;
    const positions = GameState.playerPositions[turnIndex];
    const aliveStatus = GameState.playerAlive[turnIndex];

    const canAnimate = GameState.isPlaying && turnIndex < GameState.totalStates - 1;
    const alpha = canAnimate ? (GameState.playbackAlpha || 0) : 0;
    const nextPositions = canAnimate ? GameState.playerPositions[turnIndex + 1] : null;
    const transitionTurn = canAnimate && turnIndex > 0 ? GameState.gameData.turns[turnIndex - 1] : null;
    const nextTurn = canAnimate ? (GameState.gameData.turns[turnIndex] || null) : null;

    drawMap(positions, aliveStatus, canAnimate ? { alpha, nextPositions, transitionTurn, nextTurn } : null);
}

function drawMap(positions, aliveStatus, anim = null) {
    const canvas = document.getElementById('game-map');
    const ctx = canvas.getContext('2d');
    const mapConfig = GameState.mapConfigCanvas || GameState.mapConfig;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background image
    if (GameState.mapImage) {
        ctx.drawImage(GameState.mapImage, 0, 0, canvas.width, canvas.height);
    }

    const uiScale = MAP_ALIGN.s;

    // Draw translucent room overlays + labels
    for (const [room, data] of Object.entries(mapConfig.mapCoords)) {
        const coords = data.coords;
        const xCoords = coords.filter((_, i) => i % 2 === 0);
        const yCoords = coords.filter((_, i) => i % 2 !== 0);
        const midX = (Math.max(...xCoords) + Math.min(...xCoords)) / 2;
        const minY = Math.min(...yCoords);

        // Translucent white fill
        ctx.beginPath();
        ctx.moveTo(coords[0], coords[1]);
        for (let i = 2; i < coords.length; i += 2) {
            ctx.lineTo(coords[i], coords[i + 1]);
        }
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Room label badge
        const label = ROOM_LABELS[room] || room;
        ctx.font = `bold ${Math.max(10, Math.round(11 * uiScale))}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        const textW = ctx.measureText(label).width;
        const badgeX = midX - textW / 2 - 4;
        const badgeY = minY + 1;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
        ctx.beginPath();
        ctx.roundRect(badgeX, badgeY, textW + 8, 15, 4);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.fillText(label, midX, minY + 3);
    }

    // Draw player tokens
    const playerSize = PLAYER_SPRITE_SIZE * uiScale;
    const space = PLAYER_SPRITE_GAP * uiScale;
    const playersByName = new Map((GameState.gameData.players || []).map(player => [player.name, player]));

    let roomPlayers = {};
    let movingPlayers = [];

    if (anim && anim.nextPositions && anim.alpha > 0) {
        const alpha = Math.max(0, Math.min(1, anim.alpha));
        const nextPositions = anim.nextPositions;
        const moveType = buildMovementTypeMap(anim.transitionTurn);
        const isMeetingTeleport = !!(anim.nextTurn && anim.nextTurn.phase === 'meeting');

        for (const [playerName, fromRoom] of Object.entries(positions)) {
            const toRoom = nextPositions[playerName] ?? fromRoom;
            const type = (isMeetingTeleport && fromRoom !== toRoom)
                ? 'vent'
                : (moveType[playerName] ?? (fromRoom !== toRoom ? 'move' : 'stay'));

            if (fromRoom === toRoom) {
                roomPlayers[fromRoom] = roomPlayers[fromRoom] || [];
                roomPlayers[fromRoom].push(playerName);
                continue;
            }

            if (type === 'vent') {
                const roomNow = alpha < 0.5 ? fromRoom : toRoom;
                roomPlayers[roomNow] = roomPlayers[roomNow] || [];
                roomPlayers[roomNow].push(playerName);
                continue;
            }

            const pos = getInterpolatedWalkPosition(fromRoom, toRoom, alpha, mapConfig);
            movingPlayers.push({ playerName, x: pos.x, y: pos.y });
        }
    } else {
        // Discrete: group players by room
        for (const [playerName, room] of Object.entries(positions)) {
            if (!roomPlayers[room]) roomPlayers[room] = [];
            roomPlayers[room].push(playerName);
        }
    }

    for (const [room, players] of Object.entries(roomPlayers)) {
        const roomData = mapConfig.mapCoords[room];
        if (!roomData) continue;

        const coords = roomData.coords;
        const xCoords = coords.filter((_, i) => i % 2 === 0);
        const yCoords = coords.filter((_, i) => i % 2 !== 0);
        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);
        const roomW = maxX - minX;
        const roomH = maxY - minY;

        const sz = playerSize;
        const midX = (minX + maxX) / 2;
        const midY = (minY + maxY) / 2;

        // Compute layout: keep gap, pack tighter if needed
        let gapX = space;
        let gapY = space;
        const perRow = Math.max(1, Math.floor((roomW - gapX) / (sz + gapX)));
        const rowsNeeded = Math.ceil(players.length / perRow);
        if (rowsNeeded * (sz + gapY) > roomH) {
            gapY = Math.max(-sz * 0.6, (roomH - rowsNeeded * sz) / rowsNeeded);
        }

        // Compute total group size then center on room
        const lastRowCount = players.length - (rowsNeeded - 1) * perRow;
        const groupW = Math.min(players.length, perRow) * (sz + gapX) - gapX;
        const groupH = rowsNeeded * (sz + gapY) - gapY;
        const startX = midX - groupW / 2;
        const startY = midY - groupH / 2;

        let col = 0;
        let row = 0;

        players.forEach((playerName, idx) => {
            const player = playersByName.get(playerName);
            if (!player) return;

            // Center partial last row
            const isLastRow = row === rowsNeeded - 1;
            const countThisRow = isLastRow ? lastRowCount : perRow;
            const rowW = countThisRow * (sz + gapX) - gapX;
            const rowOffsetX = (groupW - rowW) / 2;

            const x = startX + rowOffsetX + col * (sz + gapX);
            const y = startY + row * (sz + gapY);

            const color = COLOR_MAP[player.color] || player.color;
            const isAlive = aliveStatus[playerName];
            const spriteSheet = GameState.playerSprites[player.color];

            if (spriteSheet) {
                const tileW = spriteSheet.width / SPRITE_SHEET_GRID;
                const tileH = spriteSheet.height / SPRITE_SHEET_GRID;
                const srcX = SPRITE_FRAME_COL * tileW;
                const srcY = SPRITE_FRAME_ROW * tileH;

                ctx.drawImage(
                    spriteSheet,
                    srcX, srcY, tileW, tileH,
                    x, y, sz, sz
                );
            } else {
                ctx.beginPath();
                ctx.arc(x + sz / 2, y + sz / 2, sz / 2, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = 'rgba(0,0,0,0.5)';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            if (!isAlive) {
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x + sz, y + sz);
                ctx.strokeStyle = '#c44';
                ctx.lineWidth = 2.5 * uiScale;
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(x + sz, y);
                ctx.lineTo(x, y + sz);
                ctx.stroke();
            }

            col++;
            if (col >= perRow) {
                col = 0;
                row++;
            }
        });
    }

    // Draw moving players (walking) centered at interpolated points
    if (movingPlayers.length > 0) {
        const sz = playerSize;
        for (const mp of movingPlayers) {
            const player = playersByName.get(mp.playerName);
            if (!player) continue;
            const spriteSheet = GameState.playerSprites[player.color];
            const isAlive = aliveStatus[mp.playerName];

            const x = mp.x - sz / 2;
            const y = mp.y - sz / 2;

            if (spriteSheet) {
                const tileW = spriteSheet.width / SPRITE_SHEET_GRID;
                const tileH = spriteSheet.height / SPRITE_SHEET_GRID;
                const srcX = SPRITE_FRAME_COL * tileW;
                const srcY = SPRITE_FRAME_ROW * tileH;

                ctx.drawImage(
                    spriteSheet,
                    srcX, srcY, tileW, tileH,
                    x, y, sz, sz
                );
            } else {
                const color = COLOR_MAP[player.color] || player.color;
                ctx.beginPath();
                ctx.arc(mp.x, mp.y, sz / 2, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = 'rgba(0,0,0,0.5)';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            if (!isAlive) {
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x + sz, y + sz);
                ctx.strokeStyle = '#c44';
                ctx.lineWidth = 2.5 * uiScale;
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(x + sz, y);
                ctx.lineTo(x, y + sz);
                ctx.stroke();
            }
        }
    }

    // Walkway overlay + magnifier are always drawn in raw canvas space.
    drawWalkwayOverlay(ctx);
    renderWalkwayMagnifier();
}

function drawWalkwayOverlay(ctx) {
    const g = GameState.walkwayGraph;
    if (!g) return;
    if (!WalkwayEditor.enabled && !WalkwayEditor.showGraph && !WalkwayEditor.showAnchors) return;

    const showGraph = WalkwayEditor.enabled || WalkwayEditor.showGraph;
    const showAnchors = WalkwayEditor.enabled || WalkwayEditor.showAnchors;

    ctx.save();

    if (showGraph) {
        // Optional preview path between two room anchors
        if (WalkwayEditor.previewPath) {
            const fromRoomSel = document.getElementById('walkway-editor-room');
            const toRoomSel = document.getElementById('walkway-editor-room-to');
            const fromRoom = fromRoomSel ? fromRoomSel.value : '';
            const toRoom = toRoomSel ? toRoomSel.value : '';
            const fromId = g.roomAnchors?.[fromRoom];
            const toId = g.roomAnchors?.[toRoom];
            if (fromRoom && toRoom && fromRoom !== toRoom && fromId && toId && g.nodes[fromId] && g.nodes[toId]) {
                const path = getOrComputeWalkwayPath(fromRoom, toRoom, fromId, toId, g, GameState.walkwayAdj || {});
                if (path && path.points && path.points.length > 1) {
                    ctx.beginPath();
                    ctx.moveTo(path.points[0].x, path.points[0].y);
                    for (let i = 1; i < path.points.length; i++) {
                        ctx.lineTo(path.points[i].x, path.points[i].y);
                    }
                    ctx.strokeStyle = 'rgba(200, 120, 255, 0.8)';
                    ctx.lineWidth = 4;
                    ctx.stroke();
                }
            }
        }

        // Edges
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(80, 200, 255, 0.45)';
        for (const e of g.edges || []) {
            const a = g.nodes[e.from];
            const b = g.nodes[e.to];
            if (!a || !b) continue;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
        }

        // Nodes
        for (const [id, pt] of Object.entries(g.nodes || {})) {
            const isSel = id === WalkwayEditor.selectedNode;
            const isEdgeStart = id === WalkwayEditor.edgeStartNode;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, isSel ? 5 : 3.5, 0, Math.PI * 2);
            ctx.fillStyle = isSel ? 'rgba(255, 220, 80, 0.95)'
                : isEdgeStart ? 'rgba(140, 255, 170, 0.9)'
                : 'rgba(255, 255, 255, 0.75)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(0,0,0,0.45)';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    if (showAnchors && g.roomAnchors) {
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        for (const [room, nodeId] of Object.entries(g.roomAnchors)) {
            const pt = g.nodes[nodeId];
            if (!pt) continue;
            const label = ROOM_LABELS[room] || room;
            ctx.fillStyle = 'rgba(0,0,0,0.65)';
            ctx.beginPath();
            ctx.roundRect(pt.x + 6, pt.y - 9, ctx.measureText(label).width + 10, 18, 4);
            ctx.fill();
            ctx.fillStyle = 'rgba(255,255,255,0.9)';
            ctx.fillText(label, pt.x + 11, pt.y);

            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 5.5, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255, 80, 80, 0.8)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }

    // Crosshair when editing
    if (WalkwayEditor.enabled && WalkwayEditor.cursor.valid) {
        const { x, y } = WalkwayEditor.cursor;
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x - 8, y);
        ctx.lineTo(x + 8, y);
        ctx.moveTo(x, y - 8);
        ctx.lineTo(x, y + 8);
        ctx.stroke();
    }

    ctx.restore();
}

function renderWalkwayMagnifier() {
    const mag = document.getElementById('walkway-magnifier');
    if (!mag) return;
    if (!WalkwayEditor.showMagnifier || !WalkwayEditor.cursor.valid) return;

    const canvas = document.getElementById('game-map');
    const ctx = mag.getContext('2d');
    const zoom = 6;
    const sw = mag.width / zoom;
    const sh = mag.height / zoom;
    let sx = WalkwayEditor.cursor.x - sw / 2;
    let sy = WalkwayEditor.cursor.y - sh / 2;
    sx = Math.max(0, Math.min(canvas.width - sw, sx));
    sy = Math.max(0, Math.min(canvas.height - sh, sy));

    ctx.clearRect(0, 0, mag.width, mag.height);
    ctx.drawImage(canvas, sx, sy, sw, sh, 0, 0, mag.width, mag.height);

    // Crosshair
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mag.width / 2 - 10, mag.height / 2);
    ctx.lineTo(mag.width / 2 + 10, mag.height / 2);
    ctx.moveTo(mag.width / 2, mag.height / 2 - 10);
    ctx.lineTo(mag.width / 2, mag.height / 2 + 10);
    ctx.stroke();
}

// ============================================================
// UI Updates
// ============================================================
function updateTaskProgress(progress) {
    const fill = document.getElementById('task-progress-fill');
    const text = document.getElementById('task-progress-text');
    const pct = Math.round(progress * 100);
    fill.style.width = `${pct}%`;
    text.textContent = `${pct}%`;

    if (GameState.gameData.gameResult && GameState.currentTurnIndex === GameState.totalStates - 1) {
        if (GameState.gameData.gameResult.includes('Impostor')) {
            fill.classList.add('game-over');
        } else {
            fill.classList.remove('game-over');
        }
    } else {
        fill.classList.remove('game-over');
    }
}

function updateTurnDisplay(turn) {
    if (!turn) {
        document.getElementById('turn-display').textContent = 'T0';
        const badge = document.getElementById('phase-display');
        badge.textContent = 'start';
        badge.className = 'phase-badge';
        return;
    }
    document.getElementById('turn-display').textContent = `T${turn.turnNumber}`;
    const badge = document.getElementById('phase-display');
    badge.textContent = turn.phase;
    badge.className = `phase-badge ${turn.phase}`;
}

function updateTimeline() {
    const timeline = document.getElementById('timeline');
    timeline.max = Math.max(0, GameState.totalStates - 1);
    timeline.value = GameState.currentTurnIndex;
}

function updateTimelinePosition() {
    document.getElementById('timeline').value = GameState.currentTurnIndex;
}

function updatePlayerRoster() {
    const roster = document.getElementById('player-roster');
    roster.classList.remove('hidden');
    roster.innerHTML = '';

    GameState.gameData.players.forEach(player => {
        const chip = document.createElement('div');
        chip.className = 'player-chip';
        chip.dataset.player = player.name;

        const dot = document.createElement('span');
        dot.className = 'color-dot';
        dot.style.backgroundColor = COLOR_MAP[player.color] || player.color;

        const name = document.createElement('span');
        name.textContent = player.name;

        const role = document.createElement('span');
        role.className = `role-badge ${player.role.toLowerCase()}`;
        role.textContent = player.role;

        chip.appendChild(dot);
        chip.appendChild(name);
        chip.appendChild(role);
        roster.appendChild(chip);
    });
}

function updatePlayerRosterStatus(aliveStatus) {
    GameState.gameData.players.forEach(player => {
        const chip = document.querySelector(`.player-chip[data-player="${player.name}"]`);
        if (chip) {
            chip.classList.toggle('dead', !aliveStatus[player.name]);
        }
    });
}

function showGameResult() {
    const el = document.getElementById('game-result');
    const result = GameState.gameData.gameResult;
    if (result) {
        el.textContent = result;
        el.classList.remove('hidden');
        if (result.toLowerCase().includes('crewmate')) {
            el.className = 'game-result crewmate-win';
        } else {
            el.className = 'game-result impostor-win';
        }
    } else {
        el.textContent = 'Game Incomplete';
        el.className = 'game-result';
    }
}

// ============================================================
// Activity Log
// ============================================================
function updateActivityLog(currentTurnIndex) {
    const container = document.getElementById('activity-log');
    container.innerHTML = '';

    GameState.gameData.turns.forEach((turn, turnIdx) => {
        turn.events.forEach(event => {
            const entry = document.createElement('div');
            entry.className = `log-entry ${event.type}`;
            if (turnIdx === currentTurnIndex - 1) entry.classList.add('current-turn');

            const text = formatEventText(turn, event);
            entry.textContent = text;
            container.appendChild(entry);
        });

        // Show voteout as a special entry
        if (turn.voteResult && turn.voteResult.ejected) {
            const entry = document.createElement('div');
            entry.className = 'log-entry voteout';
            if (turnIdx === currentTurnIndex - 1) entry.classList.add('current-turn');
            entry.textContent = `T${turn.turnNumber}: ${turn.voteResult.ejected} was voted out`;
            container.appendChild(entry);
        }
    });

    // Show game result
    if (GameState.gameData.gameResult) {
        const entry = document.createElement('div');
        entry.className = 'log-entry game-over';
        entry.textContent = GameState.gameData.gameResult;
        container.appendChild(entry);
    }

    // Scroll to current turn entries
    const currentEntries = container.querySelectorAll('.current-turn');
    if (currentEntries.length > 0) {
        currentEntries[0].scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
}

function formatEventText(turn, event) {
    const t = `T${turn.turnNumber}`;
    const p = event.player || '';

    switch (event.type) {
        case 'move':
            return `${t} ${p}: Move ${event.from} -> ${event.to}`;
        case 'complete_task':
            return `${t} ${p}: Complete Task - ${event.task}`;
        case 'fake_task':
            return `${t} ${p}: Fake Task - ${event.task}`;
        case 'kill':
            return `${t} ${p}: KILL ${event.victim} @ ${event.location}`;
        case 'vent':
            return `${t} ${p}: Vent ${event.from} -> ${event.to}`;
        case 'speak':
            return `${t} ${p}: "${event.message.substring(0, 80)}${event.message.length > 80 ? '...' : ''}"`;
        case 'vote':
            return `${t} ${p}: Vote -> ${event.target}`;
        case 'skip_vote':
            return `${t} ${p}: Skip Vote`;
        case 'report_body':
            return `${t} ${p}: Report Body @ ${event.location}`;
        case 'call_meeting':
            return `${t} ${p}: Emergency Meeting @ ${event.location}`;
        case 'view_monitor':
            return `${t} ${p}: View Monitor`;
        default:
            return `${t} ${p}: ${event.type}`;
    }
}

// ============================================================
// Meeting Panel & Speech Bubbles
// ============================================================
function updateMeetingPanel(turn) {
    const panel = document.getElementById('meeting-panel');
    const bubblesEl = document.getElementById('speech-bubbles');
    const votesEl = document.getElementById('vote-results');
    const toggleBtn = document.getElementById('btn-chat-toggle');

    if (!turn || turn.phase !== 'meeting') {
        panel.classList.add('hidden');
        toggleBtn.classList.add('hidden');
        toggleBtn.classList.remove('has-chat');
        return;
    }

    // Show the toggle button, auto-open the panel
    toggleBtn.classList.remove('hidden');
    toggleBtn.classList.add('has-chat');
    panel.classList.remove('hidden');

    // Determine meeting trigger
    const title = document.getElementById('meeting-title');
    const triggerEvent = findMeetingTrigger(turn);
    title.textContent = triggerEvent || 'Emergency Meeting';

    // Render speech bubbles
    bubblesEl.innerHTML = '';
    const discussion = turn.discussion || [];

    discussion.forEach(round => {
        // Round divider
        const divider = document.createElement('div');
        divider.className = 'round-divider';
        divider.textContent = `Discussion Round ${round.round + 1}`;
        bubblesEl.appendChild(divider);

        round.speeches.forEach(speech => {
            const bubble = createSpeechBubble(speech.player, speech.message);
            bubblesEl.appendChild(bubble);
        });
    });

    // Render votes
    const votes = turn.votes || [];
    if (votes.length > 0) {
        votesEl.classList.remove('hidden');
        votesEl.innerHTML = '<h4>Votes</h4>';

        votes.forEach(vote => {
            const entry = document.createElement('div');
            entry.className = 'vote-entry';
            entry.textContent = `${vote.voter} -> ${vote.target}`;
            votesEl.appendChild(entry);
        });

        if (turn.voteResult && turn.voteResult.ejected) {
            const result = document.createElement('div');
            result.className = 'vote-result-line';
            result.textContent = `${turn.voteResult.ejected} was voted out!`;
            votesEl.appendChild(result);
        }
    } else {
        votesEl.classList.add('hidden');
    }
}

function findMeetingTrigger(turn) {
    // Look at the previous turn for the meeting trigger (report body or call meeting)
    const turnIdx = GameState.gameData.turns.indexOf(turn);
    if (turnIdx > 0) {
        const prevTurn = GameState.gameData.turns[turnIdx - 1];
        for (const event of prevTurn.events) {
            if (event.type === 'report_body') {
                return `${event.player} reported a body at ${event.location}`;
            }
            if (event.type === 'call_meeting') {
                return `${event.player} called an emergency meeting`;
            }
        }
    }
    return 'Emergency Meeting';
}

function createSpeechBubble(playerName, message) {
    const player = GameState.gameData.players.find(p => p.name === playerName);
    const color = player ? (COLOR_MAP[player.color] || player.color) : '#888';

    const bubble = document.createElement('div');
    bubble.className = 'speech-bubble';

    const speakerRow = document.createElement('div');
    speakerRow.className = 'speaker-row';

    const dot = document.createElement('span');
    dot.className = 'speaker-color';
    dot.style.backgroundColor = color;

    const name = document.createElement('span');
    name.className = 'speaker-name';
    name.style.color = color;
    name.textContent = playerName;

    speakerRow.appendChild(dot);
    speakerRow.appendChild(name);

    const text = document.createElement('div');
    text.className = 'speech-text';
    text.textContent = message;

    bubble.appendChild(speakerRow);
    bubble.appendChild(text);
    return bubble;
}

// ============================================================
// Playback Controls
// ============================================================
function setupControls() {
    document.getElementById('btn-start').addEventListener('click', jumpToStart);
    document.getElementById('btn-end').addEventListener('click', jumpToEnd);
    document.getElementById('btn-prev-turn').addEventListener('click', prevTurn);
    document.getElementById('btn-next-turn').addEventListener('click', nextTurn);
    document.getElementById('btn-prev').addEventListener('click', prevTurn);
    document.getElementById('btn-next').addEventListener('click', nextTurn);
    document.getElementById('btn-play').addEventListener('click', togglePlay);

    document.getElementById('timeline').addEventListener('input', (e) => {
        seekToTurn(parseInt(e.target.value));
    });

    document.querySelectorAll('.speed-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setSpeed(parseFloat(btn.dataset.speed));
        });
    });

    document.getElementById('btn-fullscreen').addEventListener('click', toggleFullscreen);
    document.getElementById('btn-chat-toggle').addEventListener('click', toggleChat);
    document.getElementById('btn-close-chat').addEventListener('click', toggleChat);
}

function toggleChat() {
    const panel = document.getElementById('meeting-panel');
    panel.classList.toggle('hidden');
}

function toggleFullscreen() {
    const mapArea = document.getElementById('map-area');
    if (!document.fullscreenElement) {
        mapArea.requestFullscreen().catch(() => {});
    } else {
        document.exitFullscreen();
    }
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Don't handle shortcuts when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                togglePlay();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                prevTurn();
                break;
            case 'ArrowRight':
                e.preventDefault();
                nextTurn();
                break;
            case 'BracketLeft':
                decreaseSpeed();
                break;
            case 'BracketRight':
                increaseSpeed();
                break;
            case 'Home':
                jumpToStart();
                break;
            case 'End':
                jumpToEnd();
                break;
            case 'KeyF':
                toggleFullscreen();
                break;
            case 'KeyE': {
                // Toggle the editor panel; if already visible, toggle edit-enabled.
                const panel = document.getElementById('walkway-editor');
                if (!panel) break;
                if (panel.classList.contains('hidden')) {
                    panel.classList.remove('hidden');
                    WalkwayEditor.enabled = true;
                } else {
                    WalkwayEditor.enabled = !WalkwayEditor.enabled;
                }
                updateWalkwayEditorHint();
                renderCanvasFrame();
                break;
            }
            case 'Escape':
                if (WalkwayEditor.enabled) {
                    WalkwayEditor.edgeStartNode = null;
                    WalkwayEditor.draggingNode = null;
                    updateWalkwayEditorHint();
                    renderCanvasFrame();
                }
                break;
        }
    });
}

function togglePlay() {
    if (GameState.isPlaying) {
        pause();
    } else {
        play();
    }
}

function play() {
    if (!GameState.gameData) return;
    GameState.isPlaying = true;
    document.getElementById('btn-play').innerHTML = '&#9646;&#9646;';
    GameState.playbackSegmentIndex = GameState.currentTurnIndex;
    GameState.playbackSegmentStartTs = performance.now();
    GameState.playbackAlpha = 0;
    GameState._lastUiTurnIndex = null;
    startPlaybackRaf();
}

function pause() {
    GameState.isPlaying = false;
    document.getElementById('btn-play').innerHTML = '&#9654;';
    clearTimeout(GameState.playbackTimer);
    stopPlaybackRaf();
    GameState.playbackAlpha = 0;
}

function startPlaybackRaf() {
    stopPlaybackRaf();
    const tick = (now) => {
        if (!GameState.isPlaying) return;
        const duration = 2000 / GameState.playbackSpeed;

        // Handle end-of-replay
        if (GameState.playbackSegmentIndex >= GameState.totalStates - 1) {
            GameState.currentTurnIndex = GameState.totalStates - 1;
            GameState.playbackAlpha = 0;
            renderCurrentState();
            pause();
            return;
        }

        let alpha = (now - GameState.playbackSegmentStartTs) / duration;
        while (alpha >= 1 && GameState.playbackSegmentIndex < GameState.totalStates - 1) {
            GameState.playbackSegmentIndex++;
            GameState.playbackSegmentStartTs += duration;
            alpha = (now - GameState.playbackSegmentStartTs) / duration;
            // If we reached end, break.
            if (GameState.playbackSegmentIndex >= GameState.totalStates - 1) {
                alpha = 0;
                break;
            }
        }

        alpha = Math.max(0, Math.min(1, alpha));
        GameState.currentTurnIndex = GameState.playbackSegmentIndex;
        GameState.playbackAlpha = alpha;
        if (GameState._lastUiTurnIndex !== GameState.currentTurnIndex) {
            GameState._lastUiTurnIndex = GameState.currentTurnIndex;
            renderCurrentState();
        } else {
            renderCanvasFrame();
        }
        GameState.playbackRafId = requestAnimationFrame(tick);
    };
    GameState.playbackRafId = requestAnimationFrame(tick);
}

function stopPlaybackRaf() {
    if (GameState.playbackRafId) {
        cancelAnimationFrame(GameState.playbackRafId);
        GameState.playbackRafId = null;
    }
}

function nextTurn() {
    if (!GameState.gameData) return;
    if (GameState.currentTurnIndex < GameState.totalStates - 1) {
        GameState.currentTurnIndex++;
        GameState.playbackSegmentIndex = GameState.currentTurnIndex;
        GameState.playbackSegmentStartTs = performance.now();
        GameState.playbackAlpha = 0;
        renderCurrentState();
    }
}

function prevTurn() {
    if (!GameState.gameData) return;
    if (GameState.currentTurnIndex > 0) {
        GameState.currentTurnIndex--;
        GameState.playbackSegmentIndex = GameState.currentTurnIndex;
        GameState.playbackSegmentStartTs = performance.now();
        GameState.playbackAlpha = 0;
        renderCurrentState();
    }
}

function jumpToStart() {
    if (!GameState.gameData) return;
    GameState.currentTurnIndex = 0;
    GameState.playbackSegmentIndex = 0;
    GameState.playbackSegmentStartTs = performance.now();
    GameState.playbackAlpha = 0;
    renderCurrentState();
}

function jumpToEnd() {
    if (!GameState.gameData) return;
    GameState.currentTurnIndex = GameState.totalStates - 1;
    GameState.playbackSegmentIndex = GameState.currentTurnIndex;
    GameState.playbackSegmentStartTs = performance.now();
    GameState.playbackAlpha = 0;
    renderCurrentState();
}

function seekToTurn(index) {
    if (!GameState.gameData) return;
    GameState.currentTurnIndex = Math.max(0, Math.min(index, GameState.totalStates - 1));
    GameState.playbackSegmentIndex = GameState.currentTurnIndex;
    GameState.playbackSegmentStartTs = performance.now();
    GameState.playbackAlpha = 0;
    renderCurrentState();
}

function setSpeed(speed) {
    GameState.playbackSpeed = speed;
    document.querySelectorAll('.speed-btn').forEach(btn => {
        btn.classList.toggle('active', parseFloat(btn.dataset.speed) === speed);
    });
    // If playing, adjust segment start time to preserve alpha.
    if (GameState.isPlaying) {
        const now = performance.now();
        const newDuration = 2000 / speed;
        const alpha = GameState.playbackAlpha || 0;
        // Re-anchor so that current alpha stays roughly consistent.
        GameState.playbackSegmentStartTs = now - alpha * newDuration;
    }
}

function decreaseSpeed() {
    const speeds = [0.5, 1, 2, 4];
    const idx = speeds.indexOf(GameState.playbackSpeed);
    if (idx > 0) setSpeed(speeds[idx - 1]);
}

function increaseSpeed() {
    const speeds = [0.5, 1, 2, 4];
    const idx = speeds.indexOf(GameState.playbackSpeed);
    if (idx < speeds.length - 1) setSpeed(speeds[idx + 1]);
}

// ============================================================
// Utilities
// ============================================================
function showLoading(show) {
    document.getElementById('loading-overlay').classList.toggle('hidden', !show);
}

// ============================================================
// Start the app
// ============================================================
document.addEventListener('DOMContentLoaded', init);
