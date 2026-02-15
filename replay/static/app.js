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
const PLAYER_SPRITE_SIZE = 36;
const PLAYER_SPRITE_GAP = 2;

// ============================================================
// Game State
// ============================================================
const GameState = {
    gameData: null,
    mapConfig: null,
    taskConfig: null,
    mapImage: null,
    playerSprites: {},

    // Precomputed per-turn state
    playerPositions: [],  // [{playerName: roomName, ...}, ...]
    playerAlive: [],      // [{playerName: bool, ...}, ...]
    taskProgress: [],     // [float, ...]

    // Playback
    currentTurnIndex: 0,
    isPlaying: false,
    playbackSpeed: 1,
    playbackTimer: null,

    get totalTurns() {
        return this.gameData ? this.gameData.turns.length : 0;
    },

    get currentTurn() {
        if (!this.gameData || this.currentTurnIndex >= this.gameData.turns.length) return null;
        return this.gameData.turns[this.currentTurnIndex];
    },
};

// ============================================================
// Initialization
// ============================================================
async function init() {
    setupControls();
    setupKeyboardShortcuts();
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

    const [gameRes, mapRes, taskRes] = await Promise.all([
        fetch(`/api/game/${encodeURIComponent(filename)}`),
        fetch('/api/map-config'),
        fetch('/api/task-config'),
    ]);

    GameState.gameData = await gameRes.json();
    GameState.mapConfig = await mapRes.json();
    GameState.taskConfig = await taskRes.json();

    // Load map image
    GameState.mapImage = await loadImage('/assets/skeld.png');
    GameState.playerSprites = {};

    // Load color sprite sheets used by this game.
    const usedColors = [...new Set((GameState.gameData.players || []).map(p => p.color))];
    await Promise.all(usedColors.map(async color => {
        try {
            GameState.playerSprites[color] = await loadImage(`/assets/${color}.png`);
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

    data.turns.forEach(turn => {
        // Clone current state at start of turn
        const turnPositions = { ...currentPositions };
        const turnAlive = { ...currentAlive };

        if (turn.phase === 'meeting') {
            // During meetings, all alive players are in Cafeteria
            Object.keys(turnPositions).forEach(name => {
                if (turnAlive[name]) turnPositions[name] = 'Cafeteria';
            });
        }

        // Process events to update state
        turn.events.forEach(event => {
            const player = event.player;
            if (event.type === 'move') {
                turnPositions[player] = event.to;
                currentPositions[player] = event.to;
            } else if (event.type === 'vent') {
                turnPositions[player] = event.to;
                currentPositions[player] = event.to;
            } else if (event.type === 'kill') {
                turnAlive[event.victim] = false;
                currentAlive[event.victim] = false;
            } else if (event.type === 'complete_task') {
                // Decrement remaining duration; mark complete when <= 0
                if (taskDurations[player] && taskDurations[player][event.task] !== undefined) {
                    taskDurations[player][event.task]--;
                    if (taskDurations[player][event.task] <= 0) {
                        taskCompleted[player][event.task] = true;
                    }
                }
            }
        });

        // Handle voteout
        if (turn.voteResult && turn.voteResult.ejected) {
            turnAlive[turn.voteResult.ejected] = false;
            currentAlive[turn.voteResult.ejected] = false;
        }

        // After meeting, reset positions to Cafeteria for alive players
        if (turn.phase === 'meeting') {
            Object.keys(currentPositions).forEach(name => {
                if (currentAlive[name]) currentPositions[name] = 'Cafeteria';
            });
        }

        positions.push(turnPositions);
        alive.push({ ...turnAlive });
        progress.push(computeTaskProgress(turnAlive));
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
    const positions = GameState.playerPositions[turnIndex];
    const aliveStatus = GameState.playerAlive[turnIndex];
    const taskProg = GameState.taskProgress[turnIndex];

    drawMap(positions, aliveStatus);
    updateTaskProgress(taskProg);
    updateTurnDisplay(turn);
    updateActivityLog(turnIndex);
    updateMeetingPanel(turn);
    updatePlayerRosterStatus(aliveStatus);
    updateTimelinePosition();
}

function drawMap(positions, aliveStatus) {
    const canvas = document.getElementById('game-map');
    const ctx = canvas.getContext('2d');
    const mapConfig = GameState.mapConfig;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background image
    if (GameState.mapImage) {
        ctx.drawImage(GameState.mapImage, 0, 0, canvas.width, canvas.height);
    }

    // Transform overlays/players to align with skeld.png
    ctx.save();
    ctx.translate(75, 8);
    ctx.scale(0.92, 0.92);

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
        ctx.font = 'bold 11px sans-serif';
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

    // Group players by room
    const roomPlayers = {};
    for (const [playerName, room] of Object.entries(positions)) {
        if (!roomPlayers[room]) roomPlayers[room] = [];
        roomPlayers[room].push(playerName);
    }

    // Draw player tokens
    const playerSize = PLAYER_SPRITE_SIZE;
    const space = PLAYER_SPRITE_GAP;
    const playersByName = new Map((GameState.gameData.players || []).map(player => [player.name, player]));

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

        // Figure out how many fit per row and how many rows needed
        const perRow = Math.max(1, Math.floor((roomW - space) / (playerSize + space)));
        const rowsNeeded = Math.ceil(players.length / perRow);
        const availH = roomH - 16; // leave room for label at top

        // Shrink sprites if they'd overflow the room vertically
        let sz = playerSize;
        if (rowsNeeded * (sz + space) > availH) {
            sz = Math.max(16, Math.floor((availH - space) / rowsNeeded) - space);
        }

        const startY = minY + 16; // below label
        let x = minX + space;
        let y = startY;

        players.forEach(playerName => {
            const player = playersByName.get(playerName);
            if (!player) return;

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

            // Draw X for dead players
            if (!isAlive) {
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x + sz, y + sz);
                ctx.strokeStyle = '#e94560';
                ctx.lineWidth = 2.5;
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(x + sz, y);
                ctx.lineTo(x, y + sz);
                ctx.stroke();
            }

            // Advance position, wrap to next row within room bounds
            x += sz + space;
            if (x + sz > maxX - space) {
                x = minX + space;
                y += sz + space;
            }
        });
    }

    ctx.restore();
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

    if (GameState.gameData.gameResult && GameState.currentTurnIndex === GameState.totalTurns - 1) {
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
    if (!turn) return;
    document.getElementById('turn-display').textContent = `T${turn.turnNumber}`;
    const badge = document.getElementById('phase-display');
    badge.textContent = turn.phase;
    badge.className = `phase-badge ${turn.phase}`;
}

function updateTimeline() {
    const timeline = document.getElementById('timeline');
    timeline.max = Math.max(0, GameState.totalTurns - 1);
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
            if (turnIdx === currentTurnIndex) entry.classList.add('current-turn');

            const text = formatEventText(turn, event);
            entry.textContent = text;
            container.appendChild(entry);
        });

        // Show voteout as a special entry
        if (turn.voteResult && turn.voteResult.ejected) {
            const entry = document.createElement('div');
            entry.className = 'log-entry voteout';
            if (turnIdx === currentTurnIndex) entry.classList.add('current-turn');
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
    scheduleNext();
}

function pause() {
    GameState.isPlaying = false;
    document.getElementById('btn-play').innerHTML = '&#9654;';
    clearTimeout(GameState.playbackTimer);
}

function scheduleNext() {
    const delay = 2000 / GameState.playbackSpeed;
    GameState.playbackTimer = setTimeout(() => {
        if (GameState.currentTurnIndex < GameState.totalTurns - 1) {
            GameState.currentTurnIndex++;
            renderCurrentState();
            if (GameState.isPlaying) scheduleNext();
        } else {
            pause();
        }
    }, delay);
}

function nextTurn() {
    if (!GameState.gameData) return;
    if (GameState.currentTurnIndex < GameState.totalTurns - 1) {
        GameState.currentTurnIndex++;
        renderCurrentState();
    }
}

function prevTurn() {
    if (!GameState.gameData) return;
    if (GameState.currentTurnIndex > 0) {
        GameState.currentTurnIndex--;
        renderCurrentState();
    }
}

function jumpToStart() {
    if (!GameState.gameData) return;
    GameState.currentTurnIndex = 0;
    renderCurrentState();
}

function jumpToEnd() {
    if (!GameState.gameData) return;
    GameState.currentTurnIndex = GameState.totalTurns - 1;
    renderCurrentState();
}

function seekToTurn(index) {
    if (!GameState.gameData) return;
    GameState.currentTurnIndex = Math.max(0, Math.min(index, GameState.totalTurns - 1));
    renderCurrentState();
}

function setSpeed(speed) {
    GameState.playbackSpeed = speed;
    document.querySelectorAll('.speed-btn').forEach(btn => {
        btn.classList.toggle('active', parseFloat(btn.dataset.speed) === speed);
    });
    // If playing, restart timer with new speed
    if (GameState.isPlaying) {
        clearTimeout(GameState.playbackTimer);
        scheduleNext();
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
