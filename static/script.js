let valueChart;

function selectTab(btn) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const algo = btn.dataset.algo || 'policy_iteration';
    setTimeout(() => runAlgorithm(algo), 30);
}

async function runAlgorithm(algorithm) {
    // Get parameters
    const params = {
        rows: document.getElementById('rows').value,
        cols: document.getElementById('cols').value,
        slip: (typeof envName !== 'undefined' && envName === 'frozenlake') ? document.getElementById('slip').value : 0.0,
        gamma: document.getElementById('gamma').value,
        theta: document.getElementById('theta').value,
        episodes: document.getElementById('episodes') ? document.getElementById('episodes').value : undefined,
        alpha: document.getElementById('alpha') ? document.getElementById('alpha').value : undefined,
        epsilon: document.getElementById('epsilon') ? document.getElementById('epsilon').value : undefined,
        algorithm: algorithm,
        env: (typeof envName !== 'undefined') ? envName : 'frozenlake',
        trace: document.getElementById('trace')?.checked ? 1 : 0
    };
    // Update displays
    document.getElementById('rows-val').textContent = params.rows;
    document.getElementById('cols-val').textContent = params.cols;
    document.getElementById('slip-val').textContent = params.slip;
    document.getElementById('gamma-val').textContent = params.gamma;
    document.getElementById('theta-val').textContent = params.theta;
    if (typeof params.episodes !== 'undefined') document.getElementById('episodes-val').textContent = params.episodes;
    if (typeof params.alpha !== 'undefined') document.getElementById('alpha-val').textContent = Number(params.alpha).toFixed(2);
    if (typeof params.epsilon !== 'undefined') document.getElementById('epsilon-val').textContent = Number(params.epsilon).toFixed(2);

    const url = `/api/run_algorithm?${new URLSearchParams(params)}`;
    let res;
    try {
        res = await fetch(url);
    } catch (err) {
        alert('Network error: ' + err);
        return;
    }
    const data = await res.json();

    if (data.error) {
        alert('Error: ' + data.error);
        return;
    }

    // store original data for later playback
    window._lastRunData = data;

    // basic validation to help debug GridWorld
    if (!data.map || data.map.length !== data.rows || data.map.some(r => r.length !== data.cols)) {
        console.error('Invalid map shape', data.map, data.rows, data.cols);
        // show non-blocking warning
        const warn = `Map size mismatch: expected ${data.rows}x${data.cols}`;
        alert(warn);
        
    }

    if (data.history && data.history.length > 0) {
        createIterationControl(data.history, data.map, data.rows, data.cols, algorithm);
        const last = data.history.length - 1;
        renderIteration(last, data.history, data.map, data.rows, data.cols, algorithm);
        // rewards are static per env
        renderRewardsGrid(data.rewards, data.rows, data.cols);
    } else {
        
        clearIterationControl();
        drawGrid(data.map, data.holes, data.policy, data.rows, data.cols);
        drawValueChart(data.values);
        renderValuesGrid(data.values, data.rows, data.cols);
        renderRewardsGrid(data.rewards, data.rows, data.cols);
    }
    updateStats(data, algorithm);

    if (data.episode_returns && Array.isArray(data.episode_returns)) {
        drawEpisodeChart(data.episode_returns);
    } else {
        clearEpisodeChart();
    }
}

function drawGrid(map, holes, policy, rows, cols) {
    const grid = document.getElementById('grid');
    if (!grid) return;
    grid.style.gridTemplateColumns = `repeat(${cols}, 70px)`;
    grid.innerHTML = '';

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const cell = document.createElement('div');
            const rawType = (map && map[i]) ? map[i][j] : '';
            const type = rawType || 'grid-empty';
            cell.className = `cell ${type}`.trim();
            cell.textContent = (rawType && rawType !== '') ? rawType : '';

            const key = `(${i},${j})`;
            if (policy && policy[key]) {
                cell.classList.add(`arrow-${policy[key]}`);
            }

            
            grid.appendChild(cell);
        }
    }
    const oldAgent = document.querySelector('.agent');
    if (oldAgent && oldAgent.parentNode) oldAgent.parentNode.removeChild(oldAgent);
}

function createIterationControl(history, map, rows, cols, algorithm) {
    const container = document.getElementById('iteration-control');
    if (!container) return;
    container.innerHTML = '';
    const label = document.createElement('label');
    label.textContent = 'Iteration: ';
    const valueSpan = document.createElement('span');
    valueSpan.id = 'iteration-value';
    valueSpan.textContent = history.length;
    label.appendChild(valueSpan);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = 1;
    slider.max = history.length;
    slider.value = history.length;
    slider.style.width = '100%';
    slider.oninput = function() {
        const idx = parseInt(this.value) - 1;
        document.getElementById('iteration-value').textContent = (idx+1);
        renderIteration(idx, history, map, rows, cols, algorithm);
    };

    container.appendChild(label);
    container.appendChild(slider);
}

function clearIterationControl() {
    const container = document.getElementById('iteration-control');
    if (container) container.innerHTML = '';
}

function renderIteration(idx, history, map, rows, cols, algorithm) {
    const item = history[idx];
    if (!item) return;
    // render policy and values from snapshot
    drawGrid(map, [], item.policy, rows, cols);
    drawValueChart(item.values);
    renderValuesGrid(item.values, rows, cols);
}

function renderValuesGrid(values, rows, cols) {
    const table = document.getElementById('values-grid');
    if (!table) return;
    table.innerHTML = '';
    for (let i = 0; i < rows; i++) {
        const tr = document.createElement('tr');
        for (let j = 0; j < cols; j++) {
            const td = document.createElement('td');
            const key = `(${i},${j})`;
            const v = (values && typeof values[key] !== 'undefined') ? values[key] : '';
            td.textContent = (v === '' || v === null) ? '' : Number(v).toFixed(2);
            tr.appendChild(td);
        }
        table.appendChild(tr);
    }
}

async function simulatePolicy() {
    const algo = document.querySelector('.tab-btn.active')?.dataset?.algo || 'policy_iteration';
    const rows = document.getElementById('rows').value;
    const cols = document.getElementById('cols').value;
    const slip = (typeof envName !== 'undefined' && envName === 'frozenlake') ? document.getElementById('slip').value : 0.0;
    const stochastic = document.getElementById('sim-stochastic')?.checked;
    const useLast = document.getElementById('use-last-policy')?.checked;
    const episodes = document.getElementById('episodes') ? Number(document.getElementById('episodes').value) : undefined;

    if (useLast && window._lastRunData && window._lastRunData.policy) {
        // ensure lastRun corresponds to same env & rows/cols
        if (window._lastRunData.env === (typeof envName !== 'undefined' ? envName : 'frozenlake') && window._lastRunData.rows == rows && window._lastRunData.cols == cols) {
            try {
                const body = {
                    env: window._lastRunData.env,
                    rows: Number(rows),
                    cols: Number(cols),
                    slip: Number(slip),
                    policy: window._lastRunData.policy,
                    stochastic: !!stochastic
                };
                const res = await fetch('/api/simulate_policy', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
                const data = await res.json();
                if (data.error) { alert('Error: '+data.error); return; }
                if (!data.trajectory || data.trajectory.length === 0) { alert('No trajectory returned'); return; }
                await animateTrajectory(data.trajectory, data.map || window._lastRunData.map, Number(rows), Number(cols));
                return;
            } catch (e) {
                alert('Simulation failed: '+e);
                return;
            }
        }
    }

    
    const params = {
        rows: rows,
        cols: cols,
        slip: slip,
        gamma: document.getElementById('gamma').value,
        theta: document.getElementById('theta').value,
        algorithm: algo,
        env: (typeof envName !== 'undefined') ? envName : 'frozenlake',
        simulate: 1,
        stochastic: stochastic ? 1 : 0
    };
    const url = `/api/run_algorithm?${new URLSearchParams(params)}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.error) { alert('Error: '+data.error); return; }
    if (!data.trajectory || data.trajectory.length === 0) { alert('No trajectory returned'); return; }

   
    await animateTrajectory(data.trajectory, data.map, data.rows, data.cols);
}

async function animateTrajectory(traj, map, rows, cols) {
    
    pauseSimulation();
    removeAgent();
    clearVisitedRewards();

    const grid = document.getElementById('grid');
    if (!grid) return;
    const agent = document.createElement('div');
    agent.className = 'agent';
    agent.textContent = 'A';
    agent.style.position = 'absolute';
    grid.appendChild(agent);

 
    function cellCenter(i, j) {
        const cell = grid.children[i*cols + j];
        if (!cell) return {x:0,y:0};
        const rGrid = grid.getBoundingClientRect();
        const rCell = cell.getBoundingClientRect();
        const x = rCell.left - rGrid.left + (rCell.width/2) - (agent.offsetWidth/2);
        const y = rCell.top - rGrid.top + (rCell.height/2) - (agent.offsetHeight/2);
        return {x, y};
    }

    let idx = 0;
    let playing = true;
    const speedControl = document.getElementById('sim-speed');
    const stepPause = () => parseInt(speedControl?.value || 450);

    setSimulationState({traj, map, rows, cols, agent, idx, playing});

   
    while (idx < traj.length) {
        const step = traj[idx];
        const [x, y] = step.state.replace(/[()]/g,'').split(',').map(Number);
        const center = cellCenter(x, y);
       
        agent.style.transition = `left ${stepPause()}ms linear, top ${stepPause()}ms linear`;
        agent.style.left = center.x + 'px';
        agent.style.top = center.y + 'px';
        const rewardEl = document.getElementById('current-reward');
        if (rewardEl) rewardEl.textContent = (Number(step.cum_reward) || 0).toFixed(2);
        try {
            updateRewardCell(x, y, typeof step.reward !== 'undefined' ? step.reward : 0, typeof step.cum_reward !== 'undefined' ? step.cum_reward : 0);
        } catch (e) {
            console.warn('Failed to update reward cell', e);
        }


        let waited = 0;
        const waitStep = stepPause();
        while (waited < waitStep) {
            await new Promise(r => setTimeout(r, 50));
            waited += 50;
            const state = getSimulationState();
            if (!state.playing) {
                // paused; wait until resume
                while (!getSimulationState().playing) {
                    await new Promise(r => setTimeout(r, 100));
                }
            }
        }
        idx = getSimulationState().idx + 1 || idx + 1;
        setSimulationState({idx});
    }
    
    try {
        const tot = traj.length ? (traj[traj.length-1].cum_reward || traj[traj.length-1].cum || 0) : 0;
        const totalEl = document.getElementById('total-reward');
        if (totalEl) totalEl.textContent = (Number(tot) || 0).toFixed(2);
    } catch (e) {
        console.warn('Failed to write total reward', e);
    }
    clearSimulationState();
}


let __simState = null;
function setSimulationState(partial) {
    __simState = Object.assign(__simState || {}, partial);
}
function getSimulationState() { return __simState || {playing:false, idx:0}; }
function clearSimulationState() { __simState = null; }

function playSimulation() {
    const s = getSimulationState();
    if (!s || !s.traj) return;
    setSimulationState({playing:true});
}
function pauseSimulation() { setSimulationState({playing:false}); }
function stepForward() { const s=getSimulationState(); if (!s||!s.traj) return; const idx = Math.min(s.idx+1, s.traj.length-1); setSimulationState({idx}); renderIteration(idx, s.traj.map(t=>({policy:{}, values:{}})), s.map, s.rows, s.cols); }
function stepBack() { const s=getSimulationState(); if (!s||!s.traj) return; const idx = Math.max(s.idx-1, 0); setSimulationState({idx}); renderIteration(idx, s.traj.map(t=>({policy:{}, values:{}})), s.map, s.rows, s.cols); }

function placeAgentAt(i, j) {
    const grid = document.getElementById('grid');
    if (!grid) return;
    const cols = getComputedStyle(grid).gridTemplateColumns.split(' ').length || parseInt(document.getElementById('cols').value || '4');
    const idx = i * cols + j;
    const cell = grid.children[idx];
    if (!cell) return;
    removeAgent();
    const a = document.createElement('div');
    a.className = 'agent';
    a.textContent = 'A';
    cell.appendChild(a);
}

function removeAgent() {
    const old = document.querySelector('.agent');
    if (old && old.parentNode) old.parentNode.removeChild(old);
}

function renderRewardsGrid(rewards, rows, cols) {
    const table = document.getElementById('rewards-grid');
    if (!table) return;
    table.innerHTML = '';
    for (let i = 0; i < rows; i++) {
        const tr = document.createElement('tr');
        for (let j = 0; j < cols; j++) {
            const td = document.createElement('td');
            const key = `(${i},${j})`;
            const v = (rewards && typeof rewards[key] !== 'undefined') ? rewards[key] : '';
            // store original reward on the cell so we can restore it later
            td.dataset.pos = key;
            td.dataset.orig = (v === '' || v === null) ? '' : Number(v).toFixed(2);
            td.textContent = td.dataset.orig;
            // color negative / positive values
            if (v < 0) td.style.color = '#ffd2d2';
            else if (v > 0) td.style.color = '#b7f5ff';
            tr.appendChild(td);
        }
        table.appendChild(tr);
    }
}

function clearVisitedRewards() {
    const table = document.getElementById('rewards-grid');
    if (!table) return;
    table.querySelectorAll('td').forEach(td => {
        td.classList.remove('reward-visited');
        td.style.background = '';
        const orig = td.dataset.orig;
        td.textContent = (typeof orig === 'undefined' || orig === '') ? '' : Number(orig).toFixed(2);
        delete td.dataset.visited;
    });
    // reset total reward display
    const totalEl = document.getElementById('total-reward');
    if (totalEl) totalEl.textContent = '0.00';
}

function updateRewardCell(i, j, stepReward, cumReward) {
    const table = document.getElementById('rewards-grid');
    if (!table) return;
    const key = `(${i},${j})`;
    const td = table.querySelector(`td[data-pos='${key}']`);
    if (!td) return;
    td.dataset.visited = '1';
    td.classList.add('reward-visited');
    if (typeof stepReward === 'undefined' || stepReward === null) stepReward = td.dataset.orig || '';
    // format: step (cum X)
    const stepText = (stepReward === '' || stepReward === null) ? '' : Number(stepReward).toFixed(2);
    const cumText = (typeof cumReward === 'undefined' || cumReward === null) ? '' : Number(cumReward).toFixed(2);
    td.textContent = (stepText === '' && cumText === '') ? '' : (cumText === '' ? stepText : `${stepText} (cum ${cumText})`);
    const numeric = Number(stepReward) || 0;
    if (numeric < 0) td.style.background = 'rgba(255,80,80,0.18)';
    else if (numeric > 0) td.style.background = 'rgba(80,200,255,0.12)';
    else td.style.background = 'rgba(200,200,200,0.06)';
}
let episodeChart = null;

function drawValueChart(values) {
    if (!values) return;
    const canvas = document.getElementById('valueChart');
    if (!canvas) return;
    const labels = Object.keys(values);
    const dataVals = labels.map(k => values[k]);
    if (valueChart) valueChart.destroy();
    valueChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'State values',
                data: dataVals,
                backgroundColor: 'rgba(54, 162, 235, 0.6)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { display: false },
                y: { beginAtZero: true }
            }
        }
    });
}

function clearEpisodeChart() {
    const canvas = document.getElementById('episodeChart');
    if (!canvas) return;
    if (episodeChart) {
        episodeChart.destroy();
        episodeChart = null;
    }
    // clear area
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function movingAverage(arr, windowSize) {
    const res = [];
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
        if (i >= windowSize) {
            sum -= arr[i - windowSize];
            res.push(sum / windowSize);
        } else {
            res.push(sum / (i + 1));
        }
    }
    return res;
}

function drawEpisodeChart(returns) {
    const canvas = document.getElementById('episodeChart');
    if (!canvas) return;
    if (!returns || returns.length === 0) { clearEpisodeChart(); return; }
    const labels = returns.map((_, i) => i + 1);
    const dataVals = returns.map(v => Number(v));
    const windowSize = Math.max(1, Math.floor(Math.min(100, returns.length / 10)));
    const ma = movingAverage(dataVals, windowSize);

    if (episodeChart) episodeChart.destroy();
    episodeChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'Return per episode', data: dataVals, borderColor: 'rgba(54,162,235,0.9)', pointRadius: 0, borderWidth: 1, fill: false },
                { label: `MA (w=${windowSize})`, data: ma, borderColor: 'rgba(99,102,241,0.9)', pointRadius: 0, borderWidth: 2, fill: false }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { x: { display: true, title: { display: true, text: 'Episode' } }, y: { display: true, title: { display: true, text: 'Return' } } }
        }
    });
}

function updateStats(data, algorithm) {
    const iterationsEl = document.getElementById('iterations');
    const timeEl = document.getElementById('time-taken');
    const nameEl = document.getElementById('algorithm-name');
    const rewardEl = document.getElementById('current-reward');
    if (nameEl) nameEl.textContent = (algorithm || '').replace('-', ' ').toUpperCase();
    if (iterationsEl && typeof data.iterations !== 'undefined') iterationsEl.textContent = data.iterations;
    if (timeEl && typeof data.time_taken !== 'undefined') timeEl.textContent = (data.time_taken).toFixed(4) + 's';
    if (rewardEl) rewardEl.textContent = '0.00';
    const aStat = document.getElementById('alpha-stat');
    const eStat = document.getElementById('epsilon-stat');
    if (aStat) aStat.textContent = document.getElementById('alpha') ? Number(document.getElementById('alpha').value).toFixed(2) : '-';
    if (eStat) eStat.textContent = document.getElementById('epsilon') ? Number(document.getElementById('epsilon').value).toFixed(2) : '-';
}
