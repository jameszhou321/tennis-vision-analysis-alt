"""action_annotator.py — Tennis Action Temporal Annotation Tool (Flask Web App)

Function: Cycles and extracts video clips from rallies_new into the annotation workspace by video source, supporting progress persistence.
Usage: Run `python action_annotator.py`, then navigate to http://localhost:5011 in your browser.
"""
import os
import json
import shutil
from flask import Flask, jsonify, request, send_file, render_template_string

app = Flask(__name__)

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_UTILS_DIR))

SOURCE_FOLDER = os.path.join(_PROJECT_DIR, "data", "rallies_new")
WORK_FOLDER   = os.path.join(_PROJECT_DIR, "data", "rallies_annotating")
PROGRESS_FILE = os.path.join(WORK_FOLDER, "_progress.json")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    os.makedirs(WORK_FOLDER, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def get_source_folders():
    if not os.path.exists(SOURCE_FOLDER):
        return []
    return sorted([
        d for d in os.listdir(SOURCE_FOLDER)
        if os.path.isdir(os.path.join(SOURCE_FOLDER, d))
    ])


def get_source_rallies(source_name):
    src_dir = os.path.join(SOURCE_FOLDER, source_name)
    return sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
        and os.path.exists(os.path.join(src_dir, d, "raw_clip.mp4"))
    ])


def get_work_rallies(source_name):
    work_src_dir = os.path.join(WORK_FOLDER, source_name)
    if not os.path.exists(work_src_dir):
        return []
    return sorted([
        d for d in os.listdir(work_src_dir)
        if os.path.isdir(os.path.join(work_src_dir, d))
        and os.path.exists(os.path.join(work_src_dir, d, "raw_clip.mp4"))
    ])


def get_deleted_set(source_name):
    progress = load_progress()
    return set(progress.get("deleted", {}).get(source_name, []))


def mark_deleted(source_name, rally_name):
    progress = load_progress()
    deleted = progress.setdefault("deleted", {})
    deleted.setdefault(source_name, [])
    if rally_name not in deleted[source_name]:
        deleted[source_name].append(rally_name)
    save_progress(progress)


def _transcode_to_h264(src_mp4, dst_mp4):
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', src_mp4,
        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
        '-c:a', 'aac', '-movflags', '+faststart',
        dst_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_next_rally(source_name):
    all_rallies = get_source_rallies(source_name)
    already = set(get_work_rallies(source_name)) | get_deleted_set(source_name)
    for rally_name in all_rallies:
        if rally_name not in already:
            src_path = os.path.join(SOURCE_FOLDER, source_name, rally_name)
            dst_path = os.path.join(WORK_FOLDER, source_name, rally_name)
            os.makedirs(os.path.join(WORK_FOLDER, source_name), exist_ok=True)
            shutil.copytree(src_path, dst_path)
            mp4 = os.path.join(dst_path, 'raw_clip.mp4')
            tmp = mp4 + '.tmp.mp4'
            try:
                _transcode_to_h264(mp4, tmp)
                os.replace(tmp, mp4)
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
            return rally_name
    return None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Tennis Action Temporal Annotator</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f5f7fa; display: flex; gap: 20px; height: 90vh; }
        .sidebar { flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); overflow-y: auto; min-width: 220px; }
        .main-panel { flex: 3; display: flex; flex-direction: column; gap: 20px; overflow-y: auto; }
        .video-container, .data-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        video { width: 100%; max-height: 50vh; background: #000; outline: none; border-radius: 4px; }
        .video-wrapper { position: relative; display: inline-block; width: 100%; }
        #actionOverlay { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.55); color: #fff; font-size: 18px; font-weight: bold; padding: 6px 14px; border-radius: 6px; pointer-events: none; display: none; }
        .source-group { margin-bottom: 12px; }
        .source-header { font-weight: bold; font-size: 12px; color: #555; padding: 5px 4px; border-bottom: 2px solid #e0e0e0; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; gap: 4px; }
        .source-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .btn-extract { font-size: 11px; padding: 2px 8px; background: #67c23a; color: white; border: none; border-radius: 3px; cursor: pointer; white-space: nowrap; }
        .btn-extract:hover { background: #4fa82a; }
        .playlist-item { padding: 7px 10px; border-bottom: 1px solid #eee; cursor: pointer; border-radius: 4px; font-size: 12px; display: flex; justify-content: space-between; align-items: center; }
        .playlist-item:hover { background-color: #f0f7ff; }
        .playlist-item.active { background-color: #e1f0ff; border-left: 4px solid #409eff; font-weight: bold; }
        .badge { font-size: 11px; padding: 2px 6px; border-radius: 10px; color: white; }
        .badge-has-json { background: #67c23a; }
        .badge-no-json { background: #909399; }
        .controls { margin-top: 15px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        button { padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; background: #409eff; color: white; }
        button:hover { background: #66b1ff; }
        .btn-speed { background: #e6a23c; }
        .btn-speed.active { background: #b57a24; font-weight: bold; }
        .btn-danger { background: #f56c6c; font-size: 12px; }
        .btn-delete-clip { background: #c0392b; font-size: 12px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; text-align: center; }
        th, td { border: 1px solid #ebeef5; padding: 8px; }
        th { background-color: #f8f9fa; }
        tr.selected-row td { background-color: #fff3cd; outline: 2px solid #e6a23c; }
    </style>
</head>
<body>
<div class="sidebar">
    <h3 style="margin-top:0;">Clip List</h3>
    <button onclick="loadPlaylist()" style="margin-bottom:12px; width:100%; background:#67c23a;">Refresh Workspace</button>
    <div id="playlist">Loading...</div>
</div>

<div class="main-panel">
    <div class="video-container">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <h2 id="currentVideoTitle" style="margin:0; color:#303133; font-size:16px;">Select a clip</h2>
            <button id="btnDeleteClip" class="btn-delete-clip" onclick="deleteClip()" style="display:none;">Delete Clip</button>
        </div>
        <div class="video-wrapper">
            <video id="videoPlayer" controls></video>
            <div id="actionOverlay"></div>
        </div>
        <div class="controls">
            <button onclick="setPoint('start')">Set Start Point [Q]</button>
            <button onclick="setPoint('end')">Set End & Pause [E]</button>
            <span style="margin-left:15px; font-size:14px;">
                Current Range: <span id="lblStart" style="color:#409eff;font-weight:bold;">0.000</span> →
                <span id="lblEnd" style="color:#f56c6c;font-weight:bold;">Undetermined</span>
            </span>
            <span style="margin-left:auto;">
                Playback Speed:
                <button id="btnSpeed05" class="btn-speed active" onclick="setSpeed(0.5)">0.5x</button>
                <button id="btnSpeed10" class="btn-speed" onclick="setSpeed(1.0)">1.0x</button>
            </span>
        </div>
        <div class="controls" style="margin-top:15px;">
            <button onclick="addAction('Forehand',1)" style="background:#67c23a;">Forehand [1]</button>
            <button onclick="addAction('Backhand',2)" style="background:#e6a23c;">Backhand [2]</button>
            <button onclick="addAction('Serve',3)" style="background:#f56c6c;">Serve [3]</button>
            <button onclick="addAction('Movement',4)" style="background:#909399;">Movement [4]</button>
            <button onclick="addAction('Idle',0)" style="background:#303133;">Idle [5]</button>
        </div>
    </div>
    <div class="data-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="margin:0;">Annotation Data</h3>
            <span id="saveStatus" style="color:#67c23a; font-weight:bold; display:none;">Autosaved</span>
        </div>
        <table id="annotationTable">
            <thead><tr><th>Start (s)</th><th>End (s)</th><th>Action</th><th>Operations</th></tr></thead>
            <tbody></tbody>
        </table>
    </div>
</div>
<script>
    let currentSource = null;
    let currentFolder = null;
    let annotations = [];
    let tempStart = 0.000;
    let tempEnd = null;
    let selectedRowIdx = null;
    const video = document.getElementById('videoPlayer');

    async function loadPlaylist() {
        const res = await fetch('/api/list');
        const data = await res.json();
        const playlistDiv = document.getElementById('playlist');
        playlistDiv.innerHTML = '';

        data.forEach(src => {
            const group = document.createElement('div');
            group.className = 'source-group';

            const header = document.createElement('div');
            header.className = 'source-header';
            const btn = document.createElement('button');
            btn.className = 'btn-extract';
            btn.textContent = 'Extract';
            btn.dataset.source = src.source_name;
            btn.addEventListener('click', () => extractOne(src.source_name));
            header.innerHTML = '<span class="source-name" title="' + src.source_name + '">' + src.source_name + '</span>';
            header.appendChild(btn);
            group.appendChild(header);

            src.rallies.forEach(r => {
                const item = document.createElement('div');
                item.className = 'playlist-item';
                const badge = r.has_json
                    ? '<span class="badge badge-has-json">Labeled</span>'
                    : '<span class="badge badge-no-json">Unlabeled</span>';
                item.innerHTML = '<span>' + r.rally_name + '</span>' + badge;
                item.onclick = () => loadVideo(src.source_name, r.rally_name, item);
                group.appendChild(item);
            });

            playlistDiv.appendChild(group);
        });
    }

    async function extractOne(sourceName) {
        const res = await fetch('/api/extract/' + encodeURIComponent(sourceName), {method:'POST'});
        const data = await res.json();
        if (data.extracted) {
            await loadPlaylist();
        } else {
            alert(sourceName + ' has been fully extracted.');
        }
    }

    async function loadVideo(sourceName, folderName, element) {
        document.querySelectorAll('.playlist-item').forEach(el => el.classList.remove('active'));
        if (element) element.classList.add('active');

        currentSource = sourceName;
        currentFolder = folderName;
        document.getElementById('currentVideoTitle').innerText = sourceName + ' / ' + folderName;
        document.getElementById('btnDeleteClip').style.display = 'inline-block';

        video.src = '/api/video/' + encodeURIComponent(sourceName) + '/' + encodeURIComponent(folderName);
        setSpeed(0.5);

        const res = await fetch('/api/json/' + encodeURIComponent(sourceName) + '/' + encodeURIComponent(folderName));
        const data = await res.json();
        annotations = data.annotations || [];
        renderTable();

        tempStart = annotations.length > 0 ? annotations[annotations.length - 1].end_time : 0.000;
        tempEnd = null;
        updateUI();
        selectedRowIdx = null;
        video.play().catch(e => console.log('Awaiting user interaction'));
    }

    async function deleteClip() {
        if (!currentFolder || !currentSource) return;
        if (!confirm('Are you sure you want to delete clip ' + currentFolder + '? Once deleted, the next clip from the same source will be automatically extracted.')) return;

        const res = await fetch('/api/delete/' + encodeURIComponent(currentSource) + '/' + encodeURIComponent(currentFolder), {method:'DELETE'});
        const data = await res.json();

        video.src = '';
        document.getElementById('currentVideoTitle').innerText = 'Select a clip';
        document.getElementById('btnDeleteClip').style.display = 'none';
        annotations = [];
        renderTable();
        currentFolder = null;
        currentSource = null;

        await loadPlaylist();

        if (data.next_extracted) {
            alert('Deleted. A new clip has been extracted from the same video source: ' + data.next_extracted);
        } else {
            alert('Deleted. No more clips available to extract from this video source.');
        }
    }

    function setSpeed(speed) {
        video.playbackRate = speed;
        document.getElementById('btnSpeed05').classList.toggle('active', speed === 0.5);
        document.getElementById('btnSpeed10').classList.toggle('active', speed === 1.0);
    }

    function setPoint(type) {
        if (!currentFolder) return;
        const time = parseFloat(video.currentTime.toFixed(3));
        if (type === 'start') {
            tempStart = time;
        } else {
            video.pause();
            tempEnd = time;
        }
        updateUI();
    }

    function addAction(name, id) {
        if (tempStart === null || tempEnd === null || tempStart >= tempEnd) return;
        annotations.push({ start_time: tempStart, end_time: tempEnd, action_name: name, action_id: id });
        annotations.sort((a, b) => a.start_time - b.start_time);
        tempStart = tempEnd;
        tempEnd = null;
        updateUI();
        renderTable();
        saveToServer();
    }

    window.deleteAction = function(idx) {
        annotations.splice(idx, 1);
        if (idx === annotations.length && annotations.length > 0) {
            tempStart = annotations[annotations.length - 1].end_time;
            updateUI();
        } else if (annotations.length === 0) {
            tempStart = 0.000;
            updateUI();
        }
        renderTable();
        saveToServer();
    };

    function updateUI() {
        document.getElementById('lblStart').innerText = tempStart !== null ? tempStart.toFixed(3) : 'Undetermined';
        document.getElementById('lblEnd').innerText = tempEnd !== null ? tempEnd.toFixed(3) : 'Undetermined';
    }

    function renderTable() {
        const tbody = document.querySelector('#annotationTable tbody');
        tbody.innerHTML = '';
        annotations.forEach((item, idx) => {
            const tr = document.createElement('tr');
            if (idx === selectedRowIdx) tr.classList.add('selected-row');
            tr.style.cursor = 'pointer';
            tr.innerHTML = '<td>' + item.start_time.toFixed(3) + '</td><td>' + item.end_time.toFixed(3) + '</td><td>' + item.action_name + '</td>' +
                '<td><button class="btn-danger" onclick="deleteAction(' + idx + ')">Delete</button></td>';
            tr.onclick = (e) => {
                if (e.target.tagName === 'BUTTON') return;
                selectedRowIdx = (selectedRowIdx === idx) ? null : idx;
                renderTable();
            };
            tbody.appendChild(tr);
        });
    }

    async function saveToServer() {
        if (!currentFolder || !currentSource) return;
        const res = await fetch('/api/save/' + encodeURIComponent(currentSource) + '/' + encodeURIComponent(currentFolder), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(annotations)
        });
        if (res.ok) {
            const status = document.getElementById('saveStatus');
            status.style.display = 'block';
            setTimeout(() => status.style.display = 'none', 1500);
            loadPlaylist();
        }
    }

    const ACTION_MAP = { '1': ['Forehand',1], '2': ['Backhand',2], '3': ['Serve',3], '4': ['Movement',4], '5': ['Idle',0] };
    document.addEventListener('keydown', function(e) {
        if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
        const k = e.key.toLowerCase();
        if (k === 'q') { setPoint('start'); e.preventDefault(); }
        if (k === 'e') { setPoint('end'); e.preventDefault(); }
        if (ACTION_MAP[k]) {
            const [name, id] = ACTION_MAP[k];
            if (selectedRowIdx !== null) {
                annotations[selectedRowIdx].action_name = name;
                annotations[selectedRowIdx].action_id = id;
                selectedRowIdx = null;
                renderTable();
                saveToServer();
            } else {
                addAction(name, id);
            }
        }
        if (k === ' ') { e.preventDefault(); video.paused ? video.play() : video.pause(); }
    });

    const ACTION_COLORS = { 'Forehand': '#67c23a', 'Backhand': '#e6a23c', 'Serve': '#f56c6c', 'Movement': '#909399', 'Idle': '#303133' };
    video.addEventListener('timeupdate', function() {
        const t = video.currentTime;
        const overlay = document.getElementById('actionOverlay');
        const hit = annotations.find(a => t >= a.start_time && t < a.end_time);
        if (hit) {
            overlay.innerText = hit.action_name;
            overlay.style.background = (ACTION_COLORS[hit.action_name] || '#555') + 'cc';
            overlay.style.display = 'block';
        } else {
            overlay.style.display = 'none';
        }
    });

    loadPlaylist();
</script>
</body>
</html>
"""


# ─── Flask Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/list')
def get_video_list():
    result = []
    for source_name in get_source_folders():
        rallies = []
        for rally_name in get_work_rallies(source_name):
            work_dir = os.path.join(WORK_FOLDER, source_name, rally_name)
            json_path = os.path.join(work_dir, "annotations.json")
            rallies.append({
                "rally_name": rally_name,
                "has_json": os.path.exists(json_path) and os.path.getsize(json_path) > 5
            })
        result.append({"source_name": source_name, "rallies": rallies})
    return jsonify(result)


@app.route('/api/extract/<path:source_name>', methods=['POST'])
def extract_one(source_name):
    extracted = extract_next_rally(source_name)
    return jsonify({"extracted": extracted})


@app.route('/api/video/<path:source_name>/<path:folder_name>')
def serve_video(source_name, folder_name):
    file_path = os.path.join(WORK_FOLDER, source_name, folder_name, "raw_clip.mp4")
    return send_file(file_path, conditional=True)


@app.route('/api/json/<path:source_name>/<path:folder_name>')
def get_json(source_name, folder_name):
    json_path = os.path.join(WORK_FOLDER, source_name, folder_name, "annotations.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                return jsonify({"annotations": json.load(f)})
            except Exception:
                pass
    return jsonify({"annotations": []})


@app.route('/api/save/<path:source_name>/<path:folder_name>', methods=['POST'])
def save_json(source_name, folder_name):
    json_path = os.path.join(WORK_FOLDER, source_name, folder_name, "annotations.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(request.json, f, ensure_ascii=False, indent=4)
    return jsonify({"status": "success"})


@app.route('/api/delete/<path:source_name>/<path:folder_name>', methods=['DELETE'])
def delete_clip(source_name, folder_name):
    clip_dir = os.path.join(WORK_FOLDER, source_name, folder_name)
    if os.path.exists(clip_dir):
        import shutil as _shutil
        _shutil.rmtree(clip_dir)
    mark_deleted(source_name, folder_name)
    next_rally = extract_next_rally(source_name)
    return jsonify({"next_extracted": next_rally})


if __name__ == '__main__':
    print("Service started successfully. Please access: https://127.0.0.1:5011")
    app.run(host='0.0.0.0', port=5011, debug=False, threaded=True, ssl_context='adhoc')