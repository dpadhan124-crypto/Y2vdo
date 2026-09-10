import os
import uuid
import time
import math
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import pysrt
from gtts import gTTS
from pydub import AudioSegment
from sqlalchemy import create_engine, Column, String, LargeBinary, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./voiceover.db") # Defaulted to SQLite for local ease

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConversionTask(Base):
    __tablename__ = "conversion_tasks"
    task_id = Column(String, primary_key=True, index=True)
    original_filename = Column(String)
    language = Column(String, default="hi")
    status = Column(String, default="processing")
    progress = Column(String, default="Queued...")
    merged_audio_data = Column(LargeBinary, nullable=True)
    created_at = Column(Integer, default=lambda: int(time.time()))

class SubtitleLine(Base):
    __tablename__ = "subtitle_lines"
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, index=True)
    line_index = Column(Integer)
    text = Column(Text)
    start_ms = Column(Integer)
    end_ms = Column(Integer)
    audio_data = Column(LargeBinary, nullable=True)
    status = Column(String, default="pending")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Ai Voiceover & Vocal Remover")

def srt_time_to_ms(t):
    return (t.hours * 3600 + t.minutes * 60 + t.seconds) * 1000 + t.milliseconds

def process_srt_in_background(task_id: str, input_path: str, language: str):
    db = SessionLocal()
    temp_files = [input_path]
    try:
        try:
            subs = pysrt.open(input_path, encoding='utf-8')
        except Exception:
            subs = pysrt.open(input_path, encoding='latin-1')

        if not subs:
            update_task_status(db, task_id, "failed", "Invalid or empty SRT file")
            return

        total_subs = len(subs)
        for idx, sub in enumerate(subs):
            clean_text = sub.text.replace('\n', ' ').strip()
            db_line = SubtitleLine(
                task_id=task_id, line_index=idx, text=clean_text,
                start_ms=srt_time_to_ms(sub.start), end_ms=srt_time_to_ms(sub.end),
                status="pending"
            )
            db.add(db_line)
        db.commit()

        # Generate Audio Chunks
        lines = db.query(SubtitleLine).filter(SubtitleLine.task_id == task_id).order_by(SubtitleLine.line_index).all()
        for i, line in enumerate(lines):
            if not line.text:
                continue
            
            update_task_status(db, task_id, "processing", f"Generating audio: {int(((i+1)/total_subs)*100)}%")
            temp_chunk = f"chunk_{task_id}_{i}.mp3"
            temp_files.append(temp_chunk)
            
            try:
                tts = gTTS(text=line.text, lang=language, slow=False)
                tts.save(temp_chunk)
                with open(temp_chunk, "rb") as f:
                    line.audio_data = f.read()
                db.commit()
            except Exception:
                time.sleep(2) # rate limit backoff
                try:
                    tts = gTTS(text=line.text, lang=language, slow=False)
                    tts.save(temp_chunk)
                    with open(temp_chunk, "rb") as f:
                        line.audio_data = f.read()
                    db.commit()
                except Exception:
                    pass

        # Stitching Phase
        final_timeline = AudioSegment.silent(duration=0)
        current_cursor = 0
        
        for i, line in enumerate(lines):
            if i % max(1, (len(lines)//10)) == 0:
                pct = int((i / len(lines)) * 100)
                update_task_status(db, task_id, "processing", f"Stitching timeline: {pct}%")

            target_duration = max(line.end_ms - line.start_ms, 500)
            
            if line.start_ms > current_cursor:
                final_timeline += AudioSegment.silent(duration=line.start_ms - current_cursor)
                current_cursor = line.start_ms

            if not line.audio_data:
                final_timeline += AudioSegment.silent(duration=target_duration)
                current_cursor += target_duration
                continue

            temp_audio = f"temp_line_{task_id}_{i}.mp3"
            temp_files.append(temp_audio)
            with open(temp_audio, "wb") as f:
                f.write(line.audio_data)
            
            segment = AudioSegment.from_mp3(temp_audio)
            audio_len = len(segment)
            
            # Speed boundaries: 0.8x to 1.2x
            if audio_len != target_duration:
                speed_factor = audio_len / target_duration
                speed_factor = max(0.8, min(1.2, speed_factor))
                new_framerate = int(segment.frame_rate * speed_factor)
                segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_framerate}).set_frame_rate(44100)

            final_timeline += segment
            current_cursor += len(segment)

        # Export Blob
        update_task_status(db, task_id, "processing", "Stitching timeline: 100%")
        merged_output = f"final_{task_id}.mp3"
        temp_files.append(merged_output)
        
        final_timeline.export(merged_output, format="mp3")
        with open(merged_output, "rb") as f:
            merged_bytes = f.read()

        task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
        if task:
            task.status = "completed"
            task.progress = "Completed successfully!"
            task.merged_audio_data = merged_bytes
            db.commit()

        # Cleanup DB
        db.query(SubtitleLine).filter(SubtitleLine.task_id == task_id).delete()
        db.commit()

    except Exception as e:
        update_task_status(db, task_id, "failed", f"Error: {str(e)}")
    finally:
        db.close()
        # Force Delete ALL temporary files
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

def update_task_status(db, task_id, status, progress):
    task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
    if task:
        task.status = status
        task.progress = progress
        db.commit()

def process_vocal_removal(task_id: str, input_path: str, filename: str):
    db = SessionLocal()
    temp_files = [input_path]
    try:
        update_task_status(db, task_id, "processing", "Processing phase cancellation...")
        sound = AudioSegment.from_file(input_path)
        
        # Basic Vocal Remover: Phase Cancellation (Subtract Left from Right)
        if sound.channels == 2:
            left = sound.split_to_mono()[0]
            right = sound.split_to_mono()[1]
            vocal_less = left.overlay(right.invert_phase())
        else:
            vocal_less = sound # Mono files can't use this trick
            
        output_path = f"novocal_{task_id}.mp3"
        temp_files.append(output_path)
        vocal_less.export(output_path, format="mp3")
        
        with open(output_path, "rb") as f:
            merged_bytes = f.read()
            
        task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
        task.status = "completed"
        task.progress = "Vocal removal complete!"
        task.merged_audio_data = merged_bytes
        db.commit()
    except Exception as e:
        update_task_status(db, task_id, "failed", f"Error: {str(e)}")
    finally:
        db.close()
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)


# --- HTML UI ---
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Ai Voiceover</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .view-section { display: none; }
        .view-section.active { display: block; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
</head>
<body class="bg-slate-50 text-slate-900 h-screen flex flex-col overflow-hidden">
    
    <!-- Navbar -->
    <header class="bg-indigo-600 text-white shadow-md flex justify-between items-center p-4 shrink-0">
        <h1 class="text-xl font-bold tracking-wide"><i class="fa-solid fa-microphone-lines mr-2"></i>Ai Voiceover</h1>
        <button onclick="toggleMenu()" class="text-2xl focus:outline-none"><i class="fa-solid fa-bars"></i></button>
    </header>

    <!-- Side Menu Overlay -->
    <div id="sideMenu" class="fixed inset-0 bg-black/50 z-40 hidden" onclick="toggleMenu()"></div>
    <div id="menuDrawer" class="fixed top-0 right-0 h-full w-64 bg-white shadow-2xl z-50 transform translate-x-full transition-transform duration-300 flex flex-col">
        <div class="p-4 border-b flex justify-between items-center">
            <span class="font-bold text-lg text-indigo-600">Menu</span>
            <button onclick="toggleMenu()" class="text-xl"><i class="fa-solid fa-xmark"></i></button>
        </div>
        <nav class="flex-1 p-4 space-y-2">
            <button onclick="switchView('home')" class="w-full text-left p-3 rounded-lg hover:bg-slate-100 font-medium"><i class="fa-solid fa-file-audio w-6"></i> SRT to TTS</button>
            <button onclick="switchView('vocals')" class="w-full text-left p-3 rounded-lg hover:bg-slate-100 font-medium"><i class="fa-solid fa-music w-6"></i> Remove Vocals</button>
            <button onclick="switchView('settings')" class="w-full text-left p-3 rounded-lg hover:bg-slate-100 font-medium"><i class="fa-solid fa-gear w-6"></i> Settings</button>
            <button onclick="switchView('dashboard')" class="w-full text-left p-3 rounded-lg hover:bg-slate-100 font-medium" onclick="loadDashboard()"><i class="fa-solid fa-database w-6"></i> Dashboard</button>
        </nav>
    </div>

    <!-- Main Content Area -->
    <main class="flex-1 overflow-y-auto p-4 sm:p-6 bg-slate-50">
        
        <!-- View: SRT to TTS -->
        <div id="view-home" class="view-section active max-w-md mx-auto">
            <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                <h2 class="text-xl font-bold mb-2">Timestamp-Synced TTS</h2>
                <p class="text-sm text-slate-500 mb-6">Upload SRT. Audio stretches securely in background (0.8x-1.2x bounds).</p>
                
                <form id="uploadForm" class="space-y-4">
                    <input type="file" id="srtFile" name="file" accept=".srt" required class="w-full text-sm border p-2 rounded-xl bg-slate-50"/>
                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 rounded-xl shadow-sm transition">Generate MP3</button>
                </form>

                <div id="statusBox" class="hidden mt-6 space-y-3 border-t pt-4">
                    <p id="progressText" class="text-sm font-semibold text-indigo-600 text-center">Initializing...</p>
                    <div class="w-full bg-slate-200 rounded-full h-2.5 mt-2 overflow-hidden hidden" id="progressContainer">
                        <div id="progressBar" class="bg-indigo-600 h-2.5 rounded-full transition-all duration-300" style="width: 0%"></div>
                    </div>
                    <a id="downloadBtn" class="hidden block w-full bg-emerald-500 text-white text-center font-bold py-3 rounded-xl">Download MP3</a>
                </div>
            </div>
        </div>

        <!-- View: Vocal Remover -->
        <div id="view-vocals" class="view-section max-w-md mx-auto">
            <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                <h2 class="text-xl font-bold mb-2">Vocal Remover</h2>
                <p class="text-sm text-slate-500 mb-6">Extract instrumental track using phase-cancellation.</p>
                <form id="vocalForm" class="space-y-4">
                    <input type="file" id="audioFile" name="file" accept="audio/*" required class="w-full text-sm border p-2 rounded-xl bg-slate-50"/>
                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 rounded-xl shadow-sm">Remove Vocals</button>
                </form>
                <div id="vocalStatusBox" class="hidden mt-6 space-y-3 border-t pt-4">
                    <p id="vocalProgressText" class="text-sm font-semibold text-indigo-600 text-center">Processing...</p>
                    <a id="vocalDownloadBtn" class="hidden block w-full bg-emerald-500 text-white text-center font-bold py-3 rounded-xl">Download Instrumental</a>
                </div>
            </div>
        </div>

        <!-- View: Settings -->
        <div id="view-settings" class="view-section max-w-md mx-auto">
            <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                <h2 class="text-xl font-bold mb-4">Voice Settings</h2>
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Target Language</label>
                        <select id="langSelect" class="w-full border-slate-300 rounded-xl p-3 bg-slate-50 border outline-none focus:border-indigo-500">
                            <option value="hi">Hindi</option>
                            <option value="bn">Bengali</option>
                            <option value="ta">Tamil</option>
                            <option value="te">Telugu</option>
                            <option value="mr">Marathi</option>
                            <option value="gu">Gujarati</option>
                            <option value="kn">Kannada</option>
                            <option value="ml">Malayalam</option>
                            <option value="en">English (India)</option>
                        </select>
                    </div>
                    <div class="bg-blue-50 text-blue-800 p-3 rounded-lg text-xs">
                        <i class="fa-solid fa-circle-info"></i> Note: Speed limits are locked at 0.8x (min) and 1.2x (max) to prevent audio distortion.
                    </div>
                </div>
            </div>
        </div>

        <!-- View: Dashboard -->
        <div id="view-dashboard" class="view-section max-w-md mx-auto">
            <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                <h2 class="text-xl font-bold mb-4">Storage & History</h2>
                
                <div class="mb-6">
                    <div class="flex justify-between text-xs font-bold text-slate-500 mb-1">
                        <span>Database Used</span>
                        <span id="storageLabel">0 MB / 500 MB</span>
                    </div>
                    <div class="w-full bg-slate-200 rounded-full h-3">
                        <div id="storageBar" class="bg-indigo-600 h-3 rounded-full" style="width: 0%"></div>
                    </div>
                </div>

                <div class="flex justify-between items-center mb-3">
                    <h3 class="font-bold text-slate-700">Saved Files</h3>
                    <button onclick="clearDatabase()" class="text-xs bg-red-100 text-red-600 px-3 py-1.5 rounded-lg font-bold hover:bg-red-200">Clear All</button>
                </div>
                
                <ul id="fileList" class="space-y-2 max-h-64 overflow-y-auto bg-slate-50 p-2 rounded-xl border border-slate-100">
                    <li class="text-sm text-slate-500 text-center py-4">No files found.</li>
                </ul>
            </div>
        </div>
    </main>

    <script>
        // UI Navigation Logic
        function toggleMenu() {
            const drawer = document.getElementById('menuDrawer');
            const overlay = document.getElementById('sideMenu');
            if (drawer.classList.contains('translate-x-full')) {
                drawer.classList.remove('translate-x-full');
                overlay.classList.remove('hidden');
            } else {
                drawer.classList.add('translate-x-full');
                overlay.classList.add('hidden');
            }
        }

        function switchView(viewId) {
            document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
            document.getElementById('view-' + viewId).classList.add('active');
            toggleMenu();
            if(viewId === 'dashboard') loadDashboard();
        }

        // Settings Persistance
        const langSelect = document.getElementById('langSelect');
        langSelect.value = localStorage.getItem('ttsLang') || 'hi';
        langSelect.addEventListener('change', (e) => localStorage.setItem('ttsLang', e.target.value));

        // Background Polling Logic
        function startPolling(taskId, textElement, progressContainer, progressBar, dlBtn, endpoint) {
            const interval = setInterval(async () => {
                const res = await fetch(`/status/${taskId}`);
                const data = await res.json();
                
                textElement.textContent = data.progress;
                
                // Parse percentage for progress bar if stitching
                if(progressBar && data.progress.includes('%')) {
                    progressContainer.classList.remove('hidden');
                    const pct = data.progress.match(/\d+/)[0];
                    progressBar.style.width = pct + '%';
                }
                
                if(data.status === 'completed') {
                    clearInterval(interval);
                    if(progressContainer) progressContainer.classList.add('hidden');
                    textElement.textContent = "Ready!";
                    dlBtn.href = `/download/${taskId}`;
                    dlBtn.classList.remove('hidden');
                    if(endpoint === 'dashboard') loadDashboard();
                } else if(data.status === 'failed') {
                    clearInterval(interval);
                    if(progressContainer) progressContainer.classList.add('hidden');
                }
            }, 2000);
        }

        // SRT Upload
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            formData.append('language', langSelect.value);
            
            const sb = document.getElementById('statusBox');
            const pt = document.getElementById('progressText');
            const pCont = document.getElementById('progressContainer');
            const pBar = document.getElementById('progressBar');
            const db = document.getElementById('downloadBtn');
            
            sb.classList.remove('hidden'); db.classList.add('hidden'); pCont.classList.add('hidden');
            pt.textContent = "Uploading...";

            const res = await fetch('/convert/', { method: 'POST', body: formData });
            const data = await res.json();
            if(!res.ok) { pt.textContent = data.detail; return; }
            
            startPolling(data.task_id, pt, pCont, pBar, db, 'dashboard');
        });

        // Vocal Remover Upload
        document.getElementById('vocalForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const sb = document.getElementById('vocalStatusBox');
            const pt = document.getElementById('vocalProgressText');
            const db = document.getElementById('vocalDownloadBtn');
            
            sb.classList.remove('hidden'); db.classList.add('hidden');
            pt.textContent = "Uploading audio...";

            const res = await fetch('/remove_vocals/', { method: 'POST', body: formData });
            const data = await res.json();
            if(!res.ok) { pt.textContent = data.detail; return; }
            
            startPolling(data.task_id, pt, null, null, db, 'dashboard');
        });

        // Dashboard Data
        async function loadDashboard() {
            const res = await fetch('/api/dashboard');
            const data = await res.json();
            
            // Render Storage Bar (Max assumed 500MB for UI demo purposes)
            const mbUsed = (data.storage_bytes / (1024*1024)).toFixed(2);
            document.getElementById('storageLabel').innerText = `${mbUsed} MB / 500 MB`;
            const pct = Math.min((mbUsed / 500) * 100, 100);
            document.getElementById('storageBar').style.width = pct + '%';

            // Render List
            const list = document.getElementById('fileList');
            list.innerHTML = '';
            if(data.files.length === 0) {
                list.innerHTML = '<li class="text-sm text-slate-500 text-center py-4">No files found.</li>';
                return;
            }
            data.files.forEach(f => {
                const li = document.createElement('li');
                li.className = "flex justify-between items-center p-3 bg-white border border-slate-200 rounded-lg shadow-sm";
                li.innerHTML = `
                    <div class="overflow-hidden">
                        <p class="text-sm font-semibold truncate w-40 text-slate-700">${f.filename}</p>
                        <p class="text-xs text-slate-400">${new Date(f.created_at * 1000).toLocaleDateString()}</p>
                    </div>
                    <a href="/download/${f.task_id}" class="bg-indigo-100 text-indigo-700 px-3 py-1.5 rounded-lg text-xs font-bold hover:bg-indigo-200"><i class="fa-solid fa-download"></i></a>
                `;
                list.appendChild(li);
            });
        }

        async function clearDatabase() {
            if(!confirm("Are you sure you want to delete all saved files?")) return;
            await fetch('/api/clear', { method: 'POST' });
            loadDashboard();
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTMLResponse(content=HTML_UI)

@app.post("/convert/")
async def convert_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...), language: str = Form("hi")):
    if not file.filename.lower().endswith('.srt'):
        raise HTTPException(status_code=400, detail="Only .srt files accepted.")
    
    task_id = str(uuid.uuid4())
    input_path = f"temp_{task_id}.srt"
    
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)
        
    db = SessionLocal()
    new_task = ConversionTask(
        task_id=task_id, 
        original_filename=file.filename,
        language=language,
        status="processing", 
        progress="Queued in database..."
    )
    db.add(new_task)
    db.commit()
    db.close()
    
    background_tasks.add_task(process_srt_in_background, task_id, input_path, language)
    return {"task_id": task_id}

@app.post("/remove_vocals/")
async def remove_vocals_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    input_path = f"temp_vocal_{task_id}.mp3"
    
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    db = SessionLocal()
    new_task = ConversionTask(
        task_id=task_id,
        original_filename=file.filename,
        language="instrumental",
        status="processing",
        progress="Uploading to worker..."
    )
    db.add(new_task)
    db.commit()
    db.close()

    background_tasks.add_task(process_vocal_removal, task_id, input_path, file.filename)
    return {"task_id": task_id}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    db = SessionLocal()
    task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
    db.close()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": task.status, "progress": task.progress}

@app.get("/download/{task_id}")
def download_file(task_id: str):
    db = SessionLocal()
    task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
    db.close()
    
    if not task or not task.merged_audio_data:
        raise HTTPException(status_code=404, detail="File not ready or missing.")
    
    output_filename = f"dl_{task_id}.mp3"
    with open(output_filename, "wb") as f:
        f.write(task.merged_audio_data)
    
    # Calculate output name based on original extension
    ext_stripped = task.original_filename.rsplit('.', 1)[0] if task.original_filename else "synced_audio"
    final_name = f"{ext_stripped}.mp3"
        
    return FileResponse(output_filename, media_type="audio/mpeg", filename=final_name, background=BackgroundTasks().add_task(os.remove, output_filename))

@app.get("/api/dashboard")
def get_dashboard_data():
    db = SessionLocal()
    tasks = db.query(ConversionTask).filter(ConversionTask.status == "completed").order_by(ConversionTask.created_at.desc()).all()
    
    total_bytes = sum([len(t.merged_audio_data) for t in tasks if t.merged_audio_data])
    
    files = [{
        "task_id": t.task_id,
        "filename": t.original_filename.rsplit('.', 1)[0] + ".mp3" if t.original_filename else f"{t.task_id}.mp3",
        "created_at": t.created_at
    } for t in tasks]
    
    db.close()
    return {"storage_bytes": total_bytes, "files": files}

@app.post("/api/clear")
def clear_db():
    db = SessionLocal()
    db.query(ConversionTask).delete()
    db.query(SubtitleLine).delete()
    db.commit()
    db.close()
    return {"status": "cleared"}
