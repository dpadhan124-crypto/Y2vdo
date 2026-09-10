import os
import uuid
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, FileResponse
import pysrt
import edge_tts
from pydub import AudioSegment
from sqlalchemy import create_engine, Column, String, LargeBinary, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@host:port/dbname")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConversionTask(Base):
    __tablename__ = "conversion_tasks"
    task_id = Column(String, primary_key=True, index=True)
    filename = Column(String, default="audio.mp3")
    status = Column(String, default="processing")
    progress = Column(String, default="Queued...")
    merged_audio_data = Column(LargeBinary, nullable=True)

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

app = FastAPI(title="Precise Timestamp SRT to Hindi Neural TTS")

def srt_time_to_ms(t):
    return (t.hours * 3600 + t.minutes * 60 + t.seconds) * 1000 + t.milliseconds

def update_task_status(db, task_id, status, progress):
    task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
    if task:
        task.status = status
        task.progress = progress
        db.commit()

async def process_srt_in_background(task_id: str, input_path: str, voice: str):
    db = SessionLocal()
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
            start_ms = srt_time_to_ms(sub.start)
            end_ms = srt_time_to_ms(sub.end)
            clean_text = sub.text.replace('\n', ' ').strip()
            
            db_line = SubtitleLine(
                task_id=task_id,
                line_index=idx,
                text=clean_text,
                start_ms=start_ms,
                end_ms=end_ms,
                status="pending"
            )
            db.add(db_line)
        db.commit()

        update_task_status(db, task_id, "processing", f"Parsed {total_subs} lines. Generating neural chunks...")

        lines = db.query(SubtitleLine).filter(SubtitleLine.task_id == task_id).order_by(SubtitleLine.line_index).all()
        
        for i, line in enumerate(lines):
            if not line.text:
                line.status = "done"
                db.commit()
                continue
            
            update_task_status(db, task_id, "processing", f"Generating audio: Line {i+1} of {total_subs}")
            
            temp_chunk = f"chunk_{task_id}_{i}.mp3"
            try:
                communicate = edge_tts.Communicate(line.text, voice)
                await communicate.save(temp_chunk)
                
                with open(temp_chunk, "rb") as f:
                    line.audio_data = f.read()
                line.status = "done"
                db.commit()
                
                if os.path.exists(temp_chunk):
                    os.remove(temp_chunk)
            except Exception:
                await asyncio.sleep(2)
                try:
                    communicate = edge_tts.Communicate(line.text, voice)
                    await communicate.save(temp_chunk)
                    with open(temp_chunk, "rb") as f:
                        line.audio_data = f.read()
                    line.status = "done"
                    db.commit()
                    if os.path.exists(temp_chunk):
                        os.remove(temp_chunk)
                except Exception:
                    pass
            
            await asyncio.sleep(0.2)

        update_task_status(db, task_id, "processing", "Stitching timeline matching exact SRT timestamps...")
        
        final_timeline = AudioSegment.silent(duration=0)
        current_timeline_cursor = 0

        for line in lines:
            target_duration = max(line.end_ms - line.start_ms, 500)
            
            if line.start_ms > current_timeline_cursor:
                gap_duration = line.start_ms - current_timeline_cursor
                final_timeline += AudioSegment.silent(duration=gap_duration)
                current_timeline_cursor = line.start_ms

            if not line.audio_data:
                final_timeline += AudioSegment.silent(duration=target_duration)
                current_timeline_cursor += target_duration
                continue

            temp_audio_file = "temp_line.mp3"
            with open(temp_audio_file, "wb") as f:
                f.write(line.audio_data)
            
            segment = AudioSegment.from_mp3(temp_audio_file)
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)

            audio_len = len(segment)
            
            if audio_len > target_duration:
                speed_factor = audio_len / target_duration
                if speed_factor > 1.2:
                    speed_factor = 1.2
                new_framerate = int(segment.frame_rate * speed_factor)
                segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_framerate}).set_frame_rate(44100)
            elif audio_len < target_duration:
                speed_factor = audio_len / target_duration
                if speed_factor < 0.8:
                    speed_factor = 0.8
                new_framerate = int(segment.frame_rate * speed_factor)
                segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_framerate}).set_frame_rate(44100)

            final_timeline += segment
            current_timeline_cursor += len(segment)

        merged_output_path = f"final_merged_{task_id}.mp3"
        final_timeline.export(merged_output_path, format="mp3")

        with open(merged_output_path, "rb") as f:
            merged_bytes = f.read()

        task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
        if task:
            task.status = "completed"
            task.progress = "Completed successfully!"
            task.merged_audio_data = merged_bytes
            db.commit()

        if os.path.exists(merged_output_path):
            os.remove(merged_output_path)

        db.query(SubtitleLine).filter(SubtitleLine.task_id == task_id).delete()
        db.commit()

    except Exception as e:
        update_task_status(db, task_id, "failed", f"Error: {str(e)}")
    finally:
        db.close()
        if os.path.exists(input_path):
            os.remove(input_path)

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hindi Neural TTS & SRT Sync Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col">
    <!-- Navigation Bar -->
    <nav class="bg-slate-900 border-b border-slate-800 px-4 sm:px-6 py-3 flex justify-between items-center">
        <span class="font-bold text-base sm:text-lg text-indigo-400">HindiVoice Studio</span>
        <div class="flex gap-2">
            <button onclick="switchPage('converter')" id="navConverter" class="px-3 py-1.5 sm:px-4 sm:py-2 rounded-xl text-xs sm:text-sm font-medium bg-indigo-600 text-white transition">Converter</button>
            <button onclick="switchPage('dashboard')" id="navDashboard" class="px-3 py-1.5 sm:px-4 sm:py-2 rounded-xl text-xs sm:text-sm font-medium bg-slate-800 text-slate-400 hover:text-white transition">Dashboard</button>
        </div>
    </nav>

    <main class="flex-grow flex items-center justify-center p-3 sm:p-4 my-auto w-full max-w-xl mx-auto">
        <!-- Converter Page -->
        <div id="converterPage" class="w-full bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-4 sm:p-6">
            <div class="flex gap-2 mb-6">
                <button id="tabSrt" onclick="switchTab('srt')" class="flex-1 py-2 text-xs sm:text-sm rounded-xl font-medium bg-indigo-600 text-white transition">SRT to MP3</button>
                <button id="tabText" onclick="switchTab('text')" class="flex-1 py-2 text-xs sm:text-sm rounded-xl font-medium bg-slate-800 text-slate-400 transition">TEXT to MP3</button>
            </div>

            <!-- SRT Form -->
            <div id="srtSection">
                <form id="uploadForm" class="space-y-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Select Voice Speaker</label>
                        <select name="voice" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-600">
                            <option value="hi-IN-MadhurNeural">Madhur (Male)</option>
                            <option value="hi-IN-SwaraNeural">Swara (Female)</option>
                            <option value="hi-IN-AaravNeural">Aarav (Male)</option>
                            <option value="hi-IN-AnanyaNeural">Ananya (Female)</option>
                        </select>
                    </div>
                    <input type="file" id="srtFile" name="file" accept=".srt" required class="w-full text-xs sm:text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs sm:file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer"/>
                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 rounded-xl transition text-sm">Start Processing</button>
                </form>
            </div>

            <!-- Text Form -->
            <div id="textSection" class="hidden">
                <form id="textForm" class="space-y-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Select Voice Speaker</label>
                        <select name="voice" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-600">
                            <option value="hi-IN-MadhurNeural">Madhur (Male)</option>
                            <option value="hi-IN-SwaraNeural">Swara (Female)</option>
                            <option value="hi-IN-AaravNeural">Aarav (Male)</option>
                            <option value="hi-IN-AnanyaNeural">Ananya (Female)</option>
                        </select>
                    </div>
                    <div>
                        <textarea name="text_content" rows="4" placeholder="Enter Hindi text here..." required class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-200 focus:outline-none focus:border-indigo-600"></textarea>
                    </div>
                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 rounded-xl transition text-sm">Generate Audio</button>
                </form>
            </div>

            <!-- Progress & Status Box -->
            <div id="statusBox" class="hidden mt-6 space-y-3 border-t border-slate-800 pt-4">
                <p id="progressText" class="text-xs sm:text-sm text-indigo-400 font-medium text-center animate-pulse">Initializing...</p>
                <div class="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
                    <div id="progressBar" class="bg-indigo-600 h-2.5 w-0 transition-all duration-500"></div>
                </div>
                <a id="downloadBtn" class="hidden block w-full bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-2.5 rounded-xl transition text-center text-sm">Download MP3</a>
            </div>
        </div>

        <!-- Dashboard Page -->
        <div id="dashboardPage" class="hidden w-full bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-4 sm:p-6">
            <h2 class="text-base sm:text-lg font-bold mb-4">File Processing History</h2>
            <div id="taskList" class="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
                <p class="text-sm text-slate-500 text-center py-4">Loading tasks...</p>
            </div>
        </div>
    </main>

    <script>
        function switchPage(page) {
            const convPage = document.getElementById('converterPage');
            const dashPage = document.getElementById('dashboardPage');
            const navConv = document.getElementById('navConverter');
            const navDash = document.getElementById('navDashboard');

            if(page === 'converter') {
                convPage.classList.remove('hidden');
                dashPage.classList.add('hidden');
                navConv.className = "px-3 py-1.5 sm:px-4 sm:py-2 rounded-xl text-xs sm:text-sm font-medium bg-indigo-600 text-white transition";
                navDash.className = "px-3 py-1.5 sm:px-4 sm:py-2 rounded-xl text-xs sm:text-sm font-medium bg-slate-800 text-slate-400 hover:text-white transition";
            } else {
                convPage.classList.add('hidden');
                dashPage.classList.remove('hidden');
                navDash.className = "px-3 py-1.5 sm:px-4 sm:py-2 rounded-xl text-xs sm:text-sm font-medium bg-indigo-600 text-white transition";
                navConv.className = "px-3 py-1.5 sm:px-4 sm:py-2 rounded-xl text-xs sm:text-sm font-medium bg-slate-800 text-slate-400 hover:text-white transition";
                loadDashboardTasks();
            }
        }

        function switchTab(tab) {
            const srtSec = document.getElementById('srtSection');
            const txtSec = document.getElementById('textSection');
            const tabSrt = document.getElementById('tabSrt');
            const tabText = document.getElementById('tabText');
            document.getElementById('statusBox').classList.add('hidden');

            if(tab === 'srt') {
                srtSec.classList.remove('hidden');
                txtSec.classList.add('hidden');
                tabSrt.className = "flex-1 py-2 text-xs sm:text-sm rounded-xl font-medium bg-indigo-600 text-white transition";
                tabText.className = "flex-1 py-2 text-xs sm:text-sm rounded-xl font-medium bg-slate-800 text-slate-400 transition";
            } else {
                srtSec.classList.add('hidden');
                txtSec.classList.remove('hidden');
                tabText.className = "flex-1 py-2 text-xs sm:text-sm rounded-xl font-medium bg-indigo-600 text-white transition";
                tabSrt.className = "flex-1 py-2 text-xs sm:text-sm rounded-xl font-medium bg-slate-800 text-slate-400 transition";
            }
        }

        async function loadDashboardTasks() {
            const res = await fetch('/tasks');
            const tasks = await res.json();
            const listEl = document.getElementById('taskList');
            
            if(tasks.length === 0) {
                listEl.innerHTML = '<p class="text-sm text-slate-500 text-center py-4">No tasks found.</p>';
                return;
            }

            listEl.innerHTML = tasks.map(t => `
                <div class="bg-slate-950 border border-slate-800 p-3 sm:p-4 rounded-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                    <div class="w-full sm:w-auto overflow-hidden">
                        <p class="text-xs sm:text-sm font-medium text-slate-200 truncate">${t.filename}</p>
                        <p class="text-[11px] sm:text-xs text-slate-400 mt-0.5">Status: <span class="${t.status === 'completed' ? 'text-emerald-400' : t.status === 'failed' ? 'text-rose-400' : 'text-amber-400'}">${t.status}</span> - ${t.progress}</p>
                    </div>
                    <div class="w-full sm:w-auto text-right">
                        ${t.status === 'completed' ? `<a href="/download/${t.task_id}" class="inline-block w-full sm:w-auto bg-emerald-600 hover:bg-emerald-500 text-white text-xs px-3 py-2 rounded-lg font-medium transition text-center">Download</a>` : ''}
                    </div>
                </div>
            `).join('');
        }

        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const statusBox = document.getElementById('statusBox');
            const progressText = document.getElementById('progressText');
            const progressBar = document.getElementById('progressBar');
            
            statusBox.classList.remove('hidden');
            progressText.textContent = "Uploading & queuing...";
            progressBar.style.width = "10%";

            const res = await fetch('/convert/', { method: 'POST', body: formData });
            const data = await res.json();
            
            if(!res.ok) { progressText.textContent = data.detail; return; }

            const taskId = data.task_id;
            
            const interval = setInterval(async () => {
                const statusRes = await fetch(`/status/${taskId}`);
                const statusData = await statusRes.json();
                
                progressText.textContent = statusData.progress;
                
                if(statusData.status === 'completed') {
                    clearInterval(interval);
                    progressBar.style.width = "100%";
                    progressText.textContent = "Sync complete!";
                    const dlBtn = document.getElementById('downloadBtn');
                    dlBtn.href = `/download/${taskId}`;
                    dlBtn.classList.remove('hidden');
                } else if(statusData.status === 'failed') {
                    clearInterval(interval);
                    progressBar.style.backgroundColor = "#f43f5e";
                    progressText.textContent = statusData.progress;
                } else {
                    progressBar.style.width = "50%";
                }
            }, 3000);
        });

        document.getElementById('textForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const statusBox = document.getElementById('statusBox');
            const progressText = document.getElementById('progressText');
            const progressBar = document.getElementById('progressBar');
            const dlBtn = document.getElementById('downloadBtn');
            
            statusBox.classList.remove('hidden');
            progressText.textContent = "Generating audio from text...";
            progressBar.style.width = "50%";
            dlBtn.classList.add('hidden');

            const res = await fetch('/convert-text/', { method: 'POST', body: formData });
            const data = await res.json();
            
            if(!res.ok) { progressText.textContent = data.detail; return; }

            progressBar.style.width = "100%";
            progressText.textContent = "Conversion complete!";
            dlBtn.href = `/download/${data.task_id}`;
            dlBtn.classList.remove('hidden');
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTMLResponse(content=HTML_UI)

@app.post("/convert/")
async def convert_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...), voice: str = Form("hi-IN-MadhurNeural")):
    if not file.filename.lower().endswith('.srt'):
        raise HTTPException(status_code=400, detail="Only .srt files accepted.")
    
    task_id = str(uuid.uuid4())
    input_path = f"temp_{task_id}.srt"
    
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)
        
    db = SessionLocal()
    new_task = ConversionTask(task_id=task_id, filename=file.filename, status="processing", progress="Queued...")
    db.add(new_task)
    db.commit()
    db.close()
    
    background_tasks.add_task(process_srt_in_background, task_id, input_path, voice)
    return {"task_id": task_id}

@app.post("/convert-text/")
async def convert_text_endpoint(text_content: str = Form(...), voice: str = Form("hi-IN-MadhurNeural")):
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
    
    task_id = str(uuid.uuid4())
    temp_audio = f"text_{task_id}.mp3"
    
    try:
        communicate = edge_tts.Communicate(text_content, voice)
        await communicate.save(temp_audio)
        
        with open(temp_audio, "rb") as f:
            audio_bytes = f.read()
            
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            
        db = SessionLocal()
        new_task = ConversionTask(
            task_id=task_id, 
            filename="text_conversion.mp3",
            status="completed", 
            progress="Completed successfully!", 
            merged_audio_data=audio_bytes
        )
        db.add(new_task)
        db.commit()
        db.close()
        
        return {"task_id": task_id}
    except Exception as e:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
def get_status(task_id: str):
    db = SessionLocal()
    task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
    db.close()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": task.status, "progress": task.progress}

@app.get("/tasks")
def get_tasks():
    db = SessionLocal()
    tasks = db.query(ConversionTask).order_by(ConversionTask.task_id.desc()).all()
    db.close()
    return [{"task_id": t.task_id, "filename": t.filename, "status": t.status, "progress": t.progress} for t in tasks]

@app.get("/download/{task_id}")
def download_file(task_id: str):
    db = SessionLocal()
    task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
    db.close()
    
    if not task or not task.merged_audio_data:
        raise HTTPException(status_code=404, detail="File not ready or missing.")
    
    output_filename = f"synced_{task_id}.mp3"
    with open(output_filename, "wb") as f:
        f.write(task.merged_audio_data)
        
    download_name = task.filename.rsplit('.', 1)[0] + "_synced.mp3" if task.filename else "audio_synced.mp3"
    return FileResponse(output_filename, media_type="audio/mpeg", filename=download_name)
