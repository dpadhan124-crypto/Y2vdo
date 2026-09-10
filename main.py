import os
import re
import math
import asyncio
import tempfile
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from pydub import AudioSegment
import edge_tts

# ==========================================
# CONFIGURATION & SUPABASE DATABASE SETUP
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/tts_db")

# Automatically handle SSL requirements for Supabase / remote PostgreSQL providers
connect_args = {}
if "supabase.co" in DATABASE_URL or "neon.tech" in DATABASE_URL or "render.com" in DATABASE_URL:
    connect_args = {"sslmode": "require"}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConversionTask(Base):
    __tablename__ = "conversion_tasks"

    id = Column(Integer, primary_key=True, index=True)
    mode = Column(String(50), nullable=False, default="srt")  # 'srt' or 'text'
    filename = Column(String(255), nullable=True)
    voice = Column(String(50), nullable=False, default="Swara")
    status = Column(String(50), default="Pending")  # Pending, Processing, Completed, Failed
    progress_message = Column(String(255), default="Initializing...")
    output_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    lines = relationship("SubtitleLine", back_populates="task", cascade="all, delete-orphan")

class SubtitleLine(Base):
    __tablename__ = "subtitle_lines"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("conversion_tasks.id"))
    line_index = Column(Integer, nullable=False)
    start_ms = Column(Integer, nullable=False)
    end_ms = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    task = relationship("ConversionTask", back_populates="lines")

def init_db():
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    
    # Safely auto-migrate missing columns if the table already existed with an older schema
    if "conversion_tasks" in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('conversion_tasks')]
        migrations = {
            'filename': "VARCHAR(255)",
            'mode': "VARCHAR(50) NOT NULL DEFAULT 'srt'",
            'voice': "VARCHAR(50) NOT NULL DEFAULT 'Swara'",
            'status': "VARCHAR(50) DEFAULT 'Pending'",
            'progress_message': "VARCHAR(255) DEFAULT 'Initializing...'",
            'output_path': "VARCHAR(500)",
            'created_at': "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        }
        with engine.begin() as conn:
            for col_name, col_type in migrations.items():
                if col_name not in columns:
                    conn.execute(text(f"ALTER TABLE conversion_tasks ADD COLUMN {col_name} {col_type};"))

init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

OUTPUT_DIR = os.path.join(os.getcwd(), "generated_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Hindi Neural TTS & SRT Studio")

# ==========================================
# VOICE MAP
# ==========================================
VOICE_MAP = {
    "Madhur": "hi-IN-MadhurNeural",
    "Swara": "hi-IN-SwaraNeural",
    "Aarav": "hi-IN-AaravNeural",
    "Ananya": "hi-IN-AnanyaNeural"
}

# ==========================================
# SRT PARSING UTILITIES
# ==========================================
def parse_timestamp(ts_str: str) -> int:
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts_str.strip())
    if not match:
        return 0
    h, m, s, ms = map(int, match.groups())
    return (h * 3600 + m * 60 + s) * 1000 + ms

def parse_srt_content(content: str) -> List[dict]:
    pattern = re.compile(
        r"(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\s*\n\d+\s*\n|\Z)",
        re.DOTALL
    )
    matches = pattern.findall(content)
    parsed_lines = []
    
    for idx, (index_str, start_str, end_str, text_block) in enumerate(matches):
        cleaned_text = " ".join(text_block.strip().replace("\n", " ").split())
        start_ms = parse_timestamp(start_str)
        end_ms = parse_timestamp(end_str)
        if end_ms > start_ms:
            parsed_lines.append({
                "index": int(index_str),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": cleaned_text
            })
    return parsed_lines

# ==========================================
# BACKGROUND TTS GENERATION ENGINE
# ==========================================
async def generate_speech_chunk(text: str, voice_name: str, output_filepath: str, retries: int = 3):
    voice_id = VOICE_MAP.get(voice_name, "hi-IN-SwaraNeural")
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, voice_id)
            await communicate.save(output_filepath)
            if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
                return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"TTS Chunk failure after {retries} attempts: {e}")
        await asyncio.sleep(1)
    return False

def change_audio_speed(segment: AudioSegment, speed: float) -> AudioSegment:
    if speed == 1.0:
        return segment
    try:
        speed = max(0.8, min(1.2, speed))
        new_sample_rate = int(segment.frame_rate * speed)
        altered = segment._spawn(segment.raw_data, overrides={'frame_rate': new_sample_rate})
        return altered.set_frame_rate(segment.frame_rate)
    except Exception:
        return segment

async def process_srt_task(task_id: int):
    with SessionLocal() as db:
        task = db.query(ConversionTask).filter(ConversionTask.id == task_id).first()
        if not task:
            return

        try:
            task.status = "Processing"
            db.commit()

            lines = db.query(SubtitleLine).filter(SubtitleLine.task_id == task_id).order_by(SubtitleLine.line_index).all()
            total_lines = len(lines)
            
            if total_lines == 0:
                raise ValueError("No valid subtitle slots found in file.")

            master_track = AudioSegment.silent(duration=0)
            current_timeline_cursor = 0

            with tempfile.TemporaryDirectory() as tmpdir:
                for idx, line in enumerate(lines, start=1):
                    task.progress_message = f"Generating audio: Line {idx} of {total_lines}"
                    db.commit()

                    chunk_path = os.path.join(tmpdir, f"chunk_{idx}.mp3")
                    success = await generate_speech_chunk(line.text, task.voice, chunk_path)
                    
                    if not success or not os.path.exists(chunk_path):
                        slot_duration = max(500, line.end_ms - line.start_ms)
                        segment = AudioSegment.silent(duration=slot_duration)
                    else:
                        segment = AudioSegment.from_file(chunk_path, format="mp3")

                    target_slot_duration = line.end_ms - line.start_ms
                    actual_duration = len(segment)

                    if actual_duration > 0 and target_slot_duration > 0:
                        calculated_speed = actual_duration / target_slot_duration
                        clamped_speed = max(0.8, min(1.2, calculated_speed))
                        segment = change_audio_speed(segment, clamped_speed)

                    if line.start_ms > current_timeline_cursor:
                        gap_duration = line.start_ms - current_timeline_cursor
                        master_track += AudioSegment.silent(duration=gap_duration)
                        current_timeline_cursor = line.start_ms

                    master_track += segment
                    current_timeline_cursor += len(segment)

                task.progress_message = "Stitching timeline matching exact SRT timestamps..."
                db.commit()

                final_output_filename = f"task_{task_id}_synchronized.mp3"
                final_output_path = os.path.join(OUTPUT_DIR, final_output_filename)
                master_track.export(final_output_path, format="mp3")

                task.output_path = final_output_path
                task.status = "Completed"
                task.progress_message = "Conversion successfully finished!"
                db.commit()

        except Exception as e:
            task.status = "Failed"
            task.progress_message = f"Error: {str(e)}"
            db.commit()

async def process_text_task(task_id: int, raw_text: str):
    with SessionLocal() as db:
        task = db.query(ConversionTask).filter(ConversionTask.id == task_id).first()
        if not task:
            return

        try:
            task.status = "Processing"
            task.progress_message = "Generating direct neural audio..."
            db.commit()

            with tempfile.TemporaryDirectory() as tmpdir:
                chunk_path = os.path.join(tmpdir, "text_output.mp3")
                success = await generate_speech_chunk(raw_text, task.voice, chunk_path)
                
                if not success:
                    raise Exception("Failed to generate neural speech audio from text.")

                final_output_filename = f"task_{task_id}_direct.mp3"
                final_output_path = os.path.join(OUTPUT_DIR, final_output_filename)
                
                segment = AudioSegment.from_file(chunk_path, format="mp3")
                segment.export(final_output_path, format="mp3")

                task.output_path = final_output_path
                task.status = "Completed"
                task.progress_message = "Instant text conversion completed!"
                db.commit()

        except Exception as e:
            task.status = "Failed"
            task.progress_message = f"Error: {str(e)}"
            db.commit()

# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def home_ui(db: Session = Depends(get_db)):
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/convert/")
async def convert_srt(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    voice: str = Form("Swara"),
    db: Session = Depends(get_db)
):
    content_bytes = await file.read()
    try:
        content_str = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content_str = content_bytes.decode("latin-1")

    parsed_subs = parse_srt_content(content_str)
    if not parsed_subs:
        raise HTTPException(status_code=400, detail="Invalid SRT formatting or empty subtitle blocks detected.")

    new_task = ConversionTask(
        mode="srt",
        filename=file.filename,
        voice=voice,
        status="Pending",
        progress_message="Queued for SRT synchronized processing..."
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task)

    for sub in parsed_subs:
        line_item = SubtitleLine(
            task_id=new_task.id,
            line_index=sub["index"],
            start_ms=sub["start_ms"],
            end_ms=sub["end_ms"],
            text=sub["text"]
        )
        db.add(line_item)
    db.commit()

    background_tasks.add_task(process_srt_task, new_task.id)
    return {"task_id": new_task.id, "message": "SRT conversion background job initiated successfully."}

@app.post("/convert-text/")
async def convert_text(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    voice: str = Form("Swara"),
    db: Session = Depends(get_db)
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")

    new_task = ConversionTask(
        mode="text",
        filename="Direct Text Input",
        voice=voice,
        status="Pending",
        progress_message="Queued for instant neural audio generation..."
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task)

    background_tasks.add_task(process_text_task, new_task.id, text)
    return {"task_id": new_task.id, "message": "Text conversion job initiated successfully."}

@app.get("/status/{task_id}")
async def get_task_status(task_id: int, db: Session = Depends(get_db)):
    task = db.query(ConversionTask).filter(ConversionTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task record not found.")
    return {
        "id": task.id,
        "status": task.status,
        "progress_message": task.progress_message,
        "mode": task.mode,
        "voice": task.voice,
        "filename": task.filename,
        "has_output": bool(task.output_path and os.path.exists(task.output_path))
    }

@app.get("/tasks")
async def get_all_tasks(db: Session = Depends(get_db)):
    tasks = db.query(ConversionTask).order_by(ConversionTask.created_at.desc()).all()
    results = []
    for t in tasks:
        results.append({
            "id": t.id,
            "mode": t.mode,
            "filename": t.filename or "Direct Input",
            "voice": t.voice,
            "status": t.status,
            "progress_message": t.progress_message,
            "created_at": t.created_at.strftime("%Y-%m-%d %H:%M:%S") if t.created_at else "",
            "has_output": bool(t.output_path and os.path.exists(t.output_path))
        })
    return results

@app.post("/clear-db")
async def clear_database(db: Session = Depends(get_db)):
    try:
        db.query(SubtitleLine).delete()
        db.query(ConversionTask).delete()
        db.commit()
        for f in os.listdir(OUTPUT_DIR):
            fp = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)
        return {"message": "All database records and cache files cleared successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{task_id}")
async def download_audio(task_id: int, db: Session = Depends(get_db)):
    task = db.query(ConversionTask).filter(ConversionTask.id == task_id).first()
    if not task or not task.output_path or not os.path.exists(task.output_path):
        raise HTTPException(status_code=404, detail="Requested audio output file not found.")
    return FileResponse(task.output_path, media_type="audio/mpeg", filename=f"Hindi_Neural_Audio_{task.id}.mp3")

# ==========================================
# FRONTEND TEMPLATE (SINGLE-PAGE INTERFACE)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hindi Neural Audio & SRT Studio</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        darker: '#0b0f19',
                        darkcard: '#111827',
                        accent: '#6366f1',
                        accenthover: '#4f46e5'
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-darker text-gray-100 min-h-screen flex flex-col font-sans selection:bg-accent selection:text-white">

    <header class="border-b border-gray-800 bg-darkcard/50 backdrop-blur sticky top-0 z-50">
        <div class="max-w-6xl mx-auto px-4 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
            <div class="flex items-center space-x-3">
                <div class="bg-accent/20 p-2.5 rounded-xl border border-accent/30 text-accent font-black text-xl">🎙️</div>
                <div>
                    <h1 class="text-xl font-bold tracking-tight text-white">Hindi Neural Audio Studio</h1>
                    <p class="text-xs text-gray-400">High-Precision Synchronized SRT & Text-to-Speech Engine</p>
                </div>
            </div>
            <nav class="flex space-x-2 bg-gray-900/80 p-1.5 rounded-xl border border-gray-800">
                <button onclick="switchTab('convert')" id="nav-convert" class="px-4 py-2 rounded-lg text-sm font-medium transition bg-accent text-white shadow-lg">Conversion Studio</button>
                <button onclick="switchTab('dashboard')" id="nav-dashboard" class="px-4 py-2 rounded-lg text-sm font-medium transition text-gray-400 hover:text-white">Task History</button>
            </nav>
        </div>
    </header>

    <main class="max-w-4xl w-full mx-auto px-4 py-8 flex-grow">

        <section id="tab-convert" class="space-y-6">
            <div class="bg-darkcard border border-gray-800 rounded-2xl p-6 sm:p-8 shadow-2xl relative overflow-hidden">
                <div class="absolute top-0 right-0 w-32 h-32 bg-accent/5 rounded-full blur-3xl pointer-events-none"></div>

                <h2 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <span>⚡</span> Dual Mode Configuration
                </h2>

                <div class="grid grid-cols-2 gap-3 mb-6 bg-gray-900/60 p-1.5 rounded-xl border border-gray-800">
                    <button type="button" onclick="setMode('srt')" id="mode-btn-srt" class="py-2.5 rounded-lg text-sm font-semibold transition bg-accent text-white shadow">SRT File to MP3</button>
                    <button type="button" onclick="setMode('text')" id="mode-btn-text" class="py-2.5 rounded-lg text-sm font-semibold transition text-gray-400 hover:text-white">Raw Text to MP3</button>
                </div>

                <form id="conversion-form" onsubmit="handleSubmission(event)" class="space-y-5">
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Select Indian Hindi Neural Voice</label>
                        <select name="voice" id="voice-select" class="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-gray-100 focus:outline-none focus:ring-2 focus:ring-accent transition">
                            <option value="Swara">Swara (Female - Natural & Expressive)</option>
                            <option value="Madhur">Madhur (Male - Professional & Clear)</option>
                            <option value="Ananya">Ananya (Female - Soft & Melodic)</option>
                            <option value="Aarav">Aarav (Male - Deep & Dynamic)</option>
                        </select>
                    </div>

                    <div id="input-container-srt" class="space-y-2">
                        <label class="block text-sm font-medium text-gray-300">Upload Subtitle File (.srt)</label>
                        <div class="border-2 border-dashed border-gray-700 hover:border-accent rounded-2xl p-6 text-center transition bg-gray-900/40 cursor-pointer relative" onclick="document.getElementById('srt-file').click()">
                            <input type="file" name="file" id="srt-file" accept=".srt" class="hidden" onchange="updateFileName(this)">
                            <div class="text-3xl mb-2">📁</div>
                            <p class="text-sm font-medium text-gray-200" id="file-label-text">Click to browse or drop your .srt file here</p>
                            <p class="text-xs text-gray-500 mt-1">Exact timestamp alignment with 0.8x-1.2x speed stretching bounds</p>
                        </div>
                    </div>

                    <div id="input-container-text" class="space-y-2 hidden">
                        <label class="block text-sm font-medium text-gray-300">Enter Raw Hindi Text</label>
                        <textarea name="text" id="raw-text" rows="5" placeholder="यहाँ अपना हिंदी पाठ दर्ज करें..." class="w-full bg-gray-900 border border-gray-700 rounded-xl p-4 text-gray-100 focus:outline-none focus:ring-2 focus:ring-accent transition resize-none"></textarea>
                    </div>

                    <button type="submit" id="submit-btn" class="w-full bg-accent hover:bg-accenthover text-white font-medium py-3.5 px-6 rounded-xl transition shadow-lg flex items-center justify-center gap-2">
                        <span>🚀</span> Start Neural Conversion
                    </button>
                </form>

                <div id="progress-box" class="hidden mt-8 bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
                    <div class="flex justify-between items-center text-sm">
                        <span id="progress-status-label" class="font-medium text-accent">Processing background queue...</span>
                        <span id="task-id-badge" class="text-xs bg-gray-800 px-2.5 py-1 rounded-md text-gray-400">Task #--</span>
                    </div>
                    <div class="w-full bg-gray-800 rounded-full h-2.5 overflow-hidden">
                        <div id="progress-bar-fill" class="bg-accent h-2.5 rounded-full transition-all duration-300 w-full animate-pulse"></div>
                    </div>
                </div>

                <div id="result-box" class="hidden mt-8 bg-gray-900 border border-green-500/30 rounded-xl p-5 space-y-4">
                    <div class="flex items-center gap-3">
                        <div class="bg-green-500/20 p-2 rounded-lg text-green-400">✅</div>
                        <div>
                            <h3 class="font-semibold text-white">Audio Generated Successfully</h3>
                            <p class="text-xs text-gray-400">Ready for instant browser playback or high-quality export</p>
                        </div>
                    </div>
                    <audio id="audio-player" controls class="w-full rounded-lg"></audio>
                    <a id="download-link" href="#" class="block text-center bg-green-600 hover:bg-green-500 text-white font-medium py-3 rounded-xl transition shadow">
                        📥 Download MP3 Audio File
                    </a>
                </div>
            </div>
        </section>

        <section id="tab-dashboard" class="space-y-6 hidden">
            <div class="bg-darkcard border border-gray-800 rounded-2xl p-6 sm:p-8 shadow-2xl">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
                    <div>
                        <h2 class="text-lg font-semibold text-white">Conversion History Dashboard</h2>
                        <p class="text-xs text-gray-400">Manage, preview, and download your past neural speech tasks</p>
                    </div>
                    <div class="flex gap-2 w-full sm:w-auto">
                        <button onclick="loadTasks()" class="flex-1 sm:flex-none bg-gray-800 hover:bg-gray-700 text-xs px-3 py-2 rounded-lg transition border border-gray-700">🔄 Refresh</button>
                        <button onclick="clearAllData()" class="flex-1 sm:flex-none bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-500/30 text-xs px-3 py-2 rounded-lg transition">🗑️ Clear All Data</button>
                    </div>
                </div>

                <div id="tasks-container" class="space-y-3">
                    <p class="text-center text-gray-500 py-8 text-sm">Loading historical task entries...</p>
                </div>
            </div>
        </section>

    </main>

    <footer class="border-t border-gray-800 bg-darkcard/30 text-center py-4 text-xs text-gray-500">
        Powered by FastAPI, Supabase PostgreSQL, Tailwind CSS, and Microsoft Edge Neural TTS.
    </footer>

    <script>
        let currentMode = 'srt';
        let activePollingInterval = null;

        function switchTab(tab) {
            const convertSec = document.getElementById('tab-convert');
            const dashSec = document.getElementById('tab-dashboard');
            const navConv = document.getElementById('nav-convert');
            const navDash = document.getElementById('nav-dashboard');

            if (tab === 'convert') {
                convertSec.classList.remove('hidden');
                dashSec.classList.add('hidden');
                navConv.className = "px-4 py-2 rounded-lg text-sm font-medium transition bg-accent text-white shadow-lg";
                navDash.className = "px-4 py-2 rounded-lg text-sm font-medium transition text-gray-400 hover:text-white";
            } else {
                convertSec.classList.add('hidden');
                dashSec.classList.remove('hidden');
                navDash.className = "px-4 py-2 rounded-lg text-sm font-medium transition bg-accent text-white shadow-lg";
                navConv.className = "px-4 py-2 rounded-lg text-sm font-medium transition text-gray-400 hover:text-white";
                loadTasks();
            }
        }

        function setMode(mode) {
            currentMode = mode;
            const btnSrt = document.getElementById('mode-btn-srt');
            const btnText = document.getElementById('mode-btn-text');
            const boxSrt = document.getElementById('input-container-srt');
            const boxText = document.getElementById('input-container-text');

            if (mode === 'srt') {
                btnSrt.className = "py-2.5 rounded-lg text-sm font-semibold transition bg-accent text-white shadow";
                btnText.className = "py-2.5 rounded-lg text-sm font-semibold transition text-gray-400 hover:text-white";
                boxSrt.classList.remove('hidden');
                boxText.classList.add('hidden');
            } else {
                btnText.className = "py-2.5 rounded-lg text-sm font-semibold transition bg-accent text-white shadow";
                btnSrt.className = "py-2.5 rounded-lg text-sm font-semibold transition text-gray-400 hover:text-white";
                boxText.classList.remove('hidden');
                boxSrt.classList.add('hidden');
            }
        }

        function updateFileName(input) {
            const label = document.getElementById('file-label-text');
            if (input.files && input.files.length > 0) {
                label.innerText = "Selected File: " + input.files[0].name;
            } else {
                label.innerText = "Click to browse or drop your .srt file here";
            }
        }

        async function handleSubmission(event) {
            event.preventDefault();
            const voice = document.getElementById('voice-select').value;

            let endpoint = '/convert/';
            const submitPayload = new FormData();
            submitPayload.append('voice', voice);

            if (currentMode === 'srt') {
                const fileInput = document.getElementById('srt-file');
                if (!fileInput.files.length) {
                    alert('Please choose an SRT subtitle file first.');
                    return;
                }
                submitPayload.append('file', fileInput.files[0]);
            } else {
                endpoint = '/convert-text/';
                const textVal = document.getElementById('raw-text').value;
                if (!textVal.trim()) {
                    alert('Please enter some text to convert.');
                    return;
                }
                submitPayload.append('text', textVal);
            }

            const submitBtn = document.getElementById('submit-btn');
            submitBtn.disabled = true;
            submitBtn.innerText = "Initializing Background Queue...";

            document.getElementById('progress-box').classList.remove('hidden');
            document.getElementById('result-box').classList.add('hidden');

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: submitPayload
                });
                
                const responseText = await response.text();
                let data;
                try {
                    data = JSON.parse(responseText);
                } catch (e) {
                    throw new Error("Server response: " + responseText);
                }

                if (!response.ok) throw new Error(data.detail || 'Failed to start conversion task.');

                document.getElementById('task-id-badge').innerText = "Task #" + data.task_id;
                pollTaskStatus(data.task_id);
            } catch (err) {
                alert(err.message);
                submitBtn.disabled = false;
                submitBtn.innerText = "🚀 Start Neural Conversion";
                document.getElementById('progress-box').classList.add('hidden');
            }
        }

        function pollTaskStatus(taskId) {
            if (activePollingInterval) clearInterval(activePollingInterval);

            activePollingInterval = setInterval(async () => {
                try {
                    const res = await fetch(`/status/${taskId}`);
                    const statusData = await res.json();

                    document.getElementById('progress-status-label').innerText = statusData.progress_message;

                    if (statusData.status === 'Completed') {
                        clearInterval(activePollingInterval);
                        resetSubmitButton();
                        document.getElementById('progress-box').classList.add('hidden');
                        
                        const audioPlayer = document.getElementById('audio-player');
                        const downloadLink = document.getElementById('download-link');
                        audioPlayer.src = `/download/${taskId}`;
                        downloadLink.href = `/download/${taskId}`;
                        document.getElementById('result-box').classList.remove('hidden');
                    } else if (statusData.status === 'Failed') {
                        clearInterval(activePollingInterval);
                        resetSubmitButton();
                        document.getElementById('progress-box').classList.add('hidden');
                        alert("Conversion Error: " + statusData.progress_message);
                    }
                } catch (e) {
                    console.error("Polling error:", e);
                }
            }, 1500);
        }

        function resetSubmitButton() {
            const submitBtn = document.getElementById('submit-btn');
            submitBtn.disabled = false;
            submitBtn.innerText = "🚀 Start Neural Conversion";
        }

        async function loadTasks() {
            const container = document.getElementById('tasks-container');
            container.innerHTML = '<p class="text-center text-gray-500 py-8 text-sm">Fetching task records...</p>';

            try {
                const res = await fetch('/tasks');
                const tasks = await res.json();

                if (tasks.length === 0) {
                    container.innerHTML = '<p class="text-center text-gray-500 py-8 text-sm">No historical conversion tasks found.</p>';
                    return;
                }

                let html = '';
                tasks.forEach(t => {
                    let statusBadge = '<span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">Pending</span>';
                    if (t.status === 'Processing') {
                        statusBadge = '<span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-500/20 text-blue-400 border border-blue-500/30 animate-pulse">Processing</span>';
                    } else if (t.status === 'Completed') {
                        statusBadge = '<span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-green-500/20 text-green-400 border border-green-500/30">Completed</span>';
                    } else if (t.status === 'Failed') {
                        statusBadge = '<span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-red-500/20 text-red-400 border border-red-500/30">Failed</span>';
                    }

                    html += `
                        <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                            <div class="space-y-1">
                                <div class="flex items-center gap-2">
                                    <span class="text-xs font-mono text-gray-400">#${t.id}</span>
                                    <span class="text-xs uppercase px-2 py-0.5 bg-gray-800 text-gray-300 rounded">${t.mode}</span>
                                    ${statusBadge}
                                </div>
                                <h4 class="font-medium text-white text-sm truncate max-w-xs sm:max-w-md">${t.filename}</h4>
                                <p class="text-xs text-gray-400">Voice: <span class="text-gray-200">${t.voice}</span> | Created: ${t.created_at}</p>
                                <p class="text-xs text-accent italic">${t.progress_message}</p>
                            </div>
                            <div class="flex items-center gap-2 w-full sm:w-auto justify-end">
                                ${t.has_output ? `
                                    <a href="/download/${t.id}" class="bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs px-3.5 py-2 rounded-lg transition border border-gray-700 flex items-center gap-1">📥 Download</a>
                                ` : ''}
                            </div>
                        </div>
                    `;
                });
                container.innerHTML = html;
            } catch (err) {
                container.innerHTML = '<p class="text-center text-red-400 py-8 text-sm">Failed to retrieve historical items.</p>';
            }
        }

        async function clearAllData() {
            if (!confirm("Are you sure you want to delete all historical tasks and cached audio data?")) return;
            try {
                const res = await fetch('/clear-db', { method: 'POST' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail);
                alert(data.message);
                loadTasks();
            } catch (err) {
                alert("Error clearing data: " + err.message);
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
