import os
import uuid
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
import pysrt
from gtts import gTTS
from pydub import AudioSegment
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- DATABASE SETUP (Supabase / Neon Free Postgres URL) ---
# Replace with your free PostgreSQL connection string from Supabase or Neon
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@host:port/dbname")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConversionTask(Base):
    __tablename__ = "conversion_tasks"
    task_id = Column(String, primary_key=True, index=True)
    status = Column(String, default="processing") # processing, completed, failed
    progress = Column(String, default="0/0 parts")
    download_path = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Large SRT to Hindi MP3 Converter")

# --- BACKGROUND PROCESSOR ---
def process_srt_in_background(task_id: str, input_path: str):
    db = SessionLocal()
    chunks_dir = f"chunks_{task_id}"
    output_mp3_path = f"output_{task_id}.mp3"
    
    try:
        os.makedirs(chunks_dir, exist_ok=True)
        try:
            subs = pysrt.open(input_path, encoding='utf-8')
        except Exception:
            subs = pysrt.open(input_path, encoding='latin-1')

        if not subs:
            update_task(db, task_id, status="failed", progress="Invalid SRT file")
            return

        # Group subtitles into ~2-minute clusters (120 seconds)
        clusters = []
        current_cluster = []
        cluster_duration_ms = 0
        
        for sub in subs:
            sub_duration = (sub.end.to_time().hour * 3600 + sub.end.to_time().minute * 60 + sub.end.to_time().second) * 1000 - \
                           (sub.start.to_time().hour * 3600 + sub.start.to_time().minute * 60 + sub.start.to_time().second) * 1000
            
            current_cluster.append(sub.text.replace('\n', ' ').strip())
            cluster_duration_ms += max(sub_duration, 1000)
            
            # If cluster reaches ~2 minutes (120,000 ms), push and reset
            if cluster_duration_ms >= 120000:
                clusters.append(" ".join(current_cluster))
                current_cluster = []
                cluster_duration_ms = 0
                
        if current_cluster:
            clusters.append(" ".join(current_cluster))

        total_parts = len(clusters)
        combined_audio = AudioSegment.empty()
        silence = AudioSegment.silent(duration=300)

        for i, text_chunk in enumerate(clusters):
            if not text_chunk.strip():
                continue
            
            update_task(db, task_id, progress=f"Processing part {i+1} of {total_parts}")
            
            chunk_path = os.path.join(chunks_dir, f"part_{i}.mp3")
            tts = gTTS(text=text_chunk, lang='hi', slow=False)
            tts.save(chunk_path)
            
            segment = AudioSegment.from_mp3(chunk_path)
            combined_audio += segment + silence
            
            # Brief pause to prevent hitting Google TTS rate limits (HTTP 429)
            time.sleep(1)

        combined_audio.export(output_mp3_path, format="mp3")
        
        # Mark complete
        task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
        if task:
            task.status = "completed"
            task.progress = f"Completed {total_parts} parts"
            task.download_path = output_mp3_path
            db.commit()

    except Exception as e:
        update_task(db, task_id, status="failed", progress=str(e))
    finally:
        db.close()
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(chunks_dir):
            import shutil
            shutil.rmtree(chunks_dir, ignore_errors=True)

def update_task(db, task_id, status=None, progress=None):
    task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
    if task:
        if status: task.status = status
        if progress: task.progress = progress
        db.commit()

# --- FRONTEND UI ---
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Large SRT to Hindi MP3 Converter</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex items-center justify-center p-4">
    <div class="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-6">
        <h1 class="text-2xl font-bold text-center mb-2">Large SRT to Hindi Audio</h1>
        <p class="text-xs text-slate-400 text-center mb-6">Splits file into 2-minute parts, processes securely via DB background queue.</p>
        
        <form id="uploadForm" class="space-y-4">
            <input type="file" id="srtFile" name="file" accept=".srt" required class="w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer"/>
            <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 rounded-xl transition">Start Chunk Processing</button>
        </form>

        <div id="statusBox" class="hidden mt-6 space-y-3 text-center border-t border-slate-800 pt-4">
            <p id="progressText" class="text-sm text-indigo-400 font-medium animate-pulse">Initializing...</p>
            <a id="downloadBtn" class="hidden block w-full bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-2.5 rounded-xl transition text-center">Download Merged MP3</a>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const statusBox = document.getElementById('statusBox');
            const progressText = document.getElementById('progressText');
            
            statusBox.classList.remove('hidden');
            progressText.textContent = "Uploading & setting up database task...";

            const res = await fetch('/convert/', { method: 'POST', body: formData });
            const data = await res.json();
            
            if(!res.ok) { progressText.textContent = data.detail; return; }

            const taskId = data.task_id;
            
            // Poll status every 3 seconds
            const interval = setInterval(async () => {
                const statusRes = await fetch(`/status/${taskId}`);
                const statusData = await statusRes.json();
                
                progressText.textContent = statusData.progress;
                
                if(statusData.status === 'completed') {
                    clearInterval(interval);
                    progressText.textContent = "Processing complete! Ready to download.";
                    const dlBtn = document.getElementById('downloadBtn');
                    dlBtn.href = `/download/${taskId}`;
                    dlBtn.classList.remove('hidden');
                } else if(statusData.status === 'failed') {
                    clearInterval(interval);
                    progressText.textContent = "Error: " + statusData.progress;
                }
            }, 3000);
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
py_root = lambda: HTMLResponse(content=HTML_UI)
app.add_api_route("/", py_root, methods=["GET"])

@app.post("/convert/")
async def convert_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.srt'):
        raise HTTPException(status_code=400, detail="Only .srt files accepted.")
    
    task_id = str(uuid.uuid4())
    input_path = f"temp_{task_id}.srt"
    
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)
        
    db = SessionLocal()
    new_task = ConversionTask(task_id=task_id, status="processing", progress="Queued in database...")
    db.add(new_task)
    db.commit()
    db.close()
    
    background_tasks.add_task(process_srt_in_background, task_id, input_path)
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
    if not task or not task.download_path or not os.path.exists(task.download_path):
        raise HTTPException(status_code=404, detail="File not ready or missing.")
    return FileResponse(task.download_path, media_type="audio/mpeg", filename="hindi_full_audio.mp3")
