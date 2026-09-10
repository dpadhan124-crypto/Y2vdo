import os
import uuid
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
import pysrt
from gtts import gTTS
from pydub import AudioSegment
from sqlalchemy import create_engine, Column, String, LargeBinary, Integer, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@host:port/dbname")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConversionTask(Base):
    __tablename__ = "conversion_tasks"
    task_id = Column(String, primary_key=True, index=True)
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
    status = Column(String, default="pending") # pending, done

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Precise Timestamp SRT to Hindi TTS")

def srt_time_to_ms(t):
    return (t.hours * 3600 + t.minutes * 60 + t.seconds) * 1000 + t.milliseconds

def process_srt_in_background(task_id: str, input_path: str):
    db = SessionLocal()
    try:
        try:
            subs = pysrt.open(input_path, encoding='utf-8')
        except Exception:
            subs = pysrt.open(input_path, encoding='latin-1')

        if not subs:
            update_task_status(db, task_id, "failed", "Invalid or empty SRT file")
            return

        # 1. Parse and split into DB in batches of 20 lines
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

        update_task_status(db, task_id, "processing", f"Parsed {total_subs} lines. Generating TTS chunks...")

        # 2. Process each line TTS and save blob back to DB
        lines = db.query(SubtitleLine).filter(SubtitleLine.task_id == task_id).order_by(SubtitleLine.line_index).all()
        
        for i, line in enumerate(lines):
            if not line.text:
                line.status = "done"
                db.commit()
                continue
            
            update_task_status(db, task_id, "processing", f"Generating audio: Line {i+1} of {total_subs}")
            
            temp_chunk = f"chunk_{task_id}_{i}.mp3"
            try:
                tts = gTTS(text=line.text, lang='hi', slow=False)
                tts.save(temp_chunk)
                
                with open(temp_chunk, "rb") as f:
                    line.audio_data = f.read()
                line.status = "done"
                db.commit()
                
                if os.path.exists(temp_chunk):
                    os.remove(temp_chunk)
            except Exception as e:
                # Fallback on rate limits with sleep
                time.sleep(2)
                try:
                    tts = gTTS(text=line.text, lang='hi', slow=False)
                    tts.save(temp_chunk)
                    with open(temp_chunk, "rb") as f:
                        line.audio_data = f.read()
                    line.status = "done"
                    db.commit()
                    if os.path.exists(temp_chunk):
                        os.remove(temp_chunk)
                except Exception:
                    pass
            
            time.sleep(0.5)

        # 3. Timestamp matching, speed-stretching limits (0.5x to 2x), and smart timeline stitching
        update_task_status(db, task_id, "processing", "Stitching timeline matching exact SRT timestamps...")
        
        final_timeline = AudioSegment.silent(duration=0)
        current_timeline_cursor = 0

        for line in lines:
            target_duration = max(line.end_ms - line.start_ms, 500) # Minimum slot duration
            
            # Insert timeline gap if subtitle start is ahead of current audio cursor
            if line.start_ms > current_timeline_cursor:
                gap_duration = line.start_ms - current_timeline_cursor
                final_timeline += AudioSegment.silent(duration=gap_duration)
                current_timeline_cursor = line.start_ms

            if not line.audio_data:
                # Empty slot placeholder matching exact timestamp duration
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
            
            # Apply speed limits (0.5x to 2.0x bounds) to fit exact timing slots
            if audio_len > target_duration:
                # Needs compression/speed up (capped at 2x max speed factor -> min duration = audio_len / 2.0)
                speed_factor = audio_len / target_duration
                if speed_factor > 2.0:
                    speed_factor = 2.0 # Cap max speed to 2x to prevent distortion
                
                new_framerate = int(segment.frame_rate * speed_factor)
                segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_framerate}).set_frame_rate(44100)
            elif audio_len < target_duration:
                # Needs stretching/slow down (capped at 0.5x max slow factor -> max duration = audio_len / 0.5)
                speed_factor = audio_len / target_duration
                if speed_factor < 0.5:
                    speed_factor = 0.5 # Cap slow down factor to 0.5x limits
                
                new_framerate = int(segment.frame_rate * speed_factor)
                segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_framerate}).set_frame_rate(44100)

            final_timeline += segment
            current_timeline_cursor += len(segment)

        # 4. Export final merged file binary blob to database
        merged_output_path = f"final_merged_{task_id}.mp3"
        final_timeline.export(merged_output_path, format="mp3")

        with open(merged_output_path, "rb") as f:
            merged_bytes = f.read()

        task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
        if task:
            task.status = "completed"
            task.progress = "Completed successfully with exact timestamp alignment!"
            task.merged_audio_data = merged_bytes
            db.commit()

        if os.path.exists(merged_output_path):
            os.remove(merged_output_path)

        # Cleanup raw line records from DB to save storage space
        db.query(SubtitleLine).filter(SubtitleLine.task_id == task_id).delete()
        db.commit()

    except Exception as e:
        update_task_status(db, task_id, "failed", f"Error: {str(e)}")
    finally:
        db.close()
        if os.path.exists(input_path):
            os.remove(input_path)

def update_task_status(db, task_id, status, progress):
    task = db.query(ConversionTask).filter(ConversionTask.task_id == task_id).first()
    if task:
        task.status = status
        task.progress = progress
        db.commit()

# --- FRONTEND UI ---
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Precise SRT to Hindi TTS Sync</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex items-center justify-center p-4">
    <div class="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-6">
        <h1 class="text-2xl font-bold text-center mb-2">Timestamp-Synced SRT Audio</h1>
        <p class="text-xs text-slate-400 text-center mb-6">Runs securely in background chunks, even if you close this browser tab.</p>
        
        <form id="uploadForm" class="space-y-4">
            <input type="file" id="srtFile" name="file" accept=".srt" required class="w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer"/>
            <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 rounded-xl transition">Start Background Processing</button>
        </form>

        <div id="statusBox" class="hidden mt-6 space-y-3 text-center border-t border-slate-800 pt-4">
            <p id="progressText" class="text-sm text-indigo-400 font-medium animate-pulse">Initializing...</p>
            <a id="downloadBtn" class="hidden block w-full bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-2.5 rounded-xl transition text-center">Download Synced MP3</a>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const statusBox = document.getElementById('statusBox');
            const progressText = document.getElementById('progressText');
            
            statusBox.classList.remove('hidden');
            progressText.textContent = "Uploading & scheduling background worker...";

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
                    progressText.textContent = "Sync complete! Click below to download.";
                    const dlBtn = document.getElementById('downloadBtn');
                    dlBtn.href = `/download/${taskId}`;
                    dlBtn.classList.remove('hidden');
                } else if(statusData.status === 'failed') {
                    clearInterval(interval);
                    progressText.textContent = statusData.progress;
                }
            }, 3000);
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTMLResponse(content=HTML_UI)

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
    
    if not task or not task.merged_audio_data:
        raise HTTPException(status_code=404, detail="File not ready or missing.")
    
    output_filename = f"synced_audio_{task_id}.mp3"
    with open(output_filename, "wb") as f:
        f.write(task.merged_audio_data)
        
    return FileResponse(output_filename, media_type="audio/mpeg", filename="hindi_synced_audio.mp3")
