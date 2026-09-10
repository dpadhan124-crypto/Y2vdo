import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import pysrt
from gtts import gTTS
from pydub import AudioSegment

app = FastAPI(title="SRT to Hindi MP3 Converter")

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SRT to Hindi MP3 Converter</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex items-center justify-center p-4">

    <div class="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-6 md:p-8">
        <div class="text-center mb-6">
            <h1 class="text-2xl font-bold tracking-tight text-white">SRT to Hindi Audio</h1>
            <p class="text-sm text-slate-400 mt-1">Convert subtitle files (.srt) into high-quality Hindi MP3 voiceovers.</p>
        </div>

        <form id="uploadForm" class="space-y-4">
            <div class="relative border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-6 text-center cursor-pointer transition bg-slate-950/50" id="dropZone">
                <input type="file" id="srtFile" name="file" accept=".srt" class="absolute inset-0 opacity-0 cursor-pointer w-full h-full" required>
                <div class="space-y-2 pointer-events-none">
                    <svg class="mx-auto h-10 w-10 text-slate-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                    <div class="text-sm text-slate-300">
                        <span class="font-semibold text-indigo-400">Click to upload</span> or drag and drop
                    </div>
                    <p class="text-xs text-slate-500" id="fileNameDisplay">Only .srt files accepted</p>
                </div>
            </div>

            <button type="submit" id="submitBtn" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 px-4 rounded-xl transition shadow-lg shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed">
                Convert to MP3
            </button>
        </form>

        <div id="loadingState" class="hidden mt-6 text-center space-y-3">
            <div class="inline-block animate-spin rounded-full h-8 w-8 border-4 border-indigo-500 border-t-transparent"></div>
            <p class="text-sm text-slate-400 animate-pulse">Processing subtitles & generating Hindi audio chunks...</p>
        </div>

        <div id="errorBox" class="hidden mt-4 p-3 bg-red-950/50 border border-red-800 rounded-xl text-red-300 text-sm"></div>

        <div id="resultBox" class="hidden mt-6 space-y-4 border-t border-slate-800 pt-6">
            <div class="flex items-center justify-between">
                <span class="text-sm font-medium text-emerald-400 flex items-center gap-1.5">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                    Conversion Successful!
                </span>
            </div>
            <audio id="audioPlayer" controls class="w-full rounded-lg"></audio>
            <a id="downloadLink" class="block w-full text-center bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-2.5 px-4 rounded-xl transition shadow-lg shadow-emerald-600/20">
                Download MP3 File
            </a>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('srtFile');
        const fileNameDisplay = document.getElementById('fileNameDisplay');
        const uploadForm = document.getElementById('uploadForm');
        const submitBtn = document.getElementById('submitBtn');
        const loadingState = document.getElementById('loadingState');
        const errorBox = document.getElementById('errorBox');
        const resultBox = document.getElementById('resultBox');
        const audioPlayer = document.getElementById('audioPlayer');
        const downloadLink = document.getElementById('downloadLink');

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                fileNameDisplay.textContent = e.target.files[0].name;
            }
        });

        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            errorBox.classList.add('hidden');
            resultBox.classList.add('hidden');
            loadingState.classList.remove('hidden');
            submitBtn.disabled = true;

            const formData = new FormData(uploadForm);

            try {
                const response = await fetch('/convert/', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to process file.');
                }

                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);

                audioPlayer.src = audioUrl;
                downloadLink.href = audioUrl;
                downloadLink.download = "hindi_audio_output.mp3";

                resultBox.classList.remove('hidden');
            } catch (err) {
                errorBox.textContent = err.message;
                errorBox.classList.remove('hidden');
            } finally {
                loadingState.classList.add('hidden');
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTMLResponse(content=HTML_CONTENT)

@app.post("/convert/")
async def convert_srt_to_mp3(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.srt'):
        raise HTTPException(status_code=400, detail="Only .srt files are allowed.")
    
    unique_id = str(uuid.uuid4())
    input_path = f"temp_{unique_id}.srt"
    output_mp3_path = f"output_{unique_id}.mp3"
    chunks_dir = f"chunks_{unique_id}"
    
    try:
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)
            
        try:
            subs = pysrt.open(input_path, encoding='utf-8')
        except Exception:
            subs = pysrt.open(input_path, encoding='latin-1')
            
        if not subs:
            raise HTTPException(status_code=400, detail="The uploaded .srt file is empty or invalid.")
            
        os.makedirs(chunks_dir, exist_ok=True)
        combined_audio = AudioSegment.empty()
        silence = AudioSegment.silent(duration=300)
        
        for i, sub in enumerate(subs):
            text = sub.text.replace('\n', ' ').strip()
            if not text:
                continue
                
            chunk_path = os.path.join(chunks_dir, f"part_{i}.mp3")
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(chunk_path)
            
            segment = AudioSegment.from_mp3(chunk_path)
            combined_audio += segment + silence
            
        if len(combined_audio) == 0:
            raise HTTPException(status_code=400, detail="No readable text found inside the .srt file.")
            
        combined_audio.export(output_mp3_path, format="mp3")
        
        return FileResponse(
            output_mp3_path, 
            media_type="audio/mpeg", 
            filename="hindi_audio_output.mp3"
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Conversion processing error: {str(e)}")
        
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(chunks_dir):
            shutil.rmtree(chunks_dir)
