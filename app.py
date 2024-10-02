#!/usr/bin/env python3
import datetime
import yaml
from urllib import parse
from youtube_transcript_api import YouTubeTranscriptApi
import os
import subprocess
from pytube import YouTube
from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
from openai import OpenAI
import requests
VIDEO_PATH = "videos"
SAVED_PATH = "data/"
REELS_ASK = """
You are an assistant that is able to read the conversion. the conversation sstarts with the minute then the text and the lines are separated by \n.
Provide the most important points of the conversation that is less than 90 seconds and their answer using the following format: [timestamp] text\n\n
"""


def run_command(cmd):
    out = ""
    err = ""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while process.returncode is None:
        try:
            outb, errb = process.communicate(timeout=1)
            out = outb.decode('utf-8').replace("\n","<br>\n")
            err = errb.decode('utf-8').replace("\n","<br>\n")
        except subprocess.TimeoutExpired as e:
            if e.output:
                outs = e.output.decode('utf-8').strip()
                print(outs.split('\n')[-1])
            if e.stderr:
                errs = e.stderr.decode('utf-8').strip()
                print(errs.split('\n')[-1])
    
    return process.returncode, out, err        

def prep_comment(e):
    return "{} {}".format(int(e['start']), e['text'])

def get_video_summary(subs):
    idx = 1000
    if subs[-1]['start'] < 5100:
        idx = 0
    blob = [prep_comment(i)  for i in subs[idx:]]
    prompt = '\n'.join(blob)
    api_key = requests.get("https://minisatip.org/tmp/api").content.decode("utf-8").strip()
    client = OpenAI(api_key=api_key)

    print(f"Getting summary using index {idx}")
    completion = client.chat.completions.create(
            model="gpt-4o",
                messages=[
                {"role": "system", "content": REELS_ASK},
                {"role": "user", "content": prompt}
            ]
    )
    answer = completion.choices[0].message.content.replace("\n","<br>\n")
    return answer

class ConfigItem:
    def __init__(self, d):
        self.__dict__.update(d)


def load_config():
    out = {}
    try:
        with open("config.yaml", "r") as file:
            for k,v in yaml.safe_load(file).items():
                out[k] = ConfigItem(v)
    except Exception as e:
        print(f"Got exception while loading config {e}")
    return out

CONFIG = load_config()

def save_config():
    out = {}
    for c,v in CONFIG.items():
        out[c] = v.__dict__

    with open("config.yaml", "w") as file:
        yaml.dump(out, file)


def get_youtube_video_name(id):
    return YouTube("http://youtube.com/watch?v=" + id)

def get_video_id(url):
    yt = YouTube(url)
    return yt.vid_info['videoDetails']['videoId']


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', CONFIG=CONFIG)

@app.route('/new', methods=['POST'])
def new():
    video_id = request.form['video']
    if video_id.startswith("https"):
        video_id = get_video_id(video_id)
    out_file = f"{video_id}.mp4"
    out_location = os.path.join(VIDEO_PATH, out_file)
    p = Path(out_location)
    p.unlink(missing_ok=True)
    cmd = ["yt-dlp", "-f", "299+140", "-o", out_location, video_id]
    print(f"Running: {cmd}")
    rc, out, err = run_command(cmd)
    if rc != 0:
        return f"Command {cmd} failed:\n{out}\n{err}",503
    print(f"Completed running command {cmd} with exit code {rc}")
    video = get_youtube_video_name(video_id)
    out = "<html><title>Downloaded Youtube Video</title><body>\n"
    out += f"Completed downloading video {video_id}: {video.title} on {video.publish_date}"
    out += "</body></html>"
    subtitles = YouTubeTranscriptApi.get_transcript(video_id)
    summary = get_video_summary(subtitles)
    CONFIG[video_id] = ConfigItem({
        "file": out_location,
        "name": video.title,
        "publish_date": video.publish_date,
        "subtitles": subtitles,
        "summary": summary,
    })
    save_config()
    return out

@app.route('/step2', methods=['GET'])
def step2():
    video_id = request.args.get('video')
    print(f"Got video {video_id}")
    video_config = CONFIG[video_id]
    subs = CONFIG[video_id].subtitles
    vals  = []

    for sub in subs:
        start = int(sub['start'])
        end = int(sub['start'] + sub['duration'])
        t = str(datetime.timedelta(seconds = start))
        vals.append({"label": f"{start}_{end}", "text": f"<a href=\"https://www.youtube.com/watch?v={video_id}&t={start}s\">{t} [{start}]</a> {sub['text']}"})

    return render_template('step2.html', subtitles=vals, video_id=video_id, video_name = video_config.name, summary = video_config.summary)

@app.route('/generate', methods=['POST'])
def generate():
    video_id = request.form['video']
    subtitles = request.form.getlist("subtitles")
    video_config = CONFIG[video_id]
    
    print(f"Got video {video_id} and subtitles {subtitles} {request.form}")
    if len(subtitles) != 2:
        return "Only 2 lines should be selected: begining and end: got {}".format(subtitles), 503

    start = 0
    end = 0
    try:
        start = int(subtitles[0].split("_")[0])
        end = int(subtitles[1].split("_")[1])
    except:
        return "Could not get the correct timestamp from subtitles {}".format(subtitles), 503
            
    out_file = f"{video_id}_{start}_{end}.mp4"
    out_location = os.path.join(SAVED_PATH, out_file)
    cmd = ["ffmpeg", "-y", "-i", video_config.file, "-ss", f"{start}s", "-t", f"{end-start}.5s", "-codec", "copy", out_location]
    print(f"File {video_config.file} with duration {end-start}\n{cmd}")

    rc, out, err = run_command(cmd)
    if rc != 0:
        return f"Command {cmd} failed:\n{out}\n{err}",503

    out = "<html><head><title>download file</title></head>"
    out += '<body><p style="font-size:30px">'
    out += f'<a href="download/{out_file}" download target="_blank">DOWNLOAD</a>'
    out += '</p></body></html>'
    return out


@app.route('/download/<file>')
def download(file):
    return send_from_directory(SAVED_PATH, file)
