#!/usr/bin/env python3
import datetime
import yaml
import urllib
import os
import subprocess
from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
import json
import time
import whisper
from pprint import pprint
import markdown2

import requests
from indexer import index_video, collection, get_video_summary, search_and_answer


class ConfigItem:
    def __init__(self, d):
        self.__dict__.update(d)

class YoutubeVideo:
    def __init__(self, id):
        params = {'id': id, 'key': os.environ.get("APIKEY"),
                'part': 'snippet,statistics'}

        url = 'https://www.googleapis.com/youtube/v3/videos'

        query_string = urllib.parse.urlencode(params)
        url = url + "?" + query_string

        with urllib.request.urlopen(url) as response:
            response_text = response.read()
            data = json.loads(response_text.decode())

        self.id = id
        self.publish_date = data['items'][0]['snippet']['publishedAt']
        self.title = data['items'][0]['snippet']['title']




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

def get_video_id(url):
    if "watch?" in url:
        return url.split("v=")[1].split("&")[0]
    if "live/" in url:
        return url.split("live/")[1].split("?")[0]
    return url


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', CONFIG=CONFIG)

def ecast(elem):
    try:
        return elem.item()
    except:
        return elem
def get_sub_dict(e):
    return {
            'start': ecast(e['start']),
            'end': ecast(e['end']),
            'text': e['word'],
            }

def get_subtitles(url):
    model = whisper.load_model("turbo")
    result = model.transcribe(url, word_timestamps=True)
    subs = [get_sub_dict(word) for segment in result['segments'] for word in segment["words"]]
    pprint(subs)
    return subs

@app.route('/new', methods=['POST'])
def new():
    video_id = request.form['video']
    url = video_id
    if video_id.startswith("https"):
        video_id = get_video_id(video_id)
    out_file = f"{video_id}.mp4"
    out_location = os.path.join(VIDEO_PATH, out_file)
    p = Path(out_location)
    p.unlink(missing_ok=True)

    video_format = 312
    audio_format = 234
    audio_format = 140
    video_format = 299
    cmd = ["yt-dlp", "-f", f"{video_format}+{audio_format}", "-o", out_location, url]
    
    print(f"Running: {cmd}")
    rc, out, err = run_command(cmd)
    if rc != 0:
        return f"Command {cmd} failed:\n{out}\n{err}",503
    print(f"Completed running command {cmd} with exit code {rc}")
    subtitles = get_subtitles(out_location)
    summary = get_video_summary(subtitles)
    video = YoutubeVideo(video_id)
    out = "<html><title>Downloaded Youtube Video</title><body>\n"
    out += f"Completed downloading video {video_id}: {video.title} on {video.publish_date}"
    out += "</body></html>"
    CONFIG[video_id] = ConfigItem({
        "file": out_location,
        "name": video.title,
        "publish_date": video.publish_date,
        "subtitles": subtitles,
        "summary": summary,
    })
    save_config()
    index_video(video_id, subtitles, video.title)
    return out

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    answer, context = search_and_answer(query)
    sources_html = context.replace("\n","\n<br>")
    answer = markdown2.markdown(answer)
    return answer
    return f"<h3>Answer:</h3>{answer}<br><br><h3>Sources:</h3>{sources_html}"

@app.route('/step2', methods=['GET'])
def step2():
    video_id = request.args.get('video')
    print(f"Got video {video_id}")
    video_config = CONFIG[video_id]
    subs = CONFIG[video_id].subtitles
    vals  = []
    words = 0
    oldbr = True

    for sub in subs:
        start = float(sub['start'])
        if "duration" in sub:
            end = float(sub['start'] + sub['duration'])
        else:
            end = float(sub['end'])
        starti = int(start)
        endi = int(end)
        t = str(datetime.timedelta(seconds = starti))
        text = sub['text']
        words += len(text.strip().split(" "))
        br = ""
        if "." in text:
            br = "<br>"
            words = 0
        if words > 5:
            words = 0
            br = "<br>"

        href = ""
        if oldbr:
            href = f"<a href=\"https://www.youtube.com/watch?v={video_id}&t={starti}s\">{t} [{starti}] </a>"

        v = {
            "href": href,
            "label": f"{start:.3f}_{end:.3f}", 
            "text": text,
            "br": br,
        }
        vals.append(v)
        oldbr = br == "<br>"

    return render_template('step2.html', subtitles=vals, video_id=video_id, video_name = video_config.name, summary = video_config.summary)

@app.route('/generate', methods=['GET'])
def generate():
    video_id = request.args['video']
    subtitles = request.args.getlist("subtitles")
    video_config = CONFIG[video_id]
    
    print(f"Got video {video_id} and subtitles {subtitles} {request.form}")
    if len(subtitles) != 2:
        return "Only 2 lines should be selected: begining and end: got {}".format(subtitles), 503

    start = 0
    end = 0
    try:
        start = float(subtitles[0].split("_")[0])
        end = float(subtitles[1].split("_")[1])
    except:
        return "Could not get the correct timestamp from subtitles {}".format(subtitles), 503
            
    out_file = f"{video_id}_{start}_{end}.mp4"
    out_location = os.path.join(SAVED_PATH, out_file)
    cmd = ["ffmpeg", "-y", "-i", video_config.file, "-ss", f"{start}s", "-t", f"{end-start}s",  out_location]
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
