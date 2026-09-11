#!/usr/bin/env python3
import datetime
import yaml
import urllib
import os
import subprocess
from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
import json
import sys
import time
import whisper
from pprint import pprint
import markdown2

import requests
from indexer import index_video, collection, search_and_answer, get_video_summary, get_video_title_and_date

VIDEO_PATH = "videos"
SAVED_PATH = "data/"

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
    video_title, video_publish_date = get_video_title_and_date(video_id)
    subtitles = get_subtitles(out_location)
    summary = get_video_summary(subtitles)
    out = "<html><title>Downloaded Youtube Video</title><body>\n"
    out += f"Completed downloading video {video_id}: {video_title} on {video_publish_date}"
    out += "</body></html>"
    CONFIG[video_id] = ConfigItem({
        "file": out_location,
        "name": video_title,
        "publish_date": video_publish_date,
        "subtitles": subtitles,
        "summary": summary,
    })
    save_config()
    index_video(video_id, subtitles)
    return out

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    answer, context = search_and_answer(query)
    sources_html = context.replace("\n","\n<br>")
    answer = markdown2.markdown(answer)
    return f'<h3>Answer:</h3>{answer}<br><hr><br><form action="/search" method="post"><h2><label>Search in videos:</label></h2><br><input type="text" name="query" style="width: 300px;"><br><br><input type="submit" value="Search"></form>'
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

    print(vals, file=sys.stderr)
    return render_template('step2.html', subtitles=vals, video_id=video_id, video_name = video_config.name, summary = markdown2.markdown(video_config.summary))

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

    # Produce the 1:1 speaker-following captioned version with the
    # shared burn_subtitles renderer (Libre Baskerville, karaoke highlight).
    # Only the captioned file is stored; the plain cut is an intermediate
    # used as render input and removed on success.
    try:
        from burn_subtitles import render_captioned_square, apply_cut_start_case
        words = []
        for sub in video_config.subtitles:
            try:
                s = float(sub['start'])
            except (KeyError, TypeError, ValueError):
                continue
            if "end" in sub:
                e = float(sub['end'])
            elif "duration" in sub:
                e = s + float(sub['duration'])
            else:
                continue
            if e <= start + 1e-6 or s >= end - 1e-6:
                # zero overlap with the cut: "life."-type words ending
                # exactly at `start` (or starting exactly at `end`) have
                # no audio inside the clip, so they must not be captioned
                continue
            text = str(sub.get('text', '')).strip()
            if not text:
                continue
            words.append({"start": max(s - start, 0.0), "end": e - start,
                          "text": text})
        words.sort(key=lambda w: w["start"])
        apply_cut_start_case(video_config.subtitles, words, start)
        square_file = f"{video_id}_{start}_{end}_square_captioned.mp4"
        square_location = os.path.join(SAVED_PATH, square_file)
        render_captioned_square(out_location, words, square_location)
        os.remove(out_location)
    except Exception as ex:
        print(f"Square captioned render failed: {ex}")
        out = "<html><head><title>download file</title></head>"
        out += '<body><p style="font-size:30px">'
        out += f'<a href="download/{out_file}" download target="_blank">DOWNLOAD</a>'
        out += f"<br>Square captioned version failed: {ex}"
        out += '</p></body></html>'
        return out

    out = "<html><head><title>download file</title></head>"
    out += '<body><p style="font-size:30px">'
    out += f'<a href="download/{square_file}" download target="_blank">DOWNLOAD</a>'
    out += '</p></body></html>'
    return out


@app.route('/download/<file>')
def download(file):
    return send_from_directory(SAVED_PATH, file)
