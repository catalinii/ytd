import os
import argparse
import chromadb
import urllib.parse
import urllib.request
import json
import subprocess
import httpx
import sys
import time
from openai import OpenAI
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

CHROMA_PATH = "chroma_index"
COLLECTION_NAME = "video_subtitles_free"

REELS_ASK = """
You are an assistant that is able to read the conversion. the conversation sstarts with the minute then the text and the lines are separated by \n.
Provide the most practical advices that the text emphasisrs using the following format: [timestamp] text\n\n
"""

SUBTITLE_CACHE_PATH = "subtitle_cache"
DEFAULT_MODEL = "gemini-3-flash-preview"

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=default_ef)

def get_openai_completion(messages, model="gpt-5-mini"):
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise Exception("OpenAI API Key not found:  https://openai.com/")
    client = OpenAI(api_key=api_key, http_client=httpx.Client())
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content

def get_gemini_completion(messages, model="gemini-3-flash-preview"):
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise Exception("Gemini API Key not found: https://aistudio.google.com/")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    contents = []
    system_instruction = None
    for msg in messages:
        if msg['role'] == 'system':
            system_instruction = {"parts": [{"text": msg['content']}]}
        elif msg['role'] == 'user':
            contents.append({"role": "user", "parts": [{"text": msg['content']}]})
        elif msg['role'] == 'assistant':
            contents.append({"role": "model", "parts": [{"text": msg['content']}]})
            
    payload = {"contents": contents}
    if system_instruction:
        payload["system_instruction"] = system_instruction
        
    try:
        response = httpx.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Gemini error: {str(e)}"

def get_completion(messages, model=None):
    model = model or DEFAULT_MODEL
    model_lower = model.lower()
    if "gpt" in model_lower:
        return get_openai_completion(messages, model=model)
    elif "gemini" in model_lower:
        return get_gemini_completion(messages, model=model)
    else:
        return f"Unsupported model: {model}"

def get_video_title_and_date(video_id):
    """Infers video title using YouTube API or yt-dlp."""
    api_key = os.environ.get("APIKEY")
    if api_key is None:
        raise Exception("Google APIKEY is required https://developers.google.com/youtube/v3/getting-started")
    if api_key:
        try:
            params = {'id': video_id, 'key': api_key, 'part': 'snippet'}
            url = f"https://www.googleapis.com/youtube/v3/videos?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                if data['items']:
                    return data['items'][0]['snippet']['title'], data['items'][0]['snippet']['publishedAt']
        except Exception as e:
            print(f"Error fetching title via API: {e}")

    return None, None

def get_subtitles_from_youtube(video_id):
    """Fetches subtitles using YouTubeTranscriptApi, with caching."""
    os.makedirs(SUBTITLE_CACHE_PATH, exist_ok=True)
    cache_file = os.path.join(SUBTITLE_CACHE_PATH, f"{video_id}.json")

    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"Loading subtitles for {video_id} from cache...")
        with open(cache_file, 'r') as f:
            return json.load(f)

    # If not in cache, fetch from YouTube
    print(f"Fetching subtitles for {video_id} from YouTube...")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(transcript, f)
        return transcript
    except Exception as e:
        print(f"Error fetching subtitles for {video_id}: {e}")
        return None

def index_video(video_id, subtitles=None, video_title=None):
    """Indexes a video's subtitles into ChromaDB with metadata."""
    # Prevent adding duplicate videos
    existing = collection.get(where={"video_id": video_id}, limit=1)
    if existing and existing['ids']:
        print(f"Video {video_id} is already indexed. Skipping.")
        return False

    if subtitles is None:
        subtitles = get_subtitles_from_youtube(video_id)
    
    if not subtitles:
        print(f"No subtitles found for video {video_id}")
        return False

    video_date = None
    if video_title is None or video_date is None:
        video_title, video_date  = get_video_title_and_date(video_id)

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Build full text and track offsets for timestamps
    full_text = ""
    offsets = [] # List of (char_start, timestamp)
    for s in subtitles:
        offsets.append((len(full_text), s['start']))
        full_text += s['text'] + " "
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True # This gives us the character offset in full_text
    )

    print(f"Indexing {video_id} {video_title} {video_date}")
    
    chunks = text_splitter.create_documents([full_text])
    
    documents = []
    metadatas = []
    ids = []
    
    for i, doc in enumerate(chunks):
        char_index = doc.metadata['start_index']
        
        # Find the timestamp corresponding to this char_index
        # We find the largest offset that is <= char_index
        timestamp = 0
        for offset_char, offset_time in reversed(offsets):
            if offset_char <= char_index:
                timestamp = offset_time
                break

        documents.append(doc.page_content)
        metadatas.append({
            "video_id": video_id,
            "video_title": video_title,
            "date": video_date,
            "url": f"{video_url}&t={int(timestamp)}s",
            "start_time": timestamp
        })
        ids.append(f"{video_id}_{i}")
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully indexed video '{video_title}' ({len(chunks)} chunks).")
    return True

def search_videos(query, n_results=5):
    """Searches indexed videos for the given query."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

local_word = 0
def prep_comment(e):
    global local_word
    local_word += 1
    if local_word > 5:
        return "{} {}".format(int(e['start']), e['text'])
    return e['text']

def get_video_summary(subs):
    idx = 1000
    if not subs:
        return "No subtitles available for summary."
    if subs[-1]['start'] < 5100:
        idx = 0
    blob = [prep_comment(i)  for i in subs[idx:]]
    prompt = '\n'.join(blob)

    print(f"Getting summary using index {idx}")
    messages = [
        {"role": "system", "content": REELS_ASK},
        {"role": "user", "content": prompt}
    ]
    # For OpenAI we use gpt-4o-mini as it is cheaper/faster for summaries
    # For others, we let get_completion use the default or we could specify
    return get_completion(messages)

def search_and_answer(query, n_results=10):
    """Searches videos and generates an answer."""
    start = time.time()
    results = search_videos(query, n_results=n_results)
    took = time.time() - start
    print(f"Completed chromadb search in {took}s", file=sys.stderr)
    
    context = ""
    if results and 'documents' in results and results['documents']:
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            context += f"From video '{meta['video_title']}' ({meta['url']}) on '{meta.get('date')}':\n{doc}\n\n"
    
    prompt = f"Using the following snippets answer the user's question: {query} by summarizing the content from the transcripts. Once done include the urls as references at the bottom including the date they were done, but do not include in the summary. Also geberate markdown answer. Get the answer from the first 5 transcripts and use the remaining ones only if really needed. Convert all the URLs into links. Follow up with 3 questions on simple topics after the references section. \n\nTranscripts:\n{context}"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions based on video transcripts"},
        {"role": "user", "content": prompt}
    ]
    
    answer = get_completion(messages)
    return answer, context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Subtitle Indexer")
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a video")
    index_parser.add_argument("video_id", help="YouTube Video ID")
    index_parser.add_argument("--title", help="Video Title", default=None)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed videos")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--n_results", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if args.command == "index":
        index_video(args.video_id, video_title=args.title)
    elif args.command == "search":
        results = search_videos(args.query, n_results=args.n_results)
        if results and 'documents' in results and results['documents']:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                print(f"--- Result {i+1} ---")
                print(f"Video: {meta.get('video_title')} ({meta.get('video_id')})")
                print(f"URL: {meta.get('url')}")
                print(f"Snippet: {doc}")
                print()
        else:
            print("No results found.")
    else:
        parser.print_help()
