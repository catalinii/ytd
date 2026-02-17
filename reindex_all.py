import os
import argparse
import subprocess
import traceback
from indexer import index_video, collection # Import collection to allow cleaning

def get_video_id(url):
    if "watch?" in url:
        return url.split("v=")[1].split("&")[0]
    if "live/" in url:
        return url.split("live/")[1].split("?")[0]
    return url.strip()

def reindex(playlist_url=None, clean_db=False):
    if clean_db:
        print("Cleaning ChromaDB collection...")
        collection.delete()
        print("ChromaDB collection cleaned.")

    if playlist_url:
        print(f"Fetching video URLs from playlist: {playlist_url}")
        cmd = ["yt-dlp", "--flat-playlist", "--print", "url", playlist_url]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            video_urls = result.stdout.strip().split('\n')
            for url in video_urls:
                if url:
                    video_id = get_video_id(url)
                    print(f"Indexing video from playlist: {video_id}")
                    try:
                        index_video(video_id)
                    except Exception as e:
                        traceback.print_exc()
        except subprocess.CalledProcessError as e:
            print(f"Error fetching playlist URLs: {e}")
            print(f"Stderr: {e.stderr}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print("No playlist URL provided. Nothing to re-index.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-index YouTube videos.")
    parser.add_argument("--playlist_url", help="URL of the YouTube playlist to index.", default=None)
    parser.add_argument("--clean", action="store_true", help="Clean the ChromaDB collection before re-indexing.")
    args = parser.parse_args()

    reindex(playlist_url=args.playlist_url, clean_db=args.clean)
