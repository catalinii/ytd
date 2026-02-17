There are at least 2 environment variables:
- APIKEY - Youtube API KEY - You can get one from https://developers.google.com/youtube/v3/getting-started using your google account
- GEMINI_KEY - Google API KEY for Gemini: https://developers.google.com/youtube/v3/getting-started 
- OPENAI_API_KEY - OpenAI_API_KEY can be obtained from openai.com

yt-dlp is also required, but is being downloaded by "run" script
In indexer.py there is a setting related to the default model, currently Gemini or GPT* models are supported, and the corresponding API KEY will be used based on thed default model.

There are 3 files currently in the repository:
- app.py (flask) - downloads videos and create subtitles using openai-whisper and allows selecting the 2 words which are using as start and end point in the video that will be downloaded
- reindex_all - Indexes all videos from an youtube playlist (requires OPENAI_KEY or GEMINI_KEY and APIKEY - youtube key)
- indexer.py - contains the code that indexes and searches in the existing index (created with reindex_all) and then provides the output of the search (requires OPENAI_KEY or GEMINI_KEY depending on the model)



