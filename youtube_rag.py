# youtube_rag.py

import os
import re
import time
import requests
import json

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

from youtube_transcript_api import YouTubeTranscriptApi
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
# from dotenv import load_dotenv

# load_dotenv()

CLIENT_SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS_JSON", "client_secrets.json")
TOKEN_FILE = os.getenv("YOUTUBE_TOKEN_FILE", "token.json")
SCOPES = [
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube.readonly",
]
REDIRECT_URI = "http://127.0.0.1:8000/oauth2callback"

from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleRequest


_cached_token = None
_cached_expiry = None
_cached_creds = None

def generate_service_account_token():
    global _cached_token, _cached_expiry, _cached_creds

    # If token exists & not expired → reuse
    if _cached_token and _cached_expiry and datetime.utcnow() < _cached_expiry:
        return _cached_token

    # Load service account JSON from env
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON env variable missing")

    sa_info = json.loads(sa_json)

    SCOPES = ["https://www.googleapis.com/auth/generative-language"]

    # Recreate credentials only when needed
    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=SCOPES
    )
    creds.refresh(GoogleRequest())

    # Cache token + expiry
    _cached_token = creds.token
    # Token expires in 1 hour → refresh early after 50 mins
    _cached_expiry = datetime.utcnow() + timedelta(minutes=50)

    return _cached_token


def call_gemini_with_service_account(prompt: str):
    token = generate_service_account_token()

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()
def extract_gemini_text(resp: dict) -> str:
    """
    Safely extract text from Gemini response.
    Returns fallback empty string if candidates are missing.
    """
    try:
        candidates = resp.get("candidates", [])
        if not candidates:
            return ""  # safety block or empty response

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""

        return parts[0].get("text", "").strip()

    except Exception:
        return ""


class GeminiRateLimiter:
    def __init__(self, max_calls_per_minute: int = 14):
        self.max_calls_per_minute = max_calls_per_minute
        self.call_times = deque()

    def wait_if_needed(self):
        current_time = time.time()
        while self.call_times and (current_time - self.call_times[0]) > 60:
            self.call_times.popleft()
        if len(self.call_times) >= self.max_calls_per_minute:
            wait_time = 60 - (current_time - self.call_times[0]) + 1
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f}s …")
                time.sleep(wait_time)
            current_time = time.time()
            while self.call_times and (current_time - self.call_times[0]) > 60:
                self.call_times.popleft()
        self.call_times.append(current_time)

    def get_calls_in_last_minute(self) -> int:
        current_time = time.time()
        while self.call_times and (current_time - self.call_times[0]) > 60:
            self.call_times.popleft()
        return len(self.call_times)

@dataclass
class Config:
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    DEV_INDEX_NAME: str = "youtube-videos-dev"
    PROD_INDEX_NAME: str = "youtube-videos-prod"
    PINECONE_DIMENSION: int = 384
    MAX_CHUNK_WORDS: int = 500
    OVERLAP_WORDS: int = 50
    GEMINI_MAX_CALLS_PER_MINUTE: int = 14

class YouTubeRAGSystem:
    def __init__(self, config: Config):
        self.config = config

        # import google.generativeai as genai
        # genai.configure(api_key=config.GEMINI_API_KEY)
        # self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        # self.gemini_rate_limiter = GeminiRateLimiter(config.GEMINI_MAX_CALLS_PER_MINUTE)
        self.gemini_model = None
        self.gemini_rate_limiter = GeminiRateLimiter(config.GEMINI_MAX_CALLS_PER_MINUTE)

        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.dev_index = self._setup_index(config.DEV_INDEX_NAME)
        self.prod_index = self._setup_index(config.PROD_INDEX_NAME)

        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.ytt_api = YouTubeTranscriptApi()

        self.credentials: Optional[Credentials] = None
        self._load_credentials()

    def _setup_index(self, index_name: str):
        try:
            existing = self.pc.list_indexes().names()
            if index_name not in existing:
                self.pc.create_index(
                    name=index_name,
                    dimension=self.config.PINECONE_DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                print(f"Created Pinecone index: {index_name}")
            return self.pc.Index(index_name)
        except Exception as e:
            print(f"Error setting up index {index_name}: {e}")
            raise

    def _load_credentials(self):
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            try:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(GoogleAuthRequest())
            except Exception as e:
                print("Failed to refresh credentials:", e)
            self.credentials = creds

    def save_credentials(self, creds: Credentials):
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        self.credentials = creds

    def _build_youtube_api(self):
        if not self.credentials:
            return None
        try:
            yt = build("youtube", "v3", credentials=self.credentials)
            return yt
        except Exception as e:
            print("Error building YouTube API client:", e)
            return None

    def _parse_vtt(self, vtt_text: str) -> List[Dict[str, Any]]:
        lines = vtt_text.splitlines()
        out = []
        prev_start = None
        prev_duration = None
        buffer = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "-->" in line:
                parts = line.split("-->")
                start_s = parts[0].strip()
                end_s = parts[1].strip().split()[0]
                def to_secs(t: str) -> float:
                    ps = t.split(":")
                    if len(ps) == 3:
                        h, m, s = ps
                        return int(h)*3600 + int(m)*60 + float(s)
                    elif len(ps) == 2:
                        m, s = ps
                        return int(m)*60 + float(s)
                    else:
                        return float(ps[0])
                st = to_secs(start_s)
                ed = to_secs(end_s)
                dur = ed - st
                if buffer and prev_start is not None and prev_duration is not None:
                    out.append({"text": " ".join(buffer), "start": prev_start, "duration": prev_duration})
                    buffer = []
                prev_start = st
                prev_duration = dur
            else:
                buffer.append(line)
        if buffer and prev_start is not None and prev_duration is not None:
            out.append({"text": " ".join(buffer), "start": prev_start, "duration": prev_duration})
        return out

    def fetch_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        yt = self._build_youtube_api()
        if yt:
            try:
                resp = yt.captions().list(part="snippet", videoId=video_id).execute()
                items = resp.get("items", [])
                if items:
                    sel = None
                    for it in items:
                        if it["snippet"].get("trackKind") == "ASR":
                            sel = it
                            break
                    if not sel:
                        sel = items[0]
                    cap_id = sel["id"]
                    dl = yt.captions().download(id=cap_id, tfmt="vtt").execute()
                    if dl:
                        parsed = self._parse_vtt(dl)
                        if parsed:
                            return parsed
            except HttpError as he:
                print("HTTP error on captions API:", he)
            except Exception as e:
                print("Error in YouTube caption API path:", e)

        try:
            fetched = self.ytt_api.fetch(video_id)
            out = []
            for s in fetched:
                out.append({"text": s.text, "start": s.start, "duration": s.duration})
            return out
        except Exception as e:
            print(f"youtube_transcript_api fallback failed for {video_id}: {e}")
            return []

    def chunk_transcript(self, transcript_data: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_start_time = 0

        for snippet in transcript_data:
            words = snippet["text"].split()
            wcount = len(words)
            if current_word_count + wcount > self.config.MAX_CHUNK_WORDS and current_chunk:
                chunk_text = " ".join(s["text"] for s in current_chunk)
                chunk_end_time = current_chunk[-1]["start"] + current_chunk[-1]["duration"]
                chunks.append({
                    "video_id": video_id,
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "start_time": chunk_start_time,
                    "end_time": chunk_end_time,
                    "duration": chunk_end_time - chunk_start_time,
                    "word_count": current_word_count
                })
                overlap = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap
                current_word_count = sum(len(s["text"].split()) for s in overlap)
                chunk_start_time = overlap[0]["start"] if overlap else snippet["start"]
            current_chunk.append(snippet)
            current_word_count += wcount
            if not chunks and len(current_chunk) == 1:
                chunk_start_time = snippet["start"]

        if current_chunk:
            chunk_text = " ".join(s["text"] for s in current_chunk)
            chunk_end_time = current_chunk[-1]["start"] + current_chunk[-1]["duration"]
            chunks.append({
                "video_id": video_id,
                "chunk_id": len(chunks),
                "text": chunk_text,
                "start_time": chunk_start_time,
                "end_time": chunk_end_time,
                "duration": chunk_end_time - chunk_start_time,
                "word_count": current_word_count
            })

        return chunks

    def generate_metadata(self, chunk_text: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following video transcript chunk and provide:
        1. A concise summary (2-3 sentences)
        2. Main topics/themes (3-5 keywords)
        3. Content category (educational, entertainment, tutorial, news, etc.)
        4. Key concepts mentioned

        Transcript chunk:
        {chunk_text}

        Provide response in this format:
        Summary: [summary]
        Topics: [topic1, topic2, topic3]
        Category: [category]

        Concepts: [concept1, concept2, concept3]
        """
        try:
            self.gemini_rate_limiter.wait_if_needed()
            # resp = self.gemini_model.generate_content(prompt)
            # text = resp.text
            resp = call_gemini_with_service_account(prompt)
            text = extract_gemini_text(resp)


            meta = {"summary": "", "topics": [], "category": "", "concepts": []}
            for line in text.split("\n"):
                if line.startswith("Summary:"):
                    meta["summary"] = line.replace("Summary:", "").strip()
                elif line.startswith("Topics:"):
                    meta["topics"] = [t.strip() for t in line.replace("Topics:", "").strip().split(",") if t.strip()]
                elif line.startswith("Category:"):
                    meta["category"] = line.replace("Category:", "").strip()
                elif line.startswith("Concepts:"):
                    meta["concepts"] = [c.strip() for c in line.replace("Concepts:", "").strip().split(",") if c.strip()]
            return meta
        except Exception as e:
            print("Error in generate_metadata:", e)
            return {"summary": "Unable to generate summary", "topics": [], "category": "unknown", "concepts": []}

    def create_embedding(self, text: str) -> List[float]:
        try:
            emb = self.embedding_model.encode(text)
            return emb.tolist()
        except Exception as e:
            print("Error creating embedding:", e)
            return [0.0] * self.config.PINECONE_DIMENSION

    def process_video(self, video_id: str) -> bool:
        transcript = self.fetch_transcript(video_id)
        if not transcript:
            print(f"No transcript available for video {video_id}")
            return False

        full_text = "\n".join(s["text"] for s in transcript)
        from db import SessionLocal, Video
        db = SessionLocal()
        try:
            new = Video(video_id=video_id,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        transcript=full_text)
            db.add(new)
            db.commit()
        except Exception as e:
            db.rollback()
            print("Database insert failed:", e)
            db.close()
            return False
        db.close()

        chunks = self.chunk_transcript(transcript, video_id)
        vectors = []
        for chunk in chunks:
            meta = self.generate_metadata(chunk["text"])
            emb = self.create_embedding(chunk["text"] + " " + meta["summary"])
            vid_chunk = f"{video_id}_chunk_{chunk['chunk_id']}"
            vect_meta = {
                "video_id": video_id,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"][:1000],
                "summary": meta["summary"],
                "topics": meta["topics"],
                "category": meta["category"],
                "concepts": meta["concepts"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "duration": chunk["duration"],
                "word_count": chunk["word_count"],
                "video_url": f"https://www.youtube.com/watch?v={video_id}&t={int(chunk['start_time'])}s"
            }
            vectors.append({"id": vid_chunk, "values": emb, "metadata": vect_meta})

        try:
            self.dev_index.upsert(vectors=vectors)
            self.prod_index.upsert(vectors=vectors)
        except Exception as e:
            print("Upsert error in process_video:", e)
            return False

        return True

    def search_videos(self, query: str, top_k: int = 5, use_prod: bool = True) -> List[Dict[str, Any]]:
        emb = self.create_embedding(query)
        idx = self.prod_index if use_prod else self.dev_index
        try:
            resp = idx.query(vector=emb, top_k=top_k, include_metadata=True)
            return resp.get("matches", [])
        except Exception as e:
            print("Search error:", e)
            return []

    def recommend_videos(self, query: str, top_k: int = 3, use_prod: bool = True) -> str:
        matches = self.search_videos(query, top_k * 2, use_prod)
        if not matches:
            return "No relevant videos found for your query."
        video_groups: Dict[str, List[Dict[str, Any]]] = {}
        for m in matches:
            vid = m["metadata"]["video_id"]
            video_groups.setdefault(vid, []).append(m)
        top = list(video_groups.items())[:top_k]
        rec_text = f"User Query: {query}\n\nTop Video Recommendations:\n\n"
        for i, (vid, chunks) in enumerate(top, start=1):
            best = max(chunks, key=lambda x: x["score"])
            md = best["metadata"]
            prompt = f"""
            Based on the user query "{query}" and this video content, explain why this video is relevant:

            Video Summary: {md['summary']}
            Topics: {', '.join(md['topics'])}
            Category: {md['category']}
            Key Concepts: {', '.join(md['concepts'])}

            Provide a brief explanation (2-3 sentences).
            """
            try:
                self.gemini_rate_limiter.wait_if_needed()
                # resp = self.gemini_model.generate_content(prompt)
                # explanation = resp.text.strip()
                resp = call_gemini_with_service_account(prompt)
                explanation = extract_gemini_text(resp)



            except Exception as e:
                explanation = f"This video covers {md['category']} content related to {', '.join(md['topics'][:3])}."
            rec_text += (
                f"{i}. **Video ID: {vid}**\n"
                f"   **URL:** {md['video_url']}\n"
                f"   **Relevance Score:** {best['score']:.3f}\n"
                f"   **Category:** {md['category']}\n"
                f"   **Why it suits query:** {explanation}\n"
                f"   **Summary:** {md['summary']}\n\n"
            )
        return rec_text

    @staticmethod
    def extract_video_id(url_or_id: str) -> Optional[str]:
        if not url_or_id:
            return None
        if re.fullmatch(r"[A-Za-z0-9_-]{10,20}", url_or_id):
            return url_or_id
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url_or_id)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
        if "youtube.com" in hostname:
            qs = parse_qs(query)
            if "v" in qs:
                return qs["v"][0]
            if path.startswith("/embed/"):
                parts = path.split("/")
                if len(parts) >= 3:
                    return parts[2]
            if path.startswith("/v/"):
                parts = path.split("/")
                if len(parts) >= 3:
                    return parts[2]
        if "youtu.be" in hostname:
            vid = path.lstrip("/")
            if vid:
                return vid.split("?")[0]
        m = re.search(r"v=([^&]+)", url_or_id)
        if m:
            return m.group(1)
        m2 = re.search(r"youtu\.be/([^?&]+)", url_or_id)
        if m2:
            return m2.group(1)
        return None

    def batch_process_videos(
        self,
        video_ids: List[str],
        batch_size: int = 5,
        pause_between_batches: int = 60
    ):
        """
        Process YouTube videos in batches to reduce API quota pressure.
        """
        total = len(video_ids)
        print(f" Starting batch processing for {total} videos (batch size = {batch_size})")

        for i in range(0, total, batch_size):
            batch = video_ids[i : i + batch_size]
            batch_number = i // batch_size + 1
            print(f"\n Processing batch {batch_number}: videos {i+1}–{min(i+batch_size, total)}")

            for vid in batch:
                try:
                    success = self.process_video(vid)
                    if success:
                        print(f" Processed: {vid}")
                    else:
                        print(f" Skipped (no transcript): {vid}")
                except Exception as e:
                    print(f" Error processing {vid}: {e}")
                    if "quota" in str(e).lower() or "403" in str(e):
                        print("⏳ API quota warning: waiting 60 seconds before continuing…")
                        time.sleep(60)

            print(f"⏸ Waiting {pause_between_batches}s before next batch…")
            time.sleep(pause_between_batches)
            processed = min(i + batch_size, total)
            print(f" Progress: {processed}/{total} videos processed")

        print("\n All batches completed!")
