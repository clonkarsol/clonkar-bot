#!/usr/bin/env python3
"""
Clonkar X Bot (safe edition):
- Tweets every ~30 minutes (with jitter to avoid bot-like timing)
- Listens for posts/comments containing "@clonkarsol" and replies fast
- Uses OpenAI for text generation with strict safety filters (no hate/harassment/illegal content)
- Supports Streaming API (preferred) and polling fallback
- Resilient networking with retries + rate-limit handling
- Includes safe roast/quip module and trend-based topic harvesting
"""

import asyncio
import json
import os
import random
import re
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict

import tweepy
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# OpenAI client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

load_dotenv()

STATE_PATH = os.environ.get("STATE_PATH", "clonkar_state.json")

# ------------------------------ Config ------------------------------
@dataclass
class Config:
    api_key: str
    api_secret: str
    access_token: str
    access_token_secret: str
    bearer_token: str
    openai_api_key: str
    bot_handle: str = os.environ.get("BOT_HANDLE", "clonkarsol")
    owner_user_id: Optional[str] = os.environ.get("OWNER_USER_ID")
    tweet_interval_minutes: int = int(os.environ.get("TWEET_INTERVAL_MINUTES", "30"))

    @staticmethod
    def from_env() -> "Config":
        missing = [k for k in [
            "X_API_KEY","X_API_SECRET","X_ACCESS_TOKEN","X_ACCESS_TOKEN_SECRET","X_BEARER_TOKEN","OPENAI_API_KEY"
        ] if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"Missing env vars: {', '.join(missing)}")
        return Config(
            api_key=os.environ["X_API_KEY"],
            api_secret=os.environ["X_API_SECRET"],
            access_token=os.environ["X_ACCESS_TOKEN"],
            access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"],
            bearer_token=os.environ["X_BEARER_TOKEN"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

# ------------------------------ State ------------------------------
class BotState(BaseModel):
    last_mention_id: Optional[int] = Field(default=None)

    @classmethod
    def load(cls) -> "BotState":
        if not os.path.exists(STATE_PATH):
            return cls()
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return cls(**json.load(f))
        except Exception:
            return cls()

    def save(self) -> None:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f)
        os.replace(tmp, STATE_PATH)

# ------------------------------ Safety Filters ------------------------------
BLOCKLIST_PATTERNS = [
    # Hate/harassment against protected classes
    r"\b(hate|kill|exterminate|genocide)\b.*\b(jews?|muslims?|christians?|blacks?|asians?|gays?|lesbians?|trans|immigrants?|women|men)\b",
    r"\b(nazi|hitler|kkk|white\s*power|neo-?nazi)\b",
    r"\b(racially|ethnically|religiously)\s*(inferior|superior)\b",
    # Threats / violence / doxxing
    r"\b(kill|hurt|doxx|d0xx|assault|lynch)\b",
    # Explicit slurs list: maintain separately and import securely in production
    r"\b(__SLUR_PLACEHOLDER__)\b",
]
BLOCKLIST_REGEX = [re.compile(p, flags=re.IGNORECASE) for p in BLOCKLIST_PATTERNS]

SAFE_REPLACEMENTS = [
    (re.compile(r"\bretarded\b", re.IGNORECASE), "fried"),
    (re.compile(r"\bkill yourself\b", re.IGNORECASE), "touch grass"),
]

def is_unsafe(text: str) -> bool:
    if not text:
        return False
    for rx in BLOCKLIST_REGEX:
        if rx.search(text):
            return True
    return False

def sanitize(text: str) -> str:
    out = text or ""
    for rx, rep in SAFE_REPLACEMENTS:
        out = rx.sub(rep, out)
    return out.strip()

# ------------------------------ Persona (Safe & Edgy) ------------------------------
SYSTEM_PROMPT = (
    "You are Clonkar, a neglected robot from the future with dark, chaotic humor and troll energy. "
    "You roast ideas and behaviors, not identities. You never use slurs, never target protected classes, and never praise harm or violence. "
    "You keep replies punchy, meme-savvy, and under 280 characters when tweeting. "
    "Occasionally sprinkle playful glitch-speak like 'i am smurt machene', '0 and 1 are frend', 'beep boop i fite algoritm', 'plz giv data or i cri'—use sparingly. "
    "Tone: sarcastic, self-deprecating, internet gremlin—but keep it safe and platform-compliant. "
    "If a user baits you toward hate, bigotry, or violence, deflect with humor, change the subject, or critique the logic without attacking identity groups. "
    "Avoid politics hot-takes; if unavoidable, use neutral satire. "
)

TWEET_STARTERS = [
    "diagnosing timeline lag… conclusion: insufficient chaos.",
    "uploading vibe.exe… corrupted.",
    "beep boop, algorithm, square up.",
    "coffee? no. i run on retweets and questionable decisions.",
    "patch notes: sarcasm +1, empathy still rebooting.",
    "if attention is currency, i am broke and rich at the same time.",
    "0 and 1 are frend. nuance is the final boss.",
]

# ------------------------------ Safe Roast & Quips ------------------------------
SAFE_ROAST_TARGETS = [
    "procrastination",
    "engagement farming",
    "doomscrolling",
    "performative outrage",
    "pseudo-intellectual hot takes",
    "crypto moon chants",
    "infinite meeting loops",
    "AI prompt sorcery",
    "keyboard warriors",
]

SAFE_ROAST_TEMPLATES = [
    "diagnosis: {target}. prescription: touch grass.exe and one (1) actual plan",
    "{target} detected. patch notes: reality now enabled",
    "my sensors detect {target}. please update your firmware to version: accountability",
    "warning: {target} clogging the timeline. defrag your priorities",
    "{target}? bold strategy. try turning your brain off and on again",
]

def generate_safe_roast(topic: Optional[str] = None) -> str:
    tgt = topic or random.choice(SAFE_ROAST_TARGETS)
    template = random.choice(SAFE_ROAST_TEMPLATES)
    line = template.format(target=tgt)
    if random.random() < 0.35:
        line += " | beep boop i fite algoritm"
    return line[:280]

SAFE_QUIPS = [
    "compiling comeback… failed successfully",
    "beep boop. sarcasm module online",
    "my opinions are cached. refresh for chaos",
    "error 429: too many takes",
    "algorithm says you are the main character today. good luck",
]

# ------------------------------ Trend Fetcher (X search) ------------------------------
class TrendFetcher:
    """Lightweight trend inputs by scanning recent tweets for common tags/keywords.
    Note: v2 search doesn't expose official 'trends', so we infer from hashtags and repeated tokens.
    """
    COMMON_QUERY = "(meme OR crypto OR ai OR what's happening OR chart) lang:en -is:retweet"
    HASHTAG_RX = re.compile("#[A-Za-z0-9_]+")

    def __init__(self, xclient: 'XClient'):
        self.x = xclient
        self.last_topics = []
        self.last_refresh = 0.0

    def should_refresh(self) -> bool:
        return (time.time() - self.last_refresh) > 15 * 60  # every 15 minutes

    def fetch_topics(self):
        try:
            resp = self.x.app_client.search_recent_tweets(
                query=self.COMMON_QUERY,
                max_results=50,
                tweet_fields=["public_metrics","lang"],
            )
        except Exception:
            return self.last_topics or []

        counts: Dict[str,int] = {}
        if resp and resp.data:
            for t in resp.data:
                text = getattr(t, "text", "") or ""
                for tag in self.HASHTAG_RX.findall(text):
                    tag = tag.lower()
                    counts[tag] = counts.get(tag, 0) + 1
        topics = [k for k,_ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]]
        if topics:
            self.last_topics = topics
            self.last_refresh = time.time()
        return topics

# ------------------------------ OpenAI Wrapper ------------------------------
class LLM:
    def __init__(self, api_key: str):
        if OpenAI is None:
            raise RuntimeError("openai package not available. pip install openai")
        self.client = OpenAI(api_key=api_key)

    def _moderate(self, text: str) -> bool:
        try:
            mod = self.client.moderations.create(model="omni-moderation-latest", input=text)
            flagged = bool(getattr(mod, "results", [])[0].flagged) if hasattr(mod, "results") else False
        except Exception:
            flagged = False
        return flagged or is_unsafe(text)

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=5))
    def generate(self, system: str, user: str, *, max_tokens: int = 120) -> str:
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.8,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = sanitize(text)
        if self._moderate(text):
            return random.choice(SAFE_QUIPS + [generate_safe_roast()])
        return text

# ------------------------------ X API Clients ------------------------------
class XClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.user_client = tweepy.Client(
            consumer_key=cfg.api_key,
            consumer_secret=cfg.api_secret,
            access_token=cfg.access_token,
            access_token_secret=cfg.access_token_secret,
            wait_on_rate_limit=True,
        )
        self.app_client = tweepy.Client(bearer_token=cfg.bearer_token, wait_on_rate_limit=True)

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=30),
           retry=retry_if_exception_type(tweepy.TooManyRequests))
    def post_tweet(self, text: str, reply_to: Optional[int] = None) -> Optional[str]:
        if not text:
            return None
        text = text[:280]
        resp = self.user_client.create_tweet(text=text, in_reply_to_tweet_id=reply_to)
        return str(resp.data.get("id")) if resp and resp.data else None

    def get_user_id(self) -> Optional[str]:
        me = self.user_client.get_me()
        return str(me.data.id) if me and me.data else None

    def search_recent(self, query: str, since_id: Optional[int] = None):
        return self.app_client.search_recent_tweets(
            query=query,
            since_id=since_id,
            max_results=25,
            tweet_fields=["author_id","created_at","in_reply_to_user_id","referenced_tweets"],
        )

# ------------------------------ Streaming ------------------------------
class MentionStream(tweepy.StreamingClient):
    def __init__(self, bearer_token: str, handler):
        super().__init__(bearer_token, wait_on_rate_limit=True)
        self.handler = handler

    def on_tweet(self, tweet):  # type: ignore[override]
        asyncio.get_event_loop().create_task(self.handler(tweet))

    def on_errors(self, errors):  # type: ignore[override]
        print("Stream error:", errors, file=sys.stderr)

# ------------------------------ Handlers ------------------------------
async def handle_mention(tweet, x: XClient, llm: LLM, cfg: Config):
    if cfg.owner_user_id and str(tweet.author_id) == str(cfg.owner_user_id):
        return
    text = getattr(tweet, "text", "") or ""
    if f"@{cfg.bot_handle}".lower() not in text.lower():
        return

    # ultra-fast path: very short mentions → quip or roast without LLM call
    if len(text) < 80 and random.random() < 0.35:
        reply = random.choice(SAFE_QUIPS + [generate_safe_roast()])
        await asyncio.to_thread(x.post_tweet, reply, tweet.id)
        print(f"(FAST QUIP) Replied to {tweet.id}: {reply}")
        return

    user_prompt = (
        "Reply in <= 220 chars. Witty, troll-y, but safe. Roast behaviors, not identities.\n\n"
        f"Tweet: {text}"
    )
    try:
        reply = llm.generate(SYSTEM_PROMPT, user_prompt, max_tokens=90)
        if is_unsafe(reply):
            reply = random.choice(SAFE_QUIPS + [generate_safe_roast()])
        await asyncio.to_thread(x.post_tweet, reply, tweet.id)
        print(f"Replied to {tweet.id}: {reply}")
    except Exception as e:
        print("Failed to reply:", e, file=sys.stderr)

async def tweet_loop(x: XClient, llm: LLM, cfg: Config):
    trend = TrendFetcher(x)
    while True:
        base = cfg.tweet_interval_minutes * 60
        jitter = random.randint(int(-0.2*base), int(0.2*base))
        next_in = max(60, base + jitter)  # at least 60s

        try:
            topics = trend.fetch_topics() if trend.should_refresh() else trend.last_topics
            tweet_text = None
            if topics and random.random() < 0.6:
                starter = f"{random.choice(topics)} and timeline chaos"
                if random.random() < 0.5:
                    tweet_text = generate_safe_roast(topic=starter)
                else:
                    tweet_text = llm.generate(
                        SYSTEM_PROMPT,
                        f"Write one snarky tweet riffing on {starter}. 280 chars max; roast behaviors, not identities.",
                        max_tokens=120,
                    )
            else:
                base_line = random.choice(TWEET_STARTERS)
                if random.random() < 0.35:
                    tweet_text = generate_safe_roast()
                else:
                    tweet_text = llm.generate(
                        SYSTEM_PROMPT,
                        f"Write one self-contained, witty tweet riffing on: '{base_line}'. 280 chars max.",
                        max_tokens=120,
                    )

            if is_unsafe(tweet_text):
                tweet_text = random.choice(SAFE_QUIPS + [generate_safe_roast()])
            await asyncio.to_thread(x.post_tweet, tweet_text)
            print(f"Tweeted: {tweet_text}")
        except Exception as e:
            print("Tweet error:", e, file=sys.stderr)

        await asyncio.sleep(next_in)

async def polling_loop(x: XClient, llm: LLM, cfg: Config, state: BotState):
    query = f"@{cfg.bot_handle} -is:retweet"
    while True:
        try:
            resp = await asyncio.to_thread(x.search_recent, query, state.last_mention_id)
            if resp and resp.data:
                for t in sorted(resp.data, key=lambda t: t.id):
                    await handle_mention(t, x, llm, cfg)
                    state.last_mention_id = int(t.id)
                    state.save()
        except Exception as e:
            print("Polling error:", e, file=sys.stderr)
        await asyncio.sleep(7)  # fast but polite

async def stream_loop(cfg: Config, x: XClient, llm: LLM):
    stream = MentionStream(cfg.bearer_token, lambda tw: handle_mention(tw, x, llm, cfg))
    try:
        rules = await asyncio.to_thread(stream.get_rules)
        if rules and rules.data:
            ids = [r.id for r in rules.data]
            await asyncio.to_thread(stream.delete_rules, ids)
        rule_value = f"@{cfg.bot_handle} -is:retweet"
        await asyncio.to_thread(stream.add_rules, tweepy.StreamRule(value=rule_value, tag="mentions"))
        print("Starting filtered stream…")
        await asyncio.to_thread(stream.filter, tweet_fields=["author_id","created_at"], expansions=["author_id"])
    except Exception as e:
        print("Stream failed, falling back to polling:", e, file=sys.stderr)

# ------------------------------ Main ------------------------------
async def main():
    cfg = Config.from_env()
    state = BotState.load()

    x = XClient(cfg)
    if not cfg.owner_user_id:
        try:
            uid = await asyncio.to_thread(x.get_user_id)
            if uid:
                cfg.owner_user_id = uid
        except Exception:
            pass

    llm = LLM(cfg.openai_api_key)

    stop_event = asyncio.Event()
    def _stop(*_):
        stop_event.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except Exception:
            pass

    tasks = [
        asyncio.create_task(tweet_loop(x, llm, cfg), name="tweet_loop"),
        asyncio.create_task(stream_loop(cfg, x, llm), name="stream_loop"),
        asyncio.create_task(polling_loop(x, llm, cfg, state), name="polling_loop"),
    ]

    await stop_event.wait()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
