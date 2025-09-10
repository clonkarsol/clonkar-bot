# Clonkar Bot (X/Twitter)

An edgy-but-safe meme bot that:
- Tweets randomly (with jitter)
- Replies fast to mentions of `@clonkarsol` (quip fast-path + LLM path)
- Riffs on lightweight trends (recent hashtag scan)
- Uses OpenAI + local safety filters + moderation to avoid hateful/harassing content

> **Note:** Keep `.env` private. Never commit secrets!

## Quick Start

1) Python 3.10+ recommended
2) Install deps:
```bash
pip install -r requirements.txt
```
3) Copy `.env.example` to `.env` and fill in real values (never commit `.env`):
```bash
cp .env.example .env
```
4) Run:
```bash
python clonkar_bot.py
```

## Deploy

### Railway / Render
- Connect your GitHub repo
- Add env vars from `.env` in the dashboard
- Deploy (Procfile and Dockerfile provided)

### Heroku
```bash
heroku create
heroku buildpacks:add heroku/python
heroku config:set $(grep -v '^#' .env | xargs)   # or paste manually
git push heroku main
heroku ps:scale worker=1
```

### Fly.io / VPS (Docker)
```bash
docker build -t clonkar-bot .
docker run --env-file .env --name clonkar clonkar-bot
```

## Safety & Tone
The bot roasts behaviors/ideas, not identities. It never uses slurs or promotes hate/violence. 
If content gets flagged, it falls back to safe quips/roasts.

