# Interruption Reproduction Case



## Running locally

Create a `.env.local` with the following values:
```
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
GOOGLE_API_KEY=
```

Install project dependencies with:
```
uv sync --group dev
```

Run the agent with:
```
uv run src/agent.py console
```

## Testing interruptions

Ask the agent to tell you a story and test backchannels by saying "mhm", "okay", etc.

Interrupt the agent and ask them to stop or to tell you another story.