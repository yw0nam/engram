# Engram memory provider for Hermes

A thin [MemoryProvider](https://github.com/ly-wang19/engram) that connects a Hermes agent to a running
Engram server (see `deploy/` — `docker compose up -d`). Context-only: relevant long-term memory is
**injected** into each turn — including how facts changed (`current X · prev Y until T`) — and each turn
is written back for asynchronous consolidation. No tools to call; memory just shows up.

## Install

Symlink (or copy) this directory into your Hermes user-plugins dir, then select it:

```bash
ln -s "$(pwd)/integrations/hermes/engram" "${HERMES_HOME:-$HOME/.hermes}/plugins/engram"
```

Set `memory.provider: engram` in your Hermes config.

## Config

Env (or `$HERMES_HOME/engram/config.json` with the same keys minus the `ENGRAM_` prefix):

| Var | Default | Meaning |
|-----|---------|---------|
| `ENGRAM_URL` | `http://127.0.0.1:9178` | Engram server base URL |
| `ENGRAM_NAMESPACE` | agent identity, else `hermes` | memory namespace (= bearer token, engram `ENGRAM_OPEN` mode) |
| `ENGRAM_TIMEOUT` | `30` | request timeout (s) |

One Engram server serves many isolated namespaces; the bearer token **is** the namespace.
