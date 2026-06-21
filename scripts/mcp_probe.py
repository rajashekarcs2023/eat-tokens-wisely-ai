"""Feasibility probe: drive a REAL MCP server over stdio JSON-RPC (no SDK), call a
tool, and measure how many tokens its result would inject into an LLM's context."""
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suffix.tokens import count_tokens  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MCP:
    """Minimal newline-delimited JSON-RPC client for an MCP stdio server."""

    def __init__(self, command, args):
        self.p = subprocess.Popen([command, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._id = 0

    def _send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def _rpc(self, method, params=None, timeout=60):
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.p.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result")
        raise TimeoutError(f"no response to {method}")

    def notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def initialize(self):
        r = self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                      "clientInfo": {"name": "suffix-probe", "version": "1.0"}})
        self.notify("notifications/initialized")
        return r

    def tools(self):
        return self._rpc("tools/list").get("tools", [])

    def call(self, name, arguments):
        res = self._rpc("tools/call", {"name": name, "arguments": arguments})
        # tool results come back as a list of content blocks; concatenate text
        return "".join(b.get("text", "") for b in res.get("content", []))

    def close(self):
        try:
            self.p.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    print("launching real MCP server: @modelcontextprotocol/server-filesystem (may download via npx)…")
    m = MCP("npx", ["-y", "@modelcontextprotocol/server-filesystem", ROOT])
    info = m.initialize()
    print("  server:", info.get("serverInfo"))
    names = [t["name"] for t in m.tools()]
    print("  tools:", names)
    out = m.call("directory_tree", {"path": os.path.join(ROOT, "suffix")})
    print(f"  directory_tree(suffix/) -> {count_tokens(out)} tokens of JSON")
    print("  first 200 chars:", out[:200].replace("\n", " "))
    m.close()
    print("OK — real MCP call works")
