"""Atlas RAG — desktop chat client.

Flow:
  1. If API already running  → go straight to chat
  2. Otherwise              → startup screen → run vigilantpack run → chat
"""
from __future__ import annotations

import subprocess
import threading

import customtkinter as ctk
import httpx

API_URL       = "http://localhost:8080"
VIGILANTPACK  = "/Users/macbook/.local/bin/vigilantpack"
REPO_DIR      = "/Users/macbook/Documents/GitHub/vigilant-rag"
MANIFEST      = f"{REPO_DIR}/vigilant.yaml"

SUGGESTIONS = [
    "What topics are covered in my notes?",
    "Summarise the key ideas from last week",
    "What are my open tasks?",
    "Explain the main concept in my research notes",
]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ── Startup screen ────────────────────────────────────────────────────────────

class StartupFrame(ctk.CTkFrame):
    def __init__(self, master, on_ready, on_error):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self._on_ready = on_ready
        self._on_error = on_error
        self._build()
        threading.Thread(target=self._run, daemon=True).start()

    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        center = ctk.CTkFrame(self, fg_color="transparent")
        center.place(relx=0.5, rely=0.45, anchor="center")

        ctk.CTkLabel(
            center, text="Atlas RAG",
            font=ctk.CTkFont(size=36, weight="bold"),
        ).pack(pady=(0, 6))

        self._status = ctk.CTkLabel(
            center, text="Starting services…",
            font=ctk.CTkFont(size=13), text_color="gray",
        )
        self._status.pack(pady=(0, 18))

        self._bar = ctk.CTkProgressBar(center, width=320, mode="indeterminate")
        self._bar.pack(pady=(0, 18))
        self._bar.start()

        self._log = ctk.CTkTextbox(
            center, width=520, height=180,
            font=ctk.CTkFont(family="Menlo", size=11),
            state="disabled", text_color="gray70",
        )
        self._log.pack()

    def _run(self):
        try:
            proc = subprocess.Popen(
                [VIGILANTPACK, "run", "--file", MANIFEST],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=REPO_DIR,
            )
            for raw in proc.stdout:
                line = raw.rstrip()
                self.after(0, lambda l=line: self._append(l))
            proc.wait()
            if proc.returncode == 0:
                self.after(0, self._done)
            else:
                self.after(0, lambda: self._fail("vigilantpack run exited with an error."))
        except FileNotFoundError:
            self.after(0, lambda: self._fail(f"vigilantpack not found at {VIGILANTPACK}"))
        except Exception as exc:
            self.after(0, lambda: self._fail(str(exc)))

    def _append(self, line: str):
        self._log.configure(state="normal")
        self._log.insert("end", line + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

        lower = line.lower()
        if "infra" in lower:
            self._status.configure(text="Starting infrastructure…")
        elif "models" in lower:
            self._status.configure(text="Loading models…")
        elif "runtime" in lower:
            self._status.configure(text="Starting app…")
        elif "ready" in lower:
            self._status.configure(text="Almost ready…")

    def _done(self):
        self._bar.stop()
        self._status.configure(text="Ready!", text_color="#2ecc71")
        self.after(700, self._on_ready)

    def _fail(self, msg: str):
        self._bar.stop()
        self._status.configure(text=f"Error: {msg}", text_color="#e74c3c")
        self._on_error(msg)


# ── Chat screen ───────────────────────────────────────────────────────────────

class ChatFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._has_messages = False
        self._build()
        self._show_welcome()
        self._check_health()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        bar = ctk.CTkFrame(self, height=52, corner_radius=0)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            bar, text="Atlas RAG",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).grid(row=0, column=0, padx=18, pady=14)

        self._status = ctk.CTkLabel(
            bar, text="● online",
            font=ctk.CTkFont(size=12), text_color="#2ecc71",
        )
        self._status.grid(row=0, column=2, padx=(0, 8))

        # Clear button
        ctk.CTkButton(
            bar, text="Clear", width=60, height=30,
            corner_radius=8,
            fg_color="transparent",
            border_width=1,
            border_color=("gray70", "gray40"),
            text_color=("gray40", "gray60"),
            hover_color=("gray85", "gray25"),
            font=ctk.CTkFont(size=12),
            command=self._clear_chat,
        ).grid(row=0, column=3, padx=(0, 14))

        # Chat area
        self._chat = ctk.CTkScrollableFrame(self, corner_radius=0)
        self._chat.grid(row=1, column=0, sticky="nsew")
        self._chat.grid_columnconfigure(0, weight=1)

        # Input bar
        foot = ctk.CTkFrame(self, height=72, corner_radius=0, fg_color=("gray90", "gray17"))
        foot.grid(row=2, column=0, sticky="ew")
        foot.grid_columnconfigure(0, weight=1)

        self._entry = ctk.CTkTextbox(
            foot, height=48, corner_radius=10,
            font=ctk.CTkFont(size=13),
        )
        self._entry.grid(row=0, column=0, sticky="ew", padx=(14, 8), pady=12)
        self._entry.bind("<Return>", self._on_enter)

        self._btn = ctk.CTkButton(
            foot, text="Send", width=80, height=48,
            corner_radius=10, command=self._send,
        )
        self._btn.grid(row=0, column=1, padx=(0, 14), pady=12)

    # ── Welcome state ─────────────────────────────────────────────────────────

    def _show_welcome(self):
        self._welcome = ctk.CTkFrame(self._chat, fg_color="transparent")
        self._welcome.grid(sticky="nsew", pady=60)
        self._welcome.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self._welcome, text="Atlas RAG",
            font=ctk.CTkFont(size=28, weight="bold"),
        ).grid(pady=(0, 6))

        ctk.CTkLabel(
            self._welcome,
            text="Ask anything about your knowledge base.",
            font=ctk.CTkFont(size=14), text_color="gray",
        ).grid(pady=(0, 32))

        # Suggestion chips
        chips = ctk.CTkFrame(self._welcome, fg_color="transparent")
        chips.grid()

        for i, suggestion in enumerate(SUGGESTIONS):
            col = i % 2
            row = i // 2
            btn = ctk.CTkButton(
                chips,
                text=suggestion,
                width=280, height=48,
                corner_radius=10,
                fg_color=("gray88", "gray20"),
                hover_color=("gray80", "gray27"),
                text_color=("gray20", "gray80"),
                font=ctk.CTkFont(size=12),
                anchor="w",
                command=lambda s=suggestion: self._send_suggestion(s),
            )
            btn.grid(row=row, column=col, padx=6, pady=6)

    def _send_suggestion(self, text: str):
        self._entry.delete("1.0", "end")
        self._entry.insert("1.0", text)
        self._send()

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_enter(self, event) -> str | None:
        if not (event.state & 0x1):
            self._send()
            return "break"

    def _send(self):
        text = self._entry.get("1.0", "end").strip()
        if not text:
            return

        if not self._has_messages:
            self._welcome.destroy()
            self._has_messages = True

        self._entry.delete("1.0", "end")
        self._bubble_user(text)
        self._bubble_thinking()
        self._btn.configure(state="disabled")
        threading.Thread(target=self._call, args=(text,), daemon=True).start()

    def _clear_chat(self):
        for widget in self._chat.winfo_children():
            widget.destroy()
        self._has_messages = False
        self._show_welcome()

    def _call(self, query: str):
        try:
            r = httpx.post(
                f"{API_URL}/api/v1/query",
                json={"query": query, "mode": "factual", "top_k": 5},
                timeout=90.0,
            )
            r.raise_for_status()
            self.after(0, lambda: self._show_answer(r.json()))
        except httpx.ConnectError:
            self.after(0, lambda: self._show_error("Lost connection — is Atlas RAG still running?"))
        except Exception as exc:
            self.after(0, lambda: self._show_error(str(exc)))
        finally:
            self.after(0, lambda: self._btn.configure(state="normal"))

    def _check_health(self):
        def check():
            try:
                r = httpx.get(f"{API_URL}/health", timeout=3.0)
                ok = r.status_code == 200
            except Exception:
                ok = False
            text  = "● online"  if ok else "● offline"
            color = "#2ecc71"   if ok else "#e74c3c"
            self.after(0, lambda: self._status.configure(text=text, text_color=color))
        threading.Thread(target=check, daemon=True).start()

    # ── Bubbles ───────────────────────────────────────────────────────────────

    def _bubble_user(self, text: str):
        row = ctk.CTkFrame(self._chat, fg_color="transparent")
        row.grid(sticky="ew", pady=(4, 0))
        row.grid_columnconfigure(0, weight=1)

        bubble = ctk.CTkFrame(row, corner_radius=14, fg_color=("#1a6fc4", "#1a4f8a"))
        bubble.grid(sticky="e", padx=(120, 14))
        ctk.CTkLabel(
            bubble, text=text, wraplength=560,
            justify="left", font=ctk.CTkFont(size=13),
            padx=14, pady=10,
        ).pack()
        self._scroll()

    def _bubble_thinking(self):
        self._thinking = ctk.CTkFrame(self._chat, fg_color="transparent")
        self._thinking.grid(sticky="ew", pady=(4, 0))
        ctk.CTkLabel(
            self._thinking, text="  thinking…",
            font=ctk.CTkFont(size=13, slant="italic"), text_color="gray",
        ).pack(anchor="w", padx=14)
        self._scroll()

    def _show_answer(self, data: dict):
        self._thinking.destroy()

        card = ctk.CTkFrame(self._chat, corner_radius=14)
        card.grid(sticky="ew", padx=14, pady=(4, 8))
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card, text=data["answer"],
            wraplength=800, justify="left",
            font=ctk.CTkFont(size=13),
            padx=16, pady=10, anchor="w",
        ).grid(sticky="ew", pady=(4, 0))

        ctk.CTkLabel(
            card, text=f"{data.get('total_latency_ms', 0)} ms",
            font=ctk.CTkFont(size=11), text_color="gray50",
            padx=16, pady=4, anchor="w",
        ).grid(sticky="w")

        sources = data.get("sources", [])
        if sources:
            ctk.CTkFrame(card, height=1, fg_color="gray30").grid(sticky="ew", padx=14, pady=4)
            ctk.CTkLabel(
                card, text="Sources",
                font=ctk.CTkFont(size=11, weight="bold"), text_color="gray50",
                padx=16, anchor="w",
            ).grid(sticky="w")
            for src in sources:
                ctk.CTkLabel(
                    card,
                    text=f"  · {src['title']}   {src['score']:.2f}",
                    font=ctk.CTkFont(size=11), text_color="gray60",
                    padx=16, pady=2, anchor="w",
                ).grid(sticky="w")
            ctk.CTkFrame(card, height=10, fg_color="transparent").grid()

        self._scroll()

    def _show_error(self, message: str):
        self._thinking.destroy()
        card = ctk.CTkFrame(self._chat, corner_radius=14, fg_color=("#3d1414", "#3d1414"))
        card.grid(sticky="ew", padx=14, pady=(4, 8))
        ctk.CTkLabel(
            card, text=f"Error: {message}",
            text_color="#ff6b6b", wraplength=800,
            font=ctk.CTkFont(size=13), padx=16, pady=12,
        ).pack(anchor="w")
        self._scroll()

    def _scroll(self):
        self.after(60, lambda: self._chat._parent_canvas.yview_moveto(1.0))


# ── App ───────────────────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Atlas RAG")
        self.geometry("960x720")
        self.minsize(640, 480)

        if self._api_running():
            ChatFrame(self)
        else:
            StartupFrame(self, on_ready=self._to_chat, on_error=lambda _: None)

    def _api_running(self) -> bool:
        try:
            return httpx.get(f"{API_URL}/health", timeout=2.0).status_code == 200
        except Exception:
            return False

    def _to_chat(self):
        for w in self.winfo_children():
            w.destroy()
        ChatFrame(self)


if __name__ == "__main__":
    App().mainloop()
