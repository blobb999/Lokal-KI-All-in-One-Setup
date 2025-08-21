#!/usr/bin/env python3
"""
ki_setup_allinone.py

Vollständiges GUI-Tool (Tkinter) für dein lokales KI-Docker-Setup:
- STRUKTURIERTE INSTALLATION in korrekter Reihenfolge (1-8)
- Python Setup überprüfung
- Ollama lokale Installation
- Docker & Docker Compose Installation
- Erzeugt Projektstruktur
- Schreibt docker-compose.yml (n8n, postgres, ollama, vision, kyutai-voice, searxng)
- Klont kyutai repo
- Start / Stop Docker Compose
- Stream Logs
- Test Endpoints (n8n, ollama, vision, kyutai, searxng)
- Ollama Modell-Pull
- n8n Workflow Import
- Öffnen von n8n im Browser / Projektordner
- ALLE PORTS KONFIGURIERBAR im GUI

Benötigt:
- Python 3.8+
- Git in PATH
- Docker & Docker Compose (oder Docker Desktop) in PATH
- Adminrechte / docker-Gruppe
"""

import os
import sys
import threading
import subprocess
import shutil
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import time
import json
import urllib.request
import platform
from string import Template
import textwrap
from queue import Queue
try:
    import psutil
    import pynvml
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    
    # Matplotlib Backend und Stil setzen
    matplotlib.use("TkAgg")
    plt.style.use("seaborn-v0_8-darkgrid") 
    
    PSUTIL_OK = True
    PYNVML_OK = True
    MATPLOTLIB_OK = True
except ImportError as e:
    # Fallback, falls eines der Module nicht installiert ist
    print(f"Info: Monitoring-Modul nicht gefunden: {e}. Dashboard wird deaktiviert.")
    PSUTIL_OK = False
    PYNVML_OK = False
    MATPLOTLIB_OK = False

# --------------------------- Konfiguration ---------------------------
DEFAULT_PROJECT_DIR = Path.home() / "mein-ki-setup"
COMPOSE_FILENAME = "docker-compose.yml"

DEFAULT_N8N_PORT = 5678
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_VISION_PORT = 8008
DEFAULT_KYUTAI_PORT = 4005
DEFAULT_SEARXNG_PORT = 8888
DEFAULT_STABLEDIFFUSION_PORT = 7860
DEFAULT_OPENWEBUI_PORT = 7070
DEFAULT_FLOWISE_PORT = 3000

DEFAULT_N8N_USER = "admin"
DEFAULT_N8N_PASS = "deinpasswort"

# --------------------------- Hilfsfunktionen ---------------------------
def is_installed(cmd_name):
    """Prüft ob Kommando verfügbar ist und updated PATH"""
    if platform.system().lower() == "windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                system_path = winreg.QueryValueEx(key, "PATH")[0]
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                try:
                    user_path = winreg.QueryValueEx(key, "PATH")[0]
                except FileNotFoundError:
                    user_path = ""
            
            new_path = system_path + ";" + user_path if user_path else system_path
            os.environ["PATH"] = new_path
        except:
            pass 
    
    return shutil.which(cmd_name) is not None

def get_python_version():
    """Gibt Python Version als Tupel zurück (major, minor, micro)"""
    return sys.version_info[:3]

def check_python_requirements():
    """Prüft ob Python 3.8+ vorhanden ist"""
    version = get_python_version()
    return version >= (3, 8, 0)

def run_cmd_stream(cmd, cwd=None, update_log=None, shell=False, env=None):
    """
    Startet einen Subprozess und streamt stdout in update_log(callback).
    Gibt (returncode, None) zurück oder (1, errorstr).
    """
    try:
        if update_log:
            update_log(f"LAUNCH ➜ {' '.join(cmd) if isinstance(cmd, (list,tuple)) else cmd}\n", tag="info")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=shell, env=env)
    except Exception as e:
        if update_log:
            update_log(f"FEHLER beim Starten: {e}\n", tag="error")
        return 1, str(e)

    for line in proc.stdout:
        if update_log:
            update_log(line.rstrip("\n"), tag="out")
    proc.wait()
    return proc.returncode, None

def run_cmd_capture(cmd, cwd=None, shell=False):
    """
    Führt Befehl aus und gibt (rc, output) zurück.
    """
    try:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=shell)
        return proc.returncode, proc.stdout
    except Exception as e:
        return 1, str(e)


class SystemMonitor:
    def __init__(self, update_callback, log_callback):
        self.update_callback = update_callback
        self.log_callback = log_callback 
        self._stop_event = threading.Event()
        self.is_running = False
        
        self.last_net_io = psutil.net_io_counters()
        self.last_time = time.time()

        self.gpu_handle = None
        if PYNVML_OK:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                self.log_callback(f"Warnung: NVIDIA GPU nicht gefunden oder pynvml-Fehler: {e}", "warning")
                self.gpu_handle = None

    def get_stats(self):
        """Sammelt alle Systemstatistiken"""
        current_time = time.time()
        time_delta = current_time - self.last_time
        if time_delta == 0: time_delta = 1 

        current_net_io = psutil.net_io_counters()
        bytes_sent = current_net_io.bytes_sent - self.last_net_io.bytes_sent
        bytes_recv = current_net_io.bytes_recv - self.last_net_io.bytes_recv
        self.last_net_io = current_net_io
        self.last_time = current_time

        ram = psutil.virtual_memory()
        
        try:
            project_path = Path(DEFAULT_PROJECT_DIR).expanduser()
            if not project_path.exists():
                project_path.mkdir(parents=True, exist_ok=True)
            disk = psutil.disk_usage(str(project_path))
        except Exception:
            disk = psutil.disk_usage('/')

        cpu_temp = "N/A"
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                cpu_temp = temps['coretemp'][0].current
            elif temps:
                cpu_temp = list(temps.values())[0][0].current

        gpu_name, gpu_util, gpu_mem_percent, gpu_mem_used, gpu_mem_total, gpu_temp = [None] * 6
        if self.gpu_handle:
            try:
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = util.gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem_total = mem.total // 1024**2
                gpu_mem_used = mem.used // 1024**2
                gpu_mem_percent = int((mem.used / mem.total) * 100) if mem.total > 0 else 0
                gpu_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                self.gpu_handle = None 

        return {
            "cpu_percent": psutil.cpu_percent(), "cpu_temp": cpu_temp,
            "ram_percent": ram.percent, "ram_total": ram.total / (1024**3), "ram_used": ram.used / (1024**3),
            "disk_percent": disk.percent, "disk_total": disk.total / (1024**3), "disk_used": disk.used / (1024**3),
            "net_down": (bytes_recv / time_delta) / (1024**2),
            "net_up": (bytes_sent / time_delta) / (1024**2),
            "gpu_name": gpu_name, "gpu_util": gpu_util, "gpu_mem_percent": gpu_mem_percent,
            "gpu_mem_used": gpu_mem_used, "gpu_mem_total": gpu_mem_total, "gpu_temp": gpu_temp,
        }

    def run(self):
        self.is_running = True
        while not self._stop_event.is_set():
            try:
                stats = self.get_stats()
                self.update_callback(stats)
            except Exception as e:

                self.log_callback(f"Fehler im Monitor-Thread: {e}", "error")
            time.sleep(2)
        self.is_running = False
        if PYNVML_OK and self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except: pass

    def start(self):
        if not self.is_running:
            self._stop_event.clear()
            threading.Thread(target=self.run, daemon=True).start()

    def stop(self):
        self._stop_event.set()

# --------------------------- GUI Klasse ---------------------------
class AllInOneGUI:
    def __init__(self, root):
        self.root = root
        root.title("Local KI All-in-One Setup & Monitoring v2.2")
        root.geometry("1600x900") 

        self.log_queue = Queue()

        main_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_pane, width=1000)
        main_pane.add(left_frame, weight=3) 

        right_frame = ttk.Frame(main_pane, width=500)
        main_pane.add(right_frame, weight=1)

        # --- Linker Frame ---
        self.setup_frame = ttk.LabelFrame(left_frame, text="🔧 Setup-Reihenfolge (1-8)")
        self.setup_frame.pack(fill=tk.X, padx=8, pady=6)

        setup_steps = [
            ("1⃣ Python 3.8+ prüfen", self.check_python_setup), ("2⃣ Git Installation prüfen", self.check_git_setup),
            ("3⃣ Docker Installation", self.setup_docker_info), ("4⃣ Ollama lokal installieren", self.setup_ollama_local),
            ("5⃣ Projekt anlegen", self.create_project), ("6⃣ Kyutai Repo klonen", self.clone_kyutai),
            ("7⃣ Docker Compose schreiben", self.write_compose), ("8⃣ Docker Services starten", self.docker_up)
        ]
        for i, (text, command) in enumerate(setup_steps):
            btn = ttk.Button(self.setup_frame, text=text, command=self.threaded(command))
            btn.grid(column=i % 4, row=i // 4, padx=4, pady=4, sticky="ew")
        for i in range(4): self.setup_frame.columnconfigure(i, weight=1)

        topf = ttk.LabelFrame(left_frame, text="⚙️ Einstellungen")
        topf.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(topf, text="Projektordner:").grid(column=0, row=0, sticky=tk.W, padx=6, pady=4)
        self.project_var = tk.StringVar(value=str(DEFAULT_PROJECT_DIR))
        self.project_entry = ttk.Entry(topf, textvariable=self.project_var, width=70)
        self.project_entry.grid(column=1, row=0, columnspan=3, sticky=tk.EW, padx=6)
        ttk.Button(topf, text="Browse", command=self.browse_folder).grid(column=4, row=0, padx=6)
        topf.columnconfigure(1, weight=1)

        ttk.Label(topf, text="n8n Benutzer:").grid(column=0, row=1, sticky=tk.W, padx=6, pady=4)
        self.n8n_user = tk.StringVar(value=DEFAULT_N8N_USER)
        ttk.Entry(topf, textvariable=self.n8n_user, width=20).grid(column=1, row=1, sticky=tk.W, padx=6)
        ttk.Label(topf, text="n8n Passwort:").grid(column=2, row=1, sticky=tk.W, padx=6)
        self.n8n_pass = tk.StringVar(value=DEFAULT_N8N_PASS)
        ttk.Entry(topf, textvariable=self.n8n_pass, width=20, show="*").grid(column=3, row=1, sticky=tk.W, padx=6)

        portf = ttk.LabelFrame(topf, text="🔌 Port-Konfiguration")
        portf.grid(column=0, row=2, columnspan=5, sticky=tk.EW, padx=7, pady=8)

        self.n8n_port = tk.IntVar(value=DEFAULT_N8N_PORT)
        self.ollama_port = tk.IntVar(value=DEFAULT_OLLAMA_PORT)
        self.vision_port = tk.IntVar(value=DEFAULT_VISION_PORT)
        self.kyutai_port = tk.IntVar(value=DEFAULT_KYUTAI_PORT)
        self.searxng_port = tk.IntVar(value=DEFAULT_SEARXNG_PORT)
        self.stablediffusion_port = tk.IntVar(value=DEFAULT_STABLEDIFFUSION_PORT)
        self.openwebui_port = tk.IntVar(value=DEFAULT_OPENWEBUI_PORT)
        self.flowise_port = tk.IntVar(value=DEFAULT_FLOWISE_PORT)
        port_configs = [
            ("n8n:", self.n8n_port), ("Ollama:", self.ollama_port), ("Vision:", self.vision_port),
            ("Kyutai:", self.kyutai_port), ("SearxNG:", self.searxng_port), ("Stable Diffusion:", self.stablediffusion_port),
            ("Open-WebUI:", self.openwebui_port), ("Flowise:", self.flowise_port)
        ]
        items_per_row = 4
        for i, (label, var) in enumerate(port_configs):
            row = i // items_per_row
            col_base = (i % items_per_row) * 2
            ttk.Label(portf, text=label).grid(column=col_base, row=row, sticky=tk.W, padx=(10, 2), pady=2)
            ttk.Entry(portf, textvariable=var, width=7).grid(column=col_base + 1, row=row, sticky=tk.W, padx=(0, 10), pady=2)

        # Reset-Button platzieren und Spalten konfigurieren
        num_rows = (len(port_configs) + items_per_row - 1) // items_per_row
        reset_button_col = items_per_row * 2
        portf.columnconfigure(reset_button_col - 1, weight=1) # Spacer-Spalte
        ttk.Button(portf, text="Reset", command=self.reset_ports).grid(column=reset_button_col, row=num_rows - 1, padx=10, sticky=tk.E)

        mgmtf = ttk.LabelFrame(left_frame, text="🎛️ Management & Tests")
        mgmtf.pack(fill=tk.X, padx=8, pady=6)

        mgmt_buttons_config = [
            ("🛑 Docker Stop", self.docker_down), 
            ("📋 Logs streamen", self.stream_logs),
            ("⏹️ Logs stoppen", self.stop_logs), 
            ("📊 Docker Status", self.docker_status),
            
            ("🧪 Endpunkte testen", self.test_endpoints), 
            ("📥 Ollama Modell Pull", self.ollama_pull_dialog),
            ("📤 n8n Workflow import", self.import_n8n_workflow_dialog),
            ("🔍 SearxNG Info", self.show_searx_info),

            ("🌐 Open n8n", self.open_n8n), 
            ("🖼️ Open Stable Diffusion", self.open_stablediffusion),
            ("🌊 Open Flowise", self.open_flowise),
            ("🌐 Open-WebUI", self.open_openwebui),
            ("📁 Projektordner", self.open_project_dir)
        ]
        
        for i, (text, command) in enumerate(mgmt_buttons_config):
            # Die is_threaded Logik muss Open-WebUI und Flowise berücksichtigen
            is_threaded = text not in [
                "📥 Ollama Modell Pull", "⏹️ Logs stoppen", "🌐 Open n8n", 
                "🖼️ Open Stable Diffusion", "🌊 Open Flowise", "🌐 Open-WebUI", 
                "📁 Projektordner"
            ]
            btn_command = self.threaded(command) if is_threaded else command
            btn = ttk.Button(mgmtf, text=text, command=btn_command)
            # Dynamisches Grid-Layout für eine variable Anzahl von Buttons
            btn.grid(column=i % 4, row=i // 4, padx=4, pady=4, sticky="ew")
        
        for i in range(4): mgmtf.columnconfigure(i, weight=1)

        logf = ttk.LabelFrame(left_frame, text="📝 Status & Logs")
        logf.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.logbox = scrolledtext.ScrolledText(logf, wrap=tk.WORD, height=15, bg="#2b2b2b", fg="white", insertbackground="white")
        self.logbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.logbox.tag_config("info", foreground="#a9b7c6")
        self.logbox.tag_config("ok", foreground="#6a8759")
        self.logbox.tag_config("error", foreground="#ff6b68")
        self.logbox.tag_config("warning", foreground="#cc7832")
        self.logbox.tag_config("out", foreground="#d3d3d3")

        # --- Rechter Frame ---
        self.dashboard_frame = ttk.LabelFrame(right_frame, text="💻 System-Dashboard")
        self.dashboard_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # --- Initialisierungslogik ---
        self._stop_logs = threading.Event()
        self.log("🚀 Willkommen zum Local KI Setup Tool v2.2", tag="info")
        
        self.system_monitor = None
        if PSUTIL_OK and MATPLOTLIB_OK:
            self.setup_dashboard()
            self.system_monitor = SystemMonitor(self.update_dashboard_widgets, self.log)
            self.threaded(self.system_monitor.start)()
        else:
            ttk.Label(self.dashboard_frame, text="Dashboard deaktiviert.\n\nModule nicht gefunden:\npsutil, pynvml, matplotlib", justify=tk.CENTER).pack(pady=50)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_log_queue()

    def process_log_queue(self):
        try:
            while not self.log_queue.empty():
                text, tag = self.log_queue.get_nowait()
                self.logbox.configure(state=tk.NORMAL)
                if not text.endswith("\n"): text += "\n"
                self.logbox.insert(tk.END, text, tag)
                self.logbox.see(tk.END)
                self.logbox.configure(state=tk.DISABLED)
        except Exception: pass
        finally:
            self.root.after(100, self.process_log_queue)

    def log(self, text, tag="info"):
        self.log_queue.put((text, tag))

    def on_closing(self):
        if self.system_monitor: self.system_monitor.stop()
        self.root.destroy()

    def setup_dashboard(self):
        self.dashboard_frame.columnconfigure(0, weight=1)
        self.dashboard_frame.rowconfigure(0, weight=1)
        try:
            rgb_16bit = self.root.winfo_rgb(self.root.cget('bg'))
            r, g, b = (c >> 8 for c in rgb_16bit)
            bg_color = f'#{r:02x}{g:02x}{b:02x}'
        except tk.TclError:
            bg_color = "#f0f0f0"

        self.fig = Figure(figsize=(4, 6), dpi=90, facecolor=bg_color)
        self.fig.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1, hspace=0.6)

        self.ax_cpu = self.fig.add_subplot(3, 1, 1)
        self.ax_ram = self.fig.add_subplot(3, 1, 2)
        self.ax_gpu = self.fig.add_subplot(3, 1, 3)
        
        for ax in [self.ax_cpu, self.ax_ram, self.ax_gpu]:
            ax.set_facecolor(bg_color)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.dashboard_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", pady=5)

        text_info_frame = ttk.Frame(self.dashboard_frame)
        text_info_frame.grid(row=1, column=0, sticky="ew", pady=5, padx=10)
        
        self.disk_label = ttk.Label(text_info_frame, text="Disk: --", font=("Segoe UI", 9))
        self.disk_label.pack(fill=tk.X)
        self.net_label = ttk.Label(text_info_frame, text="Netzwerk: --", font=("Segoe UI", 9))
        self.net_label.pack(fill=tk.X)

    def update_dashboard_widgets(self, data):
        if not data or not hasattr(self, 'canvas'): return

        font_props = {'fontsize': 9, 'fontweight': 'light'}
        title_props = {'fontsize': 10, 'fontweight': 'normal'}

        # --- CPU Plot ---
        self.ax_cpu.clear()
        cpu_percent = data.get('cpu_percent', 0)
        cpu_temp = data.get('cpu_temp', 'N/A')
        cpu_temp_str = f"{cpu_temp:.0f}°C" if isinstance(cpu_temp, (int, float)) else cpu_temp
        self.ax_cpu.barh(["CPU"], [cpu_percent], color='#569cd6', height=0.5)
        self.ax_cpu.set_title(f"CPU Auslastung", loc='left', **title_props)
        self.ax_cpu.text(cpu_percent + 2, 0, f"{cpu_percent:.1f}% | {cpu_temp_str}", va='center', ha='left', **font_props)
        
        # --- RAM Plot ---
        self.ax_ram.clear()
        ram_percent = data.get('ram_percent', 0)
        self.ax_ram.barh(["RAM"], [ram_percent], color='#4ec9b0', height=0.5)
        self.ax_ram.set_title(f"RAM: {data.get('ram_used', 0):.1f} / {data.get('ram_total', 0):.1f} GB", loc='left', **title_props)
        self.ax_ram.text(ram_percent + 2, 0, f"{ram_percent:.1f}%", va='center', ha='left', **font_props)

        # --- GPU Plot ---
        self.ax_gpu.clear()
        if data.get("gpu_name"):
            gpu_util = data.get("gpu_util", 0) or 0
            gpu_mem_percent = data.get("gpu_mem_percent", 0) or 0
            gpu_temp = data.get('gpu_temp', 'N/A')
            gpu_temp_str = f"{gpu_temp}°C" if isinstance(gpu_temp, (int, float)) else gpu_temp
            
            self.ax_gpu.barh(["VRAM", "GPU"], [gpu_mem_percent, gpu_util], color=['#d16969', '#ce9178'], height=0.6)
            self.ax_gpu.set_title(f"GPU: {data['gpu_name']}", loc='left', **title_props)
            self.ax_gpu.text(gpu_mem_percent + 2, 0, f"{data.get('gpu_mem_used', 0)}/{data.get('gpu_mem_total', 0)} MB ({gpu_mem_percent}%)", va='center', ha='left', **font_props)
            self.ax_gpu.text(gpu_util + 2, 1, f"{gpu_util}% | {gpu_temp_str}", va='center', ha='left', **font_props)
        else:
            self.ax_gpu.barh(["GPU"], [0], color='grey', height=0.5)
            self.ax_gpu.set_title("GPU: Nicht verfügbar", loc='left', **title_props)
        
        for ax in [self.ax_cpu, self.ax_ram, self.ax_gpu]:
            ax.set_xlim(0, 105)
            ax.tick_params(axis='y', length=0, labelsize=9)
            ax.tick_params(axis='x', length=0, labelsize=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_color('#bbbbbb')

        self.disk_label.config(text=f"💾 Disk: {data.get('disk_used', 0):.1f}/{data.get('disk_total', 0):.1f} GB ({data.get('disk_percent', 0):.1f}%)")
        self.net_label.config(text=f"🌐 Netz: ↓ {data.get('net_down', 0):.2f} MB/s | ↑ {data.get('net_up', 0):.2f} MB/s")

        self.canvas.draw()

    # ------------------- UI Hilfsfunktionen -------------------
    def browse_folder(self):
        d = filedialog.askdirectory(initialdir=self.project_var.get())
        if d:
            self.project_var.set(d)

    def reset_ports(self):
        """Setzt alle Ports auf die Standardwerte zurück"""
        self.n8n_port.set(DEFAULT_N8N_PORT)
        self.ollama_port.set(DEFAULT_OLLAMA_PORT)
        self.vision_port.set(DEFAULT_VISION_PORT)
        self.kyutai_port.set(DEFAULT_KYUTAI_PORT)
        self.searxng_port.set(DEFAULT_SEARXNG_PORT)
        self.stablediffusion_port.set(DEFAULT_STABLEDIFFUSION_PORT)
        self.openwebui_port.set(DEFAULT_OPENWEBUI_PORT)
        self.flowise_port.set(DEFAULT_FLOWISE_PORT)
        self.log("🔄 Ports auf Standardwerte zurückgesetzt (SearxNG: 8888, Stable Diffusion: 7860).", tag="info")

    def log(self, text, tag="info"):
        self.logbox.configure(state=tk.NORMAL)
        if not text.endswith("\n"):
            text += "\n"
        self.logbox.insert(tk.END, text, tag)
        self.logbox.see(tk.END)
        self.logbox.configure(state=tk.DISABLED)


    def threaded(self, fn):
        def wrapper(*args, **kwargs):
            t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
            t.start()
        return wrapper

    def get_docker_compose_content(self):
        """Generiert docker-compose.yml Inhalt mit aktuellen Port-Einstellungen - VISION SERVICE FIX"""
        n8n_port = self.n8n_port.get()
        ollama_port = self.ollama_port.get()
        vision_port = self.vision_port.get()
        kyutai_port = self.kyutai_port.get()
        searxng_port = self.searxng_port.get()
        stablediffusion_webui_port = self.stablediffusion_port.get()
        stablediffusion_api_port = stablediffusion_webui_port - 1
        openwebui_port = self.openwebui_port.get()
        flowise_port = self.flowise_port.get()
        n8n_user = self.n8n_user.get()
        n8n_pass = self.n8n_pass.get()
        
        return """services:
          postgres:
            image: postgres:15
            restart: unless-stopped
            environment:
              POSTGRES_USER: n8n
              POSTGRES_PASSWORD: n8n
              POSTGRES_DB: n8n
            volumes:
              - ./postgres_data:/var/lib/postgresql/data

          ollama:
            image: ollama/ollama
            restart: unless-stopped
            ports:
              - "{ollama_port}:11434"
            volumes:
              - ./ollama_data:/root/.ollama
            deploy:
              resources:
                reservations:
                  devices:
                    - driver: nvidia
                      count: all
                      capabilities: [gpu]
            environment:
              - NVIDIA_VISIBLE_DEVICES=all
              - NVIDIA_DRIVER_CAPABILITIES=compute,utility

          stable-diffusion:
            image: python:3.11-slim
            restart: unless-stopped
            working_dir: /app
            volumes:
              - ./sd_data:/app/data
              - ./sd_models:/app/models
              - ./sd_outputs:/app/outputs
              - ./sd_cache:/root/.cache
            ports:
              - "{stablediffusion_webui_port}:7861"
              - "{stablediffusion_api_port}:7860"
            environment:
              - TORCH_HOME=/app/models/torch
              - HF_HOME=/app/models/huggingface
              - TRANSFORMERS_CACHE=/app/models/transformers
              - DIFFUSERS_CACHE=/app/models/diffusers
              - PYTHONDONTWRITEBYTECODE=1
              - PIP_CACHE_DIR=/root/.cache/pip
              - GRADIO_SERVER_NAME=0.0.0.0
              - GRADIO_SERVER_PORT=7861
            deploy:
              resources:
                reservations:
                  devices:
                    - driver: nvidia
                      count: all
                      capabilities: [gpu]
            command:
              - bash
              - -c
              - |
                mkdir -p /app/models/torch /app/models/huggingface /app/models/transformers /app/models/diffusers
                mkdir -p /app/outputs /root/.cache/pip
                apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*
                pip install --upgrade pip --cache-dir=/root/.cache/pip
                pip install --cache-dir=/root/.cache/pip torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                pip install --cache-dir=/root/.cache/pip diffusers transformers accelerate safetensors gradio pillow numpy scipy fastapi uvicorn python-multipart
                
                cat > server.py << 'PYEOF'
                import gradio as gr
                from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
                import torch
                from PIL import Image
                import os
                from datetime import datetime
                import uuid
                from fastapi import FastAPI, HTTPException
                from fastapi.responses import FileResponse
                from pydantic import BaseModel
                import uvicorn
                from contextlib import asynccontextmanager
                import threading
                
                pipe = None
                
                class GenerateRequest(BaseModel ):
                    prompt: str
                    negative_prompt: str = ""
                    width: int = 512
                    height: int = 512
                    num_inference_steps: int = 20
                    guidance_scale: float = 7.5
                    seed: int = -1
                    num_images: int = 1
                
                def load_model():
                    global pipe
                    try:
                        print("🎨 Lade Stable Diffusion Modell...")
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        print(f"🖥️ Device: {{device}}")
                        model_id = "runwayml/stable-diffusion-v1-5"
                        cache_dir = "/app/models/diffusers"
                        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, cache_dir=cache_dir, local_files_only=False)
                        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                        if device == "cuda":
                            pipe = pipe.to("cuda")
                            pipe.enable_attention_slicing()
                            pipe.enable_vae_slicing()
                            try:
                                pipe.enable_xformers_memory_efficient_attention()
                                print("✅ xFormers aktiviert für bessere Performance")
                            except:
                                print("ℹ️ xFormers nicht verfügbar, verwende Standard-Attention")
                        print("✅ Stable Diffusion bereit!")
                        return True
                    except Exception as e:
                        print(f"❌ Fehler beim Laden des Modells: {{e}}")
                        return False
                
                def generate_image(prompt, negative_prompt="", num_inference_steps=20, guidance_scale=7.5, width=512, height=512, seed=-1, num_images=1):
                    if pipe is None: return None, "❌ Modell nicht geladen! Bitte warten...", []
                    try:
                        if seed == -1: seed = torch.randint(0, 2**32, (1,)).item()
                        generator = torch.Generator(device=pipe.device).manual_seed(seed)
                        images, filenames = [], []
                        for i in range(num_images):
                            print(f"🎨 Generiere Bild {{i+1}}/{{num_images}}: '{{prompt[:50]}}...'")
                            result = pipe(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator if i == 0 else torch.Generator(device=pipe.device).manual_seed(seed + i))
                            image = result.images[0]
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{{timestamp}}_{{str(uuid.uuid4())[:8]}}.png"
                            output_path = f"/app/outputs/{{filename}}"
                            os.makedirs("/app/outputs", exist_ok=True)
                            image.save(output_path)
                            print(f"✅ Bild gespeichert: {{filename}}")
                            images.append(image)
                            filenames.append(filename)
                        info = f"✅ Seed: {{seed}} | Steps: {{num_inference_steps}} | CFG: {{guidance_scale}}"
                        return images, info, filenames
                    except Exception as e:
                        print(f"❌ Generierungsfehler: {{e}}")
                        return None, f"❌ Fehler: {{str(e)}}", []
                
                def list_saved_images():
                    outputs_dir = "/app/outputs"
                    if not os.path.exists(outputs_dir): return []
                    images = []
                    for file in sorted(os.listdir(outputs_dir), reverse=True)[:20]:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            file_path = os.path.join(outputs_dir, file)
                            try: images.append(Image.open(file_path))
                            except: pass
                    return images
                
                @asynccontextmanager
                async def lifespan(app: FastAPI):
                    print("🚀 Starte FastAPI...")
                    yield
                    print("👋 FastAPI wird beendet")
                
                app = FastAPI(title="Stable Diffusion API", lifespan=lifespan)
                
                @app.post("/generate")
                async def generate_image_api(request: GenerateRequest):
                    if pipe is None: raise HTTPException(status_code=503, detail="Model not loaded")
                    try:
                        images, info, filenames = generate_image(prompt=request.prompt, negative_prompt=request.negative_prompt, num_inference_steps=request.num_inference_steps, guidance_scale=request.guidance_scale, width=request.width, height=request.height, seed=request.seed, num_images=request.num_images)
                        if images is None: raise HTTPException(status_code=500, detail=info)
                        return {{"status": "success", "filenames": filenames, "seed": request.seed if request.seed != -1 else torch.randint(0, 2**32, (1,)).item(), "prompt": request.prompt, "urls": [f"/image/{{filename}}" for filename in filenames], "parameters": {{"width": request.width, "height": request.height, "steps": request.num_inference_steps, "guidance_scale": request.guidance_scale}}, "info": info}}
                    except Exception as e:
                        print(f"❌ Generierungsfehler: {{e}}")
                        raise HTTPException(status_code=500, detail=f"Generation failed: {{str(e)}}")
                
                @app.get("/image/{{filename}}")
                async def get_image(filename: str):
                    file_path = f"/app/outputs/{{filename}}"
                    if not os.path.exists(file_path): raise HTTPException(status_code=404, detail="Image not found")
                    return FileResponse(file_path, media_type="image/png")
                
                @app.get("/images")
                async def list_images():
                    outputs_dir = "/app/outputs"
                    if not os.path.exists(outputs_dir): return {{"images": [], "count": 0}}
                    images = []
                    for file in os.listdir(outputs_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            file_path = os.path.join(outputs_dir, file)
                            stat = os.stat(file_path)
                            images.append({{"filename": file, "url": f"/image/{{file}}", "size": stat.st_size, "created": datetime.fromtimestamp(stat.st_ctime).isoformat()}})
                    images.sort(key=lambda x: x["created"], reverse=True)
                    return {{"images": images, "count": len(images)}}
                
                @app.get("/health")
                async def health():
                    model_info = {{}}
                    cache_dir = "/app/models/diffusers"
                    if os.path.exists(cache_dir):
                        model_info["cached_models"] = len(os.listdir(cache_dir))
                        total_size = sum(os.path.getsize(os.path.join(root, file)) for root, _, files in os.walk(cache_dir) for file in files)
                        model_info["cache_size_mb"] = total_size // (1024*1024)
                    output_info = {{"total_images": len([f for f in os.listdir("/app/outputs") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists("/app/outputs") else 0}}
                    return {{"status": "healthy", "model_loaded": pipe is not None, "device": "cuda" if torch.cuda.is_available() else "cpu", "gpu_available": torch.cuda.is_available(), "model_info": model_info, "output_info": output_info, "service": "stable-diffusion"}}
                
                @app.get("/")
                async def root():
                    device_info = {{"device": "cuda" if torch.cuda.is_available() else "cpu", "cuda_available": torch.cuda.is_available()}}
                    if torch.cuda.is_available():
                        device_info["gpu_name"] = torch.cuda.get_device_name(0)
                        device_info["gpu_memory"] = f"{{torch.cuda.get_device_properties(0).total_memory // (1024**3)}} GB"
                    return {{"message": "Stable Diffusion API", "endpoints": ["/generate", "/image/<filename>", "/images", "/health"], "device_info": device_info, "model_status": "loaded" if pipe else "not loaded", "persistent_storage": "/app/models and /app/outputs mounted"}}
                
                def start_gradio():
                    model_loaded = load_model()
                    with gr.Blocks(title="Stable Diffusion WebUI", theme=gr.themes.Soft()) as demo:
                        gr.Markdown("# 🎨 Stable Diffusion WebUI")
                        with gr.Row():
                            with gr.Column(scale=3):
                                prompt = gr.Textbox(label="Prompt", placeholder="Beschreibe was du sehen möchtest...", lines=3)
                                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Was NICHT im Bild sein soll...", lines=2)
                                with gr.Row():
                                    with gr.Column():
                                        steps = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Sampling Steps")
                                        guidance = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="CFG Scale")
                                    with gr.Column():
                                        width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Breite")
                                        height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Höhe")
                                with gr.Row():
                                    seed = gr.Number(value=-1, label="Seed (-1 für zufällig)", precision=0)
                                    num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Anzahl Bilder")
                                generate_btn = gr.Button("🎨 Generieren", variant="primary", size="lg")
                            with gr.Column(scale=4):
                                output_gallery = gr.Gallery(label="Generierte Bilder", show_label=True, elem_id="gallery", columns=2, rows=2, object_fit="contain", height="auto")
                                output_info = gr.Textbox(label="Info", interactive=False)
                        with gr.Row(): gr.Markdown("### 📁 Zuletzt generierte Bilder")
                        with gr.Row(): history_gallery = gr.Gallery(label="Verlauf", show_label=False, elem_id="history", columns=5, rows=2, object_fit="contain", height="auto", value=list_saved_images())
                        with gr.Row(): gr.Examples(examples=[["a beautiful sunset over mountains, golden hour, photorealistic", "blurry, low quality"],["cute robot playing guitar, digital art, colorful", "realistic, photo"],["magical forest with glowing mushrooms, fantasy art", "dark, scary"],["cyberpunk city at night, neon lights, rain", "daylight, sunny"],["portrait of a wizard, detailed, oil painting style", "modern, photograph"]], inputs=[prompt, negative_prompt], label="Beispiel-Prompts")
                        generate_btn.click(fn=generate_image, inputs=[prompt, negative_prompt, steps, guidance, width, height, seed, num_images], outputs=[output_gallery, output_info, gr.State()]).then(fn=list_saved_images, inputs=[], outputs=[history_gallery])
                        if model_loaded:
                            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
                            gr.Markdown(f"✅ **Modell geladen** | Device: {{device}}")
                        else:
                            gr.Markdown("⏳ **Modell lädt...** Bitte warten...")
                    print("🚀 Starte Gradio WebUI auf Port 7861...")
                    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)
                
                if __name__ == '__main__':
                    threading.Thread(target=lambda: uvicorn.run(app, host='0.0.0.0', port=7860), daemon=True).start()
                    start_gradio()
                PYEOF
                echo "🎨 Starte Stable Diffusion Service (WebUI + API)..."
                python server.py

          vision:
            image: python:3.11-slim
            restart: unless-stopped
            working_dir: /app
            volumes:
              - ./vision_data:/app/data
              - ./vision_models:/app/models
              - ./vision_cache:/root/.cache
            ports:
              - "{vision_port}:8000"
            environment:
              - TORCH_HOME=/app/models/torch
              - YOLO_CONFIG_DIR=/app/models/yolo
              - HF_HOME=/app/models/huggingface
              - TRANSFORMERS_CACHE=/app/models/transformers
              - TORCH_EXTENSIONS_DIR=/app/models/torch_extensions
              - PYTHONDONTWRITEBYTECODE=1
              - PIP_CACHE_DIR=/root/.cache/pip
            command:
              - bash
              - -c
              - |
                mkdir -p /app/models/torch /app/models/yolo /app/models/huggingface /app/models/transformers /app/models/torch_extensions
                mkdir -p /root/.cache/pip /root/.cache/torch
                apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 wget curl git && rm -rf /var/lib/apt/lists/*
                pip install --upgrade pip --cache-dir=/root/.cache/pip
                pip install --cache-dir=/root/.cache/pip torch torchvision --index-url https://download.pytorch.org/whl/cu118
                pip install --cache-dir=/root/.cache/pip fastapi uvicorn pillow opencv-python-headless ultralytics python-multipart
                cat > server.py << 'PYEOF'
                from fastapi import FastAPI, UploadFile, File
                from ultralytics import YOLO
                import uvicorn
                from io import BytesIO
                from PIL import Image
                import os
                import shutil
                from contextlib import asynccontextmanager
                model = None
                def validate_model_file(model_path ):
                    if not os.path.exists(model_path): return False, "Datei nicht gefunden"
                    size = os.path.getsize(model_path)
                    if size < 1000000: return False, f"Datei zu klein: {{size}} bytes"
                    try:
                        import torch
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                        if isinstance(checkpoint, dict):
                            if 'model' in checkpoint or 'state_dict' in checkpoint: return True, "Valid PyTorch checkpoint"
                            else: return False, f"Missing required keys. Found: {{list(checkpoint.keys())}}"
                        else: return False, f"Unexpected checkpoint type: {{type(checkpoint)}}"
                    except Exception as e: return False, f"PyTorch load error: {{str(e)}}"
                def safe_remove_file(file_path):
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            return True
                    except: pass
                    return False
                @asynccontextmanager
                async def lifespan(app: FastAPI):
                    global model
                    model_cache_dir = '/app/models/yolo'
                    model_file = 'yolov8n.pt'
                    cached_model_path = f'{{model_cache_dir}}/{{model_file}}'
                    os.makedirs(model_cache_dir, exist_ok=True)
                    try:
                        if os.path.exists(cached_model_path):
                            print(f"🔍 Prüfe cached Modell: {{cached_model_path}}")
                            print(f"📊 Modell-Größe: {{os.path.getsize(cached_model_path) // (1024*1024)}} MB")
                            is_valid, reason = validate_model_file(cached_model_path)
                            if is_valid:
                                print("✅ Cached Modell ist gültig")
                                try:
                                    model = YOLO(cached_model_path)
                                    print("✅ Modell erfolgreich geladen!")
                                    yield
                                    return
                                except Exception as load_error:
                                    print(f"❌ Fehler beim Laden des cached Modells: {{load_error}}")
                                    if safe_remove_file(cached_model_path): print("🗑️ Defektes cached Modell entfernt")
                            else:
                                print(f"❌ Cached Modell defekt: {{reason}}")
                                if safe_remove_file(cached_model_path): print("🗑️ Defektes cached Modell entfernt")
                        print("📥 Lade YOLO-Modell frisch herunter...")
                        model = YOLO(model_file)
                        yolo_cache_paths = [f'/root/.config/Ultralytics/{{model_file}}', f'/root/.ultralytics/{{model_file}}', f'/app/{{model_file}}', f'./{{model_file}}']
                        search_dirs = ['/root/.config', '/root/.ultralytics', '/app', '.']
                        for search_dir in search_dirs:
                            if os.path.exists(search_dir):
                                for item in os.listdir(search_dir):
                                    if item == model_file:
                                        full_path = os.path.join(search_dir, item)
                                        if full_path not in yolo_cache_paths: yolo_cache_paths.append(full_path)
                        best_source, best_size = None, 0
                        for path in yolo_cache_paths:
                            if os.path.exists(path):
                                size = os.path.getsize(path)
                                print(f"📁 Gefunden: {{path}} ({{size // (1024*1024)}} MB)")
                                if size > best_size and size > 1000000:
                                    is_valid, reason = validate_model_file(path)
                                    if is_valid:
                                        best_source, best_size = path, size
                                    else: print(f"⚠️ {{path}} defekt: {{reason}}")
                        if best_source and best_source != cached_model_path:
                            print(f"💾 Cache Modell: {{best_source}} → {{cached_model_path}}")
                            try:
                                shutil.copy2(best_source, cached_model_path)
                                is_valid, reason = validate_model_file(cached_model_path)
                                if is_valid: print(f"✅ Modell erfolgreich gecacht: {{os.path.getsize(cached_model_path) // (1024*1024)}} MB")
                                else:
                                    print(f"❌ Caching fehlgeschlagen: {{reason}}")
                                    safe_remove_file(cached_model_path)
                            except Exception as cache_error: print(f"❌ Cache-Fehler: {{cache_error}}")
                        if model is None: raise RuntimeError("Vision Service konnte kein Modell laden")
                        print("✅ Vision Service erfolgreich gestartet!")
                        print(f"🤖 Modell-Typ: {{type(model).__name__}}")
                        print(f"💾 Cache-Verzeichnis: {{model_cache_dir}}")
                        if os.path.exists(model_cache_dir):
                            cached_files = os.listdir(model_cache_dir)
                            if cached_files:
                                total_size = sum(os.path.getsize(os.path.join(model_cache_dir, f)) for f in cached_files if os.path.isfile(os.path.join(model_cache_dir, f)))
                                print(f"📦 Gecachte Dateien ({{len(cached_files)}}): {{cached_files}}")
                                print(f"💽 Gesamt-Cache-Größe: {{total_size // (1024*1024)}} MB")
                    except Exception as e:
                        print(f"❌ Kritischer Startup-Fehler: {{e}}")
                        print("🔧 Mögliche Lösungen: ...")
                        raise
                    yield
                    print("👋 Vision Service wird beendet")
                app = FastAPI(title='Vision API - Robust', lifespan=lifespan)
                @app.post('/detect')
                async def detect_objects(file: UploadFile = File(...)):
                    if model is None: return {{"error": "Model not loaded", "status": "error"}}
                    try:
                        contents = await file.read()
                        image = Image.open(BytesIO(contents))
                        results = model(image)
                        detections = results[0].boxes.data.tolist() if results[0].boxes else []
                        return {{"detections": detections, "status": "success", "count": len(detections), "image_size": image.size, "model_loaded": True}}
                    except Exception as e: return {{"error": str(e), "status": "error", "model_loaded": model is not None}}
                @app.get('/health')
                async def health():
                    model_cache_path = '/app/models/yolo/yolov8n.pt'
                    cache_size = 0
                    is_cache_valid = False
                    if os.path.exists(model_cache_path):
                        cache_size = os.path.getsize(model_cache_path) // (1024*1024)
                        is_cache_valid, _ = validate_model_file(model_cache_path)
                    return {{"status": "healthy", "model_loaded": model is not None, "model_cached": os.path.exists(model_cache_path), "cache_valid": is_cache_valid, "cache_size_mb": cache_size, "cache_path": model_cache_path, "service": "vision-robust"}}
                @app.get('/')
                async def root():
                    cache_info = {{}}
                    model_dir = '/app/models/yolo'
                    if os.path.exists(model_dir):
                        files = os.listdir(model_dir)
                        cache_info["cached_files"] = files
                        cache_info["total_files"] = len(files)
                        total_size, valid_files = 0, 0
                        for f in files:
                            file_path = os.path.join(model_dir, f)
                            if os.path.isfile(file_path):
                                total_size += os.path.getsize(file_path)
                                if f.endswith('.pt'):
                                    is_valid, _ = validate_model_file(file_path)
                                    if is_valid: valid_files += 1
                        cache_info["total_size_mb"] = total_size // (1024*1024)
                        cache_info["valid_models"] = valid_files
                    return {{"message": "Vision API - Robust Version", "endpoints": ["/detect", "/health"], "cache_info": cache_info, "persistent_storage": "/app/models mounted and validated", "model_status": "loaded" if model else "not loaded"}}
                if __name__ == '__main__': uvicorn.run(app, host='0.0.0.0', port=8000)
                PYEOF
                echo "🚀 Starte Vision Service (Robust)..."
                python server.py

          kyutai-voice:
            image: python:3.11-slim
            working_dir: /app
            volumes:
              - ./kyutai:/app
              - ./kyutai_models:/app/models
              - ./kyutai_cache:/root/.cache
            environment:
              - TRANSFORMERS_CACHE=/app/models/transformers
              - HF_HOME=/app/models/huggingface
              - TORCH_HOME=/app/models/torch
              - PIP_CACHE_DIR=/root/.cache/pip
            command:
              - bash
              - -c
              - |
                mkdir -p /app/models/transformers /app/models/huggingface /app/models/torch
                mkdir -p /root/.cache/pip
                pip install --cache-dir=/root/.cache/pip fastapi uvicorn python-multipart
                cat > server.py << 'PYEOF'
                from fastapi import FastAPI, UploadFile, File, HTTPException
                import uvicorn
                import os
                app = FastAPI(title='Kyutai Voice API')
                @app.post('/transcribe')
                async def transcribe_audio(file: UploadFile = File(...)):
                    return {{"transcription": f"Placeholder transcription for {{file.filename}}", "status": "success", "model_cache": os.path.exists('/app/models/transformers')}}
                @app.post('/synthesize')
                async def synthesize_speech(text: str):
                    return {{"message": f"Would synthesize {{text}}", "audio_url": "/placeholder.wav", "status": "success"}}
                @app.get('/health')
                async def health():
                    cache_info = {{"transformers_cache": len(os.listdir('/app/models/transformers')) if os.path.exists('/app/models/transformers') else 0, "torch_cache": len(os.listdir('/app/models/torch')) if os.path.exists('/app/models/torch') else 0}}
                    return {{"status": "healthy", "service": "kyutai-voice", "cache_info": cache_info}}
                @app.get('/')
                async def root():
                    return {{"message": "Kyutai Voice API running", "endpoints": ["/transcribe", "/synthesize", "/health"], "persistent_storage": "/app/models mounted"}}
                if __name__ == '__main__': uvicorn.run(app, host='0.0.0.0', port=5005)
                PYEOF
                python server.py
            restart: unless-stopped
            ports:
              - "{kyutai_port}:5005"

          searxng:
            image: searxng/searxng
            restart: unless-stopped
            ports:
              - "{searxng_port}:8080"
            environment:
              - SEARXNG_BASE_URL=http://localhost:{searxng_port}/
              - SEARXNG_INSTANCE_NAME=LokaleSuche
            volumes:
              - ./searxng_data:/etc/searxng

          flowise:
            image: flowiseai/flowise:latest
            restart: unless-stopped
            ports:
              - "{flowise_port}:3000"
            volumes:
              - ./flowise_data:/root/.flowise
            environment:
              - PORT=3000
              - DATABASE_PATH=/root/.flowise/flowise.db
              - API_KEY_PATH=/root/.flowise/api.key
              - SECRET_KEY_PATH=/root/.flowise/secret.key
              - LOG_LEVEL=info
            depends_on:
              - ollama

          open-webui:
            image: ghcr.io/open-webui/open-webui:main
            restart: unless-stopped
            ports:
              - "{openwebui_port}:8080"
            environment:
              - OLLAMA_BASE_URL=http://ollama:11434
              - WEBUI_SECRET_KEY=your-secret-key-here
              - WEBUI_NAME=Local AI WebUI
            volumes:
              - ./openwebui_data:/app/backend/data
            depends_on:
              - ollama
            deploy:
              resources:
                reservations:
                  devices:
                    - driver: nvidia
                      count: all
                      capabilities: [gpu]

          n8n:
            image: n8nio/n8n
            restart: unless-stopped
            ports:
              - "{n8n_port}:5678"
            environment:
              - DB_TYPE=postgresdb
              - DB_POSTGRESDB_HOST=postgres
              - DB_POSTGRESDB_PORT=5432
              - DB_POSTGRESDB_DATABASE=n8n
              - DB_POSTGRESDB_USER=n8n
              - DB_POSTGRESDB_PASSWORD=n8n
              - N8N_BASIC_AUTH_ACTIVE=true
              - N8N_BASIC_AUTH_USER={n8n_user}
              - N8N_BASIC_AUTH_PASSWORD={n8n_pass}
              - N8N_HOST=0.0.0.0
              - N8N_PORT=5678
              - N8N_PROTOCOL=http
            volumes:
              - ./n8n_data:/home/node/.n8n
            depends_on:
              - postgres
              - ollama
              - vision
              - stable-diffusion
              - kyutai-voice
              - searxng
              - open-webui
              - flowise

        """.format(
                ollama_port=ollama_port,
                vision_port=vision_port,
                kyutai_port=kyutai_port,
                searxng_port=searxng_port,
                stablediffusion_webui_port=stablediffusion_webui_port,
                stablediffusion_api_port=stablediffusion_api_port,
                flowise_port=flowise_port,
                openwebui_port=openwebui_port,
                n8n_port=n8n_port,
                n8n_user=n8n_user,
                n8n_pass=n8n_pass
        )


    def write_compose(self):
        """7️⃣ Docker Compose schreiben"""
        self.log("7️⃣ Schreibe docker-compose.yml...", tag="info")
        project = Path(self.project_var.get()).expanduser()
        compose_path = project / COMPOSE_FILENAME
        
        if not project.exists():
            self.log("❌ Projektverzeichnis nicht gefunden. Bitte zuerst Schritt 5 abschließen.", tag="error")
            return False
            
        try:
            compose_content = self.get_docker_compose_content()
            # Verwende textwrap.dedent, um führende Leerzeichen aus dem heredoc-String zu entfernen
            with open(compose_path, 'w', encoding='utf-8') as f:
                f.write(textwrap.dedent(compose_content))
            
            self.log(f"✅ docker-compose.yml geschrieben: {compose_path}", tag="ok")
            
            # KORRIGIERTE LOG-AUSGABE
            self.log(f"🔌 Ports konfiguriert: n8n={self.n8n_port.get()}, ollama={self.ollama_port.get()}, vision={self.vision_port.get()}, kyutai={self.kyutai_port.get()}, searxng={self.searxng_port.get()}, stablediffusion={self.stablediffusion_port.get()}, flowise={self.flowise_port.get()}", tag="info")
            
            return True
        except Exception as e:
            self.log(f"❌ Fehler beim Schreiben der docker-compose.yml: {e}", tag="error")
            return False

    # ------------------- Setup Actions (1-8) -------------------
    def check_python_setup(self):
        """1️⃣ Python Setup prüfen und automatisch installieren"""
        self.log("1️⃣ Prüfe Python Installation...", tag="info")
        
        if not check_python_requirements():
            version = get_python_version()
            self.log(f"❌ Python {version[0]}.{version[1]}.{version[2]} gefunden - benötigt wird Python 3.8+", tag="error")
            
            # Frage nach automatischer Installation
            try_auto = messagebox.askyesno(
                "Python Installation", 
                f"Python 3.8+ wird benötigt, aber Python {version[0]}.{version[1]} gefunden.\n\n"
                "Soll Python automatisch aktualisiert werden?\n\n"
                "⚠️ Dies führt Installations-Scripts aus!"
            )
            if try_auto:
                success = self._attempt_python_install()
                if success:
                    self.log("✅ Python erfolgreich aktualisiert. Bitte Anwendung neu starten.", tag="ok")
                    messagebox.showinfo("Neustart erforderlich", "Python wurde installiert.\nBitte die Anwendung neu starten!")
                    return True
                else:
                    self.log("❌ Automatische Python-Installation fehlgeschlagen.", tag="error")
                    return False
            else:
                self.log("📥 Manuelle Installation: https://www.python.org/downloads/", tag="warning")
                return False
        
        version = get_python_version()
        self.log(f"✅ Python {version[0]}.{version[1]}.{version[2]} - OK!", tag="ok")
        
        # Prüfe wichtige Module
        required_modules = ['tkinter', 'threading', 'subprocess', 'pathlib', 'urllib']
        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            self.log(f"❌ Fehlende Python-Module: {', '.join(missing)}", tag="error")
            self.log("💡 Meist durch Python-Neuinstallation lösbar", tag="info")
        else:
            self.log("✅ Alle benötigten Python-Module verfügbar", tag="ok")
        
        return len(missing) == 0

    def check_git_setup(self):
        """2️⃣ Git Installation prüfen und automatisch installieren"""
        self.log("2️⃣ Prüfe Git Installation...", tag="info")
        
        if not is_installed("git"):
            self.log("❌ Git nicht gefunden!", tag="error")
            
            # Frage nach automatischer Installation
            try_auto = messagebox.askyesno(
                "Git Installation", 
                "Git wird benötigt, ist aber nicht installiert.\n\n"
                "Soll Git automatisch installiert werden?\n\n"
                "⚠️ Dies führt Installations-Scripts aus!"
            )
            if try_auto:
                success = self._attempt_git_install()
                if success:
                    self.log("✅ Git erfolgreich installiert!", tag="ok")
                    # Nochmal prüfen
                    if is_installed("git"):
                        rc, out = run_cmd_capture(["git", "--version"])
                        if rc == 0:
                            self.log(f"✅ {out.strip()}", tag="ok")
                        return True
                else:
                    self.log("❌ Automatische Git-Installation fehlgeschlagen.", tag="error")
                    self._show_manual_git_instructions()
                    return False
            else:
                self._show_manual_git_instructions()
                return False
        
        # Git Version prüfen
        rc, out = run_cmd_capture(["git", "--version"])
        if rc == 0:
            self.log(f"✅ {out.strip()}", tag="ok")
        
        return True

    def setup_docker_info(self):
        """3️⃣ Docker Installation prüfen und automatisch installieren"""
        self.log("3️⃣ Prüfe Docker Installation...", tag="info")
        
        docker_ok = is_installed("docker")
        compose_ok = is_installed("docker-compose") or self._check_docker_compose_plugin()
        
        if not docker_ok or not compose_ok:
            missing = []
            if not docker_ok:
                missing.append("Docker")
            if not compose_ok:
                missing.append("Docker Compose")
            
            self.log(f"❌ {', '.join(missing)} nicht gefunden!", tag="error")
            
            # Frage nach automatischer Installation
            try_auto = messagebox.askyesno(
                "Docker Installation", 
                f"{', '.join(missing)} wird/werden benötigt.\n\n"
                "Soll Docker automatisch installiert werden?\n\n"
                "⚠️ Dies führt Installations-Scripts aus!\n"
                "⚠️ Eventuell ist ein Neustart erforderlich!"
            )
            if try_auto:
                success = self._attempt_docker_install()
                if success:
                    self.log("✅ Docker Installation eingeleitet!", tag="ok")
                    self.log("🔄 Prüfe Docker-Status nach Installation...", tag="info")
                    
                    # Warte kurz und prüfe nochmal
                    time.sleep(3)
                    docker_ok = is_installed("docker")
                    compose_ok = is_installed("docker-compose") or self._check_docker_compose_plugin()
                    
                    if docker_ok and compose_ok:
                        self.log("✅ Docker erfolgreich installiert und verfügbar!", tag="ok")
                    else:
                        self.log("⚠️ Docker installiert, aber eventuell ist ein Neustart/Logout erforderlich", tag="warning")
                        self.log("💡 Nach Neustart: Benutzer zur docker-Gruppe hinzufügen falls nötig", tag="info")
                else:
                    self.log("❌ Automatische Docker-Installation fehlgeschlagen.", tag="error")
                    self._show_manual_docker_instructions()
                    return False
            else:
                self._show_manual_docker_instructions()
                return False
        
        # Docker Version prüfen
        if docker_ok:
            rc, out = run_cmd_capture(["docker", "--version"])
            if rc == 0:
                self.log(f"✅ {out.strip()}", tag="ok")
        
        # Docker Compose Version prüfen
        if compose_ok:
            rc, out = run_cmd_capture(["docker", "compose", "version"])
            if rc == 0:
                self.log(f"✅ Docker Compose Plugin: {out.strip()}", tag="ok")
            else:
                rc, out = run_cmd_capture(["docker-compose", "--version"])
                if rc == 0:
                    self.log(f"✅ {out.strip()}", tag="ok")
        
        if docker_ok and compose_ok:
            # Prüfe Docker daemon
            rc, out = run_cmd_capture(["docker", "info"])
            if rc == 0:
                self.log("✅ Docker Daemon läuft", tag="ok")
            else:
                self.log("⚠️ Docker Daemon nicht erreichbar - bitte Docker starten", tag="warning")
                self.log("💡 Linux: sudo systemctl start docker", tag="info")
                self.log("💡 macOS/Windows: Docker Desktop starten", tag="info")
        
        return docker_ok and compose_ok

    def _check_docker_compose_plugin(self):
        """Prüft ob docker compose (plugin) verfügbar ist"""
        rc, _ = run_cmd_capture(["docker", "compose", "version"])
        return rc == 0

    def setup_ollama_local(self):
        """4️⃣ Ollama lokale Installation"""
        self.log("4️⃣ Prüfe/installiere Ollama lokal...", tag="info")
        
        if is_installed("ollama"):
            rc, out = run_cmd_capture(["ollama", "--version"])
            if rc == 0:
                self.log(f"✅ Ollama bereits installiert: {out.strip()}", tag="ok")
                return True
        
        self.log("❌ Ollama nicht gefunden", tag="warning")
        
        # Frage ob automatische Installation versucht werden soll
        try_auto = messagebox.askyesno(
            "Ollama Installation", 
            "Ollama wird benötigt, ist aber nicht installiert.\n\n"
            "Soll Ollama automatisch installiert werden?\n\n"
            "⚠️ Dies führt Installations-Scripts aus!"
        )
        if try_auto:
            success = self._attempt_ollama_install()
            if success:
                self.log("✅ Ollama Installation erfolgreich!", tag="ok")
                # Nochmal prüfen
                if is_installed("ollama"):
                    rc, out = run_cmd_capture(["ollama", "--version"])
                    if rc == 0:
                        self.log(f"✅ {out.strip()}", tag="ok")
                    return True
            else:
                self.log("❌ Automatische Ollama-Installation fehlgeschlagen.", tag="error")
                self._show_manual_ollama_instructions()
                return False
        else:
            self._show_manual_ollama_instructions()
            return False

    def _show_manual_ollama_instructions(self):
        """Zeigt manuelle Ollama-Installationsanweisungen"""
        system = platform.system().lower()
        
        if "windows" in system:
            instructions = """
🔧 OLLAMA INSTALLATION (Windows):
1. Gehe zu: https://ollama.com/download
2. Lade 'Ollama for Windows' herunter
3. Führe das .exe Installationspaket aus
4. Starte neu und teste mit: ollama --version
            """
        elif "darwin" in system:
            instructions = """
🔧 OLLAMA INSTALLATION (macOS ):
Option 1 - Installer:
  1. Gehe zu: https://ollama.com/download
  2. Lade 'Ollama for macOS' herunter
  3. Installiere die .dmg Datei

Option 2 - Homebrew:
  brew install ollama
            """
        else:
            instructions = """
🔧 OLLAMA INSTALLATION (Linux ):
curl -fsSL https://ollama.com/install.sh | sh

Alternative:
1. Gehe zu: https://ollama.com/download
2. Folge den Linux-Anweisungen für deine Distribution
            """
        
        self.log(instructions, tag="warning" )

    def _attempt_python_install(self):
        """Versucht automatische Python Installation"""
        self.log("🔄 Versuche automatische Python-Installation...", tag="info")
        system = platform.system().lower()
        
        try:
            if "windows" in system:
                # Windows: Lade Python Installer herunter und führe aus
                self.log("📥 Lade Python für Windows herunter...", tag="info")
                python_url = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
                
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
                    with urllib.request.urlopen(python_url) as response:
                        shutil.copyfileobj(response, tmp)
                    installer_path = tmp.name
                
                self.log("🚀 Starte Python Installer...", tag="info")
                # Stille Installation mit pip und PATH
                rc = subprocess.run([
                    installer_path, 
                    "/quiet", 
                    "InstallAllUsers=1", 
                    "PrependPath=1", 
                    "Include_pip=1"
                ]).returncode
                
                os.unlink(installer_path)  # Cleanup
                return rc == 0
                
            elif "darwin" in system:
                # macOS: Versuche Homebrew oder lade .pkg herunter
                if is_installed("brew"):
                    self.log("🍺 Installiere Python via Homebrew...", tag="info")
                    rc, out = run_cmd_capture(["brew", "install", "python@3.11"])
                    if rc == 0:
                        self.log("✅ Python via Homebrew installiert", tag="ok")
                        return True
                
                # Fallback: .pkg Download
                self.log("📥 Lade Python .pkg für macOS herunter...", tag="info")
                python_url = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-macos11.pkg"
                
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pkg", delete=False) as tmp:
                    with urllib.request.urlopen(python_url) as response:
                        shutil.copyfileobj(response, tmp)
                    installer_path = tmp.name
                
                self.log("🚀 Starte Python Installer (benötigt Passwort)...", tag="info")
                rc = subprocess.run(["sudo", "installer", "-pkg", installer_path, "-target", "/"]).returncode
                os.unlink(installer_path)
                return rc == 0
                
            else:
                # Linux: Verwende System Package Manager
                self.log("🐧 Versuche Python Installation via System Package Manager...", tag="info")
                
                # Erkenne Package Manager
                if is_installed("apt"):
                    cmd = ["sudo", "apt", "update", "&&", "sudo", "apt", "install", "-y", "python3", "python3-pip", "python3-tk"]
                elif is_installed("yum"):
                    cmd = ["sudo", "yum", "install", "-y", "python3", "python3-pip", "python3-tkinter"]
                elif is_installed("dnf"):
                    cmd = ["sudo", "dnf", "install", "-y", "python3", "python3-pip", "python3-tkinter"]
                elif is_installed("pacman"):
                    cmd = ["sudo", "pacman", "-S", "--noconfirm", "python", "python-pip", "tk"]
                else:
                    self.log("❌ Kein unterstützter Package Manager gefunden", tag="error")
                    return False
                
                rc, out = run_cmd_capture(cmd, shell=True)
                if rc == 0:
                    self.log("✅ Python via Package Manager installiert", tag="ok")
                    return True
                else:
                    self.log(f"❌ Package Manager Installation fehlgeschlagen: {out}", tag="error")
                    return False
                    
        except Exception as e:
            self.log(f"❌ Fehler bei automatischer Python-Installation: {e}", tag="error")
            return False

    def _show_manual_ollama_instructions(self):
        """Zeigt manuelle Ollama-Installationsanweisungen"""
        system = platform.system().lower()
        
        if "windows" in system:
            instructions = """
🔧 OLLAMA INSTALLATION (Windows):
1. Gehe zu: https://ollama.com/download
2. Lade 'Ollama for Windows' herunter
3. Führe das .exe Installationspaket aus
4. Starte neu und teste mit: ollama --version
            """
        elif "darwin" in system:
            instructions = """
🔧 OLLAMA INSTALLATION (macOS ):
Option 1 - Installer:
  1. Gehe zu: https://ollama.com/download
  2. Lade 'Ollama for macOS' herunter
  3. Installiere die .dmg Datei

Option 2 - Homebrew:
  brew install ollama
            """
        else:
            instructions = """
🔧 OLLAMA INSTALLATION (Linux ):
curl -fsSL https://ollama.com/install.sh | sh

Alternative:
1. Gehe zu: https://ollama.com/download
2. Folge den Linux-Anweisungen für deine Distribution
            """
        
        self.log(instructions, tag="warning" )

    def _attempt_python_install(self):
        """Versucht automatische Python Installation"""
        self.log("🔄 Versuche automatische Python-Installation...", tag="info")
        system = platform.system().lower()
        
        try:
            if "windows" in system:
                self.log("📥 Lade Python für Windows herunter...", tag="info")
                python_url = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
                
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".exe", delete=False ) as tmp:
                    with urllib.request.urlopen(python_url) as response:
                        shutil.copyfileobj(response, tmp)
                    installer_path = tmp.name
                
                self.log("🚀 Starte Python Installer...", tag="info")
                rc = subprocess.run([
                    installer_path, "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_pip=1"
                ]).returncode
                os.unlink(installer_path)
                return rc == 0
                
            elif "darwin" in system:
                if is_installed("brew"):
                    self.log("🍺 Installiere Python via Homebrew...", tag="info")
                    rc, out = run_cmd_capture(["brew", "install", "python@3.11"])
                    if rc == 0:
                        self.log("✅ Python via Homebrew installiert", tag="ok")
                        return True
                
                self.log("📥 Lade Python .pkg für macOS herunter...", tag="info")
                python_url = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-macos11.pkg"
                
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pkg", delete=False ) as tmp:
                    with urllib.request.urlopen(python_url) as response:
                        shutil.copyfileobj(response, tmp)
                    installer_path = tmp.name
                
                self.log("🚀 Starte Python Installer (benötigt Passwort)...", tag="info")
                rc = subprocess.run(["sudo", "installer", "-pkg", installer_path, "-target", "/"]).returncode
                os.unlink(installer_path)
                return rc == 0
                
            else:
                self.log("🐧 Versuche Python Installation via System Package Manager...", tag="info")
                if is_installed("apt"):
                    cmd = ["sudo", "apt", "update", "&&", "sudo", "apt", "install", "-y", "python3", "python3-pip", "python3-tk"]
                elif is_installed("yum"):
                    cmd = ["sudo", "yum", "install", "-y", "python3", "python3-pip", "python3-tkinter"]
                elif is_installed("dnf"):
                    cmd = ["sudo", "dnf", "install", "-y", "python3", "python3-pip", "python3-tkinter"]
                elif is_installed("pacman"):
                    cmd = ["sudo", "pacman", "-S", "--noconfirm", "python", "python-pip", "tk"]
                else:
                    self.log("❌ Kein unterstützter Package Manager gefunden", tag="error")
                    return False
                
                rc, out = run_cmd_capture(cmd, shell=True)
                if rc == 0:
                    self.log("✅ Python via Package Manager installiert", tag="ok")
                    return True
                else:
                    self.log(f"❌ Package Manager Installation fehlgeschlagen: {out}", tag="error")
                    return False
                    
        except Exception as e:
            self.log(f"❌ Fehler bei automatischer Python-Installation: {e}", tag="error")
            return False

    def _attempt_git_install(self):
        """Versucht automatische Git Installation"""
        self.log("🔄 Versuche automatische Git-Installation...", tag="info")
        system = platform.system().lower()
        
        try:
            if "windows" in system:
                self.log("📥 Lade Git für Windows herunter...", tag="info")
                git_url = "https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe"
                
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".exe", delete=False ) as tmp:
                    with urllib.request.urlopen(git_url) as response:
                        shutil.copyfileobj(response, tmp)
                    installer_path = tmp.name
                
                self.log("🚀 Starte Git Installer...", tag="info")
                rc = subprocess.run([
                    installer_path, "/VERYSILENT", "/NORESTART", "/NOCANCEL", "/SP-",
                    "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS", "/COMPONENTS=icons,ext\\reg\\shellhere,assoc,assoc_sh"
                ]).returncode
                os.unlink(installer_path)
                return rc == 0
                
            elif "darwin" in system:
                if is_installed("brew"):
                    self.log("🍺 Installiere Git via Homebrew...", tag="info")
                    rc, out = run_cmd_capture(["brew", "install", "git"])
                    return rc == 0
                else:
                    self.log("🔧 Installiere Xcode Command Line Tools...", tag="info")
                    rc, out = run_cmd_capture(["xcode-select", "--install"])
                    if rc == 0:
                        self.log("✅ Xcode Command Line Tools Installation gestartet", tag="ok")
                        self.log("⏳ Bitte Installation in System-Dialog abschließen", tag="warning")
                        return True
                    return False
                    
            else:
                self.log("🐧 Installiere Git via System Package Manager...", tag="info")
                if is_installed("apt"):
                    rc, out = run_cmd_capture(["sudo", "apt", "update", "&&", "sudo", "apt", "install", "-y", "git"], shell=True)
                elif is_installed("yum"):
                    rc, out = run_cmd_capture(["sudo", "yum", "install", "-y", "git"], shell=True)
                elif is_installed("dnf"):
                    rc, out = run_cmd_capture(["sudo", "dnf", "install", "-y", "git"], shell=True)
                elif is_installed("pacman"):
                    rc, out = run_cmd_capture(["sudo", "pacman", "-S", "--noconfirm", "git"], shell=True)
                else:
                    self.log("❌ Kein unterstützter Package Manager gefunden", tag="error")
                    return False
                return rc == 0
                
        except Exception as e:
            self.log(f"❌ Fehler bei automatischer Git-Installation: {e}", tag="error")
            return False

    def _show_manual_git_instructions(self):
        """Zeigt manuelle Git-Installationsanweisungen"""
        system = platform.system().lower()
        if "windows" in system:
            self.log("📥 Windows: Git von https://git-scm.com/download/win installieren", tag="warning" )
        elif "darwin" in system:
            self.log("📥 macOS: 'brew install git' oder von https://git-scm.com/download/mac", tag="warning" )
        else:
            self.log("📥 Linux: 'sudo apt install git' oder 'sudo yum install git'", tag="warning")

    def _attempt_docker_install(self):
        """Versucht automatische Docker Installation"""
        self.log("🔄 Versuche automatische Docker-Installation...", tag="info")
        system = platform.system().lower()
        
        try:
            if "windows" in system:
                self.log("🔍 Prüfe Windows-Voraussetzungen...", tag="info")
                if not self._check_wsl2():
                    self.log("❌ WSL2 nicht verfügbar - wird für Docker Desktop benötigt", "error")
                    self.log("💡 WSL2 Installation:\n   1. Öffne PowerShell als Administrator\n   2. Führe aus: wsl --install\n   3. Starte Windows neu\n   4. Führe nochmal Setup aus", "info")
                    return False
                
                if not self._check_virtualization():
                    self.log("⚠️ Virtualisierung eventuell nicht aktiviert. Aktiviere sie im BIOS, falls Docker Probleme hat.", "warning")
                
                self.log("📥 Lade Docker Desktop für Windows herunter... (kann einige Minuten dauern)", tag="info")
                docker_url = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
                
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".exe", delete=False ) as tmp:
                    with urllib.request.urlopen(docker_url) as response:
                        shutil.copyfileobj(response, tmp)
                    installer_path = tmp.name
                
                self.log("🚀 Starte Docker Desktop Installer...", tag="info")
                rc = subprocess.run([installer_path, "install", "--quiet", "--accept-license", "--backend=wsl-2"]).returncode
                os.unlink(installer_path)
                if rc == 0:
                    self.log("✅ Docker Desktop Installation gestartet. Bitte nach Abschluss Docker Desktop starten.", "ok")
                    self.log("🔄 Eventuell ist ein Neustart erforderlich.", "warning")
                    return True
                else:
                    self.log("❌ Docker Desktop Installation fehlgeschlagen", "error")
                    return False
                
            elif "darwin" in system:
                if is_installed("brew"):
                    self.log("🍺 Installiere Docker Desktop via Homebrew...", tag="info")
                    rc, out = run_cmd_capture(["brew", "install", "--cask", "docker"])
                    if rc == 0:
                        self.log("✅ Docker Desktop via Homebrew installiert. Bitte manuell starten.", "ok")
                        return True
                
                self.log("📥 Lade Docker Desktop für macOS herunter...", tag="info")
                arch = subprocess.run(["uname", "-m"], capture_output=True, text=True).stdout.strip()
                docker_url = "https://desktop.docker.com/mac/main/arm64/Docker.dmg" if arch == "arm64" else "https://desktop.docker.com/mac/main/amd64/Docker.dmg"
                
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".dmg", delete=False ) as tmp:
                    with urllib.request.urlopen(docker_url) as response:
                        shutil.copyfileobj(response, tmp)
                    dmg_path = tmp.name
                
                self.log("🚀 Mounte Docker DMG...", tag="info")
                if subprocess.run(["hdiutil", "attach", dmg_path, "-nobrowse"]).returncode == 0:
                    rc2 = subprocess.run(["cp", "-R", "/Volumes/Docker/Docker.app", "/Applications/"]).returncode
                    subprocess.run(["hdiutil", "detach", "/Volumes/Docker"])
                    os.unlink(dmg_path)
                    if rc2 == 0:
                        self.log("✅ Docker Desktop installiert. Bitte manuell starten: /Applications/Docker.app", "ok")
                        return True
                return False
                
            else:
                self.log("🐧 Installiere Docker via offizielles Script...", tag="info")
                try:
                    with urllib.request.urlopen("https://get.docker.com" ) as response:
                        script_content = response.read().decode('utf-8')
                    
                    proc = subprocess.Popen(["sudo", "bash", "-c", script_content], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in proc.stdout: self.log(line.rstrip(), tag="out")
                    proc.wait()
                    
                    if proc.returncode == 0:
                        self.log("✅ Docker Installation erfolgreich!", "ok")
                        username = os.getenv("USER") or os.getenv("USERNAME")
                        if username:
                            self.log(f"👤 Füge Benutzer {username} zur docker-Gruppe hinzu...", "info")
                            subprocess.run(["sudo", "usermod", "-aG", "docker", username])
                            self.log("⚠️ Logout/Login erforderlich, damit die Gruppenänderung wirksam wird.", "warning")
                        
                        self.log("🚀 Starte und aktiviere Docker Service...", "info")
                        subprocess.run(["sudo", "systemctl", "enable", "--now", "docker"])
                        return True
                    else:
                        self.log("❌ Docker Installation fehlgeschlagen", "error")
                        return False
                except Exception as e:
                    self.log(f"❌ Fehler beim Docker Install-Script: {e}", "error")
                    return False
                    
        except Exception as e:
            self.log(f"❌ Fehler bei automatischer Docker-Installation: {e}", "error")
            return False

    def _check_wsl2(self):
        """Prüft ob WSL2 verfügbar ist (Windows)"""
        try:
            result = subprocess.run(["wsl", "--list", "--verbose"], 
                                  capture_output=True, text=True)
            return result.returncode == 0 and "2" in result.stdout
        except:
            return False

    def _check_virtualization(self):
        """Prüft ob Virtualisierung aktiviert ist (Windows)"""
        try:
            result = subprocess.run(["powershell", "-Command", 
                                   "Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V"], 
                                  capture_output=True, text=True)
            return "Enabled" in result.stdout
        except:
            return False

    def _show_manual_docker_instructions(self):
            """Zeigt manuelle Docker-Installationsanweisungen"""
            system = platform.system().lower()
            if "windows" in system:
                self.log("📥 Windows: Docker Desktop von https://www.docker.com/products/docker-desktop/", tag="warning")
            elif "darwin" in system:
                self.log("📥 macOS: Docker Desktop von https://www.docker.com/products/docker-desktop/", tag="warning")
            else:
                self.log("📥 Linux: https://docs.docker.com/engine/install/", tag="warning")
                self.log("   oder: curl -fsSL https://get.docker.com | sh", tag="warning")

    def _attempt_ollama_install(self):
        """Versucht automatische Ollama Installation"""
        self.log("🔄 Versuche automatische Ollama-Installation...", tag="info")
        system = platform.system().lower()
        
        try:
            if "windows" in system:
                self.log("📥 Lade Ollama für Windows herunter...", tag="info")
                ollama_url = "https://ollama.com/download/OllamaSetup.exe"
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".exe", delete=False ) as tmp:
                    with urllib.request.urlopen(ollama_url) as response:
                        shutil.copyfileobj(response, tmp)
                    installer_path = tmp.name
                
                self.log("🚀 Starte Ollama Installer...", tag="info")
                proc = subprocess.run([installer_path, "/S"], capture_output=True, text=True)
                os.unlink(installer_path)
                if proc.returncode == 0:
                    self.log("✅ Ollama Installation erfolgreich!", "ok")
                    return True
                else:
                    self.log(f"❌ Ollama Installation fehlgeschlagen: {proc.stderr}", "error")
                    return False
            
            elif "darwin" in system:
                if is_installed("brew"):
                    self.log("🍺 Installiere Ollama via Homebrew...", tag="info")
                    rc, out = run_cmd_capture(["brew", "install", "ollama"])
                    if rc == 0:
                        self.log("✅ Ollama erfolgreich über Homebrew installiert!", "ok")
                        return True
                    else:
                        self.log(f"❌ Homebrew Installation fehlgeschlagen: {out}", "error")
                        return False
                else:
                    self.log("❌ Homebrew nicht gefunden. Bitte manuell installieren.", "error")
                    return False
                        
            else:
                self.log("🐧 Führe offizielles Installations-Script für Ollama aus...", tag="info")
                try:
                    with urllib.request.urlopen("https://ollama.com/install.sh" ) as response:
                        script_content = response.read().decode('utf-8')
                    proc = subprocess.Popen(["bash", "-c", script_content], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in proc.stdout: self.log(line.rstrip(), tag="out")
                    proc.wait()
                    if proc.returncode == 0:
                        self.log("✅ Ollama Installation erfolgreich!", "ok")
                        return True
                    else:
                        self.log("❌ Installation fehlgeschlagen.", "error")
                        return False
                except Exception as e:
                    self.log(f"❌ Fehler beim Download des Install-Scripts: {e}", "error")
                    return False
                    
        except Exception as e:
            self.log(f"❌ Fehler bei automatischer Ollama-Installation: {e}", "error")
            return False

    def create_project(self):
        """5️⃣ Projekt anlegen - erweitert um persistente Modell-Verzeichnisse"""
        self.log("5️⃣ Erstelle Projektstruktur...", tag="info")
        project = Path(self.project_var.get()).expanduser()
        try:
            project.mkdir(parents=True, exist_ok=True)
            
            # Basis-Verzeichnisse
            directories = [
                "postgres_data", 
                "n8n_data", 
                "ollama_data", 
                "vision_data", 
                "kyutai", 
                "searxng_data",
                "flowise_data",
                "openwebui_data",
                # Neue persistente Modell-Verzeichnisse
                "vision_models",          # Für Vision-Modelle (YOLO, PyTorch)
                "vision_models/torch",    # PyTorch Modelle
                "vision_models/yolo",     # YOLO Modelle
                "vision_models/huggingface",  # Hugging Face Modelle
                "vision_models/transformers", # Transformers Cache
                "vision_models/torch_extensions", # PyTorch Extensions
                "vision_cache",           # Vision pip/download Cache
                "vision_cache/pip",       # Pip Cache
                "vision_cache/torch",     # PyTorch Download Cache
                "kyutai_models",          # Für Voice-Modelle
                "kyutai_models/transformers", # Transformers für Voice
                "kyutai_models/huggingface",  # Hugging Face für Voice
                "kyutai_models/torch",    # PyTorch für Voice
                "kyutai_cache",           # Kyutai pip Cache
                "sd_data",                # Neu: Stable Diffusion Daten
                "sd_models",              # Neu: Persistente Modelle
                "sd_models/torch",        # Neu: PyTorch Modelle
                "sd_models/huggingface",  # Neu: Hugging Face Modelle
                "sd_models/transformers", # Neu: Transformers Cache
                "sd_models/diffusers",    # Neu: Diffusers Cache
                "sd_outputs",             # Neu: Generierte Bilder
                "sd_cache",               # Neu: Cache für pip/downloads
                "sd_cache/pip"            # Neu: Pip Cache
            ]
            
            for d in directories:
                dir_path = project / d
                dir_path.mkdir(parents=True, exist_ok=True)
                if "models" in d or "cache" in d or "outputs" in d:
                    self.log(f"💾 {d} erstellt (persistent)", tag="info")
                else:
                    self.log(f"📁 {d} erstellt", tag="info")
            
            # .gitignore erstellen für große Modell-Dateien
            gitignore_content = """# Modell-Dateien (zu groß für Git)
    vision_models/
    kyutai_models/
    vision_cache/
    kyutai_cache/
    sd_models/
    sd_cache/
    ollama_data/
    *.pt
    *.bin
    *.safetensors
    *.onnx
    *.pkl
    *.pth

    # Docker Volumes
    postgres_data/
    n8n_data/
    searxng_data/
    sd_outputs/

    # Logs
    *.log
    """
            gitignore_path = project / ".gitignore"
            if not gitignore_path.exists():
                with open(gitignore_path, 'w', encoding='utf-8') as f:
                    f.write(gitignore_content)
                self.log("📄 .gitignore erstellt (schließt Modell-Dateien aus)", tag="info")
            
            self.log(f"✅ Projektverzeichnis angelegt: {project}", tag="ok")
            self.log("💾 Persistente Modell-Speicherung konfiguriert", tag="ok")
            self.log("🚀 Modelle werden nach erstem Download wiederverwendet", tag="info")
            return True
            
        except Exception as e:
            self.log(f"❌ Fehler beim Anlegen des Projekts: {e}", tag="error")
            return False

    def clone_kyutai(self):
        """6️⃣ Kyutai Repo klonen"""
        self.log("6️⃣ Klone Kyutai Repository...", tag="info")
        project = Path(self.project_var.get()).expanduser()
        repo_dir = project / "kyutai"
        
        if repo_dir.exists() and any(repo_dir.iterdir()):
            self.log("✅ Kyutai-Repo scheint bereits vorhanden zu sein.", tag="ok")
            return True
            
        # Git nochmal prüfen (eventuell wurde PATH nach Installation nicht neu geladen)
        if not is_installed("git"):
            self.log("❌ Git nicht gefunden. PATH neu laden...", tag="warning")
            # Nochmaliger Versuch PATH zu aktualisieren
            if platform.system().lower() == "windows":
                time.sleep(1)
                if is_installed("git"):
                    self.log("✅ Git nach PATH-Aktualisierung gefunden", tag="ok")
                else:
                    self.log("❌ Git immer noch nicht verfügbar. Eventuell ist Neustart erforderlich.", tag="error")
                    self.log("💡 Manuell: git clone https://github.com/kyutai-labs/delayed-streams-modeling.git", tag="info")
                    return False
            else:
                self.log("❌ Git nicht gefunden. Bitte zuerst Schritt 2 abschließen.", tag="error")
                return False
            
        self.log("📥 Klone kyutai repository...", tag="info")
        rc, out = run_cmd_capture(["git", "clone", "https://github.com/kyutai-labs/delayed-streams-modeling.git", str(repo_dir)])
        if rc == 0:
            self.log("✅ Kyutai Repo erfolgreich geklont.", tag="ok")
            return True
        else:
            self.log(f"❌ Fehler beim Klonen: {out}", tag="error")
            return False

    def docker_up(self):
        """8️⃣ Docker Services starten"""
        self.log("8️⃣ Starte Docker Services...", tag="info")
        
        # Docker nochmal prüfen (eventuell wurde nach Installation PATH nicht neu geladen)
        if not is_installed("docker"):
            self.log("❌ Docker nicht gefunden. PATH neu laden...", tag="warning")
            time.sleep(1)
            if not is_installed("docker"):
                self.log("❌ Docker immer noch nicht verfügbar.", tag="error")
                self.log("💡 Mögliche Lösungen:", tag="info")
                self.log("   • Docker Desktop manuell starten", tag="info")
                self.log("   • Windows/macOS: Neustart erforderlich", tag="info")
                self.log("   • Linux: sudo systemctl start docker", tag="info")
                self.log("   • Linux: Benutzer zur docker-Gruppe hinzufügen", tag="info")
                return False
            
        project = Path(self.project_var.get()).expanduser()
        compose_path = project / COMPOSE_FILENAME
        if not compose_path.exists():
            self.log("❌ docker-compose.yml fehlt. Bitte zuerst Schritt 7 abschließen.", tag="error")
            return False
            
        # Prüfe Docker Daemon
        self.log("🔍 Prüfe Docker Daemon...", tag="info")
        rc, out = run_cmd_capture(["docker", "info"])
        if rc != 0:
            self.log("❌ Docker Daemon nicht erreichbar!", tag="error")
            self.log("💡 Mögliche Lösungen:", tag="info")
            if platform.system().lower() == "windows":
                self.log("   • Docker Desktop starten", tag="info")
                self.log("   • Windows neu starten falls Docker gerade installiert", tag="info")
            elif "darwin" in platform.system().lower():
                self.log("   • Docker Desktop aus Applications starten", tag="info")
            else:
                self.log("   • sudo systemctl start docker", tag="info")
                self.log("   • sudo usermod -aG docker $USER && logout/login", tag="info")
            return False
            
        self.log("🚀 Starte docker compose up -d ...", tag="info")
        rc, out = run_cmd_capture(["docker", "compose", "up", "-d"], cwd=str(project))
        if rc == 0:
            self.log("✅ Docker Compose erfolgreich gestartet!", tag="ok")
            self.log("🌐 Services erreichbar unter:", tag="info")
            self.log(f"   • n8n: http://localhost:{self.n8n_port.get()}", tag="info")
            self.log(f"   • Ollama API: http://localhost:{self.ollama_port.get()}", tag="info")
            self.log(f"   • Vision Service: http://localhost:{self.vision_port.get()}", tag="info")
            self.log(f"   • Kyutai Voice: http://localhost:{self.kyutai_port.get()}", tag="info")
            self.log(f"   • SearxNG: http://localhost:{self.searxng_port.get()}", tag="info")
            self.log(f"   • Open-WebUI: http://localhost:{self.openwebui_port.get()}", tag="info")
            return True
        else:
            self.log(f"❌ Fehler beim Start: {out}", tag="error")
            
            # Hilfreiche Fehlermeldungen
            if "port is already allocated" in out.lower():
                self.log("💡 Port bereits belegt - andere Services stoppen oder Ports ändern", tag="info")
            elif "no space left on device" in out.lower():
                self.log("💡 Nicht genug Speicherplatz - Docker Images aufräumen", tag="info")
            elif "permission denied" in out.lower():
                self.log("💡 Berechtigungsfehler - Benutzer zur docker-Gruppe hinzufügen", tag="info")
                
            return False

    # ------------------- Management Actions -------------------
    def docker_down(self):
        if not is_installed("docker"):
            self.log("❌ Docker nicht gefunden.", tag="error")
            return
        project = Path(self.project_var.get()).expanduser()
        self.log("🛑 Fahre Docker Compose herunter (down)...", tag="info")
        rc, out = run_cmd_capture(["docker", "compose", "down"], cwd=str(project))
        if rc == 0:
            self.log("✅ Docker Compose gestoppt.", tag="ok")
        else:
            self.log(f"❌ Fehler beim Stop: {out}", tag="error")

    def docker_status(self):
        if not is_installed("docker"):
            self.log("❌ Docker nicht installiert.", tag="error")
            return
        self.log("📊 Prüfe Docker Container Status...", tag="info")
        rc, out = run_cmd_capture(["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"])
        if rc == 0:
            self.log("🐳 Docker Services:\n" + out, tag="info")
        else:
            self.log("❌ Fehler beim Abfragen von docker ps:\n" + out, tag="error")

    def stream_logs(self):
        if not is_installed("docker"):
            self.log("❌ Docker nicht installiert.", tag="error")
            return
        project = Path(self.project_var.get()).expanduser()
        self.log("📋 Starte Streaming der Docker Compose Logs (beende mit 'Logs stoppen')...", tag="info")
        self._stop_logs.clear()

        COLORS = {
            "error": "\033[91m",
            "warning": "\033[93m",
            "info": "\033[94m",
            "out": "\033[0m"
        }

        def detect_tag(line: str) -> str:
            low = line.lower()
            if any(k in low for k in ["error", "fehler", "traceback"]):
                return "error"
            elif any(k in low for k in ["warn", "deprecated"]):
                return "warning"
            elif any(k in low for k in ["info", "started", "listening"]):
                return "info"
            else:
                return "out"

        def _stream():
            try:
                p = subprocess.Popen(
                    ["docker", "compose", "logs", "--follow", "--no-color"],
                    cwd=str(project),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace"
                )
                for line in p.stdout:
                    if self._stop_logs.is_set():
                        try:
                            p.terminate()
                        except Exception:
                            pass
                        break

                    tag = detect_tag(line)
                    line_clean = line.rstrip("\n")

                    # Konsole farbig, GUI ohne ANSI-Farben
                    if sys.stdout.isatty():
                        print(f"{COLORS[tag]}{line_clean}{COLORS['out']}")
                    else:
                        print(line_clean)

                    self.log(line_clean, tag=tag)

                self.log("⏹️ Log-Streaming beendet.", tag="info")
            except Exception as e:
                self.log(f"❌ Fehler beim Log-Streaming: {e}", tag="error")

        threading.Thread(target=_stream, daemon=True).start()

    def stop_logs(self):
        self._stop_logs.set()
        self.log("⏹️ Stoppe Log-Streaming angefordert.", tag="info")

    def test_endpoints(self):
        self.log("🧪 Teste alle Service-Endpunkte...", tag="info")
        endpoints = [
            ("n8n", f"http://localhost:{self.n8n_port.get()}"),
            ("ollama", f"http://localhost:{self.ollama_port.get()}/api/version"),
            ("vision", f"http://localhost:{self.vision_port.get()}"),
            ("kyutai", f"http://localhost:{self.kyutai_port.get()}"),
            ("searxng", f"http://localhost:{self.searxng_port.get()}"),
            ("stable-diffusion", f"http://localhost:{self.stablediffusion_port.get()}/health"),
            ("open-webui", f"http://localhost:{self.openwebui_port.get()}")
        ]
        
        success_count = 0
        for name, url in endpoints:
            self.log(f"🔍 Teste {name} @ {url}", tag="info")
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    status = resp.getcode()
                    if status == 200:
                        data = resp.read(500).decode(errors="ignore")
                        self.log(f"✅ {name} erreichbar (HTTP {status}): {data[:100]}...", tag="ok")
                        success_count += 1
                    else:
                        self.log(f"⚠️ {name} antwortet mit HTTP {status}", tag="warning")
            except Exception as e:
                self.log(f"❌ {name} NICHT erreichbar: {e}", tag="error")
        
        self.log(f"📊 Test abgeschlossen: {success_count}/{len(endpoints)} Services erreichbar", tag="info")


    def open_stablediffusion(self):
        url = f"http://localhost:{self.stablediffusion_port.get()}"
        try:
            webbrowser.open(url)
            self.log(f"🌐 Stable Diffusion geöffnet: {url}", tag="ok")
        except Exception as e:
            self.log(f"❌ Fehler beim Öffnen von Stable Diffusion: {e}", tag="error")
            self.log(f"💡 Öffne manuell: {url}", tag="info")

    def open_openwebui(self):
        url = f"http://localhost:{self.openwebui_port.get()}"
        try:
            webbrowser.open(url)
            self.log(f"🌐 Open-WebUI geöffnet: {url}", tag="ok")
            self.log("💡 Moderne Chat-Oberfläche für Ollama-Modelle", tag="info")
            self.log("🎯 Verbindet automatisch mit lokalem Ollama Service", tag="info")
        except Exception as e:
            self.log(f"❌ Fehler beim Öffnen von Open-WebUI: {e}", tag="error")
            self.log(f"💡 Öffne manuell: {url}", tag="info")

    def open_flowise(self):
        url = f"http://localhost:{self.flowise_port.get( )}"
        try:
            webbrowser.open(url)
            self.log(f"🌊 Flowise geöffnet: {url}", tag="ok")
        except Exception as e:
            self.log(f"❌ Fehler beim Öffnen von Flowise: {e}", tag="error")
            self.log(f"💡 Öffne manuell: {url}", tag="info")

    def ollama_pull_dialog(self):
        """Dialog für Ollama Modell Pull - NICHT threaded aufrufen!"""
        model = simpledialog.askstring(
            "Ollama Modell Pull", 
            "Modellname eingeben:\n\n🎮 RTX 3050 Empfehlungen:\n• deepseek-r1:1.5b (schnell, ~1.5GB)\n• llama3.2:3b (ausgewogen, ~2GB)\n• deepseek-r1:7b (sehr smart, ~4.5GB)\n• qwen2.5:3b (gut für Code, ~2GB)\n\nModell:", 
            parent=self.root
        )
        if model:
            # NUR den Pull-Prozess threaden, nicht den Dialog!
            self.threaded(self.ollama_pull)(model)

    def ollama_pull(self, model_name):
        """Ollama Modell Pull - mit verbesserter Container-Erkennung und Encoding-Fix"""
        if not is_installed("docker"):
            self.log("❌ Docker nicht installiert.", tag="error")
            return
            
        self.log(f"📥 Starte ollama Pull für Modell: {model_name}", tag="info")
        
        # Verbesserte Container-Erkennung
        self.log("🔍 Suche Ollama Container...", tag="info")
        
        # Suche nach Containern die "ollama" im Namen enthalten
        rc, out = run_cmd_capture(["docker", "ps", "--format", "{{.Names}}", "--filter", "ancestor=ollama/ollama"])
        
        if rc != 0:
            self.log(f"❌ Fehler beim Prüfen von Containern: {out}", tag="error")
            return
        
        container_names = [name.strip() for name in out.split('\n') if name.strip()]
        
        if not container_names:
            # Alternative: Suche nach beliebigen laufenden Containern
            self.log("🔍 Kein Ollama-Container gefunden, suche alle laufenden Container...", tag="info")
            rc2, out2 = run_cmd_capture(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"])
            
            if rc2 == 0 and out2.strip():
                self.log("🐳 Laufende Container:", tag="info")
                for line in out2.strip().split('\n'):
                    if line.strip():
                        self.log(f"   {line}", tag="info")
                
                # Suche speziell nach ollama
                ollama_containers = [line.split('\t')[0] for line in out2.strip().split('\n') 
                                   if 'ollama' in line.lower()]
                
                if ollama_containers:
                    container_names = ollama_containers
                    self.log(f"✅ Ollama Container gefunden: {container_names}", tag="ok")
                else:
                    self.log("❌ Kein Ollama Container läuft!", tag="error")
                    self.log("💡 Lösungen:", tag="info")
                    self.log("   • Docker Services starten (Schritt 8)", tag="info")
                    self.log("   • 'Docker Status' prüfen", tag="info")
                    self.log("   • Fallback: lokales Ollama verwenden", tag="info")
                    
                    # Fallback zu lokalem Ollama
                    if is_installed("ollama"):
                        self.log("🔄 Versuche lokales Ollama...", tag="info")
                        self.ollama_pull_local(model_name)
                    return
            else:
                self.log("❌ Keine Docker Container laufen!", tag="error")
                self.log("💡 Starte zuerst Docker Services (Schritt 8)", tag="warning")
                return
        
        # Verwende den ersten gefundenen Container
        container_name = container_names[0]
        self.log(f"🐳 Verwende Container: {container_name}", tag="info")
        
        cmd = ["docker", "exec", container_name, "ollama", "pull", model_name]
        
        # Stream the pull process mit verbessertem Encoding
        try:
            # Verwende verschiedene Encoding-Strategien
            encoding_strategies = ['utf-8', 'latin1', 'cp1252', 'ascii']
            
            for encoding in encoding_strategies:
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                          text=True, bufsize=1, universal_newlines=True, encoding=encoding)
                    
                    self.log(f"📊 Verwende Encoding: {encoding}", tag="info")
                    
                    for line in iter(proc.stdout.readline, ''):
                        if line:
                            clean_line = line.rstrip('\n')
                            if clean_line:
                                # Bereinige Sonderzeichen für saubere Anzeige
                                try:
                                    clean_display = clean_line.encode('ascii', errors='replace').decode('ascii')
                                    self.log(clean_display, tag="out")
                                except:
                                    # Fallback: nur druckbare ASCII-Zeichen anzeigen
                                    clean_display = ''.join(c for c in clean_line if ord(c) < 128)
                                    self.log(clean_display, tag="out")
                    
                    proc.wait()
                    
                    if proc.returncode == 0:
                        self.log(f"✅ Ollama Modell {model_name} erfolgreich gepullt!", tag="ok")
                        self.log(f"🎮 GPU-beschleunigt auf RTX 3050 (6GB VRAM verfügbar)", tag="ok")
                        self.threaded(self.test_ollama_model)(model_name)
                        return
                    else:
                        self.log(f"❌ Fehler beim ollama pull (Exit Code: {proc.returncode})", tag="error")
                        return
                        
                except UnicodeDecodeError as e:
                    if encoding == encoding_strategies[-1]:  # Letzter Versuch
                        self.log(f"❌ Alle Encoding-Versuche fehlgeschlagen: {e}", tag="error")
                        # Fallback: Binär-Modus verwenden
                        self.log("🔄 Fallback: Verwende binären Modus...", tag="info")
                        self.ollama_pull_binary_mode(model_name, container_name)
                        return
                    else:
                        self.log(f"⚠️ Encoding {encoding} fehlgeschlagen, versuche nächstes...", tag="warning")
                        continue
                        
        except Exception as e:
            self.log(f"❌ Fehler beim Ausführen von ollama pull: {e}", tag="error")
            self.log("🔄 Versuche alternativen Ansatz...", tag="info")
            self.ollama_pull_simple(model_name, container_name)

    def ollama_pull_binary_mode(self, model_name, container_name):
        """Fallback: Ollama Pull im binären Modus für problematische Encodings"""
        self.log("🔧 Verwende binären Modus für Ollama Pull...", tag="info")
        
        cmd = ["docker", "exec", container_name, "ollama", "pull", model_name]
        
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
            while True:
                # Lies in kleinen Blöcken
                chunk = proc.stdout.read(1024)
                if not chunk:
                    break
                    
                # Konvertiere zu String und bereinige
                try:
                    text = chunk.decode('utf-8', errors='replace')
                    # Entferne Steuerzeichen außer Newlines und Tabs
                    clean_text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
                    if clean_text.strip():
                        self.log(clean_text.strip(), tag="out")
                except Exception:
                    # Zeige nur dass Daten empfangen werden
                    self.log(".", tag="out")
            
            proc.wait()
            
            if proc.returncode == 0:
                self.log(f"✅ Ollama Modell {model_name} erfolgreich gepullt!", tag="ok")
                self.threaded(self.test_ollama_model)(model_name)
            else:
                self.log(f"❌ Pull fehlgeschlagen (Exit Code: {proc.returncode})", tag="error")
                
        except Exception as e:
            self.log(f"❌ Binärer Modus fehlgeschlagen: {e}", tag="error")

    def ollama_pull_simple(self, model_name, container_name):
        """Einfacher Pull ohne Streaming - als letzter Fallback"""
        self.log("🔧 Verwende einfachen Pull-Modus ohne Live-Streaming...", tag="info")
        
        cmd = ["docker", "exec", container_name, "ollama", "pull", model_name]
        
        try:
            # Zeige Progress-Indikator
            self.log(f"⏳ Lade {model_name} herunter... (kann einige Minuten dauern)", tag="info")
            
            # Einfacher subprocess.run ohne Streaming
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30min Timeout
            
            if result.returncode == 0:
                self.log(f"✅ Ollama Modell {model_name} erfolgreich gepullt!", tag="ok")
                if result.stdout.strip():
                    # Zeige finale Ausgabe
                    clean_output = result.stdout.encode('ascii', errors='replace').decode('ascii')
                    self.log(f"📄 Ausgabe: {clean_output[-500:]}", tag="info")  # Zeige letzten Teil
                self.threaded(self.test_ollama_model)(model_name)
            else:
                self.log(f"❌ Pull fehlgeschlagen (Exit Code: {result.returncode})", tag="error")
                if result.stderr:
                    error_clean = result.stderr.encode('ascii', errors='replace').decode('ascii')
                    self.log(f"❌ Fehler: {error_clean}", tag="error")
                    
        except subprocess.TimeoutExpired:
            self.log("⏰ Pull-Timeout nach 30 Minuten", tag="error")
            self.log("💡 Versuche es später nochmal oder verwende ein kleineres Modell", tag="info")
        except Exception as e:
            self.log(f"❌ Einfacher Pull-Modus fehlgeschlagen: {e}", tag="error")

    def ollama_pull_local(self, model_name):
        """Fallback: Lokales Ollama für Modell Pull verwenden"""
        self.log(f"🔄 Verwende lokales Ollama für Modell: {model_name}", tag="info")
        
        if not is_installed("ollama"):
            self.log("❌ Lokales Ollama nicht gefunden. Bitte zuerst Schritt 4 abschließen.", tag="error")
            return
        
        try:
            # Lokales Ollama verwenden
            proc = subprocess.Popen(["ollama", "pull", model_name], 
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                  text=True, bufsize=1, universal_newlines=True)
            
            for line in iter(proc.stdout.readline, ''):
                if line:
                    clean_line = line.rstrip('\n')
                    if clean_line:
                        self.log(clean_line, tag="out")
            
            proc.wait()
            
            if proc.returncode == 0:
                self.log(f"✅ Modell {model_name} erfolgreich mit lokalem Ollama gepullt!", tag="ok")
                self.log("💡 Modell ist jetzt sowohl lokal als auch im Container verfügbar", tag="info")
            else:
                self.log(f"❌ Lokaler Pull fehlgeschlagen (Exit Code: {proc.returncode})", tag="error")
                
        except Exception as e:
            self.log(f"❌ Fehler beim lokalen ollama pull: {e}", tag="error")

    def debug_docker_containers(self):
        """Debug-Methode um Container zu analysieren"""
        self.log("🔍 Docker Container Debug-Informationen:", tag="info")
        
        # Alle Container (auch gestoppte)
        rc, out = run_cmd_capture(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"])
        if rc == 0 and out.strip():
            self.log("📋 Alle Container:", tag="info")
            for line in out.strip().split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    name = parts[0] if len(parts) > 0 else "unknown"
                    status = parts[1] if len(parts) > 1 else "unknown"
                    image = parts[2] if len(parts) > 2 else "unknown"
                    
                    if 'ollama' in image.lower():
                        self.log(f"🎯 OLLAMA: {name} | {status} | {image}", tag="ok")
                    else:
                        self.log(f"   {name} | {status} | {image}", tag="info")
        
        # Docker Compose Services
        project = Path(self.project_var.get()).expanduser()
        if (project / "docker-compose.yml").exists():
            self.log("🐳 Docker Compose Services:", tag="info")
            rc2, out2 = run_cmd_capture(["docker", "compose", "ps"], cwd=str(project))
            if rc2 == 0 and out2.strip():
                self.log(out2, tag="info")
            else:
                self.log("⚠️ Keine Compose Services laufen", tag="warning")



    def test_ollama_model(self, model_name):
        """Testet ein Ollama Modell nach dem Pull"""
        try:
            import json as json_module
            
            self.log(f"🧪 Teste Modell {model_name}...", tag="info")
            
            # Einfacher Test-Prompt
            test_data = {
                "model": model_name,
                "prompt": "Hello! Please respond briefly to test if you're working.",
                "stream": False
            }
            
            data = json_module.dumps(test_data).encode('utf-8')
            req = urllib.request.Request(f"http://localhost:{self.ollama_port.get()}/api/generate", 
                                       data=data)
            req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.getcode() == 200:
                    result = json_module.loads(response.read().decode('utf-8'))
                    if 'response' in result:
                        response_text = result['response'].strip()
                        self.log(f"✅ Modell {model_name} funktioniert!", tag="ok")
                        self.log(f"🤖 Antwort: {response_text[:200]}{'...' if len(response_text) > 200 else ''}", tag="info")
                        
                        # GPU-Info falls verfügbar
                        if 'load_duration' in result:
                            load_time = result.get('load_duration', 0) / 1e9  # Nanosekunden zu Sekunden
                            self.log(f"⚡ Ladezeit: {load_time:.2f}s (GPU-beschleunigt)", tag="info")
                    else:
                        self.log("⚠️ Unerwartete Antwort-Struktur", tag="warning")
                else:
                    self.log(f"❌ HTTP {response.getcode()}", tag="error")
                    
        except Exception as e:
            self.log(f"❌ Fehler beim Testen des Modells: {e}", tag="error")
            self.log("💡 Das Modell wurde trotzdem heruntergeladen und sollte funktionieren", tag="info")

    def import_n8n_workflow_dialog(self):
        file_path = filedialog.askopenfilename(
            title="n8n Workflow JSON auswählen", 
            filetypes=[("JSON Dateien", "*.json"), ("Alle Dateien", "*.*")]
        )
        if file_path:
            self.threaded(self.import_n8n_workflow)(file_path)

    def import_n8n_workflow(self, file_path):
        project = Path(self.project_var.get()).expanduser()
        if not Path(file_path).exists():
            self.log("❌ Ausgewählte Workflow-Datei nicht gefunden.", tag="error")
            return
            
        self.log(f"📤 Importiere n8n Workflow: {Path(file_path).name}", tag="info")
        
        # Prüfe ob n8n Container läuft
        rc, out = run_cmd_capture(["docker", "ps", "--filter", "name=n8n", "--format", "{{.Names}}"])
        if rc != 0:
            self.log(f"❌ Fehler beim Prüfen von Containern: {out}", tag="error")
            return
            
        if "n8n" in out:
            dest_path = "/tmp/workflow_import.json"
            
            # 1. Kopiere Datei in Container
            self.log("📋 Kopiere Workflow-JSON in n8n-Container...", tag="info")
            rc2, out2 = run_cmd_capture(["docker", "cp", file_path, f"n8n:{dest_path}"])
            if rc2 != 0:
                self.log(f"❌ Fehler beim Kopieren in Container: {out2}", tag="error")
                return
            
            # 2. Versuche Import via CLI
            self.log("🔄 Versuche Workflow-Import über n8n CLI...", tag="info")
            import_commands = [
                ["docker", "exec", "n8n", "n8n", "import:workflow", "--input", dest_path],
                ["docker", "exec", "n8n", "n8n", "import", "--input", dest_path],
                ["docker", "exec", "n8n", "bash", "-c", f"n8n import:workflow --input {dest_path}"],
            ]
            
            success = False
            for cmd in import_commands:
                rc3, out3 = run_cmd_capture(cmd)
                if rc3 == 0:
                    self.log("✅ n8n Import erfolgreich!", tag="ok")
                    if out3.strip():
                        self.log(out3, tag="out")
                    success = True
                    break
                else:
                    self.log(f"⚠️ Versuch fehlgeschlagen: {' '.join(cmd[3:])}", tag="warning")
            
            if not success:
                # Fallback: API Import versuchen
                self.log("🔄 CLI-Import fehlgeschlagen. Versuche API-Import...", tag="warning")
                try:
                    import json as json_module
                    import urllib.parse
                    
                    # Lade Workflow JSON
                    with open(file_path, 'r', encoding='utf-8') as f:
                        workflow_data = json_module.load(f)
                    
                    # API Request vorbereiten
                    url = f"http://localhost:{self.n8n_port.get()}/rest/workflows"
                    data = json_module.dumps(workflow_data).encode('utf-8')
                    
                    # Basic Auth Header
                    import base64
                    auth_string = f"{self.n8n_user.get()}:{self.n8n_pass.get()}"
                    auth_bytes = auth_string.encode('ascii')
                    auth_header = base64.b64encode(auth_bytes).decode('ascii')
                    
                    # HTTP Request
                    req = urllib.request.Request(url, data=data)
                    req.add_header('Content-Type', 'application/json')
                    req.add_header('Authorization', f'Basic {auth_header}')
                    
                    with urllib.request.urlopen(req, timeout=15) as response:
                        if response.getcode() in (200, 201):
                            self.log("✅ n8n API Import erfolgreich!", tag="ok")
                            success = True
                        else:
                            resp_data = response.read().decode('utf-8')
                            self.log(f"❌ API Import Fehler: HTTP {response.getcode()}", tag="error")
                            self.log(resp_data[:500], tag="error")
                            
                except Exception as e:
                    self.log(f"❌ API-Import fehlgeschlagen: {e}", tag="error")
            
            if not success:
                self.log("💡 Manueller Import erforderlich:", tag="warning")
                self.log(f"   1. Öffne n8n: http://localhost:{self.n8n_port.get()}", tag="info")
                self.log(f"   2. Gehe zu 'Workflows' > 'Import'", tag="info")
                self.log(f"   3. Lade die Datei hoch: {file_path}", tag="info")
        else:
            self.log("❌ n8n Container läuft nicht. Bitte zuerst Docker Services starten (Schritt 8).", tag="error")

    def show_searx_info(self):
        project = Path(self.project_var.get()).expanduser()
        cfg_dir = project / "searxng_data"
        
        info_text = f"""🔍 SearxNG Konfiguration

📍 Service URL: http://localhost:{self.searxng_port.get()}
📁 Konfigurationsverzeichnis: {cfg_dir}

🔧 Konfiguration anpassen:
1. Stoppe SearxNG: docker compose stop searxng
2. Bearbeite Dateien in: {cfg_dir}
   • settings.yml - Hauptkonfiguration
   • limiter.toml - Rate Limiting
3. Starte neu: docker compose start searxng

💡 Tipps:
• Standard-Port geändert auf 8888
• Instanz-Name: "LokaleSuche"
• Für erweiterte Konfiguration siehe: 
  https://docs.searxng.org/admin/settings/"""
        
        self.log(info_text, tag="info")
        
        messagebox.showinfo("SearxNG Information", 
            f"SearxNG läuft auf Port {self.searxng_port.get()}\n\n"
            f"URL: http://localhost:{self.searxng_port.get()}\n"
            f"Konfiguration: {cfg_dir}\n\n"
            "Details siehe Log-Ausgabe."
        )

    def open_n8n(self):
        url = f"http://localhost:{self.n8n_port.get()}"
        try:
            webbrowser.open(url)
            self.log(f"🌐 n8n geöffnet: {url}", tag="ok")
            self.log(f"👤 Benutzer: {self.n8n_user.get()}", tag="info")
            self.log(f"🔒 Passwort: {'*' * len(self.n8n_pass.get())}", tag="info")
        except Exception as e:
            self.log(f"❌ Fehler beim Öffnen von n8n: {e}", tag="error")
            self.log(f"💡 Öffne manuell: {url}", tag="info")

    def open_project_dir(self):
        project = Path(self.project_var.get()).expanduser()
        if not project.exists():
            self.log("❌ Projektverzeichnis existiert nicht.", tag="error")
            return
            
        try:
            system = platform.system().lower()
            if "windows" in system:
                os.startfile(project)
            elif "darwin" in system:
                subprocess.run(["open", str(project)])
            else:
                subprocess.run(["xdg-open", str(project)])
            self.log(f"📁 Projektordner geöffnet: {project}", tag="ok")
        except Exception as e:
            self.log(f"❌ Fehler beim Öffnen: {e}", tag="error")

# --------------------------- Main ---------------------------
def main():
    """Hauptfunktion - startet die GUI"""
    try:
        # Prüfe grundlegende Python-Requirements
        if not check_python_requirements():
            print("❌ Python 3.8+ erforderlich!")
            sys.exit(1)
        
        # Starte GUI
        root = tk.Tk()
        app = AllInOneGUI(root)
        
        # Window Icon (falls verfügbar)
        try:
            # Versuche ein Icon zu setzen (optional)
            root.iconbitmap(default="")  # Leer = Standard System Icon
        except:
            pass  # Icon nicht kritisch
        
        # Starte GUI Event Loop
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\n👋 Setup abgebrochen.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
