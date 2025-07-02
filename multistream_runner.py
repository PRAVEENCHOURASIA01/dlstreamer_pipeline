import subprocess
import os
import time
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import threading
import signal
import sys

# --------------------------
# CONFIGURATION
# --------------------------

STREAM_COUNTS_TO_TEST = [1, 2, 4]

VIDEO_FILES = [
    "videos/video1.mp4",
    "videos/video2.mp4",
    "videos/video3.mp4",
    "videos/video4.mp4",
]

DETECTION_MODEL = "models/person-detection-retail-0013.xml"
DETECTION_MODEL_PROC = "models/person-detection-retail-0013.json"
GENDER_MODEL = "models/person-attributes-recognition-crossroad-0230.xml"
GENDER_MODEL_PROC = "models/person-attributes-recognition-crossroad-0230.json"
VEHICLE_MODEL = "models/vehicle-attributes-recognition-barrier-0039.xml"
VEHICLE_MODEL_PROC = "models/vehicle-attributes-recognition-barrier-0039.json"

LOG_DIR = Path("logs")
RESULTS_DIR = Path("results")
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TEST_DURATION = 30  # seconds per test
SHOW_PIPELINE_OUTPUT = True  # show output for all streams

# --------------------------
# SYSTEM VALIDATION
# --------------------------

def validate_files():
    print("[VALIDATION] Checking required files...")
    missing_files = []
    for video in VIDEO_FILES:
        if not Path(video).exists():
            missing_files.append(f"Video: {video}")

    model_files = [DETECTION_MODEL, DETECTION_MODEL_PROC, GENDER_MODEL,
                   GENDER_MODEL_PROC, VEHICLE_MODEL, VEHICLE_MODEL_PROC]
    for model in model_files:
        if not Path(model).exists():
            missing_files.append(f"Model: {model}")

    if missing_files:
        print("[ERROR] Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    print("[SUCCESS] All files validated")
    return True

def check_gstreamer():
    try:
        result = subprocess.run("gst-launch-1.0 --version", shell=True,
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"[INFO] GStreamer detected: {result.stdout.strip()}")
            return True
        else:
            print("[ERROR] GStreamer not found")
            return False
    except Exception as e:
        print(f"[ERROR] GStreamer check failed: {e}")
        return False

# --------------------------
# PIPELINE BUILDER
# --------------------------

def build_pipeline_cmd(video_file, stream_id, show_output=False):
    pipeline = [
        "gst-launch-1.0",
        f"filesrc location={video_file}",
        "! decodebin",
        "! videoconvert",
        "! video/x-raw,format=BGRx",
        f"! gvadetect model={DETECTION_MODEL} model-proc={DETECTION_MODEL_PROC} device=CPU nireq=4",
        "! queue max-size-buffers=10 leaky=2",
        f"! gvaclassify model={GENDER_MODEL} model-proc={GENDER_MODEL_PROC} device=CPU object-class=person nireq=2",
        "! queue max-size-buffers=10 leaky=2",
        f"! gvaclassify model={VEHICLE_MODEL} model-proc={VEHICLE_MODEL_PROC} device=CPU object-class=vehicle nireq=2",
        "! queue max-size-buffers=10 leaky=2",
        "! gvawatermark",
        f"! gvafpscounter name=fps_counter_{stream_id}",
    ]
    if show_output:
        pipeline.extend([
            "! videoconvert",
            "! autovideosink sync=false"
        ])
    else:
        pipeline.extend([
            "! fakesink sync=false"
        ])
    return " ".join(pipeline)

# --------------------------
# SYSTEM MONITORING
# --------------------------

class AdvancedSystemMonitor:
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.cpu_data = []
        self.memory_data = []
        self.process_pids = []

    def start_monitoring(self, process_pids, interval=2):
        self.monitoring = True
        self.process_pids = process_pids
        self.cpu_data.clear()
        self.memory_data.clear()

        def monitor_worker():
            psutil.cpu_percent(interval=None)
            while self.monitoring:
                system_cpu = psutil.cpu_percent(interval=None)
                total_process_memory = 0
                for pid in self.process_pids:
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            total_process_memory += proc.memory_info().rss / (1024 * 1024)
                    except Exception:
                        continue
                self.cpu_data.append(system_cpu)
                self.memory_data.append(total_process_memory)
                print(f"[MONITOR] CPU: {system_cpu:.1f}% | Memory: {total_process_memory:.1f}MB")
                time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
        print("[INFO] System monitoring started")

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        if not self.cpu_data:
            return {'avg_cpu': 0, 'max_cpu': 0, 'avg_memory': 0, 'max_memory': 0, 'samples': 0}

        return {
            'avg_cpu': sum(self.cpu_data) / len(self.cpu_data),
            'max_cpu': max(self.cpu_data),
            'avg_memory': sum(self.memory_data) / len(self.memory_data),
            'max_memory': max(self.memory_data),
            'samples': len(self.cpu_data)
        }

# --------------------------
# FPS PARSING
# --------------------------

def extract_fps_from_log(log_file):
    fps_values = []
    if not log_file.exists() or log_file.stat().st_size == 0:
        return fps_values

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    patterns = [
        r'fps:\s*(\d+\.?\d*)',
        r'current-fps:\s*(\d+\.?\d*)',
        r'per-stream=(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*fps',
        r'gvafpscounter.*?(\d+\.?\d*)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            try:
                fps = float(match)
                if 0.1 <= fps <= 200:
                    fps_values.append(fps)
            except ValueError:
                continue
    return fps_values

def calculate_stream_fps(fps_values):
    if not fps_values:
        return 0.0
    if len(fps_values) > 3:
        sorted_fps = sorted(fps_values)
        median = sorted_fps[len(sorted_fps) // 2]
        fps_values = [fps for fps in fps_values if abs(fps - median) < median * 0.5]
    return sum(fps_values) / len(fps_values)

# --------------------------
# PROCESS MANAGEMENT
# --------------------------

class StreamProcessManager:
    def __init__(self):
        self.processes = []
        self.log_files = []

    def start_streams(self, num_streams, timestamp, show_output=False):
        print(f"[START] Launching {num_streams} DL Streamer processes...")
        for i in range(num_streams):
            video_file = VIDEO_FILES[i % len(VIDEO_FILES)]
            stream_id = i + 1
            cmd = build_pipeline_cmd(video_file, stream_id, show_output)
            log_file = LOG_DIR / f"stream_{stream_id}_{num_streams}streams_{timestamp}.log"
            try:
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid
                    )
                self.processes.append(process)
                self.log_files.append(log_file)
                print(f"[LAUNCHED] Stream {stream_id}: PID={process.pid}, Video={Path(video_file).name}")
                time.sleep(0.5)
            except Exception as e:
                print(f"[ERROR] Failed to start stream {stream_id}: {e}")
        return [p.pid for p in self.processes]

    def terminate_all(self):
        print("[STOP] Terminating all streams...")
        for i, process in enumerate(self.processes):
            try:
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                        print(f"[STOPPED] Stream {i + 1} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        print(f"[KILLED] Stream {i + 1} force terminated")
            except Exception as e:
                print(f"[WARNING] Error stopping stream {i + 1}: {e}")
        self.processes.clear()

# --------------------------
# MAIN TEST EXECUTION
# --------------------------

def run_performance_test(num_streams, timestamp, show_output=False):
    process_manager = StreamProcessManager()
    monitor = AdvancedSystemMonitor()
    try:
        process_pids = process_manager.start_streams(num_streams, timestamp, show_output)
        if not process_pids:
            return None
        monitor.start_monitoring(process_pids, interval=2)
        time.sleep(TEST_DURATION)
        usage_stats = monitor.stop_monitoring()
        process_manager.terminate_all()

        total_fps = 0.0
        stream_fps_list = []
        successful_streams = 0
        for i, log_file in enumerate(process_manager.log_files):
            fps_values = extract_fps_from_log(log_file)
            avg_fps = calculate_stream_fps(fps_values)
            if avg_fps > 0:
                successful_streams += 1
                total_fps += avg_fps
                stream_fps_list.append(avg_fps)
            else:
                stream_fps_list.append(0.0)

        avg_fps_per_stream = total_fps / successful_streams if successful_streams else 0
        return {
            'num_streams': num_streams,
            'avg_fps_per_stream': avg_fps_per_stream,
            'avg_memory': usage_stats['avg_memory'],
        }
    finally:
        process_manager.terminate_all()
        monitor.stop_monitoring()

# --------------------------
# PLOT RESULTS
# --------------------------

def plot_results(results):
    streams = [r['num_streams'] for r in results]
    avg_fps = [r['avg_fps_per_stream'] for r in results]
    avg_mem = [r['avg_memory'] for r in results]

    plt.figure()
    plt.plot(streams, avg_fps, marker='o')
    plt.title('Average FPS per Stream')
    plt.xlabel('Number of Streams')
    plt.ylabel('Average FPS')
    plt.grid(True)
    plt.savefig(RESULTS_DIR / 'avg_fps_per_stream.png')
    plt.show()

    plt.figure()
    plt.plot(streams, avg_mem, marker='o', color='red')
    plt.title('Average Memory Usage')
    plt.xlabel('Number of Streams')
    plt.ylabel('Average Memory (MB)')
    plt.grid(True)
    plt.savefig(RESULTS_DIR / 'avg_memory_usage.png')
    plt.show()

# --------------------------
# ENTRYPOINT
# --------------------------

def main():
    if not validate_files():
        return
    if not check_gstreamer():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    for count in STREAM_COUNTS_TO_TEST:
        result = run_performance_test(count, timestamp, SHOW_PIPELINE_OUTPUT)
        if result:
            results.append(result)
    if results:
        plot_results(results)

def signal_handler(sig, frame):
    print("\n[EXIT] Cleaning up...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
