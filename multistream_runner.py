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

# Test different stream counts to find maximum supported
STREAM_COUNTS_TO_TEST = [1, 2, 4, 6, 8]

VIDEO_FILES = [
    "videos/video1.mp4",
    "videos/video2.mp4",
    "videos/video3.mp4",
    "videos/video4.mp4",
]

# Intel OpenVINO Models
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

# Test configuration
TEST_DURATION = 30  # seconds per test
SHOW_PIPELINE_OUTPUT = True  # Set True to see classification results

# --------------------------
# SYSTEM VALIDATION
# --------------------------

def validate_files():
    """Validate all required files exist"""
    print("[VALIDATION] Checking required files...")
    
    missing_files = []
    
    # Check video files
    for video in VIDEO_FILES:
        if not Path(video).exists():
            missing_files.append(f"Video: {video}")
    
    # Check model files
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
    """Check if GStreamer and DL Streamer are properly installed"""
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
    """Build optimized GStreamer DL Streamer pipeline"""
    
    # Base pipeline elements
    pipeline = [
        "gst-launch-1.0",
        f"filesrc location={video_file}",
        "! decodebin",
        "! videoconvert", 
        "! video/x-raw,format=BGRx",
        
        # Person/Vehicle Detection
        f"! gvadetect model={DETECTION_MODEL} model-proc={DETECTION_MODEL_PROC} device=CPU nireq=4",
        "! queue max-size-buffers=10 leaky=2",
        
        # Gender Classification for persons
        f"! gvaclassify model={GENDER_MODEL} model-proc={GENDER_MODEL_PROC} device=CPU object-class=person nireq=2",
        "! queue max-size-buffers=10 leaky=2",
        
        # Vehicle Classification
        f"! gvaclassify model={VEHICLE_MODEL} model-proc={VEHICLE_MODEL_PROC} device=CPU object-class=vehicle nireq=2", 
        "! queue max-size-buffers=10 leaky=2",
        
        # Add watermark to visualize detections
        "! gvawatermark",
        
        # FPS counter for performance measurement
        f"! gvafpscounter name=fps_counter_{stream_id}",
    ]
    
    # Output: either display or fakesink
    if show_output and stream_id == 1:  # Only show first stream to avoid window overload
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
# ADVANCED SYSTEM MONITORING
# --------------------------

class AdvancedSystemMonitor:
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.cpu_data = []
        self.memory_data = []
        self.process_pids = []
        
    def start_monitoring(self, process_pids, interval=2):
        """Start comprehensive system monitoring"""
        self.monitoring = True
        self.process_pids = process_pids
        self.cpu_data.clear()
        self.memory_data.clear()
        
        def monitor_worker():
            # Initialize CPU monitoring
            psutil.cpu_percent(interval=None)
            
            while self.monitoring:
                try:
                    # System-wide CPU usage
                    system_cpu = psutil.cpu_percent(interval=None)
                    
                    # Process-specific monitoring
                    total_process_memory = 0
                    active_processes = 0
                    
                    for pid in self.process_pids:
                        try:
                            proc = psutil.Process(pid)
                            if proc.is_running():
                                total_process_memory += proc.memory_info().rss / (1024 * 1024)  # MB
                                active_processes += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            continue
                    
                    # Store data
                    self.cpu_data.append(system_cpu)
                    self.memory_data.append(total_process_memory)
                    
                    print(f"[MONITOR] CPU: {system_cpu:5.1f}% | Memory: {total_process_memory:6.1f}MB | Active Processes: {active_processes}")
                    
                except Exception as e:
                    print(f"[WARNING] Monitoring error: {e}")
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
        print("[INFO] System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if not self.cpu_data:
            return {'avg_cpu': 0, 'max_cpu': 0, 'avg_memory': 0, 'max_memory': 0}
        
        stats = {
            'avg_cpu': sum(self.cpu_data) / len(self.cpu_data),
            'max_cpu': max(self.cpu_data),
            'avg_memory': sum(self.memory_data) / len(self.memory_data) if self.memory_data else 0,
            'max_memory': max(self.memory_data) if self.memory_data else 0,
            'samples': len(self.cpu_data)
        }
        
        print(f"[INFO] Monitoring stopped. Collected {stats['samples']} samples")
        return stats

# --------------------------
# ENHANCED FPS PARSING
# --------------------------

def extract_fps_from_log(log_file):
    """Extract FPS values from GStreamer DL Streamer log with enhanced parsing"""
    fps_values = []
    
    if not log_file.exists() or log_file.stat().st_size == 0:
        return fps_values
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Multiple patterns to catch different FPS output formats
        patterns = [
            r'fps:\s*(\d+\.?\d*)',
            r'current-fps:\s*(\d+\.?\d*)', 
            r'per-stream=(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*fps',
            r'FpsCounter.*?(\d+\.?\d*)',
            r'stream_fps:\s*(\d+\.?\d*)',
            # DL Streamer specific patterns
            r'gvafpscounter.*?(\d+\.?\d*)',
            r'fps_counter.*?(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    fps = float(match)
                    if 0.1 <= fps <= 200:  # Reasonable FPS range
                        fps_values.append(fps)
                except ValueError:
                    continue
        
        # If no FPS found, try parsing from the last few lines (most recent data)
        if not fps_values:
            lines = content.split('\n')[-50:]  # Last 50 lines
            for line in lines:
                if 'fps' in line.lower():
                    numbers = re.findall(r'\d+\.?\d*', line)
                    for num in numbers:
                        try:
                            fps = float(num)
                            if 0.1 <= fps <= 200:
                                fps_values.append(fps)
                        except ValueError:
                            continue
        
    except Exception as e:
        print(f"[WARNING] Error parsing log {log_file}: {e}")
    
    return fps_values

def calculate_stream_fps(fps_values):
    """Calculate representative FPS from collected values"""
    if not fps_values:
        return 0.0
    
    # Remove outliers (values that are too different from median)
    if len(fps_values) > 3:
        sorted_fps = sorted(fps_values)
        median = sorted_fps[len(sorted_fps)//2]
        filtered_fps = [fps for fps in fps_values if abs(fps - median) < median * 0.5]
        if filtered_fps:
            fps_values = filtered_fps
    
    return sum(fps_values) / len(fps_values)

# --------------------------
# PROCESS MANAGEMENT
# --------------------------

class StreamProcessManager:
    def __init__(self):
        self.processes = []
        self.log_files = []
        
    def start_streams(self, num_streams, timestamp, show_output=False):
        """Start DL Streamer processes"""
        print(f"[START] Launching {num_streams} DL Streamer processes...")
        
        for i in range(num_streams):
            video_file = VIDEO_FILES[i % len(VIDEO_FILES)]
            stream_id = i + 1
            
            # Build pipeline command
            cmd = build_pipeline_cmd(video_file, stream_id, show_output and i == 0)
            log_file = LOG_DIR / f"stream_{stream_id}_{num_streams}streams_{timestamp}.log"
            
            try:
                # Start process with proper logging
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid  # Create process group for clean termination
                    )
                
                self.processes.append(process)
                self.log_files.append(log_file)
                
                print(f"[LAUNCHED] Stream {stream_id}: PID={process.pid}, Video={Path(video_file).name}")
                
                # Small delay between process starts to avoid resource conflicts
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[ERROR] Failed to start stream {stream_id}: {e}")
        
        print(f"[SUCCESS] {len(self.processes)} streams launched successfully")
        return [p.pid for p in self.processes]
    
    def terminate_all(self):
        """Properly terminate all streams"""
        print("[STOP] Terminating all streams...")
        
        for i, process in enumerate(self.processes):
            try:
                # Send SIGTERM to process group
                if process.poll() is None:  # Process still running
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    
                # Wait for graceful termination
                try:
                    process.wait(timeout=3)
                    print(f"[STOPPED] Stream {i+1} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    print(f"[KILLED] Stream {i+1} force terminated")
                    
            except Exception as e:
                print(f"[WARNING] Error stopping stream {i+1}: {e}")
        
        self.processes.clear()
        print("[SUCCESS] All streams terminated")

# --------------------------
# MAIN TEST EXECUTION
# --------------------------

def run_performance_test(num_streams, timestamp, show_output=False):
    """Execute performance test for specified number of streams"""
    print(f"\n{'='*60}")
    print(f"TESTING {num_streams} STREAMS - DL STREAMER PERFORMANCE")
    print(f"{'='*60}")
    
    # Initialize components
    process_manager = StreamProcessManager()
    monitor = AdvancedSystemMonitor()
    
    try:
        # Start streams
        process_pids = process_manager.start_streams(num_streams, timestamp, show_output)
        
        if not process_pids:
            print("[ERROR] No streams started successfully")
            return None
        
        # Start system monitoring
        monitor.start_monitoring(process_pids, interval=2)
        
        # Let streams run for specified duration
        print(f"[RUNNING] Test duration: {TEST_DURATION} seconds...")
        if show_output:
            print("[INFO] Classification results will be displayed in video window")
        
        time.sleep(TEST_DURATION)
        
        # Stop monitoring
        usage_stats = monitor.stop_monitoring()
        
        # Terminate streams
        process_manager.terminate_all()
        
        # Parse FPS results
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
                print(f"[FPS] Stream {i+1}: {avg_fps:5.2f} fps ({len(fps_values)} samples)")
            else:
                print(f"[WARNING] Stream {i+1}: No FPS data collected")
                stream_fps_list.append(0.0)
        
        avg_fps_per_stream = total_fps / successful_streams if successful_streams > 0 else 0
        
        # Compile results
        result = {
            'num_streams': num_streams,
            'successful_streams': successful_streams,
            'total_fps': total_fps,
            'avg_fps_per_stream': avg_fps_per_stream,
            'individual_fps': stream_fps_list,
            'avg_cpu': usage_stats['avg_cpu'],
            'max_cpu': usage_stats['max_cpu'],
            'avg_memory': usage_stats['avg_memory'],
            'max_memory': usage_stats['max_memory'],
            'monitoring_samples': usage_stats['samples'],
            'success': successful_streams > 0 and avg_fps_per_stream > 1.0
        }
        
        # Print summary
        print(f"\n[RESULTS] {num_streams} Streams Summary:")
        print(f"  Successful Streams: {successful_streams}/{num_streams}")
        print(f"  Total FPS: {total_fps:6.2f}")
        print(f"  Avg FPS per Stream: {avg_fps_per_stream:6.2f}")
        print(f"  CPU Usage: {usage_stats['avg_cpu']:5.1f}% (avg) | {usage_stats['max_cpu']:5.1f}% (max)")
        print(f"  Memory Usage: {usage_stats['avg_memory']:6.1f}MB (avg) | {usage_stats['max_memory']:6.1f}MB (max)")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        return None
    finally:
        # Ensure cleanup
        process_manager.terminate_all()
        monitor.stop_monitoring()

# --------------------------
# VISUALIZATION & REPORTING
# --------------------------

def create_performance_charts(results, timestamp):
    """Generate comprehensive performance charts"""
    if not results:
        print("[WARNING] No results to visualize")
        return None
    
    # Extract data
    streams = [r['num_streams'] for r in results]
    total_fps = [r['total_fps'] for r in results]
    avg_fps = [r['avg_fps_per_stream'] for r in results]
    cpu_avg = [r['avg_cpu'] for r in results]
    cpu_max = [r['max_cpu'] for r in results]
    memory_avg = [r['avg_memory'] for r in results]
    
    # Create comprehensive chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total FPS vs Streams
    ax1.plot(streams, total_fps, 'bo-', linewidth=3, markersize=8, label='Total FPS')
    ax1.set_xlabel('Number of Streams', fontsize=12)
    ax1.set_ylabel('Total FPS', fontsize=12)
    ax1.set_title('Total FPS vs Number of Streams\n(Intel DL Streamer CPU Performance)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add value annotations
    for x, y in zip(streams, total_fps):
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    # Average FPS per Stream
    ax2.plot(streams, avg_fps, 'go-', linewidth=3, markersize=8, label='Avg FPS per Stream')
    ax2.set_xlabel('Number of Streams', fontsize=12)
    ax2.set_ylabel('FPS per Stream', fontsize=12)
    ax2.set_title('Average FPS per Stream\n(Scalability Analysis)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    for x, y in zip(streams, avg_fps):
        ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    # CPU Usage
    ax3.plot(streams, cpu_avg, 'ro-', linewidth=3, markersize=8, label='Average CPU')
    ax3.plot(streams, cpu_max, 'r--o', linewidth=2, markersize=6, label='Peak CPU')
    ax3.set_xlabel('Number of Streams', fontsize=12)
    ax3.set_ylabel('CPU Usage (%)', fontsize=12)
    ax3.set_title('CPU Usage vs Number of Streams\n(System Resource Utilization)', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Memory Usage
    ax4.plot(streams, memory_avg, 'mo-', linewidth=3, markersize=8, label='Memory Usage')
    ax4.set_xlabel('Number of Streams', fontsize=12)
    ax4.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax4.set_title('Memory Usage vs Number of Streams\n(Memory Consumption Analysis)', 
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save chart
    chart_file = RESULTS_DIR / f'intel_dlstreamer_performance_{timestamp}.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"[CHART] Performance chart saved: {chart_file}")
    
    plt.show()
    return chart_file

def generate_performance_report(results, timestamp):
    """Generate comprehensive performance report"""
    if not results:
        return None
    
    report_file = RESULTS_DIR / f'intel_dlstreamer_report_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("INTEL DL STREAMER CPU PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Duration per Configuration: {TEST_DURATION} seconds\n")
        f.write(f"Models Used: Person Detection + Gender/Vehicle Classification\n")
        f.write(f"Hardware: CPU-based inference (Intel OpenVINO)\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 20 + "\n")
        
        for result in results:
            f.write(f"\n{result['num_streams']} Streams Configuration:\n")
            f.write(f"  Total FPS: {result['total_fps']:.2f}\n")
            f.write(f"  Average FPS per Stream: {result['avg_fps_per_stream']:.2f}\n")
            f.write(f"  Successful Streams: {result['successful_streams']}/{result['num_streams']}\n")
            f.write(f"  CPU Usage: {result['avg_cpu']:.1f}% (avg), {result['max_cpu']:.1f}% (peak)\n")
            f.write(f"  Memory Usage: {result['avg_memory']:.1f}MB (avg), {result['max_memory']:.1f}MB (peak)\n")
            f.write(f"  Performance Status: {'✓ SUCCESS' if result['success'] else '✗ FAILED'}\n")
        
        # Analysis
        successful_results = [r for r in results if r['success']]
        if successful_results:
            max_streams_result = max(successful_results, key=lambda x: x['num_streams'])
            best_fps_result = max(successful_results, key=lambda x: x['total_fps'])
            
            f.write(f"\nPERFORMANCE ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Maximum Streams Supported: {max_streams_result['num_streams']}\n")
            f.write(f"Best Total FPS: {best_fps_result['total_fps']:.2f} ({best_fps_result['num_streams']} streams)\n")
            f.write(f"Optimal Configuration: {best_fps_result['num_streams']} streams @ {best_fps_result['avg_fps_per_stream']:.2f} FPS per stream\n")
            
            # Bottleneck analysis
            if max_streams_result['max_cpu'] > 90:
                f.write(f"Bottleneck: CPU (reached {max_streams_result['max_cpu']:.1f}%)\n")
            elif max_streams_result['max_memory'] > 8000:  # 8GB
                f.write(f"Bottleneck: Memory (reached {max_streams_result['max_memory']:.1f}MB)\n")
            else:
                f.write(f"Bottleneck: Not clearly identified (may be I/O or model processing)\n")
        
        f.write(f"\nRECOMMendATIONS:\n")
        f.write("-" * 15 + "\n")
        f.write("• For production deployment, use the optimal configuration identified above\n")
        f.write("• Monitor CPU usage to avoid saturation (keep below 80%)\n")
        f.write("• Consider GPU acceleration for higher stream counts\n")
        f.write("• Optimize model precision (FP16/INT8) for better performance\n")
    
    print(f"[REPORT] Performance report generated: {report_file}")
    return report_file

# --------------------------
# MAIN EXECUTION
# --------------------------

def main():
    """Main execution function"""
    print("INTEL DL STREAMER CPU PERFORMANCE TESTING SUITE")
    print("=" * 50)
    print("Senior Intel Software Engineer Implementation")
    print("=" * 50)
    
    # Validate environment
    if not validate_files():
        print("[ABORT] Required files missing")
        return
    
    if not check_gstreamer():
        print("[ABORT] GStreamer not properly configured")
        return
    
    print(f"\n[CONFIG] Testing stream counts: {STREAM_COUNTS_TO_TEST}")
    print(f"[CONFIG] Test duration per configuration: {TEST_DURATION} seconds")
    print(f"[CONFIG] Show pipeline output: {SHOW_PIPELINE_OUTPUT}")
    
    # Generate timestamp for this test session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    try:
        # Execute performance tests
        for stream_count in STREAM_COUNTS_TO_TEST:
            result = run_performance_test(stream_count, timestamp, SHOW_PIPELINE_OUTPUT)
            
            if result:
                results.append(result)
                
                # Stop testing if performance becomes unacceptable
                if result['avg_fps_per_stream'] < 3.0:
                    print(f"[STOP] Performance degraded below threshold at {stream_count} streams")
                    break
            else:
                print(f"[SKIP] Test failed for {stream_count} streams")
            
            # Brief pause between tests
            if stream_count != STREAM_COUNTS_TO_TEST[-1]:
                print("[PAUSE] 5 seconds before next test...")
                time.sleep(5)
        
        # Generate outputs
        if results:
            print(f"\n[PROCESSING] Generating performance analysis...")
            
            # Create performance charts
            create_performance_charts(results, timestamp)
            
            # Generate detailed report
            generate_performance_report(results, timestamp)
            
            # Save raw data
            df_data = []
            for result in results:
                df_data.append({
                    'Streams': result['num_streams'],
                    'Total_FPS': result['total_fps'],
                    'Avg_FPS_per_Stream': result['avg_fps_per_stream'],
                    'CPU_Avg_%': result['avg_cpu'],
                    'CPU_Max_%': result['max_cpu'],
                    'Memory_Avg_MB': result['avg_memory'],
                    'Memory_Max_MB': result['max_memory'],
                    'Success': result['success']
                })
            
            df = pd.DataFrame(df_data)
            csv_file = RESULTS_DIR / f'performance_data_{timestamp}.csv'
            df.to_csv(csv_file, index=False)
            print(f"[DATA] Raw data saved: {csv_file}")
            
            # Final summary
            print(f"\n{'='*60}")
            print("INTEL DL STREAMER PERFORMANCE TEST COMPLETE")
            print("="*60)
            
            successful_tests = [r for r in results if r['success']]
            if successful_tests:
                best_config = max(successful_tests, key=lambda x: x['total_fps'])
                print(f"OPTIMAL CONFIGURATION:")
                print(f"  Streams: {best_config['num_streams']}")
                print(f"  Total FPS: {best_config['total_fps']:.2f}")
                print(f"  FPS per Stream: {best_config['avg_fps_per_stream']:.2f}")
                print(f"  CPU Usage: {best_config['avg_cpu']:.1f}%")
                print(f"  Memory Usage: {best_config['avg_memory']:.1f}MB")
            
            print(f"\nAll results and charts saved in: {RESULTS_DIR}")
            
        else:
            print("[ERROR] No successful tests completed")
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test stopped by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    
    print("\n[COMPLETE] Intel DL Streamer performance testing finished")

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    print("\n[CLEANUP] Shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run main program
    main()