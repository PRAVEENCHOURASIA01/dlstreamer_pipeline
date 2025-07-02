Intel DL Streamer Multi-Stream CPU Performance Test

This project implements a **CPU-based video analytics pipeline** using **Intel DL Streamer** to test detection, classification, and system scalability on Intel hardware.

Objective

With the growing use of AI-powered edge cameras for city and transport surveillance, it’s important to understand how many video streams an Intel CPU can handle in real-time.  
This project builds pipelines for **person and vehicle detection and classification** using OpenVINO models and measures performance for multiple parallel streams.

Pipeline Details

- **Base Framework:** Intel DL Streamer, GStreamer
- **Pipeline:**  
  - `gvadetect` — detect persons and vehicles  
  - `gvaclassify` — classify person attributes (gender) and vehicle attributes  
  - `gvawatermark` — overlay detection results  
  - `gvafpscounter` — measure per-stream FPS  
  - Runs fully on **CPU** (`device=CPU`)

- **Models Used:**  
  - `person-detection-retail-0013`
  - `person-attributes-recognition-crossroad-0230`
  - `vehicle-attributes-recognition-barrier-0039`

Project Structure

- **Python Orchestration:** Automates running single-stream and multi-stream pipelines.
- **System Monitoring:** Tracks CPU usage, memory usage, and FPS logs.
- **Results:** Generates logs, performance charts, and summary reports.

Test Scope

✅ Build & run single-stream pipelines  
✅ Run multiple streams (1, 2, 4, 6, 8) in parallel  
✅ Measure:
- Total and per-stream FPS on **CPU**
- Average and peak CPU usage
- Memory usage
✅ Visualize results with charts  
✅ Identify bottlenecks and maximum stream capacity for CPU


Expected Outcome

Maximum number of streams supported by the CPU
Total and per-stream FPS for each configuration
Resource usage charts (CPU, Memory)


Focus
Note: This project specifically benchmarks CPU performance only — no GPU or VPU results are included.


Author
Praveen Chourasia

Feel free to clone, modify, and extend this test setup for your own hardware!
