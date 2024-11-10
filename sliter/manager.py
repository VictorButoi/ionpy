# scheduler_server.py
import time
import submitit
import threading
from typing import List, Optional, Any
from flask import Flask, request, jsonify
from ionpy.sliter.run_jobs import run_job , run_exp
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlShutdown, nvmlDeviceGetHandleByIndex

app = Flask(__name__)


class SliteGPUManager:
    def __init__(self):
        nvmlInit()
        self.num_gpus = nvmlDeviceGetCount()
        self.gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]
        self.lock = threading.Lock()
        self.gpu_status = [True] * self.num_gpus  # True means free

    def get_free_gpu(self):
        with self.lock:
            for i, status in enumerate(self.gpu_status):
                if status:
                    self.gpu_status[i] = False
                    return i
        return None

    def release_gpu(self, job_gpu):
        with self.lock:
            if (job_gpu is not None) and (0 <= job_gpu < self.num_gpus):
                self.gpu_status[job_gpu] = True

    def shutdown(self):
        nvmlShutdown()


class SliteJobScheduler:
    def __init__(self):
        self.gpu_manager = SliteGPUManager()
        self.default_executer_params = {
            "timeout_min": 60*24*7,  # 7 days
            "slurm_additional_parameters": {
                "gres": "gpu:1"  # Placeholder, not used in local executor
            }
        }
        self.lock = threading.Lock()
        self.running_jobs = {}
        self.completed_jobs = {}

    def submit_job(
        self, 
        cfg,
        job_func: Optional[Any] = None,
        exp_class: Optional[Any] = None
    ):
        executor = submitit.LocalExecutor(f'{cfg["log"]["root"]}/{cfg["log"]["uuid"]}/submitit')
        executor.update_parameters(**self.default_executer_params)

        job_gpu = self.gpu_manager.get_free_gpu()
        if job_gpu is None:
            return None  # No GPU available

        submit_kwargs = {
            "config": cfg,
            "available_gpus": job_gpu
        }
        # Submit the job to submitit
        if exp_class is not None:
            job = executor.submit(
                run_exp, 
                exp_class=exp_class,
                **submit_kwargs
            )
        else:
            job = executor.submit(
                run_job, 
                job_func=job_func,
                **submit_kwargs
            )
        # Get out the job ID
        job_id = job.job_id

        with self.lock:
            self.running_jobs[job_id] = {
                "job": job,
                "job_gpu": job_gpu,
                "status": "running"
            }

        # Start a thread to monitor job completion
        threading.Thread(target=self.monitor_job, args=(job_id, job), daemon=True).start()

        return job_id, job_gpu

    def monitor_job(self, job_id, job):
        try:
            job.result()  # This will block until the job finishes
            status = "completed"
        except Exception as e:
            status = "failed"
            with self.lock:
                self.running_jobs[job_id]["error"] = str(e)

        with self.lock:
            job_info = self.running_jobs.pop(job_id, None)
            self.completed_jobs[job_id] = {
                "job_gpu": job_info.get("job_gpu", ""),
                "status": status
            }

        # Release the GPU
        self.gpu_manager.release_gpu(self.completed_jobs[job_id]["job_gpu"])

    def list_jobs(self):
        with self.lock:
            running = {
                job_id: {
                    "job_gpu": info["job_gpu"],
                    "status": info["status"]
                } for job_id, info in self.running_jobs.items()
            }
            completed = self.completed_jobs.copy()
        return {"running": running, "completed": completed}

    def shutdown(self):
        self.gpu_manager.shutdown()
        # Optionally, cancel all running jobs
        with self.lock:
            for job_id, info in self.running_jobs.items():
                info["job"].cancel()

# Initialize the scheduler
scheduler = SliteJobScheduler()

@app.route('/submit', methods=['POST'])
def submit_job():
    data = request.get_json()
    if not data or 'config' not in data:
        return jsonify({'error': 'No config provided.'}), 400
    # Submit the job
    job_id, job_gpu = scheduler.submit_job(
        cfg=data['config'],
        job_func=data.get('job_func', None),
        exp_class=data.get('exp_class', None)
    )
    if job_id is not None:
        return jsonify({'job_id': job_id, 'job_gpu': job_gpu}), 200
    else:
        return jsonify({'error': 'No GPU available. Try again later.'}), 503

@app.route('/jobs', methods=['GET'])
def get_jobs():
    jobs = scheduler.list_jobs()
    return jsonify(jobs), 200

@app.route('/shutdown', methods=['POST'])
def shutdown_server():
    # Directly call scheduler shutdown and server shutdown within the request context
    scheduler.shutdown()  # Ensure that this line only runs if scheduler is defined
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    # Run the shutdown function in a separate thread if needed
    threading.Thread(target=func).start()
    
    return jsonify({"message": "Server is shutting down..."}), 200

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', debug=True, port=5000)
