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

    def release_gpu(self, gpu_id):
        with self.lock:
            if 0 <= gpu_id < self.num_gpus:
                self.gpu_status[gpu_id] = True

    def shutdown(self):
        nvmlShutdown()


class SliteJobScheduler:
    def __init__(self):
        self.gpu_manager = SliteGPUManager()
        self.executor = submitit.LocalExecutor("/storage/vbutoi/scratch/Submitit")
        self.executor.update_parameters(
            timeout_min=60*24*7,  # 3 days
            slurm_additional_parameters={
                "gres": "gpu:1"  # Placeholder, not used in local executor
            })
        self.lock = threading.Lock()
        self.running_jobs = {}
        self.completed_jobs = {}
        self.job_id_counter = 1

    def submit_exp(self, command):
        gpu_id = self.gpu_manager.get_free_gpu()
        if gpu_id is None:
            return None  # No GPU available

        job_id = self.job_id_counter
        self.job_id_counter += 1

        # Submit the job to submitit
        job = self.executor.submit(run_job, command, gpu_id)

        with self.lock:
            self.running_jobs[job_id] = {
                "job": job,
                "command": command,
                "gpu_id": gpu_id,
                "status": "running"
            }

        # Start a thread to monitor job completion
        threading.Thread(target=self.monitor_job, args=(job_id, job), daemon=True).start()

        return job_id

    def submit_jobs(self, jobjects):
        for job_config in jobjects['config_list']:
            job, gpu_id, _ = self.submit_job(
                job_config,
                job_func=jobjects['job_func'],
                exp_class=jobjects['exp_class']
            )
            print(f"--> Launched job-id: {job.job_id} on gpu: {gpu_id}.")
            time.sleep(jobjects['submission_delay'])

    def submit_job(
        self, 
        cfg,
        job_func: Optional[Any] = None,
        exp_class: Optional[Any] = None
    ):

        gpu_id = self.gpu_manager.get_free_gpu()
        if gpu_id is None:
            return None  # No GPU available

        job_id = self.job_id_counter
        self.job_id_counter += 1

        submit_kwargs = {
            "config": cfg,
            "available_gpus": gpu_id
        }
        # Submit the job to submitit
        if exp_class is not None:
            job = self.executor.submit(
                run_exp, 
                exp_class=exp_class,
                **submit_kwargs
            )
        else:
            job = self.executor.submit(
                run_job, 
                job_func=job_func,
                **submit_kwargs
            )

        with self.lock:
            self.running_jobs[job_id] = {
                "job": job,
                "gpu_id": gpu_id,
                "status": "running"
            }

        # Start a thread to monitor job completion
        threading.Thread(target=self.monitor_job, args=(job_id, job), daemon=True).start()

        return job, gpu_id, job_id

    def monitor_job(self, job_id, job):
        try:
            job.result()  # This will block until the job finishes
            status = "completed"
        except Exception as e:
            status = "failed"
            with self.lock:
                self.running_jobs[job_id]["error"] = str(e)

        with self.lock:
            self.running_jobs.pop(job_id, None)
            self.completed_jobs[job_id] = {
                "command": self.completed_jobs.get(job_id, {}).get("command", ""),
                "gpu_id": self.completed_jobs.get(job_id, {}).get("gpu_id", ""),
                "status": status
            }

        # Release the GPU
        self.gpu_manager.release_gpu(self.completed_jobs[job_id]["gpu_id"])

    def list_jobs(self):
        with self.lock:
            running = {
                job_id: {
                    "command": info["command"],
                    "gpu_id": info["gpu_id"],
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
def submit_jobs():
    data = request.get_json()
    if not data or 'config_list' not in data:
        return jsonify({'error': 'No configs provided.'}), 400
    job_id = scheduler.submit_jobs(data)
    if job_id is not None:
        return jsonify({'job_id': job_id}), 200
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
