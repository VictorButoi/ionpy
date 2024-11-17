# scheduler_server.py
import time
import logging
import traceback
logging.basicConfig(level=logging.DEBUG)
import submitit
import threading
from typing import List, Optional, Any
from flask import Flask, request, jsonify
from ionpy.sliter.run_jobs import run_job , run_exp
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlShutdown, nvmlDeviceGetHandleByIndex
import queue  # Import queue for the job queue

app = Flask(__name__)


class SliteGPUManager:
    def __init__(self):
        nvmlInit()
        self.num_gpus = nvmlDeviceGetCount()
        self.gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]
        self.gpu_status = [True] * self.num_gpus  # True means free

    def get_free_gpu(self):
        for i, status in enumerate(self.gpu_status):
            if status:
                self.gpu_status[i] = False
                return i
        return None

    def release_gpu(self, job_gpu):
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
        self.job_queue = queue.Queue()
        self.job_counter = 0
        self.all_jobs = {}  # Mapping from job_id to job info

    def submit_job(
        self, 
        cfg,
        job_func: Optional[Any] = None,
        exp_class: Optional[Any] = None
    ):
        with self.lock:
            job_id = str(self.job_counter)
            self.job_counter += 1
            job_info = {
                "job_id": job_id,
                "cfg": cfg,
                "job_func": job_func,
                "exp_class": exp_class,
                "status": None,  # To be set below
                "job_gpu": None,
                "submitit_root": f'{cfg["log"]["root"]}/{cfg["log"]["uuid"]}/submitit'
            }
            self.all_jobs[job_id] = job_info

            job_gpu = self.gpu_manager.get_free_gpu()
            if job_gpu is None:
                # No GPU available, queue the job
                job_info["status"] = "queued"
                self.job_queue.put(job_id)
                return job_id  # Return the job_id to the user
            else:
                # GPU available, submit the job
                job_info["job_gpu"] = job_gpu
                return self._submit_to_executor(job_id, job_info)

    def _submit_to_executor(self, job_id, job_info):
        cfg = job_info["cfg"]
        job_func = job_info["job_func"]
        exp_class = job_info["exp_class"]
        job_gpu = job_info["job_gpu"]

        executor = submitit.LocalExecutor(job_info['submitit_root'])
        executor.update_parameters(**self.default_executer_params)

        submit_kwargs = {
            "config": cfg,
            "available_gpus": job_gpu
        }

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

        job_info["status"] = "running"
        # Place the job in the running_jobs dict.
        self.running_jobs[job_id] = job_info

        threading.Thread(target=self.monitor_job, args=(job_id, job), daemon=True).start()

        return job_id

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
            job_info["status"] = status
            self.completed_jobs[job_id] = job_info

            # Release the GPU while holding self.lock
            self.gpu_manager.release_gpu(job_info["job_gpu"])

            # Check if there are queued jobs
            if not self.job_queue.empty():
                next_job_id = self.job_queue.get()
                next_job_info = self.all_jobs[next_job_id]
                # Try to get a GPU (should be available now)
                job_gpu = self.gpu_manager.get_free_gpu()
                if job_gpu is not None:
                    next_job_info["job_gpu"] = job_gpu
                    self._submit_to_executor(next_job_id, next_job_info)
                else:
                    # This shouldn't happen, but just in case
                    next_job_info["status"] = "queued"
                    self.job_queue.put(next_job_id)

    def list_jobs(self):
        with self.lock:
            jobs = {}
            # Include all jobs with their statuses
            for job_id, job_info in self.all_jobs.items():
                jobs[job_id] = {
                    "status": job_info["status"],
                    "job_gpu": job_info["job_gpu"],
                }
        return jobs

    def shutdown(self):
        self.gpu_manager.shutdown()
        # Optionally, cancel all running jobs and clear the queue
        with self.lock:
            for job_id, info in self.running_jobs.items():
                info["status"] = "cancelled"
            self.running_jobs.clear()
            # Clear queued jobs
            while not self.job_queue.empty():
                job_id = self.job_queue.get()
                self.all_jobs[job_id]["status"] = "cancelled"

# Initialize the scheduler
scheduler = SliteJobScheduler()

@app.route('/submit', methods=['POST'])
def submit_job_endpoint():
    try: 
        data = request.get_json()
        if not data or 'config' not in data:
            return jsonify({'error': 'No config provided.'}), 400
        # Submit the job
        job_id = scheduler.submit_job(
            cfg=data['config'],
            job_func=data.get('job_func', None),
            exp_class=data.get('exp_class', None)
        )
        job_info = scheduler.all_jobs[job_id]
    except Exception as e:
        logging.error(f"Failed to submit job {job_id}: {e}")
        logging.error(traceback.format_exc())
        job_info = {
            "job_id": None,
            "status": "failed",
        }     
    return jsonify({'job_id': job_id, 'status': job_info["status"], 'job_gpu': job_info.get('job_gpu')}), 200

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

@app.route('/get_job', methods=['GET'])
def get_job():
    try: 
        data = request.get_json()
        if not data or 'job_id' not in data:
            return jsonify({'error': 'No job-id provided.'}), 400
        # Submit the job
        job_info = scheduler.all_jobs[data['job_id']]
    except Exception as e:
        logging.error(f"Failed to gather job {data['job_id']}: {e}")
        logging.error(traceback.format_exc())
        job_info = {
            "job_id": None,
            "status": "failed",
        }     
    return jsonify(job_info), 200

@app.route('/jobs', methods=['GET'])
def get_jobs():
    jobs = scheduler.list_jobs()
    return jsonify(jobs), 200

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)
