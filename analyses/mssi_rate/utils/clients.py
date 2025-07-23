
import dask
from dask.distributed import Client
import os
import time
import logging

nersc_instructions = """
1. Start a compute node: salloc -N 2 -n 64 -t 30 -C cpu
2. On the compute node, start dask: ./launch_workers_nersc.sh.
"""

def get_client(client_type: str="dirac", scaling: int=5, scheduler_options: dict={"port": 8786}) -> Client:

    if client_type not in ("DIRAC", "NERSC", "LOCAL"):
        raise ValueError(f"Unsupported client type: {client_type}. Supported types are 'DIRAC', 'NERSC', 'LOCAL'")

    if client_type == "DIRAC":
        logging.info("Starting DIRAC Cluster")
        from dask_dirac import DiracCluster
        cluster = DiracCluster(scheduler_options=scheduler_options)
        cluster.scale(jobs=scaling)
        client = Client(cluster)

    elif client_type == "NERSC":
        logging.info("Starting NERSC Cluster")
        scheduler_file = os.path.join(os.environ["SCRATCH"], "scheduler_file.json")
        #scheduler_file = "scheduler_file.json"
        # check if file exists
        if not os.path.isfile(scheduler_file):
            logging.warning(f"Scheduler file {scheduler_file} does not exist. Instructions start NERSC jobs are...\n{nersc_instructions}\nWaiting until file is created...")

            # wait until file is created
            while not os.path.isfile(scheduler_file):
                time.sleep(120)
            raise FileNotFoundError(f"Scheduler file {scheduler_file} does not exist.")

        dask.config.config["distributed"]["dashboard"]["link"] = "{JUPYTERHUB_SERVICE_PREFIX}proxy/{host}:{port}/status" 
        client = Client(scheduler_file=scheduler_file)

    elif client_type == "LOCAL":
        logging.info("Starting DIRAC Cluster")
        from dask.distributed import LocalCluster
        cluster = LocalCluster()
        cluster.scale(jobs=scaling)
        client = Client(cluster)
    
    return client