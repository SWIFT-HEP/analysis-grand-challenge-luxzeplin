LOCAL_BASE="/shared/scratch/ak18773/lz/mssi"
NERSC_BASE="/global/cfs/cdirs/lz/sim/fast/background/bgSR3/BACCARAT-6.3.6_LZLAMA-3.5.6+18baad24_PROD-0"
DIRAC_HOST="root://gfe02.grid.hep.ph.ic.ac.uk"
DIRAC_BASE="/pnfs/hep.ph.ic.ac.uk/data/lz/lz/sim/fast/background/bgSR3/BACCARAT-6.3.6_LZLAMA-3.5.6+18baad24_PROD-0"

import glob

def get_file_list(cluster_type: str, file_fraction: float) -> list[str]:

    if cluster_type not in ("LOCAL", "NERSC", "DIRAC"):
        raise ValueError(f"Unsupported cluster type: {cluster_type}. Supported types are 'LOCAL', 'NERSC', 'DIRAC'")

    if cluster_type == "LOCAL":
        file_list = glob.glob(f"{LOCAL_BASE}/*/*.root")[:int(len(file_list) * file_fraction)]
    elif cluster_type == "NERSC":
        file_list = glob.glob(f"{NERSC_BASE}/*/*.root")[:int(len(file_list) * file_fraction)]
    elif cluster_type == "DIRAC":
        file_list =  get_xrootd_file_list(DIRAC_BASE, DIRAC_HOST)[:int(len(file_list) * file_fraction)]

    return file_list


def get_xrootd_file_list(base_path, host):
    from XRootD import client
    fs = client.FileSystem(host)
    status, listing = fs.dirlist(base_path)
    if not status.ok:
        raise RuntimeError(f"Failed to list directory: {status.message}")
    
    return [
        f"{host}{base_path}/{entry.name}"
        for entry in listing
        if entry.flags & client.flags.DirListFlags.kXrdIsFile
    ]