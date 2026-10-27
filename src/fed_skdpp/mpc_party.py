"""One MPC participant. Reads only its own private fixture and public inputs."""

import argparse
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
from mpyc.runtime import mpc

from .mpc_euclidean import MPCConfig
from .mpc_secure import secure_sls


def main():
    # MPyC consumes its -M/-I/-T/-B arguments when imported.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-input", type=Path, required=True)
    parser.add_argument("--private-input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    public = json.loads(args.public_input.read_text())
    local = json.loads(args.private_input.read_text())
    config = MPCConfig(**public["config"])
    coordinates = np.asarray(local["points"], dtype=float)
    if (coordinates.shape != (config.private_count//3, config.dimension)
            or not np.isfinite(coordinates).all() or (coordinates < 0).any() or (coordinates > 1).any()):
        raise ValueError("Invalid local point input")
    if args.result.exists():
        raise FileExistsError("Refusing to overwrite a party result")

    async def run():
        started = perf_counter()
        await mpc.start()
        def progress(step):
            if mpc.pid == 0:
                print(f"MPC_PROGRESS {step}/{config.max_proposals}", flush=True)
        result = await secure_sls(config, public, local, progress=progress)
        await mpc.barrier()
        result["party"] = mpc.pid
        result["elapsed_seconds"] = perf_counter()-started
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result["sent_bytes"] = sum(peer.protocol.nbytes_sent for peer in mpc.parties if peer.pid != mpc.pid)
        result["bytes_scope"] = "MPyC payloads and 12-byte message headers; excludes handshake, TCP/IP and TLS"
        # All parties see the same masked diagnostics; save one copy only.
        if mpc.pid:
            result.pop("masked_diagnostics")
        with args.result.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        await mpc.shutdown()
    mpc.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
