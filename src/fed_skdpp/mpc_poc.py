"""Trusted synthetic evaluator/launcher; NOT an MPC server or participant.

Owns the test fixtures and validation pads, launches three independent party
processes, and compares protected arithmetic with a float64 oracle. Public
run artifacts never contain private coordinates, seeds, coin shares, or pads.
Private reproducibility fixtures and detailed audit traces stay in a separate
0700 temporary directory. A single-host simulation does not protect against
the host administrator or this intentionally trusted evaluator.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import secrets
import socket
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import yaml

from .cli import allocate_run_directory
from .mpc_euclidean import (MPCConfig, WIDE_FIELD_PRIME, combined_coins, diagnostic_prime, primary_prime,
                            decode_diagnostics, generate_fixture, load_mpc_config,
                            plaintext_reference)


class PrivateRandom:
    """OS-entropy fixture RNG; retained inputs provide reproducibility."""

    def uniform(self, low, high, shape):
        values = [low+(high-low)*(secrets.randbits(53)/2**53)
                  for _ in range(math.prod(shape))]
        return np.array(values).reshape(shape)

    def permutation(self, n):
        values = list(range(n))
        random.SystemRandom().shuffle(values)
        return np.array(values, dtype=np.int64)

    def integers(self, low, high, shape):
        bounds = np.broadcast_to(high, shape).ravel()
        return np.array([low+secrets.randbelow(int(bound)-low) for bound in bounds],
                        dtype=np.int64).reshape(shape)


def _write_json(path, value, private=False):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _ports():
    # Reserve a contiguous block while checking; a small launch-time race is
    # unavoidable without passing already-bound sockets into the MPC runtime.
    for _ in range(100):
        base = 20000+secrets.randbelow(30000)
        sockets = []
        try:
            for port in range(base, base+3):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", port))
            return base
        except OSError as exc:
            if exc.errno in (1, 13):
                raise
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("Could not reserve three local MPC ports")


def launch_parties(config, public_path, private_paths, result_directory, progress=None):
    """Launch real -M3/-T1 processes; only each owner's path is passed to it."""
    base = _ports()
    processes, logs = [], []
    started = perf_counter()
    try:
        for owner in range(3):
            log = (result_directory/f"party_{owner}.log").open("x", encoding="utf-8")
            logs.append(log)
            command = [sys.executable, "-m", "fed_skdpp.mpc_party", "-M", "3",
                       "-I", str(owner), "-T", "1", "-B", str(base),
                       "-K", str(config.statistical_security_bits), "--log-level", "warning",
                       "--public-input", str(public_path),
                       "--private-input", str(private_paths[owner]),
                       "--result", str(result_directory/f"party_{owner}.json")]
            environment = os.environ.copy()
            environment["OPENBLAS_NUM_THREADS"] = "1"
            environment["OMP_NUM_THREADS"] = "1"
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               env=environment))
        last_progress = None
        while any(process.poll() is None for process in processes):
            failed = [p for p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"MPC party failed; inspect party logs in {result_directory}")
            if perf_counter()-started > config.timeout_seconds:
                raise TimeoutError("MPC trial exceeded its configured timeout")
            if progress is not None:
                lines = (result_directory/"party_0.log").read_text().splitlines()
                lines = [line for line in lines if line.startswith("MPC_PROGRESS ")]
                if lines and lines[-1] != last_progress:
                    last_progress = lines[-1]
                    progress(last_progress)
            # Waiting on a child yields promptly; this is not a busy polling loop.
            try:
                next(p for p in processes if p.poll() is None).wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        if any(process.returncode for process in processes):
            raise RuntimeError(f"MPC party failed; inspect party logs in {result_directory}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    return [json.loads((result_directory/f"party_{i}.json").read_text()) for i in range(3)], perf_counter()-started


def validate_trial(reference, decoded, parties, config):
    """Compare same data, proposal coins, slot ordering, and stopping rule."""
    logical = int(round(decoded["iterations"]))
    steps = decoded["steps"][:logical]
    reference_steps = reference["steps"]
    errors = [abs(decoded[key]-reference[key]) for key in
              ("full_logdet", "context_logdet", "initial_logdet", "final_logdet")]
    ratio_errors, decisions = [], []
    for secure, plain in zip(steps, reference_steps):
        errors.extend([abs(secure["old_logdet"]-plain["old_logdet"]),
                       abs(secure["proposed_logdet"]-plain["proposed_logdet"]),
                       abs(secure["delta"]-plain["delta"])])
        if secure["ratio_valid"] > .5:
            ratio_errors.append(abs(secure["ratio"]-plain["ratio"])/plain["ratio"])
        decisions.append(bool(round(secure["accepted"])) == plain["accepted"])
    selections = [party["selected"] for party in parties]
    selected = selections[0]
    cardinality_ok = len(selected) == config.selection_size and len(set(selected)) == len(selected)
    eligible = all(0 <= item < config.public_count for item in selected)
    max_log_error = max(errors)
    max_ratio_error = max(ratio_errors, default=float("inf"))
    full_det = math.exp(reference["full_logdet"])
    determinant_relative_error = abs(decoded["full_determinant"]-full_det)/full_det
    valid = (bool(round(decoded["valid"])) and all(p["numerically_valid"] for p in parties)
             and logical == reference["iterations"] and len(steps) == len(reference_steps)
             and selected == reference["selected"] and all(s == selected for s in selections)
             and cardinality_ok and eligible and all(decisions)
             and max_log_error <= config.logdet_absolute_tolerance
             and len(ratio_errors) == logical and max_ratio_error <= config.ratio_relative_tolerance
             and determinant_relative_error <= config.ratio_relative_tolerance)
    last = steps[-1] if steps else {}
    return {
        "validation_passed": bool(valid), "selected_public_ids": selected,
        "full_logdet_plain": reference["full_logdet"], "full_logdet_mpc": decoded["full_logdet"],
        "full_determinant_plain": full_det, "full_determinant_mpc": decoded["full_determinant"],
        "full_determinant_relative_error": determinant_relative_error,
        "context_logdet_plain": reference["context_logdet"], "context_logdet_mpc": decoded["context_logdet"],
        "initial_joint_logdet": decoded["initial_logdet"], "final_joint_logdet": decoded["final_logdet"],
        "max_logdet_absolute_error": max_log_error, "max_swap_ratio_relative_error": max_ratio_error,
        "swap_decisions_match": all(decisions), "final_selection_matches": selected == reference["selected"],
        "iterations": logical, "accepted_swaps": int(round(decoded["accepted_swaps"])),
        "stalled": int(round(decoded["stalled"])), "patience_reached": bool(round(decoded["patience_reached"])),
        "stop_reason": "patience" if decoded["patience_reached"] > .5 else "max_proposals",
        "first_swap_ratio": steps[0]["ratio"] if steps else None,
        "last_swap_ratio": last.get("ratio"),
        "first_swap_log_ratio": steps[0]["delta"] if steps else None,
        "joint_improvement": decoded["final_logdet"]-decoded["initial_logdet"],
        "kernel_condition_number": reference["kernel_condition_number"],
        "overlap_violations": 0 if eligible else None,
        "sent_bytes": sum(p["sent_bytes"] for p in parties),
        "max_party_peak_rss_bytes": max(p["peak_rss_bytes"] for p in parties),
        "explicit_barriers_per_party": config.max_proposals+1,
        "communication_rounds": None,
        "communication_rounds_note": "asynchronous MPC; actual network rounds not instrumented",
    }


def run_poc(config, output_directory, progress=print):
    run_directory = allocate_run_directory(output_directory, "mpc_euclidean_sls")
    fixture_directory = Path(tempfile.mkdtemp(prefix="fed-skdpp-private-mpc-"))
    os.chmod(fixture_directory, 0o700)
    progress(f"Private validation fixtures (sensitive, kept outside results): {fixture_directory}")
    with (run_directory/"config.yml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
    metrics = []
    started = perf_counter()
    for trial in range(config.repetitions):
        trial_dir = run_directory/f"trial_{trial+1}"
        trial_dir.mkdir()
        private_dir = fixture_directory/f"trial_{trial+1}"
        private_dir.mkdir(mode=0o700)
        points, owners, public, initial = generate_fixture(
            config, PrivateRandom(), np.random.default_rng(np.random.SeedSequence([config.public_seed, trial])))
        reference = plaintext_reference(points, initial, combined_coins(owners, config), config)
        public_input = {"config": asdict(config), "points": public.tolist(), "initial_selection": initial.tolist()}
        public_path = trial_dir/"public_input.json"
        _write_json(public_path, public_input)
        private_paths = []
        for owner, local in enumerate(owners):
            local["pads"] = [str(secrets.randbelow(diagnostic_prime(i, config))) for i in range(config.diagnostics_count)]
            path = private_dir/f"party_{owner}.json"
            _write_json(path, local, private=True)
            private_paths.append(path)
        _write_json(private_dir/"audit.json", {"reference": reference}, private=True)
        progress(f"Trial {trial+1}/{config.repetitions}: running three-party MPC, {config.max_proposals} padded proposals...")
        parties, elapsed = launch_parties(config, public_path, private_paths, trial_dir,
                                          progress=lambda text: progress(f"Trial {trial+1}: {text}"))
        decoded = decode_diagnostics(parties[0]["masked_diagnostics"], owners, config)
        _write_json(private_dir/"decoded_audit.json", decoded, private=True)
        row = validate_trial(reference, decoded, parties, config)
        row.update(trial=trial+1, elapsed_seconds=elapsed)
        metrics.append(row)
        progress(f"Trial {trial+1}: passed={row['validation_passed']}, logdet error={row['max_logdet_absolute_error']:.3g}, "
                 f"ratio error={row['max_swap_ratio_relative_error']:.3g}, logical proposals={row['iterations']}, "
                 f"stop={row['stop_reason']}, runtime={elapsed:.2f}s")
    source_digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("mpc_*.py")):
        source_digest.update(path.name.encode())
        source_digest.update(path.read_bytes())
    source_digest.update((Path(__file__).parent/"kernels.py").read_bytes())
    result = {
        "config": asdict(config), "trials": metrics,
        "all_passed": all(row["validation_passed"] for row in metrics),
        "elapsed_seconds": perf_counter()-started,
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_digest.hexdigest(),
            "versions": {name: version(name) for name in ("numpy", "mpyc", "gmpy2", "PyYAML")},
            "primary_field_prime": str(primary_prime(config)),
            "determinant_field_prime": str(WIDE_FIELD_PRIME),
            "execution": "three real MPC processes on one host; trusted synthetic evaluator outside protocol",
            "threat_model": "semi-honest, at most one corrupt MPC party, private channels assumed",
            "transport": "localhost TCP, no TLS; not safe against a network/host administrator",
            "input_model": "private coordinates; public counts and candidate pool, not public-universe membership privacy",
            "worker_outputs": "final G1 public-pool IDs, numerical validity, fixed schedule, uniformly padded diagnostics",
            "diagnostic_visibility": "trusted evaluator only; reports deliberately release synthetic aggregate validation scores",
            "sampling_claim": "SLS optimization, not exact or MCMC k-DPP sampling",
            "input_reproducibility": "private fixtures, pads and audit tapes retained separately in a 0700 /tmp directory",
            "rng": "public seed for initialization only; OS cryptographic entropy for private coordinates, partition, coins and pads",
            "communication": "sum of directed MPyC bytes, including message headers; handshake/TCP/TLS excluded",
        },
    }
    _write_json(run_directory/"results.json", result)
    _write_report(run_directory/"report.md", result)
    return result, run_directory


def _write_report(path, result):
    config = result["config"]
    rows = result["trials"]
    text = ["# Three-party MPC Euclidean SLS proof of concept", "",
            f"All numerical checks passed: **{result['all_passed']}**. "
            f"{config['total_points']} points in {config['dimension']} dimensions; "
            f"25% private in three disjoint splits; gamma={config['gamma']}; jitter={config['jitter']}.", "",
            f"Private pooled context remains fixed. Select one shared G1 of {config['selection_size']} public points. "
            f"Patience={config['patience']}; execute a fixed {config['max_proposals']}-proposal circuit, "
            "freezing the protected state at the patience stop. A safety-cap result is not a convergence certificate.", "",
            "| Trial | Full logdet: plaintext | Full logdet: MPC | Final joint logdet | Max logdet error | Max swap-ratio relative error | Logical proposals | Stop | Seconds |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for row in rows:
        text.append(f"| {row['trial']} | {row['full_logdet_plain']:.9f} | {row['full_logdet_mpc']:.9f} | "
                    f"{row['final_joint_logdet']:.9f} | {row['max_logdet_absolute_error']:.3g} | "
                    f"{row['max_swap_ratio_relative_error']:.3g} | {row['iterations']} | {row['stop_reason']} | {row['elapsed_seconds']:.2f} |")
    text.extend(["", "The full determinant check includes all private and public points. SLS scores include "
                 "the private context and only the selected public subset; these have different cardinalities.", "",
                 "For each swap, $\\Delta=\\log\\det(L_{U\\cup G_1'})-\\log\\det(L_{U\\cup G_1})$ "
                 "and $r=\\exp(\\Delta)$. The ratio divides determinants, not logdet values. "
                 "SLS accepts nondecreasing scores; patience resets only above the improvement tolerance.", "",
                 "| Trial | Full determinant: plaintext | Full determinant: MPC | First proposed swap ratio | Accepted swaps | Joint-logdet improvement | MPC bytes sent |",
                 "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        text.append(f"| {row['trial']} | {row['full_determinant_plain']:.9g} | {row['full_determinant_mpc']:.9g} | "
                    f"{row['first_swap_ratio']:.9g} | {row['accepted_swaps']} | {row['joint_improvement']:.6g} | {row['sent_bytes']:,} |")
    text.extend(["", "All final selections and active swap accept/reject decisions are checked against an "
                 "independent float64 joint-kernel oracle using the same secret coin sums and initial public set.", "",
                 "## Privacy and limits", "",
                 "This uses MPyC Shamir secret sharing, not homomorphic encryption. Three processes perform "
                 "protected kernel construction, cross-client similarities, Schur elimination, and SLS. "
                 "No MPC party receives another party's coordinates or unmasked utilities. The trusted "
                 "synthetic evaluator created the fixtures and holds all diagnostic pads; it can decode "
                 "the validation outputs. Reported synthetic scores are an intentional validation release, "
                 "not a privacy-preserving application output.", "",
                 "Protection assumes at most one passive corrupt party and private authenticated channels. "
                 "Loopback transport lacks TLS; a single-host benchmark provides no confidentiality against "
                 "the host administrator. Counts, shapes, public candidates, initialization, final G1, "
                 "validity and fixed schedule are visible. Inputs here are private coordinates, unlike "
                 "the project's original public-feature/private-membership model. No differential privacy, "
                 "malicious security, dropout support, exact sampling or optimality guarantee is claimed.", "",
                 "Fixed-point exp/log approximations and guarded LDL arithmetic are tested against float64. "
                 "No jitter, clipping or automatic repair is applied. Actual communication rounds are not "
                 "instrumented; explicit synchronization barriers and framed MPyC bytes are reported separately.", "",
                 "[MPyC security model](https://github.com/lschoe/mpyc), "
                 "[MPyC 0.11 runtime](https://github.com/lschoe/mpyc/blob/v0.11/mpyc/runtime.py). "
                 "SLS and determinant-ratio reference: expansion paper Section III Algorithm 1 and Equation (4).", ""])
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(text))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mpc_euclidean.yml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config = load_mpc_config(args.config)
    output = args.output_dir or args.config.resolve().parent/config.output_directory
    result, directory = run_poc(config, output)
    print(f"Results: {directory}")
    return 0 if result["all_passed"] else 1
