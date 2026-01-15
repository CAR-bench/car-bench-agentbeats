import argparse
import asyncio
import os, sys, time, subprocess, shlex, signal
from pathlib import Path
import tomllib
import httpx
from dotenv import load_dotenv

from a2a.client import A2ACardResolver

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
sys.path.pop(0)


# Load .env for local development only (doesn't override existing env vars from GitHub Actions)
load_dotenv(override=False)
logger = configure_logger(role="orchestrator")


async def wait_for_agents(cfg: dict, timeout: int = 30, evaluate_only: bool = False) -> bool:
    """Wait for all agents to be healthy and responding."""
    endpoints = []

    # When in evaluate-only mode, only check the green agent (host)
    # Participants are checked by the green agent itself via Docker network
    if evaluate_only:
        endpoints.append(f"http://{cfg['green_agent']['host']}:{cfg['green_agent']['port']}")
    else:
        # In normal mode, check all agents
        for p in cfg["participants"]:
            endpoints.append(f"http://{p['host']}:{p['port']}")
        endpoints.append(f"http://{cfg['green_agent']['host']}:{cfg['green_agent']['port']}")

    if not endpoints:
        return True  # No agents to wait for

    logger.info(f"Waiting for {len(endpoints)} agent(s) to be ready", num_agents=len(endpoints))
    start_time = time.time()

    async def check_endpoint(endpoint: str) -> bool:
        """Check if an endpoint is responding by fetching the agent card."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=endpoint)
                await resolver.get_agent_card()
                return True
        except Exception as e:
            # Log the specific error for debugging
            logger.debug(f"Agent check failed", endpoint=endpoint, error=f"{type(e).__name__}: {str(e)[:100]}")
            return False

    while time.time() - start_time < timeout:
        ready_status = {}
        for endpoint in endpoints:
            is_ready = await check_endpoint(endpoint)
            ready_status[endpoint] = is_ready
        
        ready_count = sum(ready_status.values())

        if ready_count == len(endpoints):
            logger.info("All agents ready", num_agents=len(endpoints))
            return True

        # Log status for agents that aren't ready yet
        for endpoint, is_ready in ready_status.items():
            if not is_ready:
                logger.debug("Agent not ready yet", endpoint=endpoint)
        
        await asyncio.sleep(1)

    logger.error("Timeout waiting for agents", ready=ready_count, total=len(endpoints), timeout=timeout)
    return False


def parse_toml(scenario_path: str) -> dict:
    path = Path(scenario_path)
    if not path.exists():
        logger.error("Scenario file not found", path=str(path))
        sys.exit(1)

    data = tomllib.loads(path.read_text())
    logger.debug("Loaded scenario file", path=str(path))

    def host_port(ep: str):
        s = (ep or "")
        s = s.replace("http://", "").replace("https://", "")
        s = s.split("/", 1)[0]
        host, port = s.split(":", 1)
        return host, int(port)

    green_ep = data.get("green_agent", {}).get("endpoint", "")
    g_host, g_port = host_port(green_ep)
    green_cmd = data.get("green_agent", {}).get("cmd", "")

    parts = []
    for p in data.get("participants", []):
        if isinstance(p, dict) and "endpoint" in p:
            h, pt = host_port(p["endpoint"])
            parts.append({
                "role": str(p.get("role", "")),
                "host": h,
                "port": pt,
                "cmd": p.get("cmd", "")
            })

    cfg = data.get("config", {})
    return {
        "green_agent": {"host": g_host, "port": g_port, "cmd": green_cmd},
        "participants": parts,
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser(description="Run agent scenario")
    parser.add_argument("scenario", help="Path to scenario TOML file")
    parser.add_argument("--show-logs", action="store_true",
                        help="Show agent stdout/stderr")
    parser.add_argument("--serve-only", action="store_true",
                        help="Start agent servers only without running evaluation")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Run evaluation only without starting agent servers")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout in seconds to wait for agents to be ready (default: 30)")
    parser.add_argument("--output", type=str, default="output/results.json",
                        help="Path to save results JSON file (default: output/results.json)")
    args = parser.parse_args()

    # Validate that --serve-only and --evaluate-only are not both set
    if args.serve_only and args.evaluate_only:
        logger.error("Cannot use both --serve-only and --evaluate-only flags")
        sys.exit(1)

    cfg = parse_toml(args.scenario)

    sink = None if args.show_logs or args.serve_only else subprocess.DEVNULL
    parent_bin = str(Path(sys.executable).parent)
    base_env = os.environ.copy()
    base_env["PATH"] = parent_bin + os.pathsep + base_env.get("PATH", "")

    procs = []
    try:
        # start participant agents (skip if --evaluate-only)
        if not args.evaluate_only:
            for p in cfg["participants"]:
                cmd_args = shlex.split(p.get("cmd", ""))
                if cmd_args:
                    logger.info(
                        "Starting participant agent",
                        role=p['role'],
                        host=p['host'],
                        port=p['port']
                    )
                    procs.append(subprocess.Popen(
                        cmd_args,
                        env=base_env,
                        stdout=sink, stderr=sink,
                        text=True,
                        start_new_session=True,
                    ))

        # start host (skip if --evaluate-only)
        if not args.evaluate_only:
            green_cmd_args = shlex.split(cfg["green_agent"].get("cmd", ""))
            if green_cmd_args:
                logger.info(
                    "Starting green agent",
                    host=cfg['green_agent']['host'],
                    port=cfg['green_agent']['port']
                )
                procs.append(subprocess.Popen(
                    green_cmd_args,
                    env=base_env,
                    stdout=sink, stderr=sink,
                    text=True,
                    start_new_session=True,
                ))

        # Wait for all agents to be ready
        if not asyncio.run(wait_for_agents(cfg, timeout=args.timeout, evaluate_only=args.evaluate_only)):
            logger.error("Not all agents became ready, exiting")
            return

        logger.info("Agents started successfully", mode="serve" if args.serve_only else "evaluate")
        if args.serve_only:
            while True:
                for proc in procs:
                    if proc.poll() is not None:
                        logger.warning("Agent exited", exit_code=proc.returncode)
                        break
                    time.sleep(0.5)
        else:
            logger.info("Starting evaluation client", output=args.output)
            client_proc = subprocess.Popen(
                [sys.executable, "-m", "agentbeats.client_cli", args.scenario, args.output],
                env=base_env,
                start_new_session=True,
            )
            procs.append(client_proc)
            client_proc.wait()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")

    finally:
        logger.info("Shutting down agents")
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(1)
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
