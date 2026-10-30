import argparse
import json

from quadsim import MODES, ROOT, TRAJECTORIES, export_csv, load_config, simulate


def parse_args():
    parser = argparse.ArgumentParser(description="Fly one quadrotor case and print its metrics.")
    parser.add_argument("--controller", choices=MODES, default=None)
    parser.add_argument("--trajectory", choices=TRAJECTORIES, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--no-disturbance", action="store_true")
    parser.add_argument("--mismatch", action="store_true")
    parser.add_argument("--td3-variant", choices=["improved", "paper"], default=None)
    return parser.parse_args()


def main():
    args, c = parse_args(), load_config()
    c.update(td3_variant=args.td3_variant or c["td3_variant"],
             enable_noise=c["enable_noise"] and not args.no_noise,
             enable_disturbance=c["enable_disturbance"] and not args.no_disturbance,
             enable_mismatch=c["enable_mismatch"] or args.mismatch)

    mode = args.controller or c["default_controller"]
    trajectory = args.trajectory or c["default_trajectory"]

    log = simulate(c, mode, trajectory, args.duration)
    export_csv(log, ROOT/"results"/f"python_{mode}_{trajectory}.csv")
    print(json.dumps(log["metrics"], indent=2))


if __name__ == "__main__":
    main()
