"""Run the point -> joint-velocity -> learned motion -> point pipeline."""

import argparse

import dots_to_rotato
import Learn
import generation


# 0: extract training features, train, generate
# 1: train, generate
# 2: generate only (requires rotato_gru.pt)
STAGE = 0


def run(stage=STAGE, seed_rows=100):
    if stage not in (0, 1, 2):
        raise ValueError("stage must be 0, 1, or 2")
    if stage <= 0:
        print("\n[stage 0/2] Extracting joint angular velocities")
        dots_to_rotato.main()
    if stage <= 1:
        print("\n[stage 1/2] Training one joint-velocity model")
        Learn.main()
    print("\n[stage 2/2] Generating angular velocities and reconstructing points")
    generation.main(seed_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=STAGE)
    parser.add_argument("--seed-rows", type=int, default=100,
                        help="number of initial source rows used to seed generation")
    args = parser.parse_args()
    run(args.stage, args.seed_rows)
