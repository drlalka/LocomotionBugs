"""Run the requested pipeline stage and all following stages."""

import argparse

import config
import dots_to_rotato
import Learn
import generate_motion
import rotato_to_dots


def run(stage=config.DEFAULT_STAGE):
    if stage not in (0, 1, 2, 3):
        stage = -1
    if stage <= 0:
        print("\n[stage 0] translating dots into body and joint angular velocities")
        dots_to_rotato.main()
    if stage <= 1:
        print("\n[stage 1] training body and legs models")
        Learn.main()
    if stage <= 2:
        print("\n[stage 2] generating velocity sequences")
        generate_motion.main()
    print("\n[stage 3] rebuilding points from generated velocities")
    rotato_to_dots.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3), default=config.DEFAULT_STAGE)
    run(parser.parse_args().stage)
