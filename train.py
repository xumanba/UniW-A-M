"""Unified UniW(A)M training entry point."""

from uniwam.cli import load_config


def main():
    cfg, args = load_config("train")
    if args.dry_run:
        return
    from groot.vla.experiment.experiment import VLAExperiment
    from groot.vla.utils.action_args_override_utils import apply_action_overrides

    VLAExperiment(apply_action_overrides(cfg)).train()


if __name__ == "__main__":
    main()
