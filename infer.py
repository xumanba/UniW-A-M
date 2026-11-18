"""Unified offline inference entry point for the five UniW(A)M interfaces."""

from uniwam.cli import load_config


def main():
    cfg, args = load_config("infer")
    if args.dry_run:
        return
    from uniwam.inference.open_loop import run

    run(cfg)


if __name__ == "__main__":
    main()
