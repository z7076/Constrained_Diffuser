import argparse # Import the argparse library

if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.evaluate_inv_parallel import evaluate
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep

    # 1. Set up argument parser
    parser = argparse.ArgumentParser(description="Run evaluation sweep.")
    parser.add_argument(
        '--sweep_file', # Argument name
        type=str,       # Expected type
        default="default_inv.jsonl", # Default value if not provided
        help='Path to the JSONL file defining the parameter sweep.' # Help text
    )
    # Add other arguments here if needed (e.g., --jaynes_mode)
    # parser.add_argument('--jaynes_mode', type=str, default="local", help='Jaynes execution mode (e.g., local, slurm)')


    # 2. Parse arguments from the command line
    args = parser.parse_args()

    # 3. Use the parsed arguments
    sweep_file_path = args.sweep_file
    # jaynes_mode = args.jaynes_mode # Example if you added another argument

    # logger.info(f"Loading sweep from: {sweep_file_path}")
    sweep = Sweep(RUN, Config).load(sweep_file_path) # Use the provided file path

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        # jaynes.config(jaynes_mode) # Example: Use parsed mode
        jaynes.config("local") # Keep it simple for now
        thunk = instr(evaluate, **kwargs)
        jaynes.run(thunk)

    jaynes.listen(timeout=6) # Set a timeout for jaynes.listen