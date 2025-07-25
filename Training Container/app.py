import argparse
import logging
import traceback

from bsp_util import Model_Creation
from pyified_resources import Standard_Vars
from config import Config

def main():
    """
    Entry point for SageMaker training container.

    Downloads CSV using username, trains model, and uploads model .zip back to S3.
    
    the format of the .h5 files and the input that they expect is [blood sugar, TOD]
    """
    parser = argparse.ArgumentParser(description="Trigger model training for a specific user.")

    if Config.IS_LOCAL:
        parser.add_argument('--username', required=False, default = "atakanka350@gmail.com", help='The username associated with the training dataset in S3.')
    else:
        parser.add_argument('--username', required=True, help='The username associated with the training dataset in S3.')
    parser.add_argument('--num_of_models', type=int, default=10, help='Number of models to generate.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per training loop.')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training.')
    parser.add_argument('--remaining_tries', type=int, default=10, help='Maximum training attempts.')
    parser.add_argument('--num_of_layers', type=int, default=7, help='Maximum training attempts.')
    parser.add_argument('--acceptable_acc_score', type=float, default=0.1, help='Minimum required accuracy.')
    parser.add_argument('--seq_len', type=int, default=12, help='Number of input timesteps')
    parser.add_argument('--early_stop_patience', type=int, default=8, help='Patience value for early stop in RNN training.')


    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Initialize global vars
    Standard_Vars.initialize(args.seq_len)

    print("Parsed args:", vars(args))

    try:
        logger.info(f"Starting training for user: {args.username}")
        message, status_code = Model_Creation.full_model_creation(
            username=args.username,
            num_of_models=args.num_of_models,
            epochs=args.epochs,
            batch_size=args.batch_size,
            remaining_tries=args.remaining_tries,
            acceptable_acc_score=args.acceptable_acc_score,
            num_of_layers=args.num_of_layers,
            early_stop_patience=args.early_stop_patience
        )
        logger.info(f"Training completed with status {status_code}: {message}")
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Training failed: {error_details}")

if __name__ == "__main__":
    main()
