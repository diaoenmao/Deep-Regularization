import config
from train import train
from lottery_ticket_train import lottery_ticket_train
from train import run_experiments

def main():
    if config.LOTTERY_TICKET['enabled']:
        print("Running Lottery Ticket training...")
        lottery_ticket_train()
    else:
        if config.RUN_MULTIPLE_EXPERIMENTS>1:
            print(f"Running {config.NUM_EXPERIMENTS} experiments with different C values...")
            run_experiments(config.NUM_EXPERIMENTS)
        else:
            print(f"Running standard training with fixed C {config.C}")
            train(config.C)

if __name__ == "__main__":
    main()
