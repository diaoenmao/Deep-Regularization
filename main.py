import config
from train import train
from lottery_ticket_train import lottery_ticket_train

def main():
    if config.LOTTERY_TICKET['enabled']:
        print("Running Lottery Ticket training...")
        lottery_ticket_train()
    else:
        print("Running standard training...")
        train()

if __name__ == "__main__":
    main()
