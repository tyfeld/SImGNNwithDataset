#!/usr/bin/env python
# coding: utf-8

from train import Trainer

def main():
    trainer = Trainer()
    trainer.prepare_for_train(batch_size = 128, epoch_num = 40, val = 0.2)
    trainer.fit()
    #trainer.save_model("model.pkl")
    #trainer.load_model("model.pkl")
    trainer.score()
    #trainer.save_record("recordfile")


if __name__ == "__main__":
    main()