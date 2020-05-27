#!/usr/bin/env python
# coding: utf-8
"""
random_id: 是否对节点标号进行随机交换
hist:  "hist":原论文方法  "none": 去掉线路二  "conv":使用卷积层来对内积图像进行分类
ifDense_GCN: 是否将多个图卷积层 的结果并在一起 综合得出 embedding
feedback:  是否启用有 error feedback 机制的 Att
"""

from train import Trainer

def main():
    trainer = Trainer()
    trainer.prepare_for_train(random_id = False, hist = "none", ifDense_GCN = False, feedback = False, batch_size = 128, epoch_num = 20)
    trainer.fit()
    #trainer.save_model("model.pkl")
    #trainer.load_model("model.pkl")
    trainer.score()
    #trainer.save_record("recordfile")


if __name__ == "__main__":
    main()
