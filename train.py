import numpy as np
from data import load_data
from cnn import ConvNet
from trainer import Trainer

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_data()

max_epochs = 20

network = ConvNet(input_dim=(1,128,128),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=4, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)


trainer.train()
