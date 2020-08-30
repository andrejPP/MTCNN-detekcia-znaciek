import json
import os
import sys
from func import save_model

class LossTracker:
    """
    Class used for tracking informations about training seasion.
    It writes gathered data into a file.
    Args:
        config: dictionary containing configuration settings of current 
            training session
        file_to_write: file path where to write tracked data
    """

    def __init__(self, config: dict, file_to_write: str = None):
        self._output_file = file_to_write
        self._data = {"training": {},
                      "validation": {},
                      "best accuracy" : 0,
                      "benchmark": [],
                      "batch size": config['batch_size']}
        self._config = config
        self._bench_best_accuracy = 0
        self._best_bench = None
        self._out_folder = "outputs_folder"
        self._lr_history = []
        try:
           os.mkdir("outputs_folder")
        except FileExistsError:
            pass

    def add_loss(self, full_loss, class_loss, regresion_loss, mode):
        """
        Add either training or validation loss record.
        Args:
            full_loss: weighted sum of classification loss and regresion loss 
            class_loss: value of classification loss
            regresion_loss: value of regresion loss
            mode: type of loss, training or validation
        """

        if mode not in ['training', 'validation']:
            raise ValueError("Unknown mode->", mode)

        loss_dict = {"full loss":full_loss,
                    "classification loss": class_loss,
                    "regresion loss": regresion_loss}

        loss_number = len(self._data['training'])
        self._data[mode][loss_number] = loss_dict

    def add_bench_data(self, bench_info: dict, model, epoch):
        """
        Save result from benchmark. In case the results are the best till
        this point, save current state of the model as .pt file.
        Args:
            bench_info: dictionary contains result of benchmark
                on model
            model: current state of the model  
        """

        self._data["benchmark"].append(bench_info)

        if bench_info['accuracy'] > self._bench_best_accuracy:
            self._bench_best_accuracy = bench_info['accuracy']
            file_name = "best_model.pt"
            path = os.path.join(self._out_folder, file_name)
            save_model(model, path)
    
    def change_output_file(self, out_file):
        self._output_file = out_file

    def save_lr_history(self, current_lr, loss):
        """
        Keep records of learning rate and the corresponding loss.
        Args:
            current_lr: learning rate
            loss: loss with current learning rate
        """
        self._lr_history.append({'lr': current_lr,
                                'loss': loss})

    def write_to_file(self):
        """
        Write information about training session into file.
        """
        file_name  = "training_data.json"
        path = os.path.join(self._out_folder,file_name)

        out_data = self._data
        out_data['best accuracy'] = self._bench_best_accuracy
        out_data['Number of epochs'] = len(self._data['training'])
        out_data['LR range history'] = self._lr_history
        out_data.update(self._data)

        from func import save_json
        save_json(
            data=out_data,
            file_name=path,
            mode="w"
        )

    