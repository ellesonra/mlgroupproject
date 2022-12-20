from argparse import ArgumentParser
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from lib.train_data import TrainData
from lib.benchmark import Benchmark

def main():
    
    parser = ArgumentParser(description="Grouparn v2.0")
    
    modes =[
        
        'train',
        'benchmark'
    ]
    
    parser.add_argument("--input-x-train-file", help="input x train file", type=str, required=True)     
    parser.add_argument("--input-y-train-file", help="input y train file", type=str, required=True)     
    parser.add_argument("--input-test-file", help="input x test file", type=str, required=True)     
    parser.add_argument("--input-labels-file", help="input y test file", type=str, required=True)     
    parser.add_argument("--mode", help="Type of action for MLTool", type=str, choices=modes, required=True)
    parser.add_argument("--output-model-file", help = "Model type", type=str, required=True)
    parser.add_argument("--input-model-file", help = "Model type", type=str, required=True)
     
    
    args = parser.parse_args()   
       
    mode                = args.mode           
    input_x_train_file  = args.input_x_train_file
    input_y_train_file  = args.input_y_train_file
    input_test_file   = args.input_test_file
    input_labels_file   = args.input_labels_file 
    output_model_file   = args.output_model_file
    input_model_file   = args.input_model_file
    

    if mode == 'train':
        
        params ={


            'input_x_train':           input_x_train_file,
            'input_y_train' :           input_y_train_file,
            'output_model'  :            output_model_file

        }
        
    
        cmd = TrainData(params)
        cmd.execute()
    
    
    elif mode == 'benchmark':
        
        params = {
            'input_model_file'  : input_model_file,
            'input_test_file'   : input_test_file,
            'input_labels_file' : input_labels_file
        }
        
        
        cmd = Benchmark(params)
        cmd.execute()
        
        
        
if __name__ == '__main__':
    main()

