import argparse
import sys

class GraformerArgumentParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--masked_encoder", type=str, required=False, default="bert-base-multilingual-cased", 
                                 help="model name or path to model checkpoint for the masked encoder, based on Hugging Face's names")
        self.parser.add_argument("--causal_decoder", type=str, required=False, default="openai-gpt",
                                  help="model name or path to model checkpoint for the causal decoder, based on Hugging Face's names")
        self.parser.add_argument('--from_checkpoint', type=str, required=False, default=None,
                                 help='path to the model checkpoint')

        self.parser.add_argument('--d_model', type=int, required=False, default=768)
        self.parser.add_argument('--n_heads', type=int, required=False, default=12)
        self.parser.add_argument('--dff', required=False, type=int, default=3072)
        
        self.parser.add_argument('--train_path_src', type=str, default=None, help='path to the training dataset')
        self.parser.add_argument('--train_path_tgt', type=str, default=None, help='path to the training dataset')
        self.parser.add_argument('--valid_path_src', type=str, default=None, help='path to the validation dataset')
        self.parser.add_argument('--valid_path_tgt', type=str, default=None, help='path to the training dataset')
        self.parser.add_argument('--test_path_src', type=str, required=True, help='path to the test dataset')
        self.parser.add_argument('--test_path_tgt', type=str, required=True, help='path to the test dataset')
        self.parser.add_argument('--test_only', 
                                 action=argparse.BooleanOptionalAction \
                                 if sys.version_info.major==3 and sys.version_info.minor>=9 else 'store_true', 
                                 help='only use the test dataset to test the model.')
        
        self.parser.add_argument('--lr', type=float, required=False, default=1e-3, 
                                 help='learning rate. not needed if only testing is required')
        self.parser.add_argument('--dropout', type=float, required=False, default=.1,
                                 help='dropout rate. not needed if only testing is required')
        self.parser.add_argument('--epoch', type=int, default=10, required=False, help='number of training epochs')
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--compile', action=argparse.BooleanOptionalAction \
                                 if sys.version_info.major==3 and sys.version_info.minor>=9 else 'store_true')

        self.args = self.parser.parse_args()
        if not self.args.test_only:
            if not self.args.train_path_src or not self.args.train_path_tgt:
                raise argparse.ArgumentError(message="Test-only scheme not on, but no training dataset path found")
            if not self.args.valid_path_src or not self.args.valid_path_tgt:
                raise argparse.ArgumentError(message="Test-only scheme not on, but no valid dataset path found")
        else:
            if not self.args.from_checkpoint:
                raise argparse.ArgumentError(message="No checkpoints to test")
    def get_args(self):
        return self.args
    
    def add_args(self, **kwargs):
        for kw in kwargs: self.args[kw] = kwargs[kw]
    

        
        

         