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
        
        self.parser.add_argument('--train_path', type=str, default=None, help='path to the training dataset')
        self.parser.add_argument('--valid_path', type=str, default=None, help='path to the validation dataset')
        self.parser.add_argument('--test_path', type=str, required=True, help='path to the test dataset')
        self.parser.add_argument('--test_only', 
                                 action=argparse.BooleanOptionalAction \
                                 if sys.version_info.major==3 and sys.version_info.minor>=9 else 'store_true', 
                                 help='only use the test dataset to test the model.')
        
        self.parser.add_argument('--lr', type=float, required=False, default=1e-5, 
                                 help='learning rate. not needed if only testing is required')
        self.parser.add_argument('--dropout', type=float, required=False, default=.1,
                                 help='dropout rate. not needed if only testing is required')

        self.args = self.parser.parse_args()
        if not self.args.test_only:
            if not self.args.train_path:
                raise argparse.ArgumentError(message="Test-only scheme not on, but no training dataset path found")
            if not self.args.valid_path:
                raise argparse.ArgumentError(message="Test-only scheme not on, but no valid dataset path found")
    def get_args(self):
        return self.args
    
    def add_args(self, **kwargs):
        for kw in kwargs: self.args[kw] = kwargs[kw]
    

        
        

         