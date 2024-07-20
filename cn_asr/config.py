class Config:
    def __init__(self):
        self.seed=42
        self.model_name = "transformer_ctc"
        self.sampling_rate = 22000
        self.hop_length= 1000
        self.mfcc_feature = 128
        self.d_model = 128
        self.num_attention_heads = 8
        self.num_layers = 16
        self.ffn_hidden_dim = 4096
        self.vocab_size = 4337
        self.device = 'cuda'
        self.learning_rate = 0.0000000000005
        self.dataloader_batch_size = 64
        self.dropout = 0
        self.blank_token = 0
        self.bos_token = 1
        self.eos_token = 2

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    
config = Config()