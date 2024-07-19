class Config:
    def __init__(self):
        self.model_name = "transformer_crossentropy"
        self.sampling_rate = 16000
        self.hop_length= 320
        self.mfcc_feature = 128
        self.d_model = 128
        self.max_sentence_len = 31
        self.max_mfcc_seqlen = 460
        self.num_attention_heads = 8
        self.num_layers = 6
        self.ffn_hidden_dim = 1024
        self.vocab_size = 4337
        self.device = 'cuda'
        self.learning_rate = 0.0001
        self.dataloader_batch_size = 32
        self.dropout = 0.1
        self.blank_token = 0
        self.bos_token = 1
        self.eos_token = 2

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    
config = Config()