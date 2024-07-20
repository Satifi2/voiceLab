class Config:
    def __init__(self):
        self.seed = 42
        self.model_name = "transformer_ctc_conv"
        self.mfcc_feature = 128
        self.max_sentence_len = 30
        self.max_mfcc_seqlen = 460
        self.num_attention_heads = 8
        self.num_layers = 12
        self.ffn_hidden_dim = 4096
        self.vocab_size = 4336
        self.device = 'cuda'
        self.learning_rate = 0.000001
        self.dataloader_batch_size = 64
        self.dropout = 0.01
        self.blank_token = 0
        self.bos_token = 1
        self.eos_token = 2 
        self.target_loss = 0.0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}

config = Config()

if __name__ == '__main__':
    print(config.to_dict())