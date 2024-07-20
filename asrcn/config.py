class Config:
    seed = 42
    model_name = "transformer_asr_51s"
    mfcc_feature = 128
    max_sentence_len = 31
    max_mfcc_seqlen = 460
    num_attention_heads = 8
    num_layers = 12
    ffn_hidden_dim = 4096
    vocab_size = 4336
    device = 'cuda'
    learning_rate = 0.000001
    dataloader_batch_size = 64
    dropout = 0.1
    pad_token = 0
    bos_token = 1
    eos_token = 2 
    target_loss = 4.5

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}

config = Config()