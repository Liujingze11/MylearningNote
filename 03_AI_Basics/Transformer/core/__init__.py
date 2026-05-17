"""
Transformer 核心组件

包含：
- 注意力机制 (attention)
- 前馈网络 (feed_forward)
- 层归一化 (layer_norm)
- 编码器层 (encoder_layer)
- 解码器层 (decoder_layer)
- 编码器 (encoder)
- 解码器 (decoder)
- Transformer 模型 (transformer)
"""

from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward, PositionWiseFeedForwardGELU
from .layer_norm import LayerNorm
from .encoder_layer import EncoderLayer, EncoderLayerPreLN
from .decoder_layer import DecoderLayer, DecoderLayerPreLN
from .encoder import Encoder, EncoderWithoutDuplicateScaling
from .decoder import Decoder
from .transformer import Transformer, create_transformer
from .masks_utils import (
    create_padding_mask,
    create_causal_mask,
    create_decoder_mask,
    count_parameters,
    print_model_info,
    generate_square_subsequent_mask,
    LabelSmoothing
)

__all__ = [
    # 基础组件
    'MultiHeadAttention',
    'PositionWiseFeedForward',
    'PositionWiseFeedForwardGELU',
    'LayerNorm',

    # 层组件
    'EncoderLayer',
    'EncoderLayerPreLN',
    'DecoderLayer',
    'DecoderLayerPreLN',

    # 完整模型
    'Encoder',
    'EncoderWithoutDuplicateScaling',
    'Decoder',
    'Transformer',
    'create_transformer',

    # 工具函数
    'create_padding_mask',
    'create_causal_mask',
    'create_decoder_mask',
    'count_parameters',
    'print_model_info',
    'generate_square_subsequent_mask',
    'LabelSmoothing',
]
