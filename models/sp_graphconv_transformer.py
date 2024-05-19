# from torch import nn
#
# import params
# from models.nn_layers import GraphConvEncoder, TransformerEncoder, Classifier
#
#
# class SPGraphConvTransformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.graphconv_encoder = GraphConvEncoder(
#             d_model=config['d_model'],
#             dropout=config['dropout'],
#             use_relu_act=config['use_relu_act'],
#         )
#         self.transformer_encoder = TransformerEncoder(
#             d_model=config['d_model'],
#             nhead=config['nhead'],
#             num_layers=config['num_layers']
#         )
#         self.classifier = Classifier(
#             d_model=config['d_model'],
#             num_class=len(params.SP_LABELS)
#         )
#
#     def forward(self, x):
#         x = self.graphconv_encoder(x)
#         x = self.transformer_encoder(x)
#         x = self.classifier(x)
#         return x
