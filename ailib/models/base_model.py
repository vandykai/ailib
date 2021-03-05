from torch import nn
import numpy as np
from ailib import tasks
from ailib.tools.utils_name_parse import parse_activation

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def loss_function(self):
        raise NotImplementedError

    def optimizer(self):
        raise NotImplementedError

    def _make_default_embedding_layer(self, **kwargs) -> nn.Module:
        """:return: an embedding module."""
        if isinstance(self.config.embedding, np.ndarray):
            self.config.embedding_input_dim = (self.config.embedding.shape[0])
            self.config.embedding_output_dim = (self.config.embedding.shape[1])
            return nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(self.config.embedding),
                freeze=self.config.embedding_freeze,
                padding_idx=self.config.padding_idx
            )
        else:
            return nn.Embedding(
                num_embeddings=self.config.embedding_input_dim,
                embedding_dim=self.config.embedding_output_dim,
                padding_idx=self.config.padding_idx
            )

    def _make_output_layer(
        self,
        in_features: int = 0
    ) -> nn.Module:
        """:return: a correctly shaped torch module for model output."""
        task = self.config.task
        if isinstance(task, tasks.Classification):
            out_features = task.num_classes
        elif isinstance(task, tasks.Ranking):
            out_features = 1
        else:
            raise ValueError(f"{task} is not a valid task type. "
                             f"Must be in `Ranking` and `Classification`.")
        if self.config.out_activation_func:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                parse_activation(self.config.out_activation_func)
            )
        else:
            return nn.Linear(in_features, out_features)

    def _make_perceptron_layer(
        self,
        in_features: int = 0,
        out_features: int = 0,
        activation: nn.Module = nn.ReLU()
    ) -> nn.Module:
        """:return: a perceptron layer."""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            activation
        )

    def _make_multi_layer_perceptron_layer(self, in_features) -> nn.Module:
        """:return: a multiple layer perceptron."""
        if not self.config.with_multi_layer_perceptron:
            raise AttributeError(
                'Parameter `with_multi_layer_perception` not set.')

        activation = parse_activation(self.config.mlp_activation_func)
        mlp_sizes = [
            in_features,
            *self.config.mlp_num_layers * [self.config.mlp_num_units],
            self.config.mlp_num_fan_out
        ]
        mlp = [
            self._make_perceptron_layer(in_f, out_f, activation)
            for in_f, out_f in zip(mlp_sizes, mlp_sizes[1:])
        ]
        return nn.Sequential(*mlp)