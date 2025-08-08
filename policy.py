import torch
from pufferlib.models import Default, LSTMWrapper


class Policy(torch.nn.Module):
    """Custom policy for the Four Rooms environment.

    This just reuses pufferlib's Default policy with an LSTM wrapper.
    It's a simple, stable interface that can be extended later.
    """

    def __init__(
        self,
        env,
        hidden_size: int = 128,
        lstm_hidden_size: int = 128,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        # Backbone feature extractor from pufferlib
        backbone = Default(env, hidden_size=hidden_size)

        # Recurrent wrapper from pufferlib
        self.inner = LSTMWrapper(
            env,
            backbone,
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
        )

        # Surface attributes expected by pufferlib trainer
        self.hidden_size = getattr(self.inner, "hidden_size", lstm_hidden_size)
        self.is_continuous = getattr(self.inner, "is_continuous", False)

        if device is not None:
            self.to(device)

    def forward_eval(self, observations, state=None):
        # Ensure state dict has the keys expected by LSTMWrapper
        if state is None:
            state = {}
        state.setdefault("lstm_h", None)
        state.setdefault("lstm_c", None)
        return self.inner.forward_eval(observations, state)

    # pufferlib calls forward during training
    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


