# Neural ODE implementation

import torch
import torch.nn as nn


class ODEBlock(nn.Module):
    """
    y' = dy/dt

    Args:
        nn (_type_): _description_
    """

    def __init__(self, variant, sigma, m, p=None):
        super(ODEBlock, self).__init__()
        self.variant = variant
        self.m = m
        self.p = p if p else m  # Default to m if p is None
        self.sigma = sigma  # activation function

        if self.variant == "standard":
            self.W = nn.Parameter(torch.randn(self.m, self.m))
            self.b = nn.Parameter(torch.randn(self.m))
        elif self.variant == "UT":
            self.U = nn.Parameter(torch.randn(self.p, self.m))
            self.W = nn.Parameter(torch.randn(self.p, self.m))
            self.b = nn.Parameter(torch.randn(self.p))
        else:
            raise ValueError("Invalid variant type")

    def forward(self, y):
        # NOTE: Need to traspose since y.shape = batch size x m

        if self.variant == "standard":
            Wy = self.W @ y.T
            Wyb = Wy.T + self.b
            return Wyb
        elif self.variant == "UT":
            Wy = self.W @ y.T
            Wyb = Wy.T + self.b
            UWyb = self.U.T @ self.sigma(Wyb.T)
            return UWyb.T


class NeuralODE(nn.Module):
    """

    FIXME: What about multiple steps?

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        hidden_internal_dim,
        output_dim,
        num_hidden_layers,
        sigma,
        method,
        variant,
    ):
        super(NeuralODE, self).__init__()
        self.method = method
        self.variant = variant
        self.num_hidden_layers = num_hidden_layers  # how many time steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_internal_dim = hidden_internal_dim
        self.output_dim = output_dim
        self.sigma = sigma()
        self.h = 0.05  # constant step size

        self.input_layer = nn.Linear(
            in_features=input_dim, out_features=hidden_dim, bias=False
        )

        self.ode = ODEBlock(
            variant=variant, sigma=self.sigma, m=hidden_dim, p=hidden_internal_dim
        )  # y'

        self.classifier = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        # logits to prob
        self.logits_to_prob = nn.Sigmoid()

    def forward(self, x):
        # track transformation of the data
        # samples in batch - num hidden units - num hidden layers + 1 # FIXME: Do i need +1??
        x_transformed = torch.empty(
            x.shape[0], self.hidden_dim, self.num_hidden_layers + 1
        )

        # input layer (increase dimension)
        x = self.input_layer(x)

        # save tranformation of x at time t0 (after dim increase)
        x_transformed[:, :, 0] = x

        # ODE solver
        for t in range(self.num_hidden_layers):  # how many time steps
            if self.method == "neural":
                x = self.ode(x)
            elif self.method == "euler":
                x = x + self.h * self.ode(x)
            elif self.method == "rk4":
                k1 = self.ode(x)
                k2 = self.ode(x + self.h * k1 / 2.0)
                k3 = self.ode(x + self.h * k2 / 2.0)
                k4 = self.ode(x + self.h * k3)

                x = x + self.h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

            else:
                raise ValueError("Invalid method type")

            # save tranformation of x for each time step
            x_transformed[:, :, t + 1] = x

        # outout layers
        # classifier to binary channel
        # and from logits to probabilities
        x = self.classifier(x)
        x = self.logits_to_prob(x)

        return x.squeeze(), x_transformed
