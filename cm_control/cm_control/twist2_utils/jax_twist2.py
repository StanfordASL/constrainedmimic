import pprint
from typing import Callable, Sequence, Optional, List

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import onnx
from onnx import numpy_helper

from cm_control.assets import CHECKPOINTS_DIR


ONNX_CHECKPOINT = CHECKPOINTS_DIR / "twist2_1017_25k.onnx"
if not ONNX_CHECKPOINT.exists():
    raise FileNotFoundError(f"Twist2 checkpoint not found at {ONNX_CHECKPOINT}")


def load_policy(debug: bool = False):

    # Load the actor model structure
    actor_model = ActorFuture(
        num_motion_observations=35,
        num_priop_observations=92,
        num_motion_steps=1,
        num_history_steps=10,
        num_future_observations=35,
        num_future_steps=1,
        motion_latent_dim=128,
        history_latent_dim=128,
        future_latent_dim=128,
        num_actions=29,
        actor_hidden_dims=[512, 512, 256, 128],
        activation=get_activation("silu"),
        layer_norm=True,
        # All else default values
    )
    # Add normalization
    model = NormalizedActorFuture(actor_model)

    # Load the onnx checkpoint into the JAX structure
    onnx_params = get_onnx_weights(ONNX_CHECKPOINT)
    jax_params = map_params_to_jax(onnx_params)

    if debug:
        # If desired, print out the parameter structure of both JAX and ONNX
        print("JAX Params")
        pprint.pprint(jax.tree_util.tree_map(lambda x: x.shape, unfreeze(jax_params)))
        print("ONNX Params")
        pprint.pprint(jax.tree_util.tree_map(lambda x: x.shape, onnx_params))

    return model, jax_params


def get_onnx_weights(path):
    model = onnx.load(path)
    weights = {}
    for initializer in model.graph.initializer:
        weights[initializer.name] = numpy_helper.to_array(initializer)
    return weights


def map_params_to_jax(onnx_params):
    jax_dict = {
        "params": {
            "Normalizer_0": {
                "mean": onnx_params["normalizer._mean"],
                "std": onnx_params["onnx::Div_163"],
            },
            "actor_cls": {
                # --- Backbone MLP ---
                "Dense_0": {
                    "kernel": onnx_params["actor.actor_backbone.0.weight"].T,
                    "bias": onnx_params["actor.actor_backbone.0.bias"],
                },
                "Dense_1": {
                    "kernel": onnx_params["actor.actor_backbone.2.weight"].T,
                    "bias": onnx_params["actor.actor_backbone.2.bias"],
                },
                "Dense_2": {
                    "kernel": onnx_params["actor.actor_backbone.4.weight"].T,
                    "bias": onnx_params["actor.actor_backbone.4.bias"],
                },
                "Dense_3": {
                    "kernel": onnx_params["actor.actor_backbone.6.weight"].T,
                    "bias": onnx_params["actor.actor_backbone.6.bias"],
                },
                "Dense_4": {
                    "kernel": onnx_params["actor.actor_backbone.9.weight"].T,
                    "bias": onnx_params["actor.actor_backbone.9.bias"],
                },
                # --- Layer Norm ---
                "LayerNorm_0": {
                    "scale": onnx_params["actor.actor_backbone.7.weight"],
                    "bias": onnx_params["actor.actor_backbone.7.bias"],
                },
                # --- Future Motion Encoder ---
                "FutureMotionEncoder_0": {
                    "Dense_0": {
                        "kernel": onnx_params[
                            "actor.future_encoder.encoder.0.weight"
                        ].T,
                        "bias": onnx_params["actor.future_encoder.encoder.0.bias"],
                    },
                    "Dense_1": {
                        "kernel": onnx_params[
                            "actor.future_encoder.encoder.3.weight"
                        ].T,
                        "bias": onnx_params["actor.future_encoder.encoder.3.bias"],
                    },
                    "Dense_2": {
                        "kernel": onnx_params[
                            "actor.future_encoder.encoder.6.weight"
                        ].T,
                        "bias": onnx_params["actor.future_encoder.encoder.6.bias"],
                    },
                },
                # --- Motion Encoder (Standard) ---
                "MotionEncoder_0": {
                    "Dense_0": {
                        "kernel": onnx_params[
                            "actor.motion_encoder.encoder.0.weight"
                        ].T,
                        "bias": onnx_params["actor.motion_encoder.encoder.0.bias"],
                    },
                    "Dense_1": {
                        "kernel": onnx_params[
                            "actor.motion_encoder.linear_output.weight"
                        ].T,
                        "bias": onnx_params["actor.motion_encoder.linear_output.bias"],
                    },
                },
                # --- Motion Encoder 1 (History/Conv Encoder) ---
                "MotionEncoder_1": {
                    "Conv_0": {
                        # ONNX (O, I, K) -> JAX (K, I, O)
                        "kernel": onnx_params[
                            "actor.history_encoder.conv_layers.0.weight"
                        ].transpose(2, 1, 0),
                        "bias": onnx_params["actor.history_encoder.conv_layers.0.bias"],
                    },
                    "Conv_1": {
                        "kernel": onnx_params[
                            "actor.history_encoder.conv_layers.2.weight"
                        ].transpose(2, 1, 0),
                        "bias": onnx_params["actor.history_encoder.conv_layers.2.bias"],
                    },
                    "Dense_0": {
                        "kernel": onnx_params[
                            "actor.history_encoder.encoder.0.weight"
                        ].T,
                        "bias": onnx_params["actor.history_encoder.encoder.0.bias"],
                    },
                    "Dense_1": {
                        "kernel": onnx_params[
                            "actor.history_encoder.linear_output.weight"
                        ].T,
                        "bias": onnx_params["actor.history_encoder.linear_output.bias"],
                    },
                },
            },
        }
    }
    return freeze(jax_dict)


# Simple normalizer to mimic the functionality of the normalizer in rsl_rl utils
class Normalizer(nn.Module):
    # features: int
    eps: float = None
    clip: float = jnp.inf

    @nn.compact
    def __call__(self, x):
        # We define these as parameters so they are stored in the variables dict
        # In inference mode, we will manually load values into these keys
        # mean = self.param("mean", nn.initializers.zeros, (self.features,))
        # std = self.param("std", nn.initializers.ones, (self.features,))
        mean = self.param("mean", lambda _, shape: jnp.zeros(shape), (x.shape[-1],))
        std = self.param("std", lambda _, shape: jnp.ones(shape), (x.shape[-1],))

        if self.eps is not None:
            norm_x = (x - mean) / (std + self.eps)
        else:
            norm_x = (x - mean) / std

        if self.clip != jnp.inf:
            norm_x = jnp.clip(norm_x, -self.clip, self.clip)

        return norm_x


# The torch version of twist2 relies on setting the xavier gain
# which is not exposed in the standard xavier jax function, but
# we can replicate this behavior via variance_scaling
def xavier_uniform_with_gain(gain):
    return nn.initializers.variance_scaling(gain**2, "fan_avg", "uniform")


def get_activation(act_name: str) -> Callable:
    """Maps string names to JAX activation functions."""
    if act_name == "elu":
        return nn.elu
    elif act_name == "selu":
        return nn.selu
    elif act_name == "relu":
        return nn.relu
    elif act_name == "crelu":
        # The original torch code mapped crelu to ReLU
        return nn.relu
    elif act_name == "lrelu":
        return nn.leaky_relu
    elif act_name == "tanh":
        return nn.tanh
    elif act_name == "sigmoid":
        return nn.sigmoid
    elif act_name == "silu":
        return nn.silu
    else:
        print("invalid activation function!")
        return None


class MotionEncoder(nn.Module):
    activation_fn: Callable
    tsteps: int
    output_size: int
    channel_size: int = 20

    @nn.compact
    def __call__(self, obs):
        """
        Args:
            obs: Input tensor of shape (batch, tsteps, input_size)
        """
        nd = obs.shape[0]
        T = self.tsteps

        # 1. Encoder Projection
        # Reshape to (Batch * Time, Features) for the initial linear projection
        x = obs.reshape((nd * T, -1))
        x = nn.Dense(features=3 * self.channel_size)(x)
        x = self.activation_fn(x)

        # 2. Reshape for Convolutional Layers
        # JAX Conv uses (Batch, Time, Channels) - no permutation needed
        x = x.reshape((nd, T, -1))

        # 3. Temporal Convolutional Blocks
        # fmt: off
        if self.tsteps == 50:
            x = nn.Conv(features=2 * self.channel_size, kernel_size=(8,), strides=(4,), padding='VALID')(x)
            x = self.activation_fn(x)
            x = nn.Conv(features=self.channel_size, kernel_size=(5,), strides=(1,), padding='VALID')(x)
            x = self.activation_fn(x)
            x = nn.Conv(features=self.channel_size, kernel_size=(5,), strides=(1,), padding='VALID')(x)
            x = self.activation_fn(x)

        elif self.tsteps == 10:
            x = nn.Conv(features=2 * self.channel_size, kernel_size=(4,), strides=(2,), padding='VALID')(x)
            x = self.activation_fn(x)
            x = nn.Conv(features=self.channel_size, kernel_size=(2,), strides=(1,), padding='VALID')(x)
            x = self.activation_fn(x)
            
        elif self.tsteps == 20:
            x = nn.Conv(features=2 * self.channel_size, kernel_size=(6,), strides=(2,), padding='VALID')(x)
            x = self.activation_fn(x)
            x = nn.Conv(features=self.channel_size, kernel_size=(4,), strides=(2,), padding='VALID')(x)
            x = self.activation_fn(x)
            
        elif self.tsteps == 1:
            # Equivalent to Flattening only
            pass
        else:
            raise ValueError(f"tsteps must be 1, 10, 20 or 50, but got {self.tsteps}")
        # fmt: on

        # FIX: Align flattening order with PyTorch.
        # PyTorch operates as (Batch, Channel, Time) and flattens that.
        # JAX currently has (Batch, Time, Channel).
        # We must transpose to (Batch, Channel, Time) before flattening
        # to ensure the elements align with the loaded weights.
        if self.tsteps > 1:
            x = x.transpose((0, 2, 1))

        # 4. Final Output Linear Layer
        # Flatten the temporal and channel dimensions
        x = x.reshape((nd, -1))
        output = nn.Dense(features=self.output_size)(x)

        return output


# In the twist2 code, the implementations for these were identical.
# So, just creating an alias to reduce repeated code
HistoryEncoder = MotionEncoder


class FutureMotionEncoder(nn.Module):
    activation_fn: Callable
    tsteps: int
    output_size: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, obs, train: bool = True):
        """
        Args:
            obs: (batch_size, tsteps, input_size + 1)
            train: Whether to apply dropout (default True)
        """
        batch_size = obs.shape[0]

        # 1. Separate mask indicator from observations
        # Torch: obs[:, :, :-1] -> JAX handles slicing the same way
        future_obs = obs[:, :, :-1]
        # mask_indicator = obs[:, :, -1] # Unused in original code, kept for reference

        # 2. Flatten future observations
        # (batch_size, tsteps * input_size)
        flattened = future_obs.reshape((batch_size, -1))

        # 3. Simple MLP encoder with custom initialization
        # We use a gain of 0.5 as specified in your xavier_uniform_ call
        # kernel_init = nn.initializers.xavier_uniform(gain=0.5)
        kernel_init = xavier_uniform_with_gain(gain=0.5)
        bias_init = nn.initializers.zeros

        x = nn.Dense(features=256, kernel_init=kernel_init, bias_init=bias_init)(
            flattened
        )
        x = self.activation_fn(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = nn.Dense(features=128, kernel_init=kernel_init, bias_init=bias_init)(x)
        x = self.activation_fn(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        output = nn.Dense(
            features=self.output_size, kernel_init=kernel_init, bias_init=bias_init
        )(x)

        return output


class ActorFuture(nn.Module):
    # Core dimensions
    num_motion_observations: int
    num_priop_observations: int
    num_motion_steps: int
    num_history_steps: int
    num_future_observations: int
    num_future_steps: int

    # Latent dimensions
    motion_latent_dim: int
    history_latent_dim: int
    future_latent_dim: int

    # Architecture config
    num_actions: int
    actor_hidden_dims: List[int]
    activation: Callable
    layer_norm: bool = False
    use_history_encoder: bool = True
    use_motion_encoder: bool = True
    future_dropout: float = 0.1
    tanh_encoder_output: bool = False

    @nn.compact
    def __call__(self, obs, train: bool = True):
        # 1. Dimension Calculations (Internal)
        num_single_motion = self.num_motion_observations // self.num_motion_steps
        num_single_history_step = (
            self.num_motion_observations + self.num_priop_observations
        )
        num_single_future = (
            (self.num_future_observations // self.num_future_steps)
            if self.num_future_observations > 0
            else 0
        )

        # 2. Slice Observations
        # Current = motion + proprioception
        current_size = self.num_motion_observations + self.num_priop_observations
        motion_obs = obs[:, : self.num_motion_observations]
        single_motion_obs = obs[:, :num_single_motion]
        priop_obs = obs[:, self.num_motion_observations : current_size]

        # History
        history_start = current_size
        history_size = self.num_history_steps * num_single_history_step
        history_obs = obs[:, history_start : history_start + history_size]

        # Future
        future_start = history_start + history_size
        future_obs = obs[:, future_start : future_start + self.num_future_observations]

        # 3. Component Encoders
        # Motion Encoder
        if self.use_motion_encoder:
            motion_latent = MotionEncoder(
                activation_fn=self.activation,
                # input_size=num_single_motion,
                tsteps=self.num_motion_steps,
                output_size=self.motion_latent_dim,
            )(motion_obs)
        else:
            motion_latent = motion_obs  # Identity logic

        # History Encoder
        if self.use_history_encoder:
            history_latent = HistoryEncoder(
                activation_fn=self.activation,
                tsteps=self.num_history_steps,
                output_size=self.history_latent_dim,
            )(history_obs)
        else:
            history_latent = history_obs

        # Future Encoder
        if num_single_future > 0:
            future_obs_reshaped = future_obs.reshape(
                (-1, self.num_future_steps, num_single_future)
            )
            future_latent = FutureMotionEncoder(
                activation_fn=self.activation,
                tsteps=self.num_future_steps,
                output_size=self.future_latent_dim,
                dropout_rate=self.future_dropout,
            )(future_obs_reshaped, train=train)
        else:
            future_latent = jnp.zeros((obs.shape[0], self.future_latent_dim))

        # 4. Concatenate for Backbone
        backbone_input = jnp.concatenate(
            [
                single_motion_obs,
                priop_obs,
                motion_latent,
                history_latent,
                future_latent,
            ],
            axis=-1,
        )

        # 5. Actor Backbone MLP
        x = backbone_input
        for i, hidden_dim in enumerate(self.actor_hidden_dims):
            gain = 0.5 if i == 0 else 1.0
            x = nn.Dense(
                features=hidden_dim,
                kernel_init=xavier_uniform_with_gain(gain=gain),
            )(x)

            if self.layer_norm and i == len(self.actor_hidden_dims) - 1:
                x = nn.LayerNorm()(x)

            x = self.activation(x)

        # 6. Action Head
        actions = nn.Dense(
            features=self.num_actions,
            kernel_init=xavier_uniform_with_gain(
                gain=0.1
            ),  # Small weights for action head
        )(x)

        if self.tanh_encoder_output:
            actions = nn.tanh(actions)

        return actions


class ActorCriticFuture(nn.Module):
    # Dimensions
    num_observations: int  # 1432
    num_critic_observations: int  # 1734
    num_motion_observations: int  # 35
    num_motion_steps: int  # 1
    num_priop_observations: int  # 92
    num_history_steps: int  # 10
    num_actions: int  # 29

    # Network Configs
    actor_hidden_dims: List[int]  # [512, 512, 256, 128]
    critic_hidden_dims: List[int]  # [512, 512, 256, 128]
    motion_latent_dim: int = 128
    history_latent_dim: int = 128
    future_latent_dim: int = 128
    activation: str = "silu"

    # Noise/Std
    init_noise_std: float = 1.0
    fix_action_std: bool = False

    # Future Encoder Params
    future_dropout: float = 0.1
    layer_norm: bool = False  # TRUE!

    # Passed through via kwargs in torch
    num_future_observations: int = None
    num_future_steps: int = 10
    tanh_encoder_output: bool = False

    def setup(self):
        # 1. Initialization Logic
        activation_fn = get_activation(self.activation)

        # Calculate derived dimensions
        single_obs_size = self.num_motion_observations + self.num_priop_observations
        expected_history_size = self.num_history_steps * single_obs_size

        if self.num_future_observations is None:
            num_future_obs = max(
                0, self.num_observations - single_obs_size - expected_history_size
            )
        else:
            num_future_obs = self.num_future_observations

        # 2. Sub-modules
        self.actor_net = ActorFuture(
            # num_observations=self.num_observations,
            num_actions=self.num_actions,
            num_motion_observations=self.num_motion_observations,
            num_priop_observations=self.num_priop_observations,
            num_motion_steps=self.num_motion_steps,
            num_future_observations=num_future_obs,
            num_future_steps=self.num_future_steps,
            num_history_steps=self.num_history_steps,
            motion_latent_dim=self.motion_latent_dim,
            future_latent_dim=self.future_latent_dim,
            history_latent_dim=self.history_latent_dim,
            actor_hidden_dims=self.actor_hidden_dims,
            activation=activation_fn,
            layer_norm=self.layer_norm,
            future_dropout=self.future_dropout,
            use_history_encoder=True,
            use_motion_encoder=True,
            tanh_encoder_output=self.tanh_encoder_output,
        )

        # Critic Logic
        num_single_motion = self.num_motion_observations // self.num_motion_steps
        self.critic_motion_encoder = MotionEncoder(
            activation_fn=activation_fn,
            # input_size=num_single_motion,
            tsteps=self.num_motion_steps,
            output_size=self.motion_latent_dim,
        )

        # 3. Action Noise (Log Std)
        # In JAX/Flax, we define the parameter here.
        # If fix_action_std is True, we will handle it during the update step (not learning it).
        self.action_std_param = self.param(
            "log_std",
            lambda key: jnp.ones((self.num_actions,)) * jnp.log(self.init_noise_std),
        )

    def __call__(self, observations, train: bool = True):
        """Standard forward pass returning the action mean."""
        return self.actor_net(observations, train=train)

    def get_value(self, critic_observations):
        """Equivalent to evaluate() in the torch code."""
        num_single_motion = self.num_motion_observations // self.num_motion_steps

        motion_obs = critic_observations[:, : self.num_motion_observations]
        motion_single_obs = critic_observations[:, :num_single_motion]

        # Encode motion for critic
        motion_latent = self.critic_motion_encoder(motion_obs)

        # Critic input: [privileged_info, single_motion, motion_latent]
        backbone_input = jnp.concatenate(
            [
                critic_observations[:, self.num_motion_observations :],
                motion_single_obs,
                motion_latent,
            ],
            axis=-1,
        )

        # Critic MLP Backbone
        x = backbone_input
        activation_fn = get_activation(self.activation)

        for i, hidden_dim in enumerate(self.critic_hidden_dims):
            x = nn.Dense(hidden_dim)(x)
            if self.layer_norm and i == len(self.critic_hidden_dims) - 2:
                x = nn.LayerNorm()(x)
            x = activation_fn(x)

        value = nn.Dense(1)(x)
        return value

    def get_dist(self, observations, train: bool = True):
        """Returns mean and std for distribution construction."""
        mean = self.actor_net(observations, train=train)
        # If std is fixed, we would typically handle that in the loss function
        # or stop gradients on the parameter.
        std = jnp.exp(self.action_std_param)
        return mean, std


class NormalizedActorFuture(nn.Module):
    actor_cls: ActorFuture

    @nn.compact
    def __call__(self, obs, train: bool = True):
        # 1. Apply Normalization
        # This creates 'Normalizer_0' in the parameter tree
        norm_obs = Normalizer()(obs)

        # 2. Pass normalized observations to the actual Actor
        # This creates 'actor' in the parameter tree
        return self.actor_cls(norm_obs, train)


def main():
    # Load the ActorCriticFuture with the standard hyperparams from Yanjie
    # acf = ActorCriticFuture(
    #     num_observations=1432,
    #     num_critic_observations=1734,
    #     num_motion_observations=35,
    #     num_motion_steps=1,
    #     num_priop_observations=92,
    #     num_history_steps=10,
    #     num_actions=29,
    #     actor_hidden_dims=[512, 512, 256, 128],
    #     critic_hidden_dims=[512, 512, 256, 128],
    #     layer_norm=True,
    #     num_future_observations=35,
    #     num_future_steps=1,
    #     # All else default values
    # )
    # print("Loaded ACF")
    af = ActorFuture(
        num_motion_observations=35,
        num_priop_observations=92,
        num_motion_steps=1,
        num_history_steps=10,
        num_future_observations=35,
        num_future_steps=1,
        motion_latent_dim=128,
        history_latent_dim=128,
        future_latent_dim=128,
        num_actions=29,
        actor_hidden_dims=[512, 512, 256, 128],
        activation=get_activation("silu"),
        layer_norm=True,
        # All else default values
    )
    naf = NormalizedActorFuture(af)
    print("Loaded NAF")


if __name__ == "__main__":
    main()
