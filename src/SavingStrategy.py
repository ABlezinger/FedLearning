from flwr.server.strategy import FedYogi, FedAvg, FedAdam, FedTrimmedAvg, FedProx
from typing import Callable, Dict, List, Optional, Tuple, Union, OrderedDict
import flwr as fl
import numpy as np
from flwr.common import Parameters, Scalar, FitRes
import torch
from models import VisionTransformer


def load_params(path: str = "round-5-weights.npz"):
    loaded_data = np.load(path)
    weights = [torch.tensor(loaded_data[key]) for key in loaded_data.files]
    return weights


class SaveModelStrategy(fl.server.strategy.FedYogi):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics