from typing import Optional, Union
import numpy as np
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes
from flwr.server.strategy import FedYogi, FedAvg, FedAdam, FedTrimmedAvg, FedProx





class SaveModelYogi(FedYogi):
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
            np.savez(f"assets/models/latest_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
    

class SaveModelFedAvg(FedAvg):
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
            np.savez(f"assets/models/latest_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
    

class SaveModelAdam(FedAdam):
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
            np.savez(f"assets/models/latest_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
    

class SaveModelTrimmedAvg(FedTrimmedAvg):
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
            np.savez(f"assets/models/latest_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
    

class SaveModelProx(FedProx):
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
            np.savez(f"assets/models/latest_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics



